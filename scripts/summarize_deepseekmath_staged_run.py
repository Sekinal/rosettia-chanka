"""Summarize one staged DeepSeekMath-language run.

The staged workflow is:

1. paid frontier generation/audit report
2. SFT seed from accepted frontier JSONL
3. GSPO policy iteration from the SFT seed

This script reads those artifacts, reports evidence, and recommends the next
command without relying on shell scrollback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", type=Path, default=None)
    parser.add_argument("--sft-dir", type=Path, default=None)
    parser.add_argument("--gspo-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--discover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When stage directories are omitted, discover latest matching artifacts under --output-root.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--fail-if-blocked", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def path_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir() if exists else False,
        "is_file": path.is_file() if exists else False,
        "bytes": path.stat().st_size if exists and path.is_file() else 0,
    }


def metric(metrics: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(metrics, dict):
        return default
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def newest_dir_with_file(output_root: Path, filename: str) -> Path | None:
    if not output_root.exists():
        return None
    candidates = [path.parent for path in output_root.rglob(filename) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / filename).stat().st_mtime)


def newest_cycle_dir(output_root: Path, stage: str) -> Path | None:
    if not output_root.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for manifest_path in output_root.rglob("cycle_manifest.json"):
        manifest = load_json(manifest_path)
        if isinstance(manifest, dict) and manifest.get("stage") == stage:
            candidates.append((manifest_path.stat().st_mtime, manifest_path.parent))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def discover_dirs(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    if not args.discover:
        return args.frontier_dir, args.sft_dir, args.gspo_dir

    frontier_dir = args.frontier_dir
    if frontier_dir is None:
        frontier_dir = newest_dir_with_file(args.output_root, "deepseek_v4_pro_paid_smoke_gate.json")
    if frontier_dir is None:
        frontier_dir = newest_dir_with_file(args.output_root, "deepseek_v4_pro_preapi_readiness.json")

    sft_dir = args.sft_dir
    if sft_dir is None:
        sft_dir = newest_cycle_dir(args.output_root, "sft_seed")

    gspo_dir = args.gspo_dir
    if gspo_dir is None:
        gspo_dir = newest_cycle_dir(args.output_root, "initial_gspo")

    return frontier_dir, sft_dir, gspo_dir


def frontier_paths(frontier_dir: Path | None) -> dict[str, Path | None]:
    if frontier_dir is None:
        return {
            "report_json": None,
            "report_md": None,
            "accepted_jsonl": None,
            "failures_jsonl": None,
            "summary_json": None,
            "paid_gate_json": None,
            "preapi_json": None,
        }
    return {
        "report_json": frontier_dir / "deepseek_v4_pro_thinking_report.json",
        "report_md": frontier_dir / "deepseek_v4_pro_thinking_report.md",
        "accepted_jsonl": frontier_dir / "deepseek_v4_pro_thinking_sft.jsonl",
        "failures_jsonl": frontier_dir / "deepseek_v4_pro_thinking_failures.jsonl",
        "summary_json": frontier_dir / "deepseek_v4_pro_thinking_sft.summary.json",
        "paid_gate_json": frontier_dir / "deepseek_v4_pro_paid_smoke_gate.json",
        "preapi_json": frontier_dir / "deepseek_v4_pro_preapi_readiness.json",
    }


def summarize_frontier(frontier_dir: Path | None) -> dict[str, Any]:
    paths = frontier_paths(frontier_dir)
    report = load_json(paths["report_json"])
    paid_gate = load_json(paths["paid_gate_json"])
    preapi = load_json(paths["preapi_json"])
    metrics = report.get("gate_metrics") if isinstance(report, dict) else {}
    summary = report.get("summary") if isinstance(report, dict) else {}
    gate_passed = bool(paid_gate.get("passed")) if isinstance(paid_gate, dict) else None
    accepted_rows = metric(metrics, "written_rows")
    expected_coverage = metric(metrics, "expected_primitive_coverage")
    ready = bool(report) and accepted_rows > 0 and gate_passed is True
    preapi_ready = preapi.get("ready_for_api") if isinstance(preapi, dict) else None
    return {
        "dir": str(frontier_dir) if frontier_dir else None,
        "ready": ready,
        "preapi_only": bool(preapi_ready) and not bool(report),
        "accepted_rows": accepted_rows,
        "failed_rows": metric(metrics, "failed_rows"),
        "accept_rate": metric(metrics, "accept_rate"),
        "distinct_primitives": metric(metrics, "distinct_primitives"),
        "expected_primitive_coverage": expected_coverage,
        "api_requests_used": summary.get("api_requests_used") if isinstance(summary, dict) else None,
        "max_api_requests": summary.get("max_api_requests") if isinstance(summary, dict) else None,
        "stopped_by_api_request_budget": summary.get("stopped_by_api_request_budget") if isinstance(summary, dict) else None,
        "paid_gate_passed": gate_passed,
        "preapi_ready": preapi_ready,
        "artifacts": {name: path_record(path) for name, path in paths.items()},
    }


def cycle_summary(cycle_dir: Path | None, manifest_path: Path | None) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    metrics = manifest.get("metrics") if isinstance(manifest, dict) else None
    promotion = manifest.get("promotion") if isinstance(manifest, dict) else None
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else {}
    policy_artifact = artifacts.get("policy_adapter") if isinstance(artifacts, dict) else None
    hardcases = manifest.get("output_hardcases") if isinstance(manifest, dict) else {}
    hardcase_artifact = artifacts.get("output_hardcases") if isinstance(artifacts, dict) else None
    return {
        "dir": str(cycle_dir) if cycle_dir else None,
        "manifest": str(manifest_path) if manifest_path else None,
        "exists": bool(manifest),
        "stage": manifest.get("stage") if isinstance(manifest, dict) else None,
        "promoted": bool(manifest.get("promoted")) if isinstance(manifest, dict) else None,
        "policy_adapter": manifest.get("policy_adapter") if isinstance(manifest, dict) else None,
        "policy_adapter_exists": bool(policy_artifact.get("exists")) if isinstance(policy_artifact, dict) else False,
        "metrics": {
            "chrf++": metric(metrics, "chrf++"),
            "bleu": metric(metrics, "bleu"),
            "token_f1": metric(metrics, "token_f1"),
            "ter": metric(metrics, "ter"),
            "required_format_rate": metric(metrics, "self_verification_required_format_rate"),
            "false_confidence_rate": metric(metrics, "self_verification_false_confidence_rate"),
            "missing_score_rate": metric(metrics, "self_verification_missing_score_rate"),
        },
        "promotion_reasons": promotion.get("reasons", []) if isinstance(promotion, dict) else [],
        "output_hardcases": hardcases,
        "output_hardcases_path": (
            str(hardcase_artifact.get("path"))
            if isinstance(hardcase_artifact, dict) and hardcase_artifact.get("path")
            else None
        ),
        "output_hardcases_exists": (
            bool(hardcase_artifact.get("exists")) if isinstance(hardcase_artifact, dict) else False
        ),
        "artifacts": artifacts if isinstance(artifacts, dict) else {},
    }


def summarize_sft(sft_dir: Path | None) -> dict[str, Any]:
    manifest_path = sft_dir / "cycle_manifest.json" if sft_dir else None
    summary = cycle_summary(sft_dir, manifest_path)
    summary["expected_metrics_path"] = str(sft_dir / "sft_only_eval" / "metrics.json") if sft_dir else None
    return summary


def summarize_gspo(gspo_dir: Path | None) -> dict[str, Any]:
    manifest_path = gspo_dir / "cycle_manifest.json" if gspo_dir else None
    return cycle_summary(gspo_dir, manifest_path)


def next_action(frontier: dict[str, Any], sft: dict[str, Any], gspo: dict[str, Any]) -> dict[str, str]:
    if not frontier.get("ready"):
        if frontier.get("preapi_ready") and frontier.get("dir"):
            frontier_dir = frontier.get("dir")
            return {
                "stage": "frontier_generation",
                "command": f"DATA_DIR={frontier_dir} experiments/gspo/run_deepseek_v4_pro_paid_generation_smoke.sh",
                "reason": "Pre-API readiness passed, but no accepted paid frontier report exists yet.",
            }
        if frontier.get("dir") and frontier.get("paid_gate_passed") is not True:
            frontier_dir = frontier.get("dir")
            artifacts = frontier.get("artifacts") if isinstance(frontier.get("artifacts"), dict) else {}
            report = artifacts.get("report_json") if isinstance(artifacts.get("report_json"), dict) else {}
            reason = (
                "A frontier report exists, but the paid-smoke data gate has not passed."
                if report.get("exists")
                else "Frontier directory exists, but no paid-smoke data gate has passed yet."
            )
            return {
                "stage": "frontier_generation",
                "command": f"DATA_DIR={frontier_dir} experiments/gspo/run_deepseek_v4_pro_paid_generation_smoke.sh",
                "reason": reason,
            }
        return {
            "stage": "frontier_generation",
            "command": "experiments/gspo/run_deepseek_v4_pro_paid_generation_smoke.sh",
            "reason": "No accepted frontier report passed the paid-smoke data gate yet.",
        }
    if not sft.get("exists"):
        frontier_dir = frontier.get("dir") or "<frontier_data_dir>"
        return {
            "stage": "sft_seed",
            "command": f"DATA_DIR={frontier_dir} experiments/gspo/run_deepseek_v4_pro_sft_from_frontier_data.sh",
            "reason": "Frontier data exists, but no SFT seed manifest was found.",
        }
    if not sft.get("policy_adapter_exists"):
        return {
            "stage": "inspect_sft_seed",
            "command": f"cat {sft.get('manifest') or '<sft_cycle_manifest>'}",
            "reason": "SFT manifest exists but the policy adapter artifact is missing.",
        }
    if not gspo.get("exists"):
        sft_dir = sft.get("dir") or "<sft_output_dir>"
        return {
            "stage": "initial_gspo",
            "command": f"THINKING_SFT_OUTPUT_DIR={sft_dir} experiments/gspo/run_deepseek_v4_pro_gspo_from_sft_seed.sh",
            "reason": "SFT seed exists, but no initial GSPO cycle manifest was found.",
        }
    if gspo.get("promoted") and gspo.get("policy_adapter_exists"):
        manifest = gspo.get("manifest") or "<gspo_cycle_manifest>"
        return {
            "stage": "promoted_policy",
            "command": f"BASE_CYCLE_MANIFEST={manifest} experiments/gspo/run_hardcase_meta_then_followup_gspo_cycle.sh",
            "reason": "Initial GSPO is promoted; continue from its manifest so lineage and hardcases are preserved.",
        }
    hardcase_count = 0
    if isinstance(gspo.get("output_hardcases"), dict):
        try:
            hardcase_count = int(gspo["output_hardcases"].get("valid_records", 0) or 0)
        except (TypeError, ValueError):
            hardcase_count = 0
    if gspo.get("output_hardcases_path") and gspo.get("output_hardcases_exists") and hardcase_count > 0:
        return {
            "stage": "hardcase_iteration",
            "command": (
                f"GSPO_META_JSONL={gspo.get('output_hardcases_path')} "
                "experiments/gspo/run_next_meta_verifier_from_hardcases.sh"
            ),
            "reason": "GSPO was not promoted; train the next meta-verifier from its mined hardcases.",
        }
    return {
        "stage": "hardcase_iteration",
        "command": "experiments/gspo/run_next_meta_verifier_from_hardcases.sh",
        "reason": "GSPO exists but was not promoted; use mined hardcases for the next verifier iteration.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    frontier_dir, sft_dir, gspo_dir = discover_dirs(args)
    frontier = summarize_frontier(frontier_dir)
    sft = summarize_sft(sft_dir)
    gspo = summarize_gspo(gspo_dir)
    action = next_action(frontier, sft, gspo)
    return {
        "discovery": {
            "enabled": bool(args.discover),
            "output_root": str(args.output_root),
            "frontier_dir": str(frontier_dir) if frontier_dir else None,
            "sft_dir": str(sft_dir) if sft_dir else None,
            "gspo_dir": str(gspo_dir) if gspo_dir else None,
        },
        "frontier": frontier,
        "sft_seed": sft,
        "initial_gspo": gspo,
        "next_action": action,
        "blocked": action["stage"] == "frontier_generation" and not frontier.get("preapi_ready"),
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frontier = report["frontier"]
    sft = report["sft_seed"]
    gspo = report["initial_gspo"]
    action = report["next_action"]
    lines = [
        "# DeepSeekMath Staged Run Status",
        "",
        f"- next stage: `{action['stage']}`",
        f"- next command: `{action['command']}`",
        f"- reason: {action['reason']}",
        "",
        "## Frontier",
        "",
        f"- directory: `{frontier.get('dir')}`",
        f"- ready: {frontier.get('ready')}",
        f"- pre-API ready: {frontier.get('preapi_ready')}",
        f"- accepted rows: {fmt(frontier.get('accepted_rows'))}",
        f"- failed rows: {fmt(frontier.get('failed_rows'))}",
        f"- accept rate: {fmt(frontier.get('accept_rate'))}",
        f"- distinct primitives: {fmt(frontier.get('distinct_primitives'))}",
        f"- expected primitive coverage: {fmt(frontier.get('expected_primitive_coverage'))}",
        f"- API requests used: {frontier.get('api_requests_used')}",
        f"- paid gate passed: {frontier.get('paid_gate_passed')}",
        "",
        "## SFT Seed",
        "",
        f"- manifest: `{sft.get('manifest')}`",
        f"- exists: {sft.get('exists')}",
        f"- promoted: {sft.get('promoted')}",
        f"- policy adapter exists: {sft.get('policy_adapter_exists')}",
        f"- chrF++: {fmt(sft['metrics']['chrf++'])}",
        f"- BLEU: {fmt(sft['metrics']['bleu'])}",
        f"- token F1: {fmt(sft['metrics']['token_f1'])}",
        f"- required format rate: {fmt(sft['metrics']['required_format_rate'])}",
        "",
        "## Initial GSPO",
        "",
        f"- manifest: `{gspo.get('manifest')}`",
        f"- exists: {gspo.get('exists')}",
        f"- promoted: {gspo.get('promoted')}",
        f"- policy adapter exists: {gspo.get('policy_adapter_exists')}",
        f"- chrF++: {fmt(gspo['metrics']['chrf++'])}",
        f"- BLEU: {fmt(gspo['metrics']['bleu'])}",
        f"- token F1: {fmt(gspo['metrics']['token_f1'])}",
        f"- required format rate: {fmt(gspo['metrics']['required_format_rate'])}",
    ]
    if gspo.get("promotion_reasons"):
        lines.extend(["", "## GSPO Promotion Reasons", ""])
        for reason in gspo["promotion_reasons"][:5]:
            lines.append(f"- {reason}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        write_markdown(report, args.output_md)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.fail_if_blocked and report["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
