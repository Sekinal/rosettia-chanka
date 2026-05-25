"""Gate a DeepSeekMath cycle manifest before using its policy as next base."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import summarize_deepseekmath_cycles as summarize


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--require-promoted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-policy-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-stage", type=str, default=None)
    parser.add_argument("--expected-policy-adapter", type=str, default=None)
    parser.add_argument("--max-missing-artifacts", type=int, default=0)
    parser.add_argument("--min-chrf", type=float, default=35.0)
    parser.add_argument("--min-bleu", type=float, default=8.0)
    parser.add_argument("--min-token-f1", type=float, default=15.0)
    return parser.parse_args(argv)


def artifact_exists(record: dict[str, Any], name: str) -> bool:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    artifact = artifacts.get(name)
    return isinstance(artifact, dict) and bool(artifact.get("exists"))


def report_for(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reasons: list[str] = []
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    missing_artifacts = record.get("missing_artifacts")
    if not isinstance(missing_artifacts, list):
        missing_artifacts = summarize.missing_artifacts(record)

    if args.require_promoted and not bool(record.get("promoted")):
        reasons.append("cycle is not promoted")
    if args.expected_stage is not None and record.get("stage") != args.expected_stage:
        reasons.append(f"stage {record.get('stage')} != {args.expected_stage}")
    if args.expected_policy_adapter is not None and str(record.get("policy_adapter")) != args.expected_policy_adapter:
        reasons.append(
            f"policy_adapter {record.get('policy_adapter')} != {args.expected_policy_adapter}"
        )
    if len(missing_artifacts) > args.max_missing_artifacts:
        reasons.append(f"missing artifacts {len(missing_artifacts)} > {args.max_missing_artifacts}: {', '.join(missing_artifacts)}")
    if args.require_policy_adapter and not artifact_exists(record, "policy_adapter"):
        reasons.append("policy_adapter artifact is missing")
    if summarize.metric(metrics, "chrf++") < args.min_chrf:
        reasons.append(f"chrf++ {summarize.metric(metrics, 'chrf++'):.4f} < {args.min_chrf:.4f}")
    if summarize.metric(metrics, "bleu") < args.min_bleu:
        reasons.append(f"bleu {summarize.metric(metrics, 'bleu'):.4f} < {args.min_bleu:.4f}")
    if summarize.metric(metrics, "token_f1") < args.min_token_f1:
        reasons.append(f"token_f1 {summarize.metric(metrics, 'token_f1'):.4f} < {args.min_token_f1:.4f}")

    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    policy_artifact = artifacts.get("policy_adapter") if isinstance(artifacts.get("policy_adapter"), dict) else {}
    return {
        "passed": not reasons,
        "reasons": reasons,
        "manifest_json": str(args.manifest_json),
        "stamp": record.get("stamp"),
        "stage": record.get("stage"),
        "promoted": bool(record.get("promoted")),
        "policy_adapter": record.get("policy_adapter"),
        "policy_adapter_exists": bool(policy_artifact.get("exists")),
        "missing_artifacts": missing_artifacts,
        "artifact_missing_count": len(missing_artifacts),
        "promotion_reasons": summarize.promotion_reasons(record, limit=10),
        "metrics": {
            "chrf++": summarize.metric(metrics, "chrf++"),
            "bleu": summarize.metric(metrics, "bleu"),
            "token_f1": summarize.metric(metrics, "token_f1"),
            "ter": summarize.metric(metrics, "ter"),
        },
        "thresholds": {
            "require_promoted": args.require_promoted,
            "require_policy_adapter": args.require_policy_adapter,
            "expected_stage": args.expected_stage,
            "expected_policy_adapter": args.expected_policy_adapter,
            "max_missing_artifacts": args.max_missing_artifacts,
            "min_chrf": args.min_chrf,
            "min_bleu": args.min_bleu,
            "min_token_f1": args.min_token_f1,
        },
    }


def missing_manifest_report(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "passed": False,
        "reasons": [reason],
        "manifest_json": str(args.manifest_json),
        "stamp": None,
        "stage": None,
        "promoted": False,
        "policy_adapter": None,
        "policy_adapter_exists": False,
        "missing_artifacts": [],
        "artifact_missing_count": 0,
        "promotion_reasons": [],
        "metrics": {"chrf++": 0.0, "bleu": 0.0, "token_f1": 0.0, "ter": 0.0},
        "thresholds": {
            "require_promoted": args.require_promoted,
            "require_policy_adapter": args.require_policy_adapter,
            "expected_stage": args.expected_stage,
            "expected_policy_adapter": args.expected_policy_adapter,
            "max_missing_artifacts": args.max_missing_artifacts,
            "min_chrf": args.min_chrf,
            "min_bleu": args.min_bleu,
            "min_token_f1": args.min_token_f1,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.manifest_json.exists():
        report = missing_manifest_report(args, f"manifest JSON not found: {args.manifest_json}")
    else:
        try:
            record = summarize.load_manifest(args.manifest_json)
        except (json.JSONDecodeError, OSError) as exc:
            report = missing_manifest_report(args, f"manifest JSON invalid: {exc}")
        else:
            report = report_for(record, args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
