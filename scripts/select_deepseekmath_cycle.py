"""Select the best promotable DeepSeekMath cycle as the next base policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts import check_deepseekmath_cycle_manifest as check_cycle
from scripts import summarize_deepseekmath_cycles as summarize


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing cycle_manifest.json files.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-chrf", type=float, default=35.0)
    parser.add_argument("--min-bleu", type=float, default=8.0)
    parser.add_argument("--min-token-f1", type=float, default=15.0)
    parser.add_argument("--max-missing-artifacts", type=int, default=0)
    parser.add_argument("--top-k-failures", type=int, default=5)
    return parser.parse_args(argv)


def gate_args(args: argparse.Namespace, manifest_path: str) -> argparse.Namespace:
    return check_cycle.parse_args(
        [
            "--manifest-json",
            manifest_path,
            "--min-chrf",
            str(args.min_chrf),
            "--min-bleu",
            str(args.min_bleu),
            "--min-token-f1",
            str(args.min_token_f1),
            "--max-missing-artifacts",
            str(args.max_missing_artifacts),
        ]
    )


def artifact_path(record: dict[str, Any], name: str) -> str | None:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    artifact = artifacts.get(name)
    if not isinstance(artifact, dict):
        return None
    path = artifact.get("path")
    return str(path) if path else None


def selection_for(args: argparse.Namespace) -> dict[str, Any]:
    records = summarize.collect_manifests(args.root)
    passing: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        manifest_path = str(record.get("manifest_path"))
        report = check_cycle.report_for(record, gate_args(args, manifest_path))
        if report["passed"]:
            passing.append((record, report))
        else:
            failures.append(
                {
                    "manifest_json": manifest_path,
                    "stamp": record.get("stamp"),
                    "stage": record.get("stage"),
                    "cycle_score": record.get("cycle_score"),
                    "reasons": report["reasons"],
                    "promotion_reasons": report.get("promotion_reasons", []),
                }
            )

    selected_record: dict[str, Any] | None = None
    selected_gate: dict[str, Any] | None = None
    if passing:
        selected_record, selected_gate = passing[0]

    return {
        "passed": selected_record is not None,
        "root": str(args.root),
        "selected": (
            {
                "manifest_json": selected_record.get("manifest_path"),
                "stamp": selected_record.get("stamp"),
                "stage": selected_record.get("stage"),
                "cycle_score": selected_record.get("cycle_score"),
                "policy_adapter": selected_record.get("policy_adapter"),
                "baseline_metrics_json": artifact_path(selected_record, "metrics"),
                "gate": selected_gate,
            }
            if selected_record is not None
            else None
        ),
        "candidate_count": len(records),
        "passing_count": len(passing),
        "failed_count": len(failures),
        "failures": failures[: max(0, args.top_k_failures)],
        "thresholds": {
            "min_chrf": args.min_chrf,
            "min_bleu": args.min_bleu,
            "min_token_f1": args.min_token_f1,
            "max_missing_artifacts": args.max_missing_artifacts,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = selection_for(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
