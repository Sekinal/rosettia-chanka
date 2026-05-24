"""Write a manifest for one DeepSeekMath-style translation iteration.

The manifest ties together the policy base, refreshed meta-verifier, follow-up
GSPO metrics, promotion gate, and mined hardcases so iterations can be audited
without reconstructing paths from shell logs.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from scripts import check_meta_hardcase_data as hardcase_check


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--meta-verifier-adapter", required=True)
    parser.add_argument("--meta-output-dir", type=Path, required=True)
    parser.add_argument("--followup-output-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--promotion-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--input-hardcase-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-hardcase-jsonl", type=Path, default=None)
    parser.add_argument("--baseline-metrics-json", type=Path, default=None)
    return parser.parse_args(argv)


def load_json_if_exists(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def file_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def manifest_for(args: argparse.Namespace) -> dict[str, Any]:
    input_hardcases = hardcase_check.count_records(args.input_hardcase_jsonl, require_existing_files=False)
    output_hardcases = (
        hardcase_check.count_records([args.output_hardcase_jsonl], require_existing_files=False)
        if args.output_hardcase_jsonl
        else None
    )
    metrics = load_json_if_exists(args.metrics_json)
    promotion = load_json_if_exists(args.promotion_json)
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "stamp": args.stamp,
        "base_model": args.base_model,
        "meta_verifier_adapter": args.meta_verifier_adapter,
        "meta_output_dir": str(args.meta_output_dir),
        "followup_output_dir": str(args.followup_output_dir),
        "artifacts": {
            "baseline_metrics": file_record(args.baseline_metrics_json),
            "metrics": file_record(args.metrics_json),
            "promotion": file_record(args.promotion_json),
            "predictions": file_record(args.predictions_jsonl),
            "input_hardcases": [file_record(path) for path in args.input_hardcase_jsonl],
            "output_hardcases": file_record(args.output_hardcase_jsonl),
        },
        "metrics": metrics,
        "promotion": promotion,
        "promoted": bool(promotion.get("promoted")) if isinstance(promotion, dict) else None,
        "input_hardcases": input_hardcases,
        "output_hardcases": output_hardcases,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = manifest_for(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
