"""Gate mined meta-verifier hardcases before training.

The DeepSeekMath-style loop should not train a refreshed meta-verifier on an
empty or tiny set of real self-analysis failures. This checker counts one or
more JSONL files produced by build_meta_verifier_from_self_outputs.py and
optionally reports the rationale/severity mix.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


REQUIRED_FIELDS = {"source", "reference", "candidate", "analysis", "label"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, action="append", default=[], help="Meta hardcase JSONL file.")
    parser.add_argument("--min-records", type=int, default=32)
    parser.add_argument(
        "--require-existing-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when any requested JSONL path is missing.",
    )
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def label_payload(record: dict[str, Any]) -> dict[str, Any]:
    label = record.get("label")
    if isinstance(label, dict):
        return label
    if isinstance(label, str):
        try:
            parsed = json.loads(label)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def count_records(paths: Sequence[Path], require_existing_files: bool = True) -> dict[str, Any]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing and require_existing_files:
        return {
            "records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "missing_files": missing,
            "label_rationales": {},
            "label_severities": {},
        }

    records = 0
    valid_records = 0
    invalid_records = 0
    rationales: collections.Counter[str] = collections.Counter()
    severities: collections.Counter[str] = collections.Counter()
    seen: set[tuple[str, str, str, str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        for record in iter_jsonl(path):
            records += 1
            if not REQUIRED_FIELDS.issubset(record):
                invalid_records += 1
                continue
            key = (
                str(record["source"]),
                str(record["reference"]),
                str(record["candidate"]),
                str(record["analysis"]),
                str(record["label"]),
            )
            if key in seen:
                continue
            seen.add(key)
            valid_records += 1
            label = label_payload(record)
            rationale = str(label.get("rationale") or "unknown")
            severity = str(label.get("severity") or "unknown")
            rationales[rationale] += 1
            severities[severity] += 1
    return {
        "records": records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "missing_files": missing,
        "label_rationales": dict(rationales),
        "label_severities": dict(severities),
    }


def check_gate(metrics: dict[str, Any], min_records: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics.get("missing_files"):
        reasons.append(f"missing_files: {', '.join(metrics['missing_files'])}")
    if int(metrics.get("valid_records", 0)) < min_records:
        reasons.append(f"valid_records {int(metrics.get('valid_records', 0))} < {min_records}")
    if int(metrics.get("invalid_records", 0)) > 0:
        reasons.append(f"invalid_records {int(metrics['invalid_records'])} > 0")
    return not reasons, reasons


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = count_records(args.jsonl, require_existing_files=args.require_existing_files)
    passed, reasons = check_gate(metrics, args.min_records)
    report = {
        **metrics,
        "jsonl": [str(path) for path in args.jsonl],
        "min_records": args.min_records,
        "passed": passed,
        "reasons": reasons,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
