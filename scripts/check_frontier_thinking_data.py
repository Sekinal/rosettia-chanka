"""Gate frontier-thinking datasets before SFT.

This prevents spending GPU time on tiny or low-yield synthetic thinking sets.
The gate can read the builder summary JSON, or count accepted/failure JSONL
files directly.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

PRIMITIVES = (
    "[SIGNIFICADO]",
    "[GRAMATICA]",
    "[ENTIDADES]",
    "[TERMINOLOGIA]",
    "[ANTI_COPIA]",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    parser.add_argument("--min-written-rows", type=int, default=64)
    parser.add_argument("--min-accept-rate", type=float, default=0.5)
    parser.add_argument(
        "--min-primitive-tags-per-row",
        type=int,
        default=0,
        help="Require this many primitive tags on average per accepted row. Disabled at 0.",
    )
    parser.add_argument(
        "--min-primitive-row-rate",
        type=float,
        default=0.0,
        help="Require this fraction of accepted rows to have at least --min-primitive-tags-per-row tags. Disabled at 0.",
    )
    parser.add_argument(
        "--min-distinct-primitives",
        type=int,
        default=0,
        help="Require at least this many distinct primitive tags across accepted rows. Disabled at 0.",
    )
    parser.add_argument(
        "--min-expected-primitive-coverage",
        type=float,
        default=0.0,
        help="Require this fraction of rows with expected_primitives to include all expected tags. Disabled at 0.",
    )
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def primitive_counts(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "primitive_rows": 0,
            "primitive_tag_total": 0,
            "primitive_tag_counts": {},
            "primitive_row_tag_counts": [],
        }
    tag_counts: collections.Counter[str] = collections.Counter()
    missing_expected_counts: collections.Counter[str] = collections.Counter()
    row_tag_counts: list[int] = []
    expected_rows = 0
    expected_covered_rows = 0
    for record in iter_jsonl(path):
        text = str(record.get("frontier_analysis") or record.get("target") or "")
        present = [tag for tag in PRIMITIVES if tag in text]
        row_tag_counts.append(len(present))
        tag_counts.update(present)
        expected = record.get("expected_primitives")
        if isinstance(expected, list) and expected:
            expected_rows += 1
            missing = [str(tag) for tag in expected if str(tag) not in text]
            if missing:
                missing_expected_counts.update(missing)
            else:
                expected_covered_rows += 1
    return {
        "primitive_rows": len(row_tag_counts),
        "primitive_tag_total": sum(row_tag_counts),
        "primitive_tag_counts": dict(tag_counts),
        "primitive_row_tag_counts": row_tag_counts,
        "expected_primitive_rows": expected_rows,
        "expected_primitive_covered_rows": expected_covered_rows,
        "missing_expected_primitive_counts": dict(missing_expected_counts),
    }


def load_counts(args: argparse.Namespace) -> dict[str, int]:
    if args.summary_json is not None and args.summary_json.exists():
        payload = json.loads(args.summary_json.read_text())
        summary = payload.get("summary", payload)
        written = int(summary.get("total_written_rows", summary.get("written_rows", summary.get("new_written_rows", 0))))
        failed = int(summary.get("total_failed_rows", summary.get("failed_rows", summary.get("new_failed_rows", 0))))
        requested = int(summary.get("requested_rows", written + failed))
        return {"written": written, "failed": failed, "requested": requested}
    written = count_jsonl(args.output_jsonl)
    failed = count_jsonl(args.failures_jsonl)
    return {"written": written, "failed": failed, "requested": written + failed}


def gate_metrics(counts: dict[str, int], primitives: dict[str, Any] | None = None, min_tags_per_row: int = 0) -> dict[str, float]:
    attempted = counts["written"] + counts["failed"]
    accept_rate = counts["written"] / max(1, attempted)
    primitives = primitives or {}
    primitive_rows = int(primitives.get("primitive_rows", 0))
    row_tag_counts = list(primitives.get("primitive_row_tag_counts", []))
    primitive_tag_total = int(primitives.get("primitive_tag_total", 0))
    primitive_tag_counts = dict(primitives.get("primitive_tag_counts", {}))
    expected_rows = int(primitives.get("expected_primitive_rows", 0))
    expected_covered_rows = int(primitives.get("expected_primitive_covered_rows", 0))
    metrics = {
        "written_rows": float(counts["written"]),
        "failed_rows": float(counts["failed"]),
        "requested_rows": float(counts["requested"]),
        "attempted_rows": float(attempted),
        "accept_rate": accept_rate,
        "primitive_rows": float(primitive_rows),
        "avg_primitive_tags": primitive_tag_total / max(1, primitive_rows),
        "distinct_primitives": float(len(primitive_tag_counts)),
        "expected_primitive_rows": float(expected_rows),
        "expected_primitive_covered_rows": float(expected_covered_rows),
        "expected_primitive_coverage": expected_covered_rows / max(1, expected_rows),
    }
    if min_tags_per_row > 0:
        rows_with_min_tags = sum(1 for count in row_tag_counts if count >= min_tags_per_row)
        metrics["primitive_row_rate"] = rows_with_min_tags / max(1, primitive_rows)
    else:
        metrics["primitive_row_rate"] = 0.0
    for tag in PRIMITIVES:
        metrics[f"primitive_{tag.strip('[]').lower()}_rows"] = float(primitive_tag_counts.get(tag, 0))
    return metrics


def check_gate(
    metrics: dict[str, float],
    min_written_rows: int,
    min_accept_rate: float,
    min_primitive_tags_per_row: int = 0,
    min_primitive_row_rate: float = 0.0,
    min_distinct_primitives: int = 0,
    min_expected_primitive_coverage: float = 0.0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics["written_rows"] < min_written_rows:
        reasons.append(f"written_rows {metrics['written_rows']:.0f} < {min_written_rows}")
    if metrics["accept_rate"] < min_accept_rate:
        reasons.append(f"accept_rate {metrics['accept_rate']:.4f} < {min_accept_rate:.4f}")
    if min_primitive_tags_per_row > 0 and metrics["avg_primitive_tags"] < min_primitive_tags_per_row:
        reasons.append(
            f"avg_primitive_tags {metrics['avg_primitive_tags']:.4f} < {min_primitive_tags_per_row:.4f}"
        )
    if min_primitive_row_rate > 0 and metrics["primitive_row_rate"] < min_primitive_row_rate:
        reasons.append(f"primitive_row_rate {metrics['primitive_row_rate']:.4f} < {min_primitive_row_rate:.4f}")
    if min_distinct_primitives > 0 and metrics["distinct_primitives"] < min_distinct_primitives:
        reasons.append(f"distinct_primitives {metrics['distinct_primitives']:.0f} < {min_distinct_primitives}")
    if min_expected_primitive_coverage > 0 and metrics["expected_primitive_coverage"] < min_expected_primitive_coverage:
        reasons.append(
            "expected_primitive_coverage "
            f"{metrics['expected_primitive_coverage']:.4f} < {min_expected_primitive_coverage:.4f}"
        )
    return not reasons, reasons


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = gate_metrics(
        load_counts(args),
        primitive_counts(args.output_jsonl),
        min_tags_per_row=args.min_primitive_tags_per_row,
    )
    passed, reasons = check_gate(
        metrics,
        args.min_written_rows,
        args.min_accept_rate,
        args.min_primitive_tags_per_row,
        args.min_primitive_row_rate,
        args.min_distinct_primitives,
        args.min_expected_primitive_coverage,
    )
    report = {
        **metrics,
        "min_written_rows": args.min_written_rows,
        "min_accept_rate": args.min_accept_rate,
        "min_primitive_tags_per_row": args.min_primitive_tags_per_row,
        "min_primitive_row_rate": args.min_primitive_row_rate,
        "min_distinct_primitives": args.min_distinct_primitives,
        "min_expected_primitive_coverage": args.min_expected_primitive_coverage,
        "passed": passed,
        "reasons": reasons,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
