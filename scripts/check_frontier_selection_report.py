"""Gate frontier source-row selection before any paid frontier API call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

PRIMITIVES = (
    "[SIGNIFICADO]",
    "[GRAMATICA]",
    "[ENTIDADES]",
    "[TERMINOLOGIA]",
    "[ANTI_COPIA]",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-selected-rows", type=int, default=64)
    parser.add_argument("--min-distinct-expected-primitives", type=int, default=5)
    parser.add_argument("--min-avg-expected-primitives", type=float, default=2.0)
    parser.add_argument(
        "--required-primitive",
        action="append",
        default=[],
        help="Primitive tag that must appear at least --min-count-per-required-primitive times. Defaults to all tags.",
    )
    parser.add_argument("--min-count-per-required-primitive", type=int, default=1)
    return parser.parse_args(argv)


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def gate_selection(
    report: dict[str, Any],
    min_selected_rows: int,
    min_distinct_expected_primitives: int,
    min_avg_expected_primitives: float,
    required_primitives: Sequence[str] | None = None,
    min_count_per_required_primitive: int = 1,
) -> tuple[dict[str, Any], bool, list[str]]:
    counts = dict(report.get("expected_primitive_counts") or {})
    selected_rows = int(report.get("selected_rows", 0))
    distinct = int(report.get("distinct_expected_primitives", 0))
    avg = float(report.get("avg_expected_primitives_per_row", 0.0))
    required = list(required_primitives or PRIMITIVES)

    metrics = {
        "selected_rows": selected_rows,
        "distinct_expected_primitives": distinct,
        "avg_expected_primitives_per_row": avg,
        "expected_primitive_counts": counts,
        "required_primitives": required,
        "min_selected_rows": min_selected_rows,
        "min_distinct_expected_primitives": min_distinct_expected_primitives,
        "min_avg_expected_primitives": min_avg_expected_primitives,
        "min_count_per_required_primitive": min_count_per_required_primitive,
    }
    reasons: list[str] = []
    if selected_rows < min_selected_rows:
        reasons.append(f"selected_rows {selected_rows} < {min_selected_rows}")
    if distinct < min_distinct_expected_primitives:
        reasons.append(f"distinct_expected_primitives {distinct} < {min_distinct_expected_primitives}")
    if avg < min_avg_expected_primitives:
        reasons.append(f"avg_expected_primitives_per_row {avg:.4f} < {min_avg_expected_primitives:.4f}")
    for tag in required:
        count = int(counts.get(tag, 0))
        if count < min_count_per_required_primitive:
            reasons.append(f"expected_primitive_counts[{tag}] {count} < {min_count_per_required_primitive}")
    return metrics, not reasons, reasons


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    required = args.required_primitive or list(PRIMITIVES)
    metrics, passed, reasons = gate_selection(
        load_report(args.selection_report_json),
        args.min_selected_rows,
        args.min_distinct_expected_primitives,
        args.min_avg_expected_primitives,
        required,
        args.min_count_per_required_primitive,
    )
    payload = {
        **metrics,
        "passed": passed,
        "reasons": reasons,
        "selection_report_json": str(args.selection_report_json),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
