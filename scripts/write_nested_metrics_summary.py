"""Write summary.json for eval dirs containing metric JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_gspo_canaries import selection_score


DEFAULT_FIELDS = ("selection_score", "chrf++", "bleu", "token_f1", "ter")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--metric-field",
        action="append",
        default=[],
        help="Extra scalar metric field to copy into each record. Repeatable.",
    )
    return parser.parse_args(argv)


def record_from_metrics(metrics_path: Path, extra_fields: Sequence[str]) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text())
    score = metrics.get("selection_score")
    if score is None:
        score = selection_score(metrics)
    if metrics_path.name == "metrics.json":
        checkpoint = metrics_path.parent.name
    else:
        checkpoint = metrics_path.stem.removesuffix("_metrics")
    record = {
        "checkpoint": checkpoint,
        "selection_score": score,
        "metrics_json": str(metrics_path),
    }
    for field in (*DEFAULT_FIELDS[1:], *extra_fields):
        record[field] = metrics.get(field)
    return record


def collect_records(eval_dir: Path, extra_fields: Sequence[str]) -> list[dict[str, Any]]:
    metric_paths = sorted({*eval_dir.glob("*/metrics.json"), *eval_dir.glob("*_metrics.json")})
    records = [record_from_metrics(metrics_path, extra_fields) for metrics_path in metric_paths]
    records.sort(
        key=lambda row: (
            row.get("selection_score") is not None,
            float(row.get("selection_score") or -1.0),
            float(row.get("chrf++") or 0.0),
        ),
        reverse=True,
    )
    return records


def write_summary(eval_dir: Path, output_json: Path | None, extra_fields: Sequence[str]) -> dict[str, Any]:
    records = collect_records(eval_dir, extra_fields)
    summary = {"records": records}
    path = output_json or (eval_dir / "summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return {"summary_json": str(path), "best": records[0] if records else None, "records": len(records)}


def main() -> None:
    args = parse_args()
    result = write_summary(args.eval_dir, args.output_json, args.metric_field)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
