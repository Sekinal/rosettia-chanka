"""Merge multiple candidate prediction JSONL pools into one grouped pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import rerank_candidate_predictions as rerank
from scripts import train_gspo_chanka_unsloth as gspo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--dedupe-normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop duplicate normalized predictions within each source/reference group.",
    )
    return parser.parse_args(argv)


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
            record["_pool_path"] = str(path)
            yield record


def candidate_key(record: dict[str, Any]) -> tuple[str, str, str | None, str | None]:
    return (
        str(record["source"]),
        str(record.get("reference", "")),
        record.get("source_name"),
        record.get("variant"),
    )


def merge_records(paths: Sequence[Path], dedupe_normalized: bool = True) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str | None, str | None], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, str | None, str | None]] = []
    seen_predictions: dict[tuple[str, str, str | None, str | None], set[str]] = {}
    for path in paths:
        for record in iter_records(path):
            key = candidate_key(record)
            if key not in grouped:
                grouped[key] = []
                seen_predictions[key] = set()
                order.append(key)
            normalized_prediction = gspo.normalize_text(str(record.get("prediction", ""))).lower()
            if dedupe_normalized and normalized_prediction in seen_predictions[key]:
                continue
            seen_predictions[key].add(normalized_prediction)
            grouped[key].append(record)

    merged: list[dict[str, Any]] = []
    for key in order:
        source, reference, source_name, variant = key
        for candidate_index, record in enumerate(grouped[key]):
            merged.append(
                {
                    "candidate_index": candidate_index,
                    "prediction": gspo.normalize_text(str(record.get("prediction", ""))),
                    "reference": reference,
                    "source": source,
                    "source_name": source_name,
                    "variant": variant,
                    "pool_path": record.get("_pool_path"),
                }
            )
    return merged


def write_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        rerank.Candidate(
            source=str(record["source"]),
            reference=str(record.get("reference", "")),
            prediction=str(record["prediction"]),
            source_name=record.get("source_name"),
            variant=record.get("variant"),
            candidate_index=int(record.get("candidate_index", 0)),
        )
        for record in records
    ]
    groups = rerank.group_candidates(candidates)
    return {
        "groups": len(groups),
        "records": len(records),
        "mean_candidates_per_group": len(records) / max(1, len(groups)),
        "min_candidates_per_group": min((len(group) for group in groups), default=0),
        "max_candidates_per_group": max((len(group) for group in groups), default=0),
    }


def main() -> None:
    args = parse_args()
    records = merge_records(args.predictions_jsonl, args.dedupe_normalized)
    if not records:
        raise ValueError("No records were merged.")
    write_records(args.output_jsonl, records)
    print(json.dumps(summarize(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
