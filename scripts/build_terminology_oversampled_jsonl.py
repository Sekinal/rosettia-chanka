"""Oversample sentence rows whose Spanish source matches glossary terms."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_gspo_chanka_unsloth as gspo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--target-field", default="target")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument(
        "--terminology-file",
        default="clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet",
    )
    parser.add_argument("--terminology-top-k", type=int, default=1)
    parser.add_argument("--terminology-min-source-chars", type=int, default=3)
    parser.add_argument("--repeat-matched", type=int, default=2)
    parser.add_argument("--max-extra-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc


def normalized_row(record: dict[str, Any], path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    source = gspo.normalize_text(str(record.get(args.source_field, "")))
    target = gspo.normalize_text(str(record.get(args.target_field, "")))
    if not source or not target:
        return None
    return {
        "source": source,
        "target": target,
        "reference": gspo.normalize_text(str(record.get(args.reference_field, ""))),
        "source_name": str(record.get("source_name") or path.name),
        "variant": str(record.get("variant") or "quy/chanka"),
        "label_type": str(record.get("label_type") or "base"),
    }


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in args.input_jsonl:
        for record in iter_jsonl(path):
            row = normalized_row(record, path, args)
            if row is not None:
                rows.append(row)
    if not rows:
        raise RuntimeError("No input rows survived normalization.")
    return rows


def matched_terms(
    source: str,
    terminology_entries: Sequence[tuple[str, str]],
    top_k: int,
) -> list[tuple[str, str]]:
    return gspo.select_terminology(source, terminology_entries, top_k)


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_rows(args)
    terminology_entries = gspo.load_terminology_entries(
        args.dataset_repo,
        args.terminology_file,
        args.terminology_min_source_chars,
    )
    if args.repeat_matched < 1:
        raise ValueError("--repeat-matched must be >= 1")

    output_rows = [dict(row) for row in rows]
    extra_rows: list[dict[str, Any]] = []
    matched_base_rows = 0
    for row in rows:
        terms = matched_terms(row["source"], terminology_entries, args.terminology_top_k)
        if not terms:
            continue
        matched_base_rows += 1
        for repeat_index in range(args.repeat_matched - 1):
            repeated = dict(row)
            repeated["label_type"] = f"{row['label_type']}_terminology_oversample"
            repeated["terminology"] = [
                {"source_term": source_term, "target_term": target_term}
                for source_term, target_term in terms
            ]
            repeated["oversample_index"] = repeat_index + 1
            extra_rows.append(repeated)
            if args.max_extra_rows is not None and len(extra_rows) >= args.max_extra_rows:
                break
        if args.max_extra_rows is not None and len(extra_rows) >= args.max_extra_rows:
            break

    output_rows.extend(extra_rows)
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(output_rows)

    metrics = {
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "extra_rows": len(extra_rows),
        "matched_base_rows": matched_base_rows,
        "repeat_matched": args.repeat_matched,
        "terminology_entries": len(terminology_entries),
        "terminology_file": args.terminology_file,
    }
    return output_rows, metrics


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rows, metrics = build_rows(args)
    write_jsonl(args.output_jsonl, rows)
    metrics["output_jsonl"] = str(args.output_jsonl)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
