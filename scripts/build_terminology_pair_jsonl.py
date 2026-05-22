"""Build compact Spanish->Chanka terminology-pair JSONL for SFT augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_gspo_chanka_unsloth as gspo


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument(
        "--terminology-file",
        default="clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet",
    )
    parser.add_argument("--min-source-chars", type=int, default=3)
    parser.add_argument("--max-source-chars", type=int, default=80)
    parser.add_argument("--max-target-chars", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args(argv)


def valid_entry(source: str, target: str, args: argparse.Namespace) -> bool:
    if not source or not target:
        return False
    if len(source) < args.min_source_chars or len(source) > args.max_source_chars:
        return False
    if len(target) > args.max_target_chars:
        return False
    if source.lower() in gspo.SPANISH_STOPWORDS:
        return False
    return True


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    entries = gspo.load_terminology_entries(args.dataset_repo, args.terminology_file, args.min_source_chars)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source, target in entries:
        source = gspo.normalize_text(source)
        target = gspo.normalize_text(target)
        if not valid_entry(source, target, args):
            continue
        key = (source.lower(), target.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "source": source,
                "target": target,
                "reference": target,
                "source_name": args.terminology_file,
                "variant": "quy/chanka_terminology",
                "label_type": "terminology_pair",
            }
        )
        if args.max_rows is not None and len(rows) >= args.max_rows:
            break
    if not rows:
        raise RuntimeError("No terminology rows survived filtering.")
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_jsonl(args.output_jsonl, rows)
    print(
        json.dumps(
            {
                "output_jsonl": str(args.output_jsonl),
                "rows": len(rows),
                "terminology_file": args.terminology_file,
                "dataset_repo": args.dataset_repo,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
