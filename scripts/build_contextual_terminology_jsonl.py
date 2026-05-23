"""Build short glossary-in-context Spanish->Chanka SFT rows.

Standalone glossary pair SFT over-steered the model in earlier experiments.
This builder keeps the same terminology signal but wraps each term in tiny,
stable translation contexts so the model sees the target term inside a Chanka
sentence rather than as a bare dictionary answer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_gspo_chanka_unsloth as gspo


@dataclass(frozen=True)
class Template:
    name: str
    source: str
    target: str


TEMPLATES = [
    Template(
        name="explain_term",
        source="Explique {source_term}.",
        target="{target_term}ta sut'ichay.",
    ),
    Template(
        name="explain_term_case",
        source="Explique el caso de {source_term}.",
        target="{target_term} kasutam sut'ichay.",
    ),
    Template(
        name="talk_about_term",
        source="Hable sobre {source_term}.",
        target="{target_term}manta rimay.",
    ),
    Template(
        name="need_term_information",
        source="Necesito información sobre {source_term}.",
        target="{target_term}manta willakuyta munani.",
    ),
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument(
        "--terminology-file",
        default="clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet",
    )
    parser.add_argument("--min-source-chars", type=int, default=3)
    parser.add_argument("--max-source-chars", type=int, default=48)
    parser.add_argument("--max-target-chars", type=int, default=64)
    parser.add_argument("--max-source-words", type=int, default=4)
    parser.add_argument("--max-target-words", type=int, default=5)
    parser.add_argument("--templates-per-term", type=int, default=2)
    parser.add_argument("--max-terms", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args(argv)


def simple_term(text: str, max_chars: int, max_words: int) -> bool:
    normalized = gspo.normalize_text(text)
    if not normalized or len(normalized) > max_chars:
        return False
    if len(gspo.word_tokens(normalized)) > max_words:
        return False
    if re.search(r"[0-9/@#*_{}\[\]<>|=()\"“”]", normalized):
        return False
    return True


def valid_entry(source: str, target: str, args: argparse.Namespace) -> bool:
    source_norm = gspo.normalize_text(source)
    target_norm = gspo.normalize_text(target)
    if len(source_norm) < args.min_source_chars:
        return False
    if source_norm.lower() in gspo.SPANISH_STOPWORDS:
        return False
    if not simple_term(source_norm, args.max_source_chars, args.max_source_words):
        return False
    if not simple_term(target_norm, args.max_target_chars, args.max_target_words):
        return False
    return True


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, int | str]]:
    entries = gspo.load_terminology_entries(args.dataset_repo, args.terminology_file, args.min_source_chars)
    rows: list[dict[str, str]] = []
    seen_entries: set[tuple[str, str]] = set()
    accepted_terms = 0
    skipped_terms = 0

    templates = TEMPLATES[: max(1, args.templates_per_term)]
    for source_term, target_term in entries:
        source_term = gspo.normalize_text(source_term)
        target_term = gspo.normalize_text(target_term)
        entry_key = (source_term.lower(), target_term.lower())
        if entry_key in seen_entries:
            continue
        seen_entries.add(entry_key)
        if not valid_entry(source_term, target_term, args):
            skipped_terms += 1
            continue

        accepted_terms += 1
        for template in templates:
            rows.append(
                {
                    "source": template.source.format(source_term=source_term),
                    "target": template.target.format(target_term=target_term),
                    "reference": template.target.format(target_term=target_term),
                    "source_name": args.terminology_file,
                    "variant": "quy/chanka_terminology_context",
                    "label_type": f"terminology_context_{template.name}",
                    "terminology_source": source_term,
                    "terminology_target": target_term,
                }
            )
            if args.max_rows is not None and len(rows) >= args.max_rows:
                break
        if args.max_rows is not None and len(rows) >= args.max_rows:
            break
        if args.max_terms is not None and accepted_terms >= args.max_terms:
            break

    if not rows:
        raise RuntimeError("No contextual terminology rows survived filtering.")

    metrics: dict[str, int | str] = {
        "rows": len(rows),
        "accepted_terms": accepted_terms,
        "skipped_terms": skipped_terms,
        "templates_per_term": len(templates),
        "terminology_file": args.terminology_file,
    }
    return rows, metrics


def write_jsonl(path: Path, rows: Sequence[dict[str, str]]) -> None:
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
