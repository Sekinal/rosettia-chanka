from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz
import polars as pl

from extract_manual_parallel_corpus import SOURCE_FILE, SOURCE_ID, normalize_text


DEFAULT_OUTPUT_CSV = Path("data/processed/manual_quechua_chanka_glossary_entries.csv")
DEFAULT_OUTPUT_PARQUET = Path("data/processed/manual_quechua_chanka_glossary_entries.parquet")
DEFAULT_SIMPLE_CSV = Path("data/processed/manual_quechua_chanka_glossary_simple_terms.csv")
DEFAULT_SIMPLE_PARQUET = Path("data/processed/manual_quechua_chanka_glossary_simple_terms.parquet")
EXAMPLE_RE = re.compile(r"[:¿¡]|\b(1\.|2\.|3\.|Abandonado|Herido|Enterrado|literalmente|ejemplo)\b", re.I)


def glossary_status(source_term: str, target_text: str) -> str:
    if "\n" in target_text or EXAMPLE_RE.search(target_text):
        return "rich_entry_needs_term_review"
    if "," in source_term or "," in target_text or "(" in target_text or ")" in target_text:
        return "simple_entry_with_alternatives_or_note"
    return "simple_term_pair"


def extract_glossary(pdf_path: Path) -> pl.DataFrame:
    doc = fitz.open(pdf_path)
    records: list[dict[str, object]] = []
    ranges = [
        (121, 129, "spa_Latn-quy_Latn", "vocabulario_castellano_quechua"),
        (133, 149, "quy_Latn-spa_Latn", "vocabulario_quechua_castellano"),
    ]

    for start, end, direction, section in ranges:
        for page_number in range(start, end + 1):
            page = doc[page_number - 1]
            for table_index, table in enumerate(page.find_tables().tables):
                for table_row_index, row in enumerate(table.extract()):
                    if len(row) < 2:
                        continue
                    source_term = normalize_text(row[0])
                    target_text = normalize_text(row[1])
                    if not source_term or not target_text:
                        continue
                    status = glossary_status(source_term, row[1] or "")
                    records.append(
                        {
                            "glossary_id": len(records) + 1,
                            "source_id": SOURCE_ID,
                            "source_file": SOURCE_FILE,
                            "page_number": page_number,
                            "table_index": table_index,
                            "table_row_index": table_row_index,
                            "section": section,
                            "direction": direction,
                            "source_term": source_term,
                            "target_text": target_text,
                            "raw_source_term": row[0],
                            "raw_target_text": row[1],
                            "glossary_status": status,
                            "training_use": "terminology_only",
                            "license_note": "Source states reproduction is permitted when the source is cited.",
                        }
                    )

    return pl.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract glossary/terminology entries from the Chanka Quechua manual.")
    parser.add_argument("--pdf", type=Path, default=Path(SOURCE_FILE))
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--simple-csv", type=Path, default=DEFAULT_SIMPLE_CSV)
    parser.add_argument("--simple-parquet", type=Path, default=DEFAULT_SIMPLE_PARQUET)
    args = parser.parse_args()

    df = extract_glossary(args.pdf)
    simple = df.filter(pl.col("glossary_status") == "simple_term_pair")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.output_csv)
    df.write_parquet(args.output_parquet)
    simple.write_csv(args.simple_csv)
    simple.write_parquet(args.simple_parquet)

    print(f"Extracted {df.height} glossary entries")
    print(f"Simple term pairs: {simple.height}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_parquet}")
    print(f"Wrote {args.simple_csv}")
    print(f"Wrote {args.simple_parquet}")


if __name__ == "__main__":
    main()
