from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz
import polars as pl


SOURCE_ID = "manual_quechua_chanka_administracion_justicia_2014"
SOURCE_FILE = "data/raw/source_documents/manual_quechua_chanka_administracion_justicia_2014.pdf"
ROLE_RE = re.compile(r"^(A|C|A/C)$")
PAREN_RE = re.compile(r"\(([^()]*)\)")
NOTE_RE = re.compile(
    r"\b(literalmente|más común|mas común|campesinos|denominan|visitante|responde|puede ser despectivo)\b",
    re.I,
)


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    value = value.replace("\u00ad", "")
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    value = value.replace(" /", "/").replace("/ ", "/")
    value = value.replace("¿ ", "¿").replace("¡ ", "¡")
    value = re.sub(r"\s+([?.!,;:])", r"\1", value)
    value = value.replace("pare ja", "pareja")
    value = value.replace("?.", "?").replace("!.", "!")
    return value.strip()


def strip_parenthetical_notes(value: str) -> tuple[str, list[str]]:
    notes = [match.group(1).strip() for match in PAREN_RE.finditer(value)]
    cleaned = PAREN_RE.sub("", value)
    cleaned = normalize_text(cleaned)
    return cleaned, [note for note in notes if note]


def has_adjacent_duplicate_tokens(value: str) -> bool:
    tokens = [token.casefold() for token in re.findall(r"\w+", value)]
    return any(left == right for left, right in zip(tokens, tokens[1:]))


def build_quality_flags(
    spanish: str,
    chanka_quechua: str,
    raw_spanish: str,
    raw_chanka_quechua: str,
    notes: list[str],
    source_was_filled: bool,
    role_was_filled: bool,
) -> list[str]:
    flags: list[str] = []
    if role_was_filled:
        flags.append("role_filled_from_previous_row")
    if source_was_filled:
        flags.append("source_filled_from_previous_row")
    if len(spanish) > 140 or len(chanka_quechua) > 160:
        flags.append("long_cell")
    if NOTE_RE.search(raw_spanish) or NOTE_RE.search(raw_chanka_quechua):
        flags.append("source_note_removed")
    if notes:
        flags.append("has_context_note")
    if has_adjacent_duplicate_tokens(spanish) or has_adjacent_duplicate_tokens(chanka_quechua):
        flags.append("adjacent_duplicate_tokens")
    if "/" in spanish or "/" in chanka_quechua:
        flags.append("contains_alternative_marker")
    if not spanish or not chanka_quechua:
        flags.append("empty_side")
    return flags


def clean_row(raw_spanish: str, raw_chanka_quechua: str) -> tuple[str, str, list[str]]:
    spanish, spanish_notes = strip_parenthetical_notes(raw_spanish)
    chanka_quechua, chanka_notes = strip_parenthetical_notes(raw_chanka_quechua)
    return spanish, chanka_quechua, spanish_notes + chanka_notes


def extract_candidates(pdf_path: Path, min_page: int = 17, max_page: int = 118) -> pl.DataFrame:
    doc = fitz.open(pdf_path)
    records: list[dict[str, object]] = []

    for page_index, page in enumerate(doc, start=1):
        if page_index < min_page or page_index > max_page:
            continue

        tables = page.find_tables().tables
        for table_index, table in enumerate(tables):
            previous_role = ""
            previous_raw_spanish = ""
            for table_row_index, row in enumerate(table.extract()):
                if len(row) < 3:
                    continue

                raw_role = normalize_text(row[0])
                role = raw_role
                raw_spanish = normalize_text(row[1])
                raw_chanka_quechua = normalize_text(row[2])
                source_was_filled = False
                role_was_filled = False

                if not ROLE_RE.match(role):
                    if raw_chanka_quechua and not raw_spanish and previous_role and previous_raw_spanish:
                        role = previous_role
                        raw_spanish = previous_raw_spanish
                        role_was_filled = True
                        source_was_filled = True
                    else:
                        continue

                if not raw_spanish and previous_raw_spanish:
                    raw_spanish = previous_raw_spanish
                    source_was_filled = True
                elif raw_spanish and not role_was_filled:
                    previous_role = role
                    previous_raw_spanish = raw_spanish

                spanish, chanka_quechua, notes = clean_row(raw_spanish, raw_chanka_quechua)
                quality_flags = build_quality_flags(
                    spanish=spanish,
                    chanka_quechua=chanka_quechua,
                    raw_spanish=raw_spanish,
                    raw_chanka_quechua=raw_chanka_quechua,
                    notes=notes,
                    source_was_filled=source_was_filled,
                    role_was_filled=role_was_filled,
                )
                needs_review = bool(quality_flags)
                review_status = "needs_review" if needs_review else "auto_passed_table_checks"

                records.append(
                    {
                        "row_id": len(records) + 1,
                        "source_id": SOURCE_ID,
                        "source_file": SOURCE_FILE,
                        "page_number": page_index,
                        "table_index": table_index,
                        "table_row_index": table_row_index,
                        "raw_role": raw_role,
                        "role": role,
                        "spanish": spanish,
                        "chanka_quechua": chanka_quechua,
                        "raw_spanish": raw_spanish,
                        "raw_chanka_quechua": raw_chanka_quechua,
                        "notes": " | ".join(notes),
                        "direction": "spa_Latn-quy_Latn",
                        "license_note": "Source states reproduction is permitted when the source is cited.",
                        "extraction_status": "draft_table_extraction",
                        "quality_flags": ";".join(quality_flags),
                        "needs_review": needs_review,
                        "review_status": review_status,
                    }
                )

    return pl.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract draft Spanish-Chanka Quechua parallel candidates from the justice manual PDF."
    )
    parser.add_argument("--pdf", type=Path, default=Path(SOURCE_FILE))
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/manual_quechua_chanka_parallel_candidates.csv"),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/processed/manual_quechua_chanka_parallel_candidates.parquet"),
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=Path("data/processed/manual_quechua_chanka_parallel_auto_passed.csv"),
    )
    parser.add_argument(
        "--clean-parquet",
        type=Path,
        default=Path("data/processed/manual_quechua_chanka_parallel_auto_passed.parquet"),
    )
    parser.add_argument("--min-page", type=int, default=17)
    parser.add_argument("--max-page", type=int, default=118)
    args = parser.parse_args()

    df = extract_candidates(args.pdf, min_page=args.min_page, max_page=args.max_page)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    args.clean_csv.parent.mkdir(parents=True, exist_ok=True)
    args.clean_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.csv)
    df.write_parquet(args.parquet)
    clean_df = df.filter(~pl.col("needs_review"))
    clean_df.write_csv(args.clean_csv)
    clean_df.write_parquet(args.clean_parquet)
    print(f"Extracted {df.height} draft parallel candidates")
    print(f"Auto-passed {clean_df.height} candidates")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.parquet}")
    print(f"Wrote {args.clean_csv}")
    print(f"Wrote {args.clean_parquet}")


if __name__ == "__main__":
    main()
