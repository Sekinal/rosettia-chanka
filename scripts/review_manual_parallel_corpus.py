from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


DEFAULT_INPUT = Path("data/processed/manual_quechua_chanka_parallel_candidates.parquet")
DEFAULT_REVIEWED_CSV = Path("data/processed/manual_quechua_chanka_parallel_reviewed.csv")
DEFAULT_REVIEWED_PARQUET = Path("data/processed/manual_quechua_chanka_parallel_reviewed.parquet")
DEFAULT_TRAINING_READY_CSV = Path("data/processed/manual_quechua_chanka_parallel_training_ready.csv")
DEFAULT_TRAINING_READY_PARQUET = Path("data/processed/manual_quechua_chanka_parallel_training_ready.parquet")
DEFAULT_REPORT = Path("data/interim/manual_review_report.md")


def correct_pair(spanish: str, chanka_quechua: str) -> tuple[str, str, list[str]]:
    corrections: list[str] = []
    reviewed_spanish = spanish.replace("pare ja", "pareja")
    reviewed_chanka_quechua = chanka_quechua.replace("?.", "?").replace("!.", "!")

    duplicated = "¿Hayka watataq kuska tiyarankichik?tiyarankichik?"
    if reviewed_chanka_quechua == duplicated:
        reviewed_chanka_quechua = "¿Hayka watataq kuska tiyarankichik?"

    if reviewed_spanish != spanish:
        corrections.append("corrected_broken_spanish_word")
    if reviewed_chanka_quechua != chanka_quechua:
        corrections.append("corrected_target_punctuation_or_duplicate")

    return reviewed_spanish, reviewed_chanka_quechua, corrections


def review_row(row: dict[str, object]) -> dict[str, object]:
    spanish = str(row["spanish"])
    chanka_quechua = str(row["chanka_quechua"])
    flags = str(row["quality_flags"])
    reviewed_spanish, reviewed_chanka_quechua, corrections = correct_pair(spanish, chanka_quechua)

    include = True
    training_ready = True
    decision = "auto_passed_table_checks"
    review_notes: list[str] = []

    is_page_104_bad_row = (
        row["page_number"] == 104
        and row["table_index"] == 0
        and row["table_row_index"] == 5
        and spanish == "Mi hija me avisó"
    )
    if is_page_104_bad_row:
        include = False
        training_ready = False
        decision = "reviewed_exclude_semantic_mismatch"
        review_notes.append(
            "Visual check: Spanish cell says 'Mi hija me avisó', but target is a Quechua question about how it was seen."
        )
    elif "contains_alternative_marker" in flags:
        training_ready = False
        decision = "reviewed_keep_needs_split"
        review_notes.append("Valid row, but slash alternatives should be split or normalized before training.")
    elif "source_filled_from_previous_row" in flags or "role_filled_from_previous_row" in flags:
        decision = "reviewed_approved_alternative_translation"
        review_notes.append("Visual/table review: blank role/source cell is an additional translation for the previous Spanish prompt.")
    elif "has_context_note" in flags:
        decision = "reviewed_approved_context_note_removed"
        review_notes.append("Context note was preserved in notes/raw fields and removed from reviewed text.")
    elif flags:
        decision = "reviewed_approved_flag_checked"
        review_notes.append("Flag checked against table extraction and accepted.")

    if corrections and include:
        decision = "reviewed_corrected" if training_ready else decision
        review_notes.extend(corrections)

    output = dict(row)
    output.update(
        {
            "reviewed_spanish": reviewed_spanish,
            "reviewed_chanka_quechua": reviewed_chanka_quechua,
            "include_in_reviewed_dataset": include,
            "training_ready": training_ready and include,
            "review_decision": decision,
            "manual_review_notes": " | ".join(review_notes),
        }
    )
    return output


def write_report(df: pl.DataFrame, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    included = df.filter(pl.col("include_in_reviewed_dataset"))
    training_ready = included.filter(pl.col("training_ready"))
    needs_split = included.filter(~pl.col("training_ready"))
    excluded = df.filter(~pl.col("include_in_reviewed_dataset"))

    lines = [
        "# Manual Corpus Review Report",
        "",
        "This is a local generated report. Data artifacts remain ignored by git.",
        "",
        f"- Total extracted rows reviewed: {df.height}",
        f"- Included after review: {included.height}",
        f"- Training-ready after review: {training_ready.height}",
        f"- Kept but needs split/normalization before training: {needs_split.height}",
        f"- Excluded after review: {excluded.height}",
        "",
        "## Decisions",
        "",
    ]

    for row in df.group_by("review_decision").len().sort("review_decision").iter_rows(named=True):
        lines.append(f"- `{row['review_decision']}`: {row['len']}")

    if excluded.height:
        lines.extend(["", "## Excluded Rows", ""])
        for row in excluded.select(
            [
                "row_id",
                "page_number",
                "table_index",
                "table_row_index",
                "spanish",
                "chanka_quechua",
                "manual_review_notes",
            ]
        ).iter_rows(named=True):
            lines.append(
                f"- row {row['row_id']} page {row['page_number']} table {row['table_index']} row {row['table_row_index']}: "
                f"{row['manual_review_notes']}"
            )
            lines.append(f"  - Spanish: {row['spanish']}")
            lines.append(f"  - Chanka Quechua: {row['chanka_quechua']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual review decisions to extracted corpus candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reviewed-csv", type=Path, default=DEFAULT_REVIEWED_CSV)
    parser.add_argument("--reviewed-parquet", type=Path, default=DEFAULT_REVIEWED_PARQUET)
    parser.add_argument("--training-ready-csv", type=Path, default=DEFAULT_TRAINING_READY_CSV)
    parser.add_argument("--training-ready-parquet", type=Path, default=DEFAULT_TRAINING_READY_PARQUET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    df = pl.read_parquet(args.input)
    reviewed = pl.DataFrame([review_row(row) for row in df.iter_rows(named=True)])
    included = reviewed.filter(pl.col("include_in_reviewed_dataset"))
    training_ready = included.filter(pl.col("training_ready"))

    args.reviewed_csv.parent.mkdir(parents=True, exist_ok=True)
    reviewed.write_csv(args.reviewed_csv)
    reviewed.write_parquet(args.reviewed_parquet)
    training_ready.write_csv(args.training_ready_csv)
    training_ready.write_parquet(args.training_ready_parquet)
    write_report(reviewed, args.report)

    print(f"Reviewed {reviewed.height} rows")
    print(f"Included {included.height} rows")
    print(f"Training-ready {training_ready.height} rows")
    print(f"Wrote {args.reviewed_csv}")
    print(f"Wrote {args.reviewed_parquet}")
    print(f"Wrote {args.training_ready_csv}")
    print(f"Wrote {args.training_ready_parquet}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
