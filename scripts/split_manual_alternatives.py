from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl


DEFAULT_INPUT = Path("data/processed/manual_quechua_chanka_parallel_reviewed.parquet")
DEFAULT_OUTPUT_CSV = Path("data/processed/manual_quechua_chanka_parallel_alternative_splits.csv")
DEFAULT_OUTPUT_PARQUET = Path("data/processed/manual_quechua_chanka_parallel_alternative_splits.parquet")
DEFAULT_AUGMENTED_CSV = Path("data/processed/manual_quechua_chanka_parallel_training_ready_augmented.csv")
DEFAULT_AUGMENTED_PARQUET = Path("data/processed/manual_quechua_chanka_parallel_training_ready_augmented.parquet")


def split_with_prefix(value: str, separator: str = "/") -> list[str]:
    parts = [part.strip() for part in value.split(separator)]
    if len(parts) <= 1:
        return [value]
    first = parts[0]
    prefix_match = re.match(r"^(.*\s)(\S+)$", first)
    if not prefix_match:
        return parts
    prefix = prefix_match.group(1)
    return [first] + [prefix + part for part in parts[1:]]


def expand_spanish(value: str) -> list[str]:
    manual: dict[str, list[str]] = {
        "Bien, doctor/a": ["Bien, doctor", "Bien, doctora"],
        "Bien, jefe/a": ["Bien, jefe", "Bien, jefa"],
        "Soy soltera/o": ["Soy soltera", "Soy soltero"],
        "Estoy separado/divorciado Estoy separada/divorciada": [
            "Estoy separado",
            "Estoy divorciado",
            "Estoy separada",
            "Estoy divorciada",
        ],
        "¿Varones?/¿Mujeres?": ["¿Varones?", "¿Mujeres?"],
        "Con lliclla/manta": ["Con lliclla", "Con manta"],
        "Bebe demasiado/es un borracho": ["Bebe demasiado", "Es un borracho"],
        "Señale cuáles fueron los bienes que le robaron/¿Qué le robaron?": [
            "Señale cuáles fueron los bienes que le robaron",
            "¿Qué le robaron?",
        ],
        "¿Con quién se escapó su hija?/¿Quién se llevó a su hija?": [
            "¿Con quién se escapó su hija?",
            "¿Quién se llevó a su hija?",
        ],
        "No conozco a ese hombre/No conozco a esa mujer.": [
            "No conozco a ese hombre.",
            "No conozco a esa mujer.",
        ],
        "Un psicólogo va a hacerle preguntas a su hija/o para ayudarla/o": [
            "Un psicólogo va a hacerle preguntas a su hija para ayudarla",
            "Un psicólogo va a hacerle preguntas a su hijo para ayudarlo",
        ],
        "Te vamos a ayudar/le vamos a ayudar": ["Te vamos a ayudar", "Le vamos a ayudar"],
    }
    if value in manual:
        return manual[value]

    replacements: list[tuple[str, tuple[str, str]]] = [
        ("él/ella", ("él", "ella")),
        ("el/ella", ("el", "ella")),
        ("hija/hijo", ("hija", "hijo")),
        ("hijo/hija", ("hijo", "hija")),
        ("hija/o", ("hija", "hijo")),
        ("niño/a", ("niño", "niña")),
        ("vecino/a", ("vecino", "vecina")),
        ("hombre/mujer", ("hombre", "mujer")),
        ("esposo/pareja/marido", ("esposo", "pareja", "marido")),
        ("esposa/pareja/mujer", ("esposa", "pareja", "mujer")),
    ]
    for needle, options in replacements:
        if needle in value:
            return [value.replace(needle, option) for option in options]

    if "/" in value:
        return split_with_prefix(value)
    return [value]


def expand_chanka(value: str) -> list[str]:
    manual: dict[str, list[str]] = {
        "Allinllam, doctor/a": ["Allinllam, doctor", "Allinllam, doctora"],
        "Allinllam, jefe/a": ["Allinllam, jefe", "Allinllam, jefa"],
        "¿Qari wawakunachu?/¿Warmi wawakunachu?": [
            "¿Qari wawakunachu?",
            "¿Warmi wawakunachu?",
        ],
        "¿Qari churikunachu?/¿Warmi churikunachu?": [
            "¿Qari churikunachu?",
            "¿Warmi churikunachu?",
        ],
        "Anqas ñawiyuq/Azul ñawiyuq": ["Anqas ñawiyuq", "Azul ñawiyuq"],
        "Anqas warayuq/Azul pantaloyuq": ["Anqas warayuq", "Azul pantaloyuq"],
        "Qumir unkuyuq/Verde chompayuq": ["Qumir unkuyuq", "Verde chompayuq"],
        "Llumpaytam machan/Tomakuqmi": ["Llumpaytam machan", "Tomakuqmi"],
    }
    if value in manual:
        return manual[value]
    if "/" in value:
        return [part.strip() for part in value.split("/")]
    return [value]


def split_row(row: dict[str, object]) -> list[dict[str, object]]:
    spanish_options = expand_spanish(str(row["reviewed_spanish"]))
    chanka_options = expand_chanka(str(row["reviewed_chanka_quechua"]))

    if len(spanish_options) == len(chanka_options):
        pairs = list(zip(spanish_options, chanka_options, strict=True))
        strategy = "pairwise"
    elif len(spanish_options) == 1:
        pairs = [(spanish_options[0], chanka) for chanka in chanka_options]
        strategy = "duplicate_spanish"
    elif len(chanka_options) == 1:
        pairs = [(spanish, chanka_options[0]) for spanish in spanish_options]
        strategy = "duplicate_chanka"
    else:
        pairs = [(spanish, chanka) for spanish in spanish_options for chanka in chanka_options]
        strategy = "cartesian_review_recommended"

    records: list[dict[str, object]] = []
    for index, (spanish, chanka) in enumerate(pairs, start=1):
        output = dict(row)
        output.update(
            {
                "source_reviewed_row_id": row["row_id"],
                "split_index": index,
                "split_strategy": strategy,
                "reviewed_spanish": spanish,
                "reviewed_chanka_quechua": chanka,
                "training_ready": strategy != "cartesian_review_recommended",
                "review_decision": "reviewed_alternative_split",
                "manual_review_notes": (
                    str(row.get("manual_review_notes") or "")
                    + " | split from slash-alternative row"
                ).strip(" |"),
            }
        )
        records.append(output)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Split reviewed slash-alternative rows into separate candidate pairs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--augmented-csv", type=Path, default=DEFAULT_AUGMENTED_CSV)
    parser.add_argument("--augmented-parquet", type=Path, default=DEFAULT_AUGMENTED_PARQUET)
    args = parser.parse_args()

    reviewed = pl.read_parquet(args.input)
    split_source = reviewed.filter(pl.col("review_decision") == "reviewed_keep_needs_split")
    split_records = [record for row in split_source.iter_rows(named=True) for record in split_row(row)]
    split_df = pl.DataFrame(split_records)

    base_training = reviewed.filter(
        pl.col("include_in_reviewed_dataset")
        & pl.col("training_ready")
        & (pl.col("review_decision") != "reviewed_keep_needs_split")
    )
    augmented = pl.concat([base_training, split_df.filter(pl.col("training_ready"))], how="diagonal_relaxed")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.write_csv(args.output_csv)
    split_df.write_parquet(args.output_parquet)
    augmented.write_csv(args.augmented_csv)
    augmented.write_parquet(args.augmented_parquet)

    print(f"Split {split_source.height} slash-alternative rows into {split_df.height} rows")
    print(f"Training-ready split rows: {split_df.filter(pl.col('training_ready')).height}")
    print(f"Augmented training-ready rows: {augmented.height}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_parquet}")
    print(f"Wrote {args.augmented_csv}")
    print(f"Wrote {args.augmented_parquet}")


if __name__ == "__main__":
    main()
