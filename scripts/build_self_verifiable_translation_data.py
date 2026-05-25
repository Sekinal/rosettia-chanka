"""Build DeepSeekMath-V2-style self-verification data for Chanka translation.

The paper trains three related behaviors: verify a solution, meta-verify the
verifier's analysis, and generate a solution with faithful self-analysis. This
script creates cold-start JSONL for the analogous translation loop from clean
Spanish -> Chanka pairs plus deterministic corruptions.
"""

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
from scripts import train_verifier_chanka_unsloth as verifier


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--dataset-file", default=gspo.CHANKA_FILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args(argv)


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = gspo.load_chanka_rows(args.dataset_repo, args.dataset_file)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    return rows


def verifier_analysis_for_label(label_json: str) -> str:
    label = json.loads(label_json)
    score = float(label["score"])
    severity = str(label["severity"])
    rationale = str(label["rationale"])
    if score >= 0.9:
        issue = "No encuentro errores importantes; conserva el significado y la forma chanka esperada."
    elif score >= 0.7:
        issue = f"Hay un problema menor: {rationale}."
    elif score >= 0.4:
        issue = f"Hay un problema mayor: {rationale}."
    else:
        issue = f"Hay un problema critico: {rationale}."
    return (
        f"Resumen de problemas: {issue}\n"
        f"Severidad: {severity}\n"
        f"Puntaje: \\boxed{{{score:.2f}}}"
    )


def flawed_analysis_for_label(label_json: str) -> str:
    label = json.loads(label_json)
    score = float(label["score"])
    if score >= 0.85:
        wrong_score = 0.0
        issue = "Afirma falsamente que la traduccion es inservible aunque coincide con la referencia."
    else:
        wrong_score = 1.0
        issue = "Afirma falsamente que no hay errores aunque la candidata tiene problemas."
    return (
        f"Resumen de problemas: {issue}\n"
        "Severidad: incorrecta\n"
        f"Puntaje: \\boxed{{{wrong_score:.2f}}}"
    )


def meta_label_for_analysis(label_json: str, analysis: str, faithful: bool) -> str:
    label = json.loads(label_json)
    if faithful:
        return verifier.verifier_target(0.98, "none", "analysis_identifies_real_translation_issues")
    severity = "critical" if float(label["score"]) < 0.5 else "major"
    return verifier.verifier_target(0.05, severity, "analysis_hallucinates_or_hides_translation_issues")


def generator_target(reference: str) -> str:
    return (
        f"Traduccion final: {gspo.normalize_text(reference)}\n"
        "Autoevaluacion: No veo errores importantes; conserva el significado, no copia el espanol y usa quechua chanka natural.\n"
        "Puntaje: \\boxed{0.98}"
    )


def thinking_generator_target(reference: str) -> str:
    return (
        "Analisis de traduccion: [SIGNIFICADO] conserva el sentido; [ANTI_COPIA] evita copiar espanol y mantiene quechua chanka natural.\n"
        f"Traduccion final: {gspo.normalize_text(reference)}\n"
        "Autoevaluacion: No veo errores importantes; conserva el significado, no copia el espanol y usa quechua chanka natural.\n"
        "Puntaje: \\boxed{0.98}"
    )


def compact_thinking_generator_target(reference: str) -> str:
    return (
        "Analisis: [SIGNIFICADO] conserva sentido; [GRAMATICA] mantiene chanka natural.\n"
        f"Final: {gspo.normalize_text(reference)}\n"
        "Puntaje: \\boxed{0.98}"
    )


def build_records(rows: Iterable[dict[str, str]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    base_rows = list(rows)
    distractors = [row["target"] for row in base_rows]
    verifier_records: list[dict[str, Any]] = []
    meta_records: list[dict[str, Any]] = []
    generator_records: list[dict[str, Any]] = []

    for row in base_rows:
        examples = verifier.verifier_examples_for_row(row, rng, distractors=distractors)
        verifier_records.extend(examples)
        generator_records.append(
            {
                "source": row["source"],
                "reference": row["target"],
                "target": generator_target(row["target"]),
                "source_name": row.get("source_name"),
                "variant": row.get("variant"),
                "task": "self_verifiable_translation_generation",
            }
        )
        for example in examples:
            faithful_analysis = verifier_analysis_for_label(example["label"])
            flawed_analysis = flawed_analysis_for_label(example["label"])
            for analysis, faithful in ((faithful_analysis, True), (flawed_analysis, False)):
                meta_records.append(
                    {
                        "source": example["source"],
                        "reference": example["reference"],
                        "candidate": example["candidate"],
                        "analysis": analysis,
                        "label": meta_label_for_analysis(example["label"], analysis, faithful),
                    }
                )

    rng.shuffle(verifier_records)
    rng.shuffle(meta_records)
    rng.shuffle(generator_records)
    return verifier_records, meta_records, generator_records


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    verifier_records, meta_records, generator_records = build_records(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "translation_verifier_cold_start.jsonl", verifier_records)
    write_jsonl(args.output_dir / "translation_meta_verifier_cold_start.jsonl", meta_records)
    write_jsonl(args.output_dir / "self_verifiable_generator_sft.jsonl", generator_records)
    thinking_generator_records = [
        {
            **record,
            "target": thinking_generator_target(record["reference"]),
            "task": "self_verifiable_thinking_translation_generation",
        }
        for record in generator_records
    ]
    write_jsonl(args.output_dir / "self_verifiable_thinking_generator_sft.jsonl", thinking_generator_records)
    compact_thinking_generator_records = [
        {
            **record,
            "target": compact_thinking_generator_target(record["reference"]),
            "task": "self_verifiable_compact_thinking_translation_generation",
        }
        for record in generator_records
    ]
    write_jsonl(
        args.output_dir / "self_verifiable_compact_thinking_generator_sft.jsonl",
        compact_thinking_generator_records,
    )
    summary = {
        "source_rows": len(rows),
        "verifier_records": len(verifier_records),
        "meta_verifier_records": len(meta_records),
        "generator_records": len(generator_records),
        "thinking_generator_records": len(thinking_generator_records),
        "compact_thinking_generator_records": len(compact_thinking_generator_records),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
