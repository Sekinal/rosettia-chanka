"""Measure oracle reranking headroom from multi-candidate prediction JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_gspo_chanka_unsloth as gspo
from scripts.summarize_gspo_canaries import selection_score


@dataclass(frozen=True)
class Candidate:
    source: str
    reference: str
    prediction: str
    source_name: str | None = None
    variant: str | None = None
    candidate_index: int = 0
    pool_path: str | None = None

    @property
    def key(self) -> tuple[str, str, str | None, str | None]:
        return (self.source, self.reference, self.source_name, self.variant)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="rerank",
        help="Prefix for metrics/prediction output files.",
    )
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


def load_candidates(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    group_counts: dict[tuple[str, str, str | None, str | None], int] = {}
    for record in iter_jsonl(path):
        candidate = Candidate(
            source=str(record["source"]),
            reference=str(record["reference"]),
            prediction=str(record["prediction"]),
            source_name=record.get("source_name"),
            variant=record.get("variant"),
            candidate_index=0,
            pool_path=record.get("pool_path"),
        )
        index = group_counts.get(candidate.key, 0)
        group_counts[candidate.key] = index + 1
        candidates.append(
            Candidate(
                source=candidate.source,
                reference=candidate.reference,
                prediction=candidate.prediction,
                source_name=candidate.source_name,
                variant=candidate.variant,
                candidate_index=index,
                pool_path=candidate.pool_path,
            )
        )
    return candidates


def group_candidates(candidates: Sequence[Candidate]) -> list[list[Candidate]]:
    grouped: dict[tuple[str, str, str | None, str | None], list[Candidate]] = {}
    order: list[tuple[str, str, str | None, str | None]] = []
    for candidate in candidates:
        if candidate.key not in grouped:
            grouped[candidate.key] = []
            order.append(candidate.key)
        grouped[candidate.key].append(candidate)
    return [grouped[key] for key in order]


def candidate_oracle_score(candidate: Candidate) -> float:
    chrf = gspo.sentence_chrfpp(candidate.prediction, candidate.reference)
    bleu = gspo.sentence_bleu(candidate.prediction, candidate.reference)
    token_f1 = gspo.token_f1(candidate.prediction, candidate.reference)
    length = gspo.length_ratio_score(candidate.prediction, candidate.reference)
    return gspo.reference_rerank_metric_score(
        candidate.prediction,
        candidate.reference,
        candidate.source,
        chrf,
        bleu,
        token_f1,
        length,
    )


def select_first(groups: Sequence[Sequence[Candidate]]) -> list[Candidate]:
    return [group[0] for group in groups if group]


def select_oracle(groups: Sequence[Sequence[Candidate]]) -> list[Candidate]:
    selected: list[Candidate] = []
    for group in groups:
        if not group:
            continue
        selected.append(
            max(
                group,
                key=lambda candidate: (
                    candidate_oracle_score(candidate),
                    -candidate.candidate_index,
                ),
            )
        )
    return selected


def metrics_for_selection(
    selected: Sequence[Candidate],
    method: str,
    source_path: Path,
    total_candidates: int,
) -> dict[str, Any]:
    predictions = [candidate.prediction for candidate in selected]
    references = [candidate.reference for candidate in selected]
    sources = [candidate.source for candidate in selected]
    metrics: dict[str, Any] = gspo.corpus_metrics(predictions, references, sources)
    metrics["selection_score"] = selection_score(metrics)
    metrics["method"] = method
    metrics["prediction_groups"] = len(selected)
    metrics["total_candidates"] = total_candidates
    metrics["mean_candidates_per_group"] = total_candidates / max(1, len(selected))
    metrics["predictions_jsonl"] = str(source_path)
    if method == "oracle":
        non_first = [candidate for candidate in selected if candidate.candidate_index != 0]
        metrics["oracle_non_first_rate"] = 100.0 * len(non_first) / max(1, len(selected))
        metrics["oracle_mean_selected_index"] = sum(candidate.candidate_index for candidate in selected) / max(1, len(selected))
    return metrics


def write_predictions(path: Path, selected: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for candidate in selected:
            handle.write(
                json.dumps(
                    {
                        "candidate_index": candidate.candidate_index,
                        "prediction": candidate.prediction,
                        "reference": candidate.reference,
                        "source": candidate.source,
                        "source_name": candidate.source_name,
                        "variant": candidate.variant,
                        "pool_path": candidate.pool_path,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_summary(path: Path, records: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Candidate Reranking Headroom",
        "",
        "The oracle method uses references and is not deployable. It measures whether sampled candidates contain better translations than the first decoded candidate.",
        "",
        "| Method | Selection | chrF++ | BLEU | token F1 | source copy % | exact copy % | leakage % | artifact % | TER | non-first % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["method"]),
                    f"{float(record.get('selection_score', 0.0)):.4f}",
                    f"{float(record.get('chrf++', 0.0)):.4f}",
                    f"{float(record.get('bleu', 0.0)):.4f}",
                    f"{float(record.get('token_f1', 0.0)):.4f}",
                    f"{float(record.get('source_copy_ratio', 0.0)):.4f}",
                    f"{float(record.get('exact_source_copy_rate', 0.0)):.4f}",
                    f"{float(record.get('spanish_leakage_penalty', 0.0)):.4f}",
                    f"{float(record.get('chat_artifact_penalty', 0.0)):.4f}",
                    f"{float(record.get('ter', 0.0)):.4f}",
                    f"{float(record.get('oracle_non_first_rate', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.predictions_jsonl)
    groups = group_candidates(candidates)
    if not groups:
        raise ValueError(f"No candidates found in {args.predictions_jsonl}")

    first = select_first(groups)
    oracle = select_oracle(groups)
    records = [
        metrics_for_selection(first, "first", args.predictions_jsonl, len(candidates)),
        metrics_for_selection(oracle, "oracle", args.predictions_jsonl, len(candidates)),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for selected, record in [(first, records[0]), (oracle, records[1])]:
        method = record["method"]
        write_predictions(args.output_dir / f"{args.prefix}_{method}_predictions.jsonl", selected)
        write_metrics(args.output_dir / f"{args.prefix}_{method}_metrics.json", record)
    write_summary(args.output_dir / f"{args.prefix}_summary.md", records)
    write_metrics(args.output_dir / f"{args.prefix}_summary.json", {"records": records})
    print(json.dumps({"records": records}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
