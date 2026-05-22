"""Reference-free MBR-style reranking for multi-candidate Chanka predictions."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_gspo_chanka_unsloth as gspo
from scripts.summarize_gspo_canaries import selection_score


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="mbr")
    return parser.parse_args(argv)


def candidate_length_score(candidate: oracle_rerank.Candidate, group: Sequence[oracle_rerank.Candidate]) -> float:
    lengths = [max(1, len(gspo.word_tokens(item.prediction))) for item in group]
    median_length = statistics.median(lengths) if lengths else 1.0
    candidate_length = max(1, len(gspo.word_tokens(candidate.prediction)))
    ratio = candidate_length / max(1.0, float(median_length))
    import math

    return max(0.0, 1.0 - abs(math.log(ratio)))


def pairwise_utility(candidate: str, pseudo_reference: str) -> float:
    chrf = gspo.sentence_chrfpp(candidate, pseudo_reference)
    f1 = gspo.token_f1(candidate, pseudo_reference)
    precision = gspo.token_precision(candidate, pseudo_reference)
    return (0.55 * chrf) + (0.30 * f1) + (0.15 * precision)


def mbr_score(candidate: oracle_rerank.Candidate, group: Sequence[oracle_rerank.Candidate]) -> float:
    peers = [item for item in group if item.prediction != candidate.prediction]
    if not peers:
        peers = list(group)
    utility = sum(pairwise_utility(candidate.prediction, peer.prediction) for peer in peers) / max(1, len(peers))
    return (
        utility
        + (0.08 * candidate_length_score(candidate, group))
        - (0.28 * gspo.source_copy_ratio(candidate.prediction, candidate.source))
        - (0.32 * gspo.spanish_leakage_penalty(candidate.prediction))
        - (0.30 * gspo.chat_artifact_penalty(candidate.prediction))
        - (0.14 * gspo.repetition_penalty(candidate.prediction))
        - (0.30 if gspo.exact_source_copy(candidate.prediction, candidate.source) else 0.0)
    )


def select_mbr(groups: Sequence[Sequence[oracle_rerank.Candidate]]) -> list[oracle_rerank.Candidate]:
    selected: list[oracle_rerank.Candidate] = []
    for group in groups:
        selected.append(
            max(
                group,
                key=lambda candidate: (
                    mbr_score(candidate, group),
                    -candidate.candidate_index,
                ),
            )
        )
    return selected


def metrics_for_selection(
    selected: Sequence[oracle_rerank.Candidate],
    method: str,
    predictions_jsonl: Path,
    total_candidates: int,
) -> dict[str, Any]:
    metrics = oracle_rerank.metrics_for_selection(selected, method, predictions_jsonl, total_candidates)
    metrics["selection_score"] = selection_score(metrics)
    non_first = [candidate for candidate in selected if candidate.candidate_index != 0]
    if method == "mbr":
        metrics["mbr_non_first_rate"] = 100.0 * len(non_first) / max(1, len(selected))
        metrics["mbr_mean_selected_index"] = sum(candidate.candidate_index for candidate in selected) / max(1, len(selected))
    return metrics


def write_summary(path: Path, records: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# MBR Candidate Reranking",
        "",
        "MBR is reference-free: it selects the candidate most similar to other sampled candidates, with copy/leakage/artifact guards.",
        "",
        "| Method | Selection | chrF++ | BLEU | token F1 | source copy % | exact copy % | leakage % | artifact % | TER | non-first % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        non_first = record.get("mbr_non_first_rate", record.get("oracle_non_first_rate", 0.0))
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
                    f"{float(non_first):.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    candidates = oracle_rerank.load_candidates(args.predictions_jsonl)
    groups = oracle_rerank.group_candidates(candidates)
    if not groups:
        raise ValueError(f"No candidates found in {args.predictions_jsonl}")
    first = oracle_rerank.select_first(groups)
    mbr = select_mbr(groups)
    oracle = oracle_rerank.select_oracle(groups)
    records = [
        metrics_for_selection(first, "first", args.predictions_jsonl, len(candidates)),
        metrics_for_selection(mbr, "mbr", args.predictions_jsonl, len(candidates)),
        metrics_for_selection(oracle, "oracle", args.predictions_jsonl, len(candidates)),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for selected, record in [(first, records[0]), (mbr, records[1]), (oracle, records[2])]:
        oracle_rerank.write_predictions(args.output_dir / f"{args.prefix}_{record['method']}_predictions.jsonl", selected)
        oracle_rerank.write_metrics(args.output_dir / f"{args.prefix}_{record['method']}_metrics.json", record)
    oracle_rerank.write_metrics(args.output_dir / f"{args.prefix}_summary.json", {"records": records})
    write_summary(args.output_dir / f"{args.prefix}_summary.md", records)
    print(json.dumps({"records": records}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
