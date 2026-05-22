"""Build preference pairs from multi-candidate Chanka predictions.

The references are used only to label training-set candidate pools. Do not use
this for held-out eval data except as an oracle diagnostic.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_gspo_chanka_unsloth as gspo
from scripts.summarize_gspo_canaries import selection_score


@dataclass(frozen=True)
class PreferencePair:
    source: str
    reference: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    chosen_index: int
    rejected_index: int
    candidate_count: int
    source_name: str | None = None
    variant: str | None = None

    @property
    def margin(self) -> float:
        return self.chosen_score - self.rejected_score


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--min-candidates", type=int, default=4)
    parser.add_argument("--min-margin", type=float, default=0.50)
    parser.add_argument("--max-source-copy-ratio", type=float, default=0.60)
    parser.add_argument("--max-spanish-leakage-penalty", type=float, default=0.25)
    parser.add_argument("--max-chat-artifact-penalty", type=float, default=0.0)
    parser.add_argument("--allow-exact-source-copy", action="store_true")
    parser.add_argument(
        "--rejected-strategy",
        choices=["hard", "worst", "first"],
        default="hard",
        help="Hard selects the highest-scoring non-chosen candidate below the chosen candidate.",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args(argv)


def candidate_passes_output_filters(
    candidate: oracle_rerank.Candidate,
    args: argparse.Namespace,
) -> bool:
    if gspo.source_copy_ratio(candidate.prediction, candidate.source) > args.max_source_copy_ratio:
        return False
    if gspo.spanish_leakage_penalty(candidate.prediction) > args.max_spanish_leakage_penalty:
        return False
    if gspo.chat_artifact_penalty(candidate.prediction) > args.max_chat_artifact_penalty:
        return False
    if not args.allow_exact_source_copy and gspo.exact_source_copy(candidate.prediction, candidate.source):
        return False
    return True


def scored_group(group: Sequence[oracle_rerank.Candidate]) -> list[tuple[oracle_rerank.Candidate, float]]:
    deduped: dict[str, tuple[oracle_rerank.Candidate, float]] = {}
    for candidate in group:
        prediction_key = gspo.normalize_text(candidate.prediction).lower()
        score = oracle_rerank.candidate_oracle_score(candidate)
        previous = deduped.get(prediction_key)
        if previous is None or score > previous[1]:
            deduped[prediction_key] = (candidate, score)
    return sorted(deduped.values(), key=lambda item: (item[1], -item[0].candidate_index), reverse=True)


def choose_rejected(
    ranked: Sequence[tuple[oracle_rerank.Candidate, float]],
    chosen: oracle_rerank.Candidate,
    strategy: str,
) -> tuple[oracle_rerank.Candidate, float] | None:
    alternatives = [(candidate, score) for candidate, score in ranked if candidate.prediction != chosen.prediction]
    if not alternatives:
        return None
    if strategy == "worst":
        return min(alternatives, key=lambda item: (item[1], item[0].candidate_index))
    if strategy == "first":
        first = min(alternatives, key=lambda item: item[0].candidate_index)
        return first
    return alternatives[0]


def build_pair(group: Sequence[oracle_rerank.Candidate], args: argparse.Namespace) -> PreferencePair | None:
    if len(group) < args.min_candidates:
        return None
    ranked = scored_group(group)
    if len(ranked) < 2:
        return None
    chosen, chosen_score = ranked[0]
    if not candidate_passes_output_filters(chosen, args):
        return None
    rejected_item = choose_rejected(ranked, chosen, args.rejected_strategy)
    if rejected_item is None:
        return None
    rejected, rejected_score = rejected_item
    if chosen_score - rejected_score < args.min_margin:
        return None
    return PreferencePair(
        source=chosen.source,
        reference=chosen.reference,
        chosen=chosen.prediction,
        rejected=rejected.prediction,
        chosen_score=chosen_score,
        rejected_score=rejected_score,
        chosen_index=chosen.candidate_index,
        rejected_index=rejected.candidate_index,
        candidate_count=len(group),
        source_name=chosen.source_name,
        variant=chosen.variant,
    )


def pair_to_record(pair: PreferencePair) -> dict[str, Any]:
    return {
        "source": pair.source,
        "reference": pair.reference,
        "chosen": pair.chosen,
        "rejected": pair.rejected,
        "chosen_score": pair.chosen_score,
        "rejected_score": pair.rejected_score,
        "score_margin": pair.margin,
        "chosen_index": pair.chosen_index,
        "rejected_index": pair.rejected_index,
        "candidate_count": pair.candidate_count,
        "source_name": pair.source_name,
        "variant": pair.variant,
        "label_type": "oracle_preference_pair",
    }


def metrics_for_pairs(
    pairs: Sequence[PreferencePair],
    args: argparse.Namespace,
    total_groups: int,
    total_candidates: int,
) -> dict[str, Any]:
    chosen_candidates = [
        oracle_rerank.Candidate(
            pair.source,
            pair.reference,
            pair.chosen,
            source_name=pair.source_name,
            variant=pair.variant,
            candidate_index=pair.chosen_index,
        )
        for pair in pairs
    ]
    rejected_candidates = [
        oracle_rerank.Candidate(
            pair.source,
            pair.reference,
            pair.rejected,
            source_name=pair.source_name,
            variant=pair.variant,
            candidate_index=pair.rejected_index,
        )
        for pair in pairs
    ]
    metrics: dict[str, Any] = {
        "predictions_jsonl": str(args.predictions_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "input_groups": total_groups,
        "input_candidates": total_candidates,
        "kept_pairs": len(pairs),
        "kept_rate": len(pairs) / max(1, total_groups),
        "filters": {
            "min_candidates": args.min_candidates,
            "min_margin": args.min_margin,
            "max_source_copy_ratio": args.max_source_copy_ratio,
            "max_spanish_leakage_penalty": args.max_spanish_leakage_penalty,
            "max_chat_artifact_penalty": args.max_chat_artifact_penalty,
            "allow_exact_source_copy": args.allow_exact_source_copy,
            "rejected_strategy": args.rejected_strategy,
            "max_pairs": args.max_pairs,
        },
        "mean_margin": statistics.fmean(pair.margin for pair in pairs) if pairs else 0.0,
        "mean_chosen_score": statistics.fmean(pair.chosen_score for pair in pairs) if pairs else 0.0,
        "mean_rejected_score": statistics.fmean(pair.rejected_score for pair in pairs) if pairs else 0.0,
    }
    if pairs:
        chosen_quality = oracle_rerank.metrics_for_selection(
            chosen_candidates,
            "chosen",
            args.predictions_jsonl,
            total_candidates,
        )
        rejected_quality = oracle_rerank.metrics_for_selection(
            rejected_candidates,
            "rejected",
            args.predictions_jsonl,
            total_candidates,
        )
        chosen_quality["selection_score"] = selection_score(chosen_quality)
        rejected_quality["selection_score"] = selection_score(rejected_quality)
        metrics["chosen_quality"] = chosen_quality
        metrics["rejected_quality"] = rejected_quality
    return metrics


def build_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = oracle_rerank.load_candidates(args.predictions_jsonl)
    groups = oracle_rerank.group_candidates(candidates)
    pairs = [pair for group in groups if (pair := build_pair(group, args)) is not None]
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    return [pair_to_record(pair) for pair in pairs], metrics_for_pairs(pairs, args, len(groups), len(candidates))


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    chosen = metrics.get("chosen_quality", {})
    rejected = metrics.get("rejected_quality", {})
    lines = [
        "# Oracle Preference Pairs",
        "",
        f"- Input groups: `{metrics['input_groups']}`",
        f"- Kept pairs: `{metrics['kept_pairs']}` (`{100 * metrics['kept_rate']:.2f}%`)",
        f"- Mean margin: `{metrics['mean_margin']:.4f}`",
        "",
        "| Side | Selection | chrF++ | BLEU | token F1 | source copy % | leakage % | TER |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, quality in [("chosen", chosen), ("rejected", rejected)]:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{quality.get('selection_score', 0.0):.4f}",
                    f"{quality.get('chrf++', 0.0):.4f}",
                    f"{quality.get('bleu', 0.0):.4f}",
                    f"{quality.get('token_f1', 0.0):.4f}",
                    f"{quality.get('source_copy_ratio', 0.0):.4f}",
                    f"{quality.get('spanish_leakage_penalty', 0.0):.4f}",
                    f"{quality.get('ter', 0.0):.4f}",
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    records, metrics = build_records(args)
    write_jsonl(args.output_jsonl, records)
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if args.summary_md:
        write_summary(args.summary_md, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
