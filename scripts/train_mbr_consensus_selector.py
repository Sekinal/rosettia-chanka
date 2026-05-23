"""Train a reference-free MBR consensus selector on candidate pools.

This is a Deep Past-style MBR selector: at inference time it chooses the
candidate with the strongest consensus against the other candidates in the same
group, plus formatting/copy guards. References are used only to tune the
consensus weights on a training candidate pool.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import mbr_candidate_predictions as default_mbr
from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_gspo_chanka_unsloth as gspo
from scripts.summarize_gspo_canaries import selection_score


_CHRF_METRIC: Any | None = None
_BLEU_METRIC: Any | None = None

SIGNAL_NAMES = [
    "pairwise_chrf",
    "pairwise_bleu",
    "pairwise_token_f1",
    "pairwise_precision",
    "pairwise_jaccard",
    "length_consensus",
    "duplicate_rate",
    "source_copy_ratio",
    "spanish_leakage",
    "chat_artifact",
    "repetition",
    "exact_source_copy",
    "candidate_index_fraction",
]


@dataclass(frozen=True)
class ConsensusRow:
    candidate: oracle_rerank.Candidate
    signals: dict[str, float]
    oracle_score: float


@dataclass(frozen=True)
class ConsensusModel:
    signal_names: list[str]
    weights: dict[str, float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--eval-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--model-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="mbr_consensus")
    parser.add_argument("--search-iterations", type=int, default=6000)
    parser.add_argument("--initial-noise", type=float, default=0.30)
    parser.add_argument("--min-noise", type=float, default=0.01)
    parser.add_argument(
        "--max-peers",
        type=int,
        default=12,
        help="Maximum peer candidates to compare against each candidate. Use <=0 for all peers.",
    )
    parser.add_argument(
        "--utility-mode",
        choices=["fast", "sacrebleu"],
        default="fast",
        help="Pairwise utility backend. Fast uses local n-gram overlap approximations; sacrebleu is slower.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(gspo.word_tokens(left))
    right_tokens = set(gspo.word_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def cached_sentence_chrfpp(hypothesis: str, reference: str) -> float:
    global _CHRF_METRIC
    if _CHRF_METRIC is None:
        _CHRF_METRIC = gspo.load_sacrebleu().metrics.CHRF(word_order=2)
    return _CHRF_METRIC.sentence_score(hypothesis, [reference]).score / 100.0


def cached_sentence_bleu(hypothesis: str, reference: str) -> float:
    global _BLEU_METRIC
    if _BLEU_METRIC is None:
        _BLEU_METRIC = gspo.load_sacrebleu().metrics.BLEU(effective_order=True)
    return _BLEU_METRIC.sentence_score(hypothesis, [reference]).score / 100.0


def char_ngrams(text: str, n: int) -> collections.Counter[str]:
    normalized = gspo.normalize_text(text).lower()
    if len(normalized) < n:
        return collections.Counter([normalized]) if normalized else collections.Counter()
    return collections.Counter(normalized[index : index + n] for index in range(len(normalized) - n + 1))


def counter_overlap(left: collections.Counter[str], right: collections.Counter[str]) -> int:
    return sum(min(count, right.get(item, 0)) for item, count in left.items())


def fbeta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)


def fast_chrf_approx(hypothesis: str, reference: str) -> float:
    scores: list[float] = []
    for n in range(1, 7):
        hyp = char_ngrams(hypothesis, n)
        ref = char_ngrams(reference, n)
        if not hyp or not ref:
            scores.append(0.0)
            continue
        overlap = counter_overlap(hyp, ref)
        scores.append(fbeta_score(overlap / sum(hyp.values()), overlap / sum(ref.values()), beta=2.0))
    return statistics.fmean(scores) if scores else 0.0


def token_ngrams(tokens: Sequence[str], n: int) -> collections.Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return collections.Counter()
    return collections.Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def fast_bleu_approx(hypothesis: str, reference: str) -> float:
    hyp_tokens = gspo.word_tokens(hypothesis)
    ref_tokens = gspo.word_tokens(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    precisions: list[float] = []
    for n in range(1, 5):
        hyp = token_ngrams(hyp_tokens, n)
        ref = token_ngrams(ref_tokens, n)
        if not hyp:
            precisions.append(0.0)
            continue
        precisions.append(counter_overlap(hyp, ref) / sum(hyp.values()))
    length_penalty = min(1.0, len(hyp_tokens) / max(1, len(ref_tokens)))
    return statistics.fmean(precisions) * length_penalty


def pairwise_signal_means(
    candidate: oracle_rerank.Candidate,
    peers: Sequence[oracle_rerank.Candidate],
    utility_mode: str = "fast",
) -> dict[str, float]:
    if not peers:
        peers = [candidate]
    values: dict[str, list[float]] = {
        "pairwise_chrf": [],
        "pairwise_bleu": [],
        "pairwise_token_f1": [],
        "pairwise_precision": [],
        "pairwise_jaccard": [],
    }
    for peer in peers:
        if utility_mode == "sacrebleu":
            chrf = cached_sentence_chrfpp(candidate.prediction, peer.prediction)
            bleu = cached_sentence_bleu(candidate.prediction, peer.prediction)
        else:
            chrf = fast_chrf_approx(candidate.prediction, peer.prediction)
            bleu = fast_bleu_approx(candidate.prediction, peer.prediction)
        values["pairwise_chrf"].append(chrf)
        values["pairwise_bleu"].append(bleu)
        values["pairwise_token_f1"].append(gspo.token_f1(candidate.prediction, peer.prediction))
        values["pairwise_precision"].append(gspo.token_precision(candidate.prediction, peer.prediction))
        values["pairwise_jaccard"].append(jaccard_similarity(candidate.prediction, peer.prediction))
    return {name: statistics.fmean(items) for name, items in values.items()}


def candidate_length_consensus(
    candidate: oracle_rerank.Candidate,
    group: Sequence[oracle_rerank.Candidate],
) -> float:
    return default_mbr.candidate_length_score(candidate, group)


def duplicate_rate(candidate: oracle_rerank.Candidate, group: Sequence[oracle_rerank.Candidate]) -> float:
    normalized = gspo.normalize_text(candidate.prediction).lower()
    if not normalized:
        return 0.0
    matches = sum(1 for item in group if gspo.normalize_text(item.prediction).lower() == normalized)
    return matches / max(1, len(group))


def peer_subset(
    peers: Sequence[oracle_rerank.Candidate],
    max_peers: int,
) -> list[oracle_rerank.Candidate]:
    if max_peers <= 0 or len(peers) <= max_peers:
        return list(peers)
    if max_peers == 1:
        return [peers[0]]
    step = (len(peers) - 1) / (max_peers - 1)
    indices = sorted({round(index * step) for index in range(max_peers)})
    return [peers[index] for index in indices]


def signal_rows_for_group(
    group: Sequence[oracle_rerank.Candidate],
    max_peers: int = 12,
    utility_mode: str = "fast",
) -> list[ConsensusRow]:
    rows: list[ConsensusRow] = []
    max_index = max((candidate.candidate_index for candidate in group), default=0)
    for candidate in group:
        peers = peer_subset([item for item in group if item.prediction != candidate.prediction], max_peers)
        signals = pairwise_signal_means(candidate, peers, utility_mode=utility_mode)
        signals.update(
            {
                "length_consensus": candidate_length_consensus(candidate, group),
                "duplicate_rate": duplicate_rate(candidate, group),
                "source_copy_ratio": gspo.source_copy_ratio(candidate.prediction, candidate.source),
                "spanish_leakage": gspo.spanish_leakage_penalty(candidate.prediction),
                "chat_artifact": gspo.chat_artifact_penalty(candidate.prediction),
                "repetition": gspo.repetition_penalty(candidate.prediction),
                "exact_source_copy": 1.0 if gspo.exact_source_copy(candidate.prediction, candidate.source) else 0.0,
                "candidate_index_fraction": candidate.candidate_index / max(1, max_index),
            }
        )
        rows.append(
            ConsensusRow(
                candidate=candidate,
                signals=signals,
                oracle_score=oracle_rerank.candidate_oracle_score(candidate),
            )
        )
    return rows


def load_consensus_groups(
    paths: Sequence[Path],
    max_peers: int = 12,
    utility_mode: str = "fast",
) -> list[list[ConsensusRow]]:
    candidates = [candidate for path in paths for candidate in oracle_rerank.load_candidates(path)]
    groups = oracle_rerank.group_candidates(candidates)
    return [signal_rows_for_group(group, max_peers=max_peers, utility_mode=utility_mode) for group in groups]


def consensus_score(row: ConsensusRow, model: ConsensusModel) -> float:
    return sum(model.weights.get(name, 0.0) * row.signals[name] for name in model.signal_names)


def select_consensus(
    groups: Sequence[Sequence[ConsensusRow]],
    model: ConsensusModel,
) -> list[ConsensusRow]:
    selected: list[ConsensusRow] = []
    for group in groups:
        if not group:
            continue
        selected.append(
            max(
                group,
                key=lambda row: (
                    consensus_score(row, model),
                    -row.candidate.candidate_index,
                ),
            )
        )
    return selected


def mean_oracle_objective(groups: Sequence[Sequence[ConsensusRow]], model: ConsensusModel) -> float:
    selected = select_consensus(groups, model)
    if not selected:
        return 0.0
    return statistics.fmean(row.oracle_score for row in selected)


def initial_weights() -> dict[str, float]:
    return {
        "pairwise_chrf": 0.55,
        "pairwise_bleu": 0.08,
        "pairwise_token_f1": 0.20,
        "pairwise_precision": 0.10,
        "pairwise_jaccard": 0.07,
        "length_consensus": 0.06,
        "duplicate_rate": 0.02,
        "source_copy_ratio": -0.28,
        "spanish_leakage": -0.32,
        "chat_artifact": -0.30,
        "repetition": -0.14,
        "exact_source_copy": -0.30,
        "candidate_index_fraction": -0.01,
    }


def clamp_weights(weights: dict[str, float]) -> dict[str, float]:
    clamped = dict(weights)
    for signal in [
        "source_copy_ratio",
        "spanish_leakage",
        "chat_artifact",
        "repetition",
        "exact_source_copy",
        "candidate_index_fraction",
    ]:
        clamped[signal] = min(0.0, clamped.get(signal, 0.0))
    for signal in [
        "pairwise_chrf",
        "pairwise_bleu",
        "pairwise_token_f1",
        "pairwise_precision",
        "pairwise_jaccard",
        "length_consensus",
        "duplicate_rate",
    ]:
        clamped[signal] = max(0.0, clamped.get(signal, 0.0))
    return clamped


def train_model(
    train_groups: Sequence[Sequence[ConsensusRow]],
    seed: int,
    search_iterations: int,
    initial_noise: float,
    min_noise: float,
) -> tuple[ConsensusModel, dict[str, Any]]:
    signal_names = SIGNAL_NAMES[:]
    best = ConsensusModel(signal_names, clamp_weights(initial_weights()))
    best_objective = mean_oracle_objective(train_groups, best)
    rng = random.Random(seed)
    accepted = 0

    for step in range(max(0, search_iterations)):
        progress = step / max(1, search_iterations - 1)
        noise = max(min_noise, initial_noise * (1.0 - progress))
        weights = dict(best.weights)
        if rng.random() < 0.8:
            signal = rng.choice(signal_names)
            weights[signal] = weights.get(signal, 0.0) + rng.gauss(0.0, noise)
        else:
            for signal in rng.sample(signal_names, k=min(4, len(signal_names))):
                weights[signal] = weights.get(signal, 0.0) + rng.gauss(0.0, noise * 0.5)
        proposal = ConsensusModel(signal_names, clamp_weights(weights))
        objective = mean_oracle_objective(train_groups, proposal)
        if objective > best_objective:
            best = proposal
            best_objective = objective
            accepted += 1

    diagnostics = {
        "training_objective": "mbr_consensus_hillclimb",
        "accepted_updates": accepted,
        "initial_objective": mean_oracle_objective(train_groups, ConsensusModel(signal_names, clamp_weights(initial_weights()))),
        "best_objective": best_objective,
        "search_iterations": search_iterations,
        "train_groups": len(train_groups),
        "train_candidates": sum(len(group) for group in train_groups),
    }
    return best, diagnostics


def model_to_json(model: ConsensusModel, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal_names": model.signal_names,
        "weights": model.weights,
        "diagnostics": diagnostics,
    }


def model_from_json(payload: dict[str, Any]) -> ConsensusModel:
    return ConsensusModel(
        signal_names=list(payload["signal_names"]),
        weights={name: float(value) for name, value in payload["weights"].items()},
    )


def metrics_for_selected(
    selected_rows: Sequence[ConsensusRow],
    method: str,
    source_path: Path,
    total_candidates: int,
) -> dict[str, Any]:
    selected = [row.candidate for row in selected_rows]
    metrics = oracle_rerank.metrics_for_selection(selected, method, source_path, total_candidates)
    metrics["selection_score"] = selection_score(metrics)
    non_first = [candidate for candidate in selected if candidate.candidate_index != 0]
    metrics[f"{method}_non_first_rate"] = 100.0 * len(non_first) / max(1, len(selected))
    metrics[f"{method}_mean_selected_index"] = (
        sum(candidate.candidate_index for candidate in selected) / max(1, len(selected))
    )
    return metrics


def write_summary(path: Path, records: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Tuned MBR Consensus Selector",
        "",
        "Weights are tuned with references on the train candidate pool, but eval-time selection is reference-free.",
        "",
        "| Method | Selection | chrF++ | BLEU | token F1 | source copy % | exact copy % | leakage % | artifact % | TER | non-first % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        non_first = record.get(f"{record['method']}_non_first_rate", record.get("oracle_non_first_rate", 0.0))
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


def evaluate_groups(
    groups: Sequence[Sequence[ConsensusRow]],
    model: ConsensusModel,
    predictions_jsonl: Path,
    output_dir: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    raw_groups = [[row.candidate for row in group] for group in groups]
    total_candidates = sum(len(group) for group in raw_groups)
    first = [group[0] for group in groups if group]
    tuned = select_consensus(groups, model)
    default_selected = [
        ConsensusRow(candidate, {}, oracle_rerank.candidate_oracle_score(candidate))
        for candidate in default_mbr.select_mbr(raw_groups)
    ]
    oracle_selected = [
        ConsensusRow(candidate, {}, oracle_rerank.candidate_oracle_score(candidate))
        for candidate in oracle_rerank.select_oracle(raw_groups)
    ]
    selections = [
        ("first", first),
        ("tuned_mbr", tuned),
        ("default_mbr", default_selected),
        ("oracle", oracle_selected),
    ]
    records = [
        metrics_for_selected(selected_rows, method, predictions_jsonl, total_candidates)
        for method, selected_rows in selections
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for (method, selected_rows), record in zip(selections, records, strict=True):
        oracle_rerank.write_predictions(
            output_dir / f"{prefix}_{method}_predictions.jsonl",
            [row.candidate for row in selected_rows],
        )
        oracle_rerank.write_metrics(output_dir / f"{prefix}_{method}_metrics.json", record)
    oracle_rerank.write_metrics(output_dir / f"{prefix}_summary.json", {"records": records})
    write_summary(output_dir / f"{prefix}_summary.md", records)
    return records


def main() -> None:
    args = parse_args()
    eval_groups = load_consensus_groups(args.eval_jsonl, max_peers=args.max_peers, utility_mode=args.utility_mode)
    if args.model_json:
        payload = json.loads(args.model_json.read_text())
        model = model_from_json(payload)
        diagnostics = dict(payload.get("diagnostics", {}))
    else:
        if not args.train_jsonl:
            raise ValueError("--train-jsonl is required unless --model-json is supplied")
        train_groups = load_consensus_groups(args.train_jsonl, max_peers=args.max_peers, utility_mode=args.utility_mode)
        model, diagnostics = train_model(
            train_groups,
            seed=args.seed,
            search_iterations=args.search_iterations,
            initial_noise=args.initial_noise,
            min_noise=args.min_noise,
        )

    diagnostics["eval_groups"] = len(eval_groups)
    diagnostics["eval_candidates"] = sum(len(group) for group in eval_groups)
    diagnostics["max_peers"] = args.max_peers
    diagnostics["utility_mode"] = args.utility_mode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.prefix}_model.json"
    model_path.write_text(json.dumps(model_to_json(model, diagnostics), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    records = evaluate_groups(eval_groups, model, args.eval_jsonl[0], args.output_dir, args.prefix)
    print(
        json.dumps(
            {
                "model_json": str(model_path),
                "diagnostics": diagnostics,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
