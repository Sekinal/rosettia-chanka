"""Train and evaluate a lightweight reference-free candidate reranker.

The trainer uses hidden references only to fit feature weights on training
candidate pools. The learned scorer itself is deployable: it sees only the
Spanish source and candidate Chanka translations in each sampled group.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import mbr_candidate_predictions as mbr
from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_gspo_chanka_unsloth as gspo
from scripts.summarize_gspo_canaries import selection_score


FEATURE_NAMES = [
    "mbr_score",
    "mbr_rank",
    "mbr_gap_to_best",
    "duplicate_rate",
    "length_consensus",
    "source_copy_ratio",
    "spanish_leakage_penalty",
    "chat_artifact_penalty",
    "repetition_penalty",
    "exact_source_copy",
    "candidate_index_fraction",
    "target_token_count",
    "target_source_token_ratio",
]


@dataclass(frozen=True)
class CandidateFeatures:
    candidate: oracle_rerank.Candidate
    raw: dict[str, float]
    oracle_score: float


@dataclass(frozen=True)
class FeatureModel:
    feature_names: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    weights: dict[str, float]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--eval-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--weights-json", type=Path, default=None, help="Existing feature weights to evaluate.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="feature_reranker")
    parser.add_argument("--search-iterations", type=int, default=3000)
    parser.add_argument("--initial-noise", type=float, default=0.35)
    parser.add_argument("--min-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def load_groups(paths: Sequence[Path]) -> list[list[oracle_rerank.Candidate]]:
    groups: list[list[oracle_rerank.Candidate]] = []
    for path in paths:
        candidates = oracle_rerank.load_candidates(path)
        groups.extend(oracle_rerank.group_candidates(candidates))
    return groups


def normalized_prediction(text: str) -> str:
    return gspo.normalize_text(text).lower()


def feature_rows_for_group(group: Sequence[oracle_rerank.Candidate]) -> list[CandidateFeatures]:
    if not group:
        return []
    group_size = len(group)
    mbr_scores = [mbr.mbr_score(candidate, group) for candidate in group]
    best_mbr = max(mbr_scores)
    sorted_mbr = sorted(((score, index) for index, score in enumerate(mbr_scores)), reverse=True)
    ranks = {index: rank for rank, (_score, index) in enumerate(sorted_mbr)}
    prediction_counts: dict[str, int] = {}
    for candidate in group:
        key = normalized_prediction(candidate.prediction)
        prediction_counts[key] = prediction_counts.get(key, 0) + 1

    rows: list[CandidateFeatures] = []
    max_index = max(1, group_size - 1)
    for candidate, mbr_score in zip(group, mbr_scores, strict=True):
        target_tokens = gspo.word_tokens(candidate.prediction)
        source_tokens = gspo.word_tokens(candidate.source)
        target_count = len(target_tokens)
        source_count = max(1, len(source_tokens))
        raw = {
            "mbr_score": mbr_score,
            "mbr_rank": 1.0 - (ranks[candidate.candidate_index] / max_index),
            "mbr_gap_to_best": mbr_score - best_mbr,
            "duplicate_rate": prediction_counts[normalized_prediction(candidate.prediction)] / group_size,
            "length_consensus": mbr.candidate_length_score(candidate, group),
            "source_copy_ratio": gspo.source_copy_ratio(candidate.prediction, candidate.source),
            "spanish_leakage_penalty": gspo.spanish_leakage_penalty(candidate.prediction),
            "chat_artifact_penalty": gspo.chat_artifact_penalty(candidate.prediction),
            "repetition_penalty": gspo.repetition_penalty(candidate.prediction),
            "exact_source_copy": 1.0 if gspo.exact_source_copy(candidate.prediction, candidate.source) else 0.0,
            "candidate_index_fraction": candidate.candidate_index / max_index,
            "target_token_count": float(target_count),
            "target_source_token_ratio": target_count / source_count,
        }
        rows.append(CandidateFeatures(candidate, raw, oracle_rerank.candidate_oracle_score(candidate)))
    return rows


def featurize_groups(groups: Sequence[Sequence[oracle_rerank.Candidate]]) -> list[list[CandidateFeatures]]:
    return [feature_rows_for_group(group) for group in groups if group]


def normalization_stats(groups: Sequence[Sequence[CandidateFeatures]]) -> tuple[dict[str, float], dict[str, float]]:
    flat = [row for group in groups for row in group]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = [row.raw[name] for row in flat]
        means[name] = statistics.fmean(values) if values else 0.0
        std = statistics.pstdev(values) if len(values) > 1 else 1.0
        stds[name] = std if std > 1e-9 else 1.0
    return means, stds


def normalized_feature(row: CandidateFeatures, model: FeatureModel, name: str) -> float:
    return (row.raw[name] - model.means[name]) / model.stds[name]


def candidate_score(row: CandidateFeatures, model: FeatureModel) -> float:
    return sum(model.weights.get(name, 0.0) * normalized_feature(row, model, name) for name in model.feature_names)


def select_feature(groups: Sequence[Sequence[CandidateFeatures]], model: FeatureModel) -> list[CandidateFeatures]:
    selected: list[CandidateFeatures] = []
    for group in groups:
        if not group:
            continue
        selected.append(
            max(
                group,
                key=lambda row: (
                    candidate_score(row, model),
                    -row.candidate.candidate_index,
                ),
            )
        )
    return selected


def mean_oracle_objective(groups: Sequence[Sequence[CandidateFeatures]], model: FeatureModel) -> float:
    selected = select_feature(groups, model)
    return statistics.fmean(row.oracle_score for row in selected) if selected else 0.0


def initial_weights() -> dict[str, float]:
    return {
        "mbr_score": 1.0,
        "mbr_rank": 0.25,
        "mbr_gap_to_best": 0.15,
        "duplicate_rate": 0.05,
        "length_consensus": 0.10,
        "source_copy_ratio": -0.10,
        "spanish_leakage_penalty": -0.15,
        "chat_artifact_penalty": -0.20,
        "repetition_penalty": -0.08,
        "exact_source_copy": -0.15,
        "candidate_index_fraction": -0.03,
        "target_token_count": 0.0,
        "target_source_token_ratio": 0.0,
    }


def train_model(
    train_groups: Sequence[Sequence[CandidateFeatures]],
    seed: int,
    search_iterations: int,
    initial_noise: float,
    min_noise: float,
) -> tuple[FeatureModel, dict[str, Any]]:
    means, stds = normalization_stats(train_groups)
    best = FeatureModel(FEATURE_NAMES[:], means, stds, initial_weights())
    best_objective = mean_oracle_objective(train_groups, best)
    rng = random.Random(seed)
    accepted = 0

    for step in range(max(0, search_iterations)):
        progress = step / max(1, search_iterations - 1)
        noise = max(min_noise, initial_noise * (1.0 - progress))
        weights = dict(best.weights)
        if rng.random() < 0.80:
            feature = rng.choice(FEATURE_NAMES)
            weights[feature] = weights.get(feature, 0.0) + rng.gauss(0.0, noise)
        else:
            for feature in rng.sample(FEATURE_NAMES, k=min(3, len(FEATURE_NAMES))):
                weights[feature] = weights.get(feature, 0.0) + rng.gauss(0.0, noise * 0.5)
        proposal = FeatureModel(FEATURE_NAMES[:], means, stds, weights)
        objective = mean_oracle_objective(train_groups, proposal)
        if objective > best_objective:
            best = proposal
            best_objective = objective
            accepted += 1

    diagnostics = {
        "accepted_updates": accepted,
        "initial_objective": mean_oracle_objective(
            train_groups,
            FeatureModel(FEATURE_NAMES[:], means, stds, initial_weights()),
        ),
        "best_objective": best_objective,
        "search_iterations": search_iterations,
    }
    return best, diagnostics


def model_to_json(model: FeatureModel, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": model.feature_names,
        "means": model.means,
        "stds": model.stds,
        "weights": model.weights,
        "diagnostics": diagnostics,
    }


def model_from_json(payload: dict[str, Any]) -> FeatureModel:
    return FeatureModel(
        feature_names=list(payload["feature_names"]),
        means={key: float(value) for key, value in payload["means"].items()},
        stds={key: float(value) for key, value in payload["stds"].items()},
        weights={key: float(value) for key, value in payload["weights"].items()},
    )


def metrics_for_selected(
    selected_rows: Sequence[CandidateFeatures],
    method: str,
    predictions_jsonl: Path,
    total_candidates: int,
) -> dict[str, Any]:
    selected = [row.candidate for row in selected_rows]
    metrics = oracle_rerank.metrics_for_selection(selected, method, predictions_jsonl, total_candidates)
    metrics["selection_score"] = selection_score(metrics)
    non_first = [candidate for candidate in selected if candidate.candidate_index != 0]
    metrics[f"{method}_non_first_rate"] = 100.0 * len(non_first) / max(1, len(selected))
    metrics[f"{method}_mean_selected_index"] = sum(candidate.candidate_index for candidate in selected) / max(1, len(selected))
    metrics[f"{method}_mean_oracle_sentence_score"] = (
        statistics.fmean(row.oracle_score for row in selected_rows) if selected_rows else 0.0
    )
    return metrics


def write_summary(path: Path, records: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Feature Candidate Reranker",
        "",
        "The feature reranker is reference-free at inference time. References are used only to fit feature weights on training candidate pools.",
        "",
        "| Method | Selection | chrF++ | BLEU | token F1 | source copy % | exact copy % | leakage % | artifact % | TER | non-first % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        method = str(record["method"])
        non_first = record.get(
            f"{method}_non_first_rate",
            record.get("mbr_non_first_rate", record.get("oracle_non_first_rate", 0.0)),
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
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
    groups: Sequence[Sequence[CandidateFeatures]],
    model: FeatureModel,
    predictions_jsonl: Path,
    output_dir: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    raw_groups = [[row.candidate for row in group] for group in groups]
    total_candidates = sum(len(group) for group in raw_groups)
    first = [CandidateFeatures(group[0], {}, oracle_rerank.candidate_oracle_score(group[0])) for group in raw_groups if group]
    feature = select_feature(groups, model)
    mbr_selected = [
        CandidateFeatures(candidate, {}, oracle_rerank.candidate_oracle_score(candidate))
        for candidate in mbr.select_mbr(raw_groups)
    ]
    oracle_selected = [
        CandidateFeatures(candidate, {}, oracle_rerank.candidate_oracle_score(candidate))
        for candidate in oracle_rerank.select_oracle(raw_groups)
    ]
    selections = [
        ("first", first),
        ("feature", feature),
        ("mbr", mbr_selected),
        ("oracle", oracle_selected),
    ]
    records = [
        metrics_for_selected(selected_rows, method, predictions_jsonl, total_candidates)
        for method, selected_rows in selections
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for (method, selected_rows), record in zip(selections, records, strict=True):
        selected = [row.candidate for row in selected_rows]
        oracle_rerank.write_predictions(output_dir / f"{prefix}_{method}_predictions.jsonl", selected)
        oracle_rerank.write_metrics(output_dir / f"{prefix}_{method}_metrics.json", record)
    oracle_rerank.write_metrics(output_dir / f"{prefix}_summary.json", {"records": records})
    write_summary(output_dir / f"{prefix}_summary.md", records)
    return records


def main() -> None:
    args = parse_args()
    eval_groups_raw = load_groups(args.eval_jsonl)
    eval_groups = featurize_groups(eval_groups_raw)
    diagnostics: dict[str, Any] = {}
    if args.weights_json:
        payload = json.loads(args.weights_json.read_text())
        model = model_from_json(payload)
        diagnostics = dict(payload.get("diagnostics", {}))
    else:
        if not args.train_jsonl:
            raise ValueError("--train-jsonl is required unless --weights-json is supplied")
        train_groups = featurize_groups(load_groups(args.train_jsonl))
        model, diagnostics = train_model(
            train_groups,
            seed=args.seed,
            search_iterations=args.search_iterations,
            initial_noise=args.initial_noise,
            min_noise=args.min_noise,
        )
        diagnostics["train_groups"] = len(train_groups)
        diagnostics["train_candidates"] = sum(len(group) for group in train_groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = model_to_json(model, diagnostics)
    weights_path = args.output_dir / f"{args.prefix}_weights.json"
    weights_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    records = evaluate_groups(
        eval_groups,
        model,
        predictions_jsonl=args.eval_jsonl[0],
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    result = {
        "weights_json": str(weights_path),
        "diagnostics": diagnostics,
        "records": records,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
