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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import mbr_candidate_predictions as mbr
from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_gspo_chanka_unsloth as gspo
from scripts.summarize_gspo_canaries import selection_score


BASE_FEATURE_NAMES = [
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
EXPERIMENTAL_FEATURE_NAMES = [
    "source_root_copy_ratio",
    "terminology_target_coverage",
    "terminology_source_root_leakage",
]
FEATURE_NAMES = BASE_FEATURE_NAMES + EXPERIMENTAL_FEATURE_NAMES


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
    bias: float = 0.0
    stumps: list[dict[str, float | str]] = field(default_factory=list)


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
    parser.add_argument("--training-objective", choices=["hillclimb", "listwise", "boosted-stumps"], default="hillclimb")
    parser.add_argument("--listwise-epochs", type=int, default=80)
    parser.add_argument("--listwise-learning-rate", type=float, default=0.03)
    parser.add_argument("--listwise-temperature", type=float, default=0.04)
    parser.add_argument("--listwise-l2", type=float, default=0.001)
    parser.add_argument("--listwise-target", choices=["soft", "best"], default="soft")
    parser.add_argument("--boosted-estimators", type=int, default=120)
    parser.add_argument("--boosted-learning-rate", type=float, default=0.08)
    parser.add_argument("--boosted-thresholds", type=int, default=32)
    parser.add_argument("--boosted-min-leaf", type=int, default=8)
    parser.add_argument("--boosted-init", choices=["mean", "hillclimb"], default="hillclimb")
    parser.add_argument(
        "--terminology-file",
        default=None,
        help="Optional dataset-repo parquet glossary used to add terminology coverage/leakage features.",
    )
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--terminology-top-k", type=int, default=6)
    parser.add_argument("--terminology-min-source-chars", type=int, default=3)
    parser.add_argument(
        "--include-source-root-copy",
        action="store_true",
        help="Opt in to an experimental feature that penalizes source-token prefix copying such as fiesta -> Fiestapi.",
    )
    parser.add_argument(
        "--include-terminology-features",
        action="store_true",
        help="Opt in to experimental glossary target-coverage and source-root leakage features.",
    )
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


def significant_term_tokens(text: str) -> list[str]:
    return [
        token
        for token in gspo.word_tokens(text)
        if len(token) >= 4 and token.lower() not in gspo.SPANISH_STOPWORDS
    ]


def token_is_morphological_match(candidate_token: str, term_token: str) -> bool:
    candidate_token = candidate_token.lower()
    term_token = term_token.lower()
    return candidate_token == term_token or candidate_token.startswith(term_token)


def term_tokens_are_covered(candidate_tokens: Sequence[str], term: str) -> bool:
    term_tokens = significant_term_tokens(term)
    if not term_tokens:
        return False
    return all(
        any(token_is_morphological_match(candidate_token, term_token) for candidate_token in candidate_tokens)
        for term_token in term_tokens
    )


def terminology_features(
    source: str,
    prediction: str,
    terminology_entries: Sequence[tuple[str, str]],
    terminology_top_k: int,
) -> tuple[float, float]:
    selected = gspo.select_terminology(source, terminology_entries, terminology_top_k)
    if not selected:
        return 0.0, 0.0

    prediction_tokens = [token.lower() for token in gspo.word_tokens(prediction)]
    covered_targets = [
        target_term
        for _source_term, target_term in selected
        if term_tokens_are_covered(prediction_tokens, target_term)
    ]
    target_coverage = len(covered_targets) / len(selected)
    covered_target_keys = {target_term.lower() for target_term in covered_targets}

    leaked_roots = 0
    checked_roots = 0
    for source_term, target_term in selected:
        source_tokens = significant_term_tokens(source_term)
        if not source_tokens:
            continue
        checked_roots += len(source_tokens)
        if target_term.lower() in covered_target_keys:
            continue
        for source_token in source_tokens:
            if any(token_is_morphological_match(prediction_token, source_token) for prediction_token in prediction_tokens):
                leaked_roots += 1

    source_root_leakage = leaked_roots / max(1, checked_roots)
    return target_coverage, source_root_leakage


def source_root_copy_ratio(source: str, prediction: str) -> float:
    source_tokens = significant_term_tokens(source)
    prediction_tokens = [token.lower() for token in gspo.word_tokens(prediction)]
    if not source_tokens or not prediction_tokens:
        return 0.0
    copied = 0
    for source_token in source_tokens:
        if any(token_is_morphological_match(prediction_token, source_token) for prediction_token in prediction_tokens):
            copied += 1
    return copied / len(source_tokens)


def feature_rows_for_group(
    group: Sequence[oracle_rerank.Candidate],
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> list[CandidateFeatures]:
    if not group:
        return []
    terminology_entries = terminology_entries or []
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
        terminology_target_coverage, terminology_source_root_leakage = terminology_features(
            candidate.source,
            candidate.prediction,
            terminology_entries,
            terminology_top_k,
        )
        raw = {
            "mbr_score": mbr_score,
            "mbr_rank": 1.0 - (ranks[candidate.candidate_index] / max_index),
            "mbr_gap_to_best": mbr_score - best_mbr,
            "duplicate_rate": prediction_counts[normalized_prediction(candidate.prediction)] / group_size,
            "length_consensus": mbr.candidate_length_score(candidate, group),
            "source_copy_ratio": gspo.source_copy_ratio(candidate.prediction, candidate.source),
            "source_root_copy_ratio": source_root_copy_ratio(candidate.source, candidate.prediction),
            "spanish_leakage_penalty": gspo.spanish_leakage_penalty(candidate.prediction),
            "chat_artifact_penalty": gspo.chat_artifact_penalty(candidate.prediction),
            "repetition_penalty": gspo.repetition_penalty(candidate.prediction),
            "exact_source_copy": 1.0 if gspo.exact_source_copy(candidate.prediction, candidate.source) else 0.0,
            "terminology_target_coverage": terminology_target_coverage,
            "terminology_source_root_leakage": terminology_source_root_leakage,
            "candidate_index_fraction": candidate.candidate_index / max_index,
            "target_token_count": float(target_count),
            "target_source_token_ratio": target_count / source_count,
        }
        rows.append(CandidateFeatures(candidate, raw, oracle_rerank.candidate_oracle_score(candidate)))
    return rows


def featurize_groups(
    groups: Sequence[Sequence[oracle_rerank.Candidate]],
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> list[list[CandidateFeatures]]:
    return [
        feature_rows_for_group(group, terminology_entries, terminology_top_k)
        for group in groups
        if group
    ]


def normalization_stats(
    groups: Sequence[Sequence[CandidateFeatures]],
    feature_names: Sequence[str] = BASE_FEATURE_NAMES,
) -> tuple[dict[str, float], dict[str, float]]:
    flat = [row for group in groups for row in group]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in feature_names:
        values = [row.raw[name] for row in flat]
        means[name] = statistics.fmean(values) if values else 0.0
        std = statistics.pstdev(values) if len(values) > 1 else 1.0
        stds[name] = std if std > 1e-9 else 1.0
    return means, stds


def normalized_feature(row: CandidateFeatures, model: FeatureModel, name: str) -> float:
    return (row.raw[name] - model.means[name]) / model.stds[name]


def candidate_score(row: CandidateFeatures, model: FeatureModel) -> float:
    score = model.bias + sum(
        model.weights.get(name, 0.0) * normalized_feature(row, model, name) for name in model.feature_names
    )
    for stump in model.stumps:
        feature = str(stump["feature"])
        value = normalized_feature(row, model, feature)
        if value <= float(stump["threshold"]):
            score += float(stump["left_value"])
        else:
            score += float(stump["right_value"])
    return score


def stable_softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not values:
        return []
    temperature = max(1e-6, temperature)
    scaled = [value / temperature for value in values]
    max_value = max(scaled)
    exp_values = [math.exp(max(-60.0, min(60.0, value - max_value))) for value in scaled]
    total = sum(exp_values)
    return [value / total for value in exp_values]


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
        "source_root_copy_ratio": -0.15,
        "spanish_leakage_penalty": -0.15,
        "chat_artifact_penalty": -0.20,
        "repetition_penalty": -0.08,
        "exact_source_copy": -0.15,
        "terminology_target_coverage": 0.15,
        "terminology_source_root_leakage": -0.25,
        "candidate_index_fraction": -0.03,
        "target_token_count": 0.0,
        "target_source_token_ratio": 0.0,
    }


def train_model(
    train_groups: Sequence[Sequence[CandidateFeatures]],
    feature_names: Sequence[str],
    seed: int,
    search_iterations: int,
    initial_noise: float,
    min_noise: float,
) -> tuple[FeatureModel, dict[str, Any]]:
    feature_names = list(feature_names)
    means, stds = normalization_stats(train_groups, feature_names)
    best = FeatureModel(feature_names, means, stds, initial_weights())
    best_objective = mean_oracle_objective(train_groups, best)
    rng = random.Random(seed)
    accepted = 0

    for step in range(max(0, search_iterations)):
        progress = step / max(1, search_iterations - 1)
        noise = max(min_noise, initial_noise * (1.0 - progress))
        weights = dict(best.weights)
        if rng.random() < 0.80:
            feature = rng.choice(feature_names)
            weights[feature] = weights.get(feature, 0.0) + rng.gauss(0.0, noise)
        else:
            for feature in rng.sample(feature_names, k=min(3, len(feature_names))):
                weights[feature] = weights.get(feature, 0.0) + rng.gauss(0.0, noise * 0.5)
        proposal = FeatureModel(feature_names, means, stds, weights)
        objective = mean_oracle_objective(train_groups, proposal)
        if objective > best_objective:
            best = proposal
            best_objective = objective
            accepted += 1

    diagnostics = {
        "accepted_updates": accepted,
        "initial_objective": mean_oracle_objective(
            train_groups,
            FeatureModel(feature_names, means, stds, initial_weights()),
        ),
        "best_objective": best_objective,
        "search_iterations": search_iterations,
    }
    return best, diagnostics


def listwise_targets(group: Sequence[CandidateFeatures], temperature: float, target_mode: str) -> list[float]:
    if not group:
        return []
    if target_mode == "best":
        best_index = max(
            range(len(group)),
            key=lambda index: (
                group[index].oracle_score,
                -group[index].candidate.candidate_index,
            ),
        )
        return [1.0 if index == best_index else 0.0 for index in range(len(group))]
    return stable_softmax([row.oracle_score for row in group], temperature)


def train_listwise_model(
    train_groups: Sequence[Sequence[CandidateFeatures]],
    feature_names: Sequence[str],
    seed: int,
    epochs: int,
    learning_rate: float,
    target_temperature: float,
    l2: float,
    target_mode: str,
) -> tuple[FeatureModel, dict[str, Any]]:
    feature_names = list(feature_names)
    means, stds = normalization_stats(train_groups, feature_names)
    weights = dict(initial_weights())
    rng = random.Random(seed)
    groups = [list(group) for group in train_groups if group]
    initial = FeatureModel(feature_names, means, stds, weights)
    initial_objective = mean_oracle_objective(groups, initial)
    last_loss = 0.0

    for _epoch in range(max(0, epochs)):
        rng.shuffle(groups)
        for group in groups:
            targets = listwise_targets(group, target_temperature, target_mode)
            model = FeatureModel(feature_names, means, stds, weights)
            scores = [candidate_score(row, model) for row in group]
            probabilities = stable_softmax(scores)
            last_loss += -sum(target * math.log(max(1e-12, probability)) for target, probability in zip(targets, probabilities, strict=True))
            gradients = {name: l2 * weights.get(name, 0.0) for name in feature_names}
            for row, probability, target in zip(group, probabilities, targets, strict=True):
                error = probability - target
                for name in feature_names:
                    gradients[name] += error * normalized_feature(row, model, name)
            scale = 1.0 / max(1, len(group))
            for name in feature_names:
                weights[name] = weights.get(name, 0.0) - (learning_rate * gradients[name] * scale)

    model = FeatureModel(feature_names, means, stds, weights)
    diagnostics = {
        "training_objective": "listwise",
        "initial_objective": initial_objective,
        "best_objective": mean_oracle_objective(groups, model),
        "listwise_epochs": epochs,
        "listwise_learning_rate": learning_rate,
        "listwise_temperature": target_temperature,
        "listwise_l2": l2,
        "listwise_target": target_mode,
        "last_accumulated_loss": last_loss,
    }
    return model, diagnostics


def flatten_rows(groups: Sequence[Sequence[CandidateFeatures]]) -> list[CandidateFeatures]:
    return [row for group in groups for row in group]


def normalized_matrix(
    rows: Sequence[CandidateFeatures],
    feature_names: Sequence[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> Any:
    import numpy as np

    return np.asarray(
        [[(row.raw[name] - means[name]) / stds[name] for name in feature_names] for row in rows],
        dtype="float64",
    )


def candidate_scores_for_rows(rows: Sequence[CandidateFeatures], model: FeatureModel) -> list[float]:
    return [candidate_score(row, model) for row in rows]


def candidate_thresholds(values: Any, threshold_count: int) -> list[float]:
    import numpy as np

    unique = np.unique(values)
    if len(unique) <= 1:
        return []
    if len(unique) <= threshold_count + 1:
        return [float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:], strict=True)]
    quantiles = np.linspace(0.0, 1.0, threshold_count + 2, dtype="float64")[1:-1]
    return [float(value) for value in np.unique(np.quantile(values, quantiles))]


def best_residual_stump(
    values: Any,
    residuals: Any,
    threshold_count: int,
    min_leaf: int,
) -> dict[str, float] | None:
    import numpy as np

    best: dict[str, float] | None = None
    best_loss = float("inf")
    row_count = len(values)
    if row_count < max(2, min_leaf * 2):
        return None
    for threshold in candidate_thresholds(values, threshold_count):
        left_mask = values <= threshold
        left_count = int(left_mask.sum())
        right_count = row_count - left_count
        if left_count < min_leaf or right_count < min_leaf:
            continue
        right_mask = ~left_mask
        left_value = float(residuals[left_mask].mean())
        right_value = float(residuals[right_mask].mean())
        loss = float(
            np.square(residuals[left_mask] - left_value).sum()
            + np.square(residuals[right_mask] - right_value).sum()
        )
        if loss < best_loss:
            best_loss = loss
            best = {
                "threshold": float(threshold),
                "left_value": left_value,
                "right_value": right_value,
                "loss": best_loss,
            }
    return best


def train_boosted_stump_model(
    train_groups: Sequence[Sequence[CandidateFeatures]],
    feature_names: Sequence[str],
    seed: int,
    estimators: int,
    learning_rate: float,
    threshold_count: int,
    min_leaf: int,
    init_mode: str,
    search_iterations: int,
    initial_noise: float,
    min_noise: float,
) -> tuple[FeatureModel, dict[str, Any]]:
    import numpy as np

    feature_names = list(feature_names)
    means, stds = normalization_stats(train_groups, feature_names)
    rows = flatten_rows(train_groups)
    y = np.asarray([row.oracle_score for row in rows], dtype="float64")
    x = normalized_matrix(rows, feature_names, means, stds)

    if init_mode == "hillclimb":
        base_model, base_diagnostics = train_model(
            train_groups,
            feature_names=feature_names,
            seed=seed,
            search_iterations=search_iterations,
            initial_noise=initial_noise,
            min_noise=min_noise,
        )
        predictions = np.asarray(candidate_scores_for_rows(rows, base_model), dtype="float64")
        weights = dict(base_model.weights)
        bias = base_model.bias
        initial_objective = base_diagnostics["best_objective"]
    else:
        base_diagnostics = {}
        bias = float(y.mean()) if len(y) else 0.0
        weights = {name: 0.0 for name in feature_names}
        predictions = np.full(len(y), bias, dtype="float64")
        initial_objective = mean_oracle_objective(
            train_groups,
            FeatureModel(feature_names, means, stds, weights, bias=bias),
        )

    stumps: list[dict[str, float | str]] = []
    rng = random.Random(seed)
    feature_order = list(range(len(feature_names)))
    for _step in range(max(0, estimators)):
        rng.shuffle(feature_order)
        residuals = y - predictions
        best: dict[str, float | str] | None = None
        best_loss = float("inf")
        for feature_index in feature_order:
            stump = best_residual_stump(x[:, feature_index], residuals, threshold_count, min_leaf)
            if stump is None:
                continue
            if float(stump["loss"]) < best_loss:
                best_loss = float(stump["loss"])
                best = {
                    "feature": feature_names[feature_index],
                    "threshold": float(stump["threshold"]),
                    "left_value": learning_rate * float(stump["left_value"]),
                    "right_value": learning_rate * float(stump["right_value"]),
                }
        if best is None:
            break
        feature_index = feature_names.index(str(best["feature"]))
        update = np.where(
            x[:, feature_index] <= float(best["threshold"]),
            float(best["left_value"]),
            float(best["right_value"]),
        )
        predictions += update
        stumps.append(best)

    model = FeatureModel(feature_names, means, stds, weights, bias=bias, stumps=stumps)
    diagnostics = {
        "training_objective": "boosted-stumps",
        "boosted_init": init_mode,
        "base_diagnostics": base_diagnostics,
        "initial_objective": initial_objective,
        "best_objective": mean_oracle_objective(train_groups, model),
        "boosted_estimators_requested": estimators,
        "boosted_estimators_fit": len(stumps),
        "boosted_learning_rate": learning_rate,
        "boosted_thresholds": threshold_count,
        "boosted_min_leaf": min_leaf,
        "train_mse": float(np.square(y - predictions).mean()) if len(y) else 0.0,
    }
    return model, diagnostics


def model_to_json(model: FeatureModel, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": model.feature_names,
        "means": model.means,
        "stds": model.stds,
        "weights": model.weights,
        "bias": model.bias,
        "stumps": model.stumps,
        "diagnostics": diagnostics,
    }


def model_from_json(payload: dict[str, Any]) -> FeatureModel:
    return FeatureModel(
        feature_names=list(payload["feature_names"]),
        means={key: float(value) for key, value in payload["means"].items()},
        stds={key: float(value) for key, value in payload["stds"].items()},
        weights={key: float(value) for key, value in payload["weights"].items()},
        bias=float(payload.get("bias", 0.0)),
        stumps=list(payload.get("stumps", [])),
    )


def load_optional_terminology(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.terminology_file:
        return []
    return gspo.load_terminology_entries(
        args.dataset_repo,
        args.terminology_file,
        args.terminology_min_source_chars,
    )


def selected_feature_names(args: argparse.Namespace) -> list[str]:
    feature_names = BASE_FEATURE_NAMES[:]
    if args.include_source_root_copy:
        feature_names.append("source_root_copy_ratio")
    if args.include_terminology_features:
        feature_names.extend(["terminology_target_coverage", "terminology_source_root_leakage"])
    return feature_names


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
    terminology_entries = load_optional_terminology(args)
    feature_names = selected_feature_names(args)
    eval_groups_raw = load_groups(args.eval_jsonl)
    eval_groups = featurize_groups(eval_groups_raw, terminology_entries, args.terminology_top_k)
    diagnostics: dict[str, Any] = {}
    if args.weights_json:
        payload = json.loads(args.weights_json.read_text())
        model = model_from_json(payload)
        diagnostics = dict(payload.get("diagnostics", {}))
    else:
        if not args.train_jsonl:
            raise ValueError("--train-jsonl is required unless --weights-json is supplied")
        train_groups = featurize_groups(load_groups(args.train_jsonl), terminology_entries, args.terminology_top_k)
        if args.training_objective == "listwise":
            model, diagnostics = train_listwise_model(
                train_groups,
                feature_names=feature_names,
                seed=args.seed,
                epochs=args.listwise_epochs,
                learning_rate=args.listwise_learning_rate,
                target_temperature=args.listwise_temperature,
                l2=args.listwise_l2,
                target_mode=args.listwise_target,
            )
        elif args.training_objective == "boosted-stumps":
            model, diagnostics = train_boosted_stump_model(
                train_groups,
                feature_names=feature_names,
                seed=args.seed,
                estimators=args.boosted_estimators,
                learning_rate=args.boosted_learning_rate,
                threshold_count=args.boosted_thresholds,
                min_leaf=args.boosted_min_leaf,
                init_mode=args.boosted_init,
                search_iterations=args.search_iterations,
                initial_noise=args.initial_noise,
                min_noise=args.min_noise,
            )
        else:
            model, diagnostics = train_model(
                train_groups,
                feature_names=feature_names,
                seed=args.seed,
                search_iterations=args.search_iterations,
                initial_noise=args.initial_noise,
                min_noise=args.min_noise,
            )
        diagnostics["train_groups"] = len(train_groups)
        diagnostics["train_candidates"] = sum(len(group) for group in train_groups)
    diagnostics["terminology_entries"] = len(terminology_entries)
    diagnostics["terminology_top_k"] = args.terminology_top_k if terminology_entries else 0

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
