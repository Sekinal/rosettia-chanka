"""GSPO-style Chanka alignment with sequence-level GRPO rewards.

TRL 0.24 exposes the GSPO paper's sequence-level importance sampling through
GRPOConfig.importance_sampling_level="sequence". This script keeps the reviewed
Chanka judicial data out of SFT and uses it only for the RL alignment stage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
from datasets import Dataset
from huggingface_hub import hf_hub_download


DATASET_REPO = "Thermostatic/rosettia-chanka-data"
DEFAULT_MODEL_ID = "unsloth/Qwen3.5-2B"
CHANKA_FILE = "clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet"
DEFAULT_SFT_CHECKPOINT = (
    "outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400"
)
REWARD_PROFILE = "baseline"
REWARD_PROFILES = (
    "baseline",
    "fg_severity_2411",
    "severity_proxy_2310",
    "self_verifier_2511",
    "vibethinker_2511",
    "mix_severity_verifier",
    "mix_verifier_vibe",
    "mix_all_strict",
    "rosettia_guard_v1",
    "rosettia_guard_v2",
    "learned_verifier_2511",
    "learned_verifier_vibe_2511",
    "learned_verifier_ensemble_vibe_2511",
    "learned_verifier_bleu_margin_vibe_2511",
    "reference_rerank_vibe_v1",
)
SPANISH_STOPWORDS = {
    "el",
    "la",
    "los",
    "las",
    "de",
    "del",
    "que",
    "y",
    "en",
    "para",
    "por",
    "con",
    "una",
    "un",
    "usted",
    "señor",
    "señora",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path(DEFAULT_SFT_CHECKPOINT),
        help="Broad SFT LoRA checkpoint to continue from. Defaults to the current LoRA baseline checkpoint.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Trainer checkpoint to resume from, preserving optimizer and scheduler state.",
    )
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--dataset-file", default=CHANKA_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen35_2b_chanka_gspo"))
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--max-prompt-length", type=int, default=96)
    parser.add_argument("--max-completion-length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--terminology-file",
        default=None,
        help="Optional dataset-repo parquet glossary for terminology-conditioned GSPO prompts.",
    )
    parser.add_argument("--terminology-top-k", type=int, default=6)
    parser.add_argument("--terminology-min-source-chars", type=int, default=3)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--attach-lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach a fresh LoRA adapter after loading the model. Useful for RL from a merged/full checkpoint.",
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--loss-type", choices=["dapo", "dr_grpo", "grpo", "bnpo"], default="dapo")
    parser.add_argument("--scale-rewards", choices=["group", "batch", "none"], default="group")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--evals-per-epoch", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reward-profile",
        choices=REWARD_PROFILES,
        default="baseline",
        help="Paper-inspired reward profile to test independently.",
    )
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--verifier-adapter-path",
        type=Path,
        default=None,
        help="Verifier LoRA path used by reward-profile learned_verifier_2511.",
    )
    parser.add_argument("--verifier-max-seq-length", type=int, default=512)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=96)
    parser.add_argument("--verifier-batch-size", type=int, default=4)
    parser.add_argument(
        "--overlong-max-words",
        type=int,
        default=None,
        help="Optional hard word-count guard for GSPO rewards. Penalizes completions at or above this many words.",
    )
    parser.add_argument(
        "--overlong-ratio-threshold",
        type=float,
        default=0.0,
        help="Optional hypothesis/reference word-count ratio guard. Disabled at 0.",
    )
    parser.add_argument(
        "--overlong-penalty-weight",
        type=float,
        default=1.0,
        help="Multiplier for the optional overlong completion penalty.",
    )
    parser.add_argument(
        "--secondary-verifier-adapter-path",
        type=Path,
        default=None,
        help="Optional second verifier LoRA for learned_verifier_ensemble_vibe_2511.",
    )
    return parser.parse_args(argv)


def download_parquet(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))


def load_chanka_rows(repo_id: str, filename: str) -> list[dict[str, str]]:
    path = download_parquet(repo_id, filename)
    frame = pl.read_parquet(path)
    rows: list[dict[str, str]] = []
    for row in frame.select(["reviewed_spanish", "reviewed_chanka_quechua"]).iter_rows(named=True):
        source = str(row["reviewed_spanish"]).strip()
        target = str(row["reviewed_chanka_quechua"]).strip()
        if not source or not target:
            continue
        rows.append(
            {
                "source": source,
                "target": target,
                "source_name": filename,
                "variant": "quy/chanka",
            }
        )
    if not rows:
        raise RuntimeError(f"No Chanka rows loaded from {filename}")
    return rows


def load_terminology_entries(
    repo_id: str,
    filename: str,
    min_source_chars: int,
) -> list[tuple[str, str]]:
    path = download_parquet(repo_id, filename)
    frame = pl.read_parquet(path)
    required = {"direction", "source_term", "target_text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Terminology file {filename} is missing columns: {sorted(missing)}")

    if "glossary_status" in frame.columns:
        frame = frame.filter(pl.col("glossary_status") == "simple_term_pair")
    frame = frame.filter(pl.col("direction") == "spa_Latn-quy_Latn")

    entries: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in frame.select(["source_term", "target_text"]).iter_rows(named=True):
        source_term = normalize_text(str(row["source_term"]))
        target_text = normalize_text(str(row["target_text"]))
        if len(source_term) < min_source_chars or not target_text:
            continue
        key = (source_term.lower(), target_text.lower())
        if key in seen:
            continue
        seen.add(key)
        entries.append((source_term, target_text))
    entries.sort(key=lambda item: (-len(item[0]), item[0].lower(), item[1].lower()))
    return entries


def split_rows(
    rows: list[dict[str, str]],
    validation_fraction: float,
    seed: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * validation_fraction))
    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    if max_train_samples is not None:
        train_rows = train_rows[:max_train_samples]
    if max_eval_samples is not None:
        eval_rows = eval_rows[:max_eval_samples]
    return train_rows, eval_rows


def optimizer_steps_per_epoch(row_count: int, batch_size: int, gradient_accumulation_steps: int) -> int:
    del gradient_accumulation_steps
    return max(1, math.ceil(row_count / max(1, batch_size)))


def configure_step_schedule(args: argparse.Namespace, train_row_count: int) -> None:
    if args.eval_steps is not None and args.save_steps is not None:
        return
    if args.max_steps and args.max_steps > 0:
        fallback_steps = max(1, args.max_steps // max(1, args.evals_per_epoch))
    else:
        fallback_steps = max(
            1,
            optimizer_steps_per_epoch(
                train_row_count,
                args.per_device_train_batch_size,
                args.gradient_accumulation_steps,
            )
            // max(1, args.evals_per_epoch),
        )
    if args.eval_steps is None:
        args.eval_steps = fallback_steps
    if args.save_steps is None:
        args.save_steps = args.eval_steps


def validate_grpo_batching(args: argparse.Namespace) -> None:
    if args.per_device_train_batch_size % args.num_generations != 0:
        raise ValueError(
            "Unsloth GRPO/GSPO expects per-device train batch size "
            f"({args.per_device_train_batch_size}) to be divisible by num_generations "
            f"({args.num_generations})."
        )
    if args.per_device_eval_batch_size % args.num_generations != 0:
        raise ValueError(
            "Unsloth GRPO/GSPO expects per-device eval batch size "
            f"({args.per_device_eval_batch_size}) to be divisible by num_generations "
            f"({args.num_generations})."
        )


def prompt_messages(
    source: str,
    terminology: Sequence[tuple[str, str]] | None = None,
    few_shot_examples: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    user_content = (
        "Traduce del español al quechua chanka. Usa una traducción directa, "
        "natural y fiel. Conserva nombres, números y entidades; evita copiar "
        "el español salvo cuando sea necesario."
    )
    if few_shot_examples:
        example_lines: list[str] = []
        for index, (example_source, example_target) in enumerate(few_shot_examples, start=1):
            example_lines.append(
                f"Ejemplo {index}\n"
                f"Español: {normalize_text(example_source)}\n"
                f"Quechua chanka: {normalize_text(example_target)}"
            )
        user_content += (
            "\n\nEjemplos de referencia: imita el estilo cuando sea pertinente, "
            "pero traduce solo el texto final.\n"
            + "\n\n".join(example_lines)
        )
    if terminology:
        glossary_lines = [
            f"- {spanish_term} = {chanka_term}"
            for spanish_term, chanka_term in terminology
        ]
        user_content += (
            "\n\nGlosario sugerido: usa estos terminos solo si aplican al texto; "
            "no fuerces terminos irrelevantes.\n"
            + "\n".join(glossary_lines)
        )
    user_content += f"\n\nEspañol: {source}"
    return [
        {"role": "system", "content": "Eres un traductor profesional español-quechua chanka."},
        {
            "role": "user",
            "content": user_content,
        },
    ]


def apply_chat_template_no_thinking(tokenizer, messages: list[dict[str, str]], **kwargs):
    """Apply chat templates with reasoning disabled when the tokenizer supports it."""
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def source_contains_term(source: str, source_term: str) -> bool:
    source_norm = normalize_text(source).lower()
    term_norm = normalize_text(source_term).lower()
    if not term_norm or term_norm in SPANISH_STOPWORDS:
        return False
    return re.search(rf"(?<!\w){re.escape(term_norm)}(?!\w)", source_norm) is not None


def select_terminology(
    source: str,
    entries: Sequence[tuple[str, str]],
    top_k: int,
) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    used_targets: set[str] = set()
    for source_term, target_term in entries:
        if not source_contains_term(source, source_term):
            continue
        target_key = target_term.lower()
        if target_key in used_targets:
            continue
        selected.append((source_term, target_term))
        used_targets.add(target_key)
        if len(selected) >= top_k:
            break
    return selected


def build_dataset(
    rows: Iterable[dict[str, str]],
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": prompt_messages(
                    row["source"],
                    select_terminology(row["source"], terminology_entries or [], terminology_top_k)
                    if terminology_entries and terminology_top_k > 0
                    else None,
                ),
                "target": row["target"],
                "source": row["source"],
                "source_name": row["source_name"],
                "variant": row["variant"],
            }
            for row in rows
        ]
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", normalize_text(text).lower())


def completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return normalize_text(completion)
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return normalize_text(str(last.get("content", "")))
    return normalize_text(str(completion))


def load_sacrebleu():
    try:
        import sacrebleu
    except ImportError as exc:  # pragma: no cover - exercised on remote env setup
        raise RuntimeError("Install sacrebleu to use chrF++/BLEU metrics: pip install sacrebleu") from exc
    return sacrebleu


def sentence_chrfpp(hypothesis: str, reference: str) -> float:
    sacrebleu = load_sacrebleu()
    return sacrebleu.metrics.CHRF(word_order=2).sentence_score(hypothesis, [reference]).score / 100.0


def sentence_bleu(hypothesis: str, reference: str) -> float:
    sacrebleu = load_sacrebleu()
    return sacrebleu.metrics.BLEU(effective_order=True).sentence_score(hypothesis, [reference]).score / 100.0


def token_f1(hypothesis: str, reference: str) -> float:
    hyp_tokens = word_tokens(hypothesis)
    ref_tokens = word_tokens(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in hyp_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_precision(hypothesis: str, reference: str) -> float:
    hyp_tokens = word_tokens(hypothesis)
    ref_tokens = word_tokens(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in hyp_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1
    return overlap / len(hyp_tokens)


def length_ratio_score(hypothesis: str, reference: str) -> float:
    hyp_len = max(1, len(normalize_text(hypothesis).split()))
    ref_len = max(1, len(normalize_text(reference).split()))
    ratio = hyp_len / ref_len
    return max(0.0, 1.0 - abs(math.log(ratio)))


def overlong_completion_penalty(
    hypothesis: str,
    reference: str,
    max_words: int | None = None,
    ratio_threshold: float = 0.0,
) -> float:
    """Penalize cap-hitting or very verbose RL completions.

    `max_completion_length` is token-based, but the reward function only sees
    decoded text. A word-count guard is an intentionally simple proxy that
    catches the failure mode observed in GSPO canaries: long, still-plausible
    completions that run to the generation cap and receive decent semantic
    reward anyway.
    """
    words = normalize_text(hypothesis).split()
    if not words:
        return 0.0
    penalty = 0.0
    hyp_len = len(words)
    ref_len = max(1, len(normalize_text(reference).split()))
    if max_words is not None and max_words > 0 and hyp_len >= max_words:
        penalty += min(0.45, 0.25 + (0.02 * (hyp_len - max_words)))
    if ratio_threshold and ratio_threshold > 0:
        ratio = hyp_len / ref_len
        if ratio >= ratio_threshold:
            penalty += min(0.45, 0.12 * (ratio - ratio_threshold + 1.0))
    return min(0.70, penalty)


def spanish_leakage_penalty(hypothesis: str) -> float:
    tokens = word_tokens(hypothesis)
    if not tokens:
        return 0.0
    leaked = sum(1 for token in tokens if token in SPANISH_STOPWORDS)
    return min(0.25, leaked / max(4, len(tokens)))


def source_copy_ratio(hypothesis: str, source: str | None) -> float:
    if not source:
        return 0.0
    hyp_tokens = [token for token in word_tokens(hypothesis) if token not in SPANISH_STOPWORDS]
    source_tokens = [token for token in word_tokens(source) if token not in SPANISH_STOPWORDS]
    if not hyp_tokens or not source_tokens:
        return 0.0
    source_set = set(source_tokens)
    copied = sum(1 for token in hyp_tokens if token in source_set)
    return copied / len(hyp_tokens)


def exact_source_copy(hypothesis: str, source: str | None) -> bool:
    return bool(source) and normalize_text(hypothesis).lower() == normalize_text(str(source)).lower()


def repetition_penalty(hypothesis: str, ngram_size: int = 3) -> float:
    tokens = word_tokens(hypothesis)
    if len(tokens) < ngram_size * 2:
        return 0.0
    ngrams = [tuple(tokens[index : index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)]
    repeated = len(ngrams) - len(set(ngrams))
    return min(0.20, repeated / max(1, len(ngrams)))


def chat_artifact_penalty(hypothesis: str) -> float:
    normalized = normalize_text(hypothesis).lower()
    artifacts = (
        "<think",
        "</think",
        "thinking process",
        "<|im_start|>",
        "<|im_end|>",
        " assistant ",
        " user ",
        " system ",
    )
    padded = f" {normalized} "
    hits = sum(padded.count(artifact) for artifact in artifacts)
    return min(0.60, 0.18 * hits)


def strip_chat_artifacts(text: str) -> str:
    normalized = normalize_text(text)
    final_answer = re.search(r"(?i)(?:final answer|respuesta final)\s*:\s*(.+)$", normalized)
    if final_answer:
        return normalize_text(final_answer.group(1))
    match = re.search(
        r"(?i)(?:thinking process|<think|</think|<\|im_start\|>|<\|im_end\|>|\b(?:assistant|user|system)\b)",
        normalized,
    )
    if not match:
        return normalized
    return normalize_text(normalized[: match.start()])


def source_entities(source: str | None) -> set[str]:
    if not source:
        return set()
    entities = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", source))
    entities.update(re.findall(r"\b[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑa-záéíóúüñ]{2,}\b", source))
    return entities


def entity_preservation_score(hypothesis: str, source: str | None) -> float:
    entities = source_entities(source)
    if not entities:
        return 1.0
    hypothesis_lower = hypothesis.lower()
    preserved = sum(1 for entity in entities if entity.lower() in hypothesis_lower)
    return preserved / len(entities)


def severity_penalty(
    hypothesis: str,
    reference: str,
    source: str | None,
    chrf: float,
    f1: float,
    length_score: float,
) -> float:
    """Transparent minor/major/critical severity map for translation failures."""
    penalty = 0.0
    copy_ratio = source_copy_ratio(hypothesis, source)
    if exact_source_copy(hypothesis, source):
        penalty += 0.55  # critical
    elif copy_ratio >= 0.55:
        penalty += 0.35  # major
    elif copy_ratio >= 0.30:
        penalty += 0.15  # minor

    leakage = spanish_leakage_penalty(hypothesis)
    if leakage >= 0.18:
        penalty += 0.30
    elif leakage >= 0.08:
        penalty += 0.15
    else:
        penalty += leakage

    if chrf < 0.18 and f1 < 0.12:
        penalty += 0.35
    elif chrf < 0.30 and f1 < 0.20:
        penalty += 0.18

    if length_score < 0.55:
        penalty += 0.20
    elif length_score < 0.70:
        penalty += 0.10

    entity_score = entity_preservation_score(hypothesis, source)
    if entity_score < 0.5:
        penalty += 0.25
    elif entity_score < 1.0:
        penalty += 0.10

    penalty += repetition_penalty(hypothesis)
    penalty += chat_artifact_penalty(hypothesis)
    del reference
    return min(1.0, penalty)


def baseline_reward_score(
    hypothesis: str,
    reference: str,
    source: str | None,
    chrf: float,
    bleu: float,
    f1: float,
    length_score: float,
) -> float:
    return (
        (0.62 * chrf)
        + (0.18 * bleu)
        + (0.12 * f1)
        + (0.08 * length_score)
        - spanish_leakage_penalty(hypothesis)
        - (0.20 * source_copy_ratio(hypothesis, source))
        - chat_artifact_penalty(hypothesis)
    )


def reference_rerank_metric_score(
    hypothesis: str,
    reference: str,
    source: str | None,
    chrf: float,
    bleu: float,
    f1: float,
    length_score: float,
) -> float:
    """Reference-aware score inspired by K-sample oracle reranking headroom."""
    exact_copy = 1.0 if exact_source_copy(hypothesis, source) else 0.0
    return (
        (0.44 * chrf)
        + (0.14 * bleu)
        + (0.24 * f1)
        + (0.08 * length_score)
        - (0.25 * source_copy_ratio(hypothesis, source))
        - (0.40 * exact_copy)
        - (0.22 * spanish_leakage_penalty(hypothesis))
        - (0.35 * chat_artifact_penalty(hypothesis))
        - (0.12 * repetition_penalty(hypothesis))
    )


def reward_score(
    hypothesis: str,
    reference: str,
    source: str | None,
    profile: str,
    overlong_max_words: int | None = None,
    overlong_ratio_threshold: float = 0.0,
    overlong_penalty_weight: float = 1.0,
) -> float:
    chrf = sentence_chrfpp(hypothesis, reference)
    bleu = sentence_bleu(hypothesis, reference)
    f1 = token_f1(hypothesis, reference)
    length_score = length_ratio_score(hypothesis, reference)
    entity_score = entity_preservation_score(hypothesis, source)
    severity = severity_penalty(hypothesis, reference, source, chrf, f1, length_score)
    copy_ratio = source_copy_ratio(hypothesis, source)
    leakage = spanish_leakage_penalty(hypothesis)
    repetition = repetition_penalty(hypothesis)
    artifacts = chat_artifact_penalty(hypothesis)
    overlong = overlong_penalty_weight * overlong_completion_penalty(
        hypothesis,
        reference,
        max_words=overlong_max_words,
        ratio_threshold=overlong_ratio_threshold,
    )
    anti_copy = 1.0 - min(1.0, copy_ratio * 1.8)
    anti_leakage = 1.0 - min(1.0, leakage * 4.0)

    baseline_score = baseline_reward_score(hypothesis, reference, source, chrf, bleu, f1, length_score)
    fg_score = (0.42 * chrf) + (0.18 * f1) + (0.12 * bleu) + (0.14 * length_score) + (0.14 * entity_score) - severity
    severity_score = (
        (0.48 * chrf)
        + (0.17 * bleu)
        + (0.15 * f1)
        + (0.10 * length_score)
        + (0.10 * entity_score)
        - (0.85 * severity)
    )
    meaning_score = (0.58 * chrf) + (0.27 * f1) + (0.15 * bleu)
    verifier_score = (
        (0.42 * meaning_score)
        + (0.18 * anti_copy)
        + (0.16 * anti_leakage)
        + (0.12 * length_score)
        + (0.08 * entity_score)
        + (0.04 * (1.0 - min(1.0, repetition * 5.0)))
    )
    if exact_source_copy(hypothesis, source):
        verifier_score -= 0.50
    verifier_score -= 0.35 * severity
    vibe_score = baseline_score - (0.25 * repetition) - artifacts

    if profile == "fg_severity_2411":
        # 2411.05986: densify sparse sentence rewards with severity-weighted token/error signals.
        return fg_score - overlong
    if profile == "severity_proxy_2310":
        # 2310.10482 motivation only: sentence score plus transparent error severity.
        # This is not xCOMET; it is a cheap ablation for severity-weighted rewards.
        return severity_score - overlong
    if profile == "self_verifier_2511":
        # 2511.22570: verifier-style rubric; reward faithful translation plus explicit failure detection proxies.
        return verifier_score - overlong
    if profile == "vibethinker_2511":
        # 2511.06221: keep a broad candidate spectrum, then amplify the best signal.
        return vibe_score - overlong
    if profile == "mix_severity_verifier":
        return (0.45 * fg_score) + (0.25 * severity_score) + (0.30 * verifier_score) - overlong
    if profile == "mix_verifier_vibe":
        return (0.58 * verifier_score) + (0.42 * vibe_score) - overlong
    if profile == "mix_all_strict":
        return (
            (0.22 * baseline_score)
            + (0.28 * fg_score)
            + (0.20 * severity_score)
            + (0.20 * verifier_score)
            + (0.10 * vibe_score)
            - (0.15 * copy_ratio)
            - (0.25 * leakage)
            - (0.30 * artifacts)
            - overlong
        )
    if profile == "rosettia_guard_v1":
        quality = (0.45 * chrf) + (0.25 * f1) + (0.10 * bleu)
        guards = (0.08 * length_score) + (0.06 * entity_score) + (0.06 * anti_copy) + (0.06 * anti_leakage)
        return quality + guards - (0.45 * severity) - (0.15 * repetition) - (0.20 * artifacts) - overlong
    if profile == "rosettia_guard_v2":
        quality = (0.34 * chrf) + (0.24 * f1) + (0.08 * bleu)
        guards = (0.12 * length_score) + (0.08 * entity_score) + (0.10 * anti_copy) + (0.08 * anti_leakage)
        return quality + guards - (0.30 * severity) - (0.20 * repetition) - (0.25 * artifacts) - overlong
    if profile == "reference_rerank_vibe_v1":
        return reference_rerank_metric_score(hypothesis, reference, source, chrf, bleu, f1, length_score) - overlong
    return baseline_score - overlong


def verifier_prompt_text(tokenizer, source: str, reference: str, candidate: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Eres un verificador experto de traducciones español a quechua chanka.",
        },
        {
            "role": "user",
            "content": (
                "Evalua si la traduccion candidata conserva el significado, evita copiar "
                "el espanol, respeta entidades/numeros y suena natural en quechua chanka. "
                "Devuelve solo JSON compacto con score entre 0 y 1, severity y rationale.\n\n"
                f"Español: {source}\n"
                f"Referencia chanka: {reference}\n"
                f"Candidata: {candidate}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_verifier_score(text: str) -> float:
    match = re.search(r'"score"\s*:\s*([01](?:\.\d+)?)', text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    loose = re.search(r"\b(?:score|puntaje)\b\D{0,12}([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if loose:
        return max(0.0, min(1.0, float(loose.group(1))))
    return 0.0


class LearnedVerifierScorer:
    def __init__(self, adapter_path: Path, max_seq_length: int, max_new_tokens: int, batch_size: int) -> None:
        from unsloth import FastLanguageModel
        import torch

        self.torch = torch
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, batch_size)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=max_seq_length,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        FastLanguageModel.for_inference(self.model)
        self.model.eval()

    def score_many(self, sources: list[str | None], references: list[str], hypotheses: list[str]) -> list[float]:
        prompts = [
            verifier_prompt_text(self.tokenizer, source or "", reference, hypothesis)
            for source, reference, hypothesis in zip(sources, references, hypotheses, strict=True)
        ]
        scores: list[float] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            inputs = self.tokenizer(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            ).to(self.model.device)
            prompt_length = inputs["input_ids"].shape[1]
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            for row_index in range(len(batch_prompts)):
                completion_ids = output_ids[row_index, prompt_length:]
                decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                scores.append(parse_verifier_score(decoded))
        return scores


def learned_verifier_rewards(
    scorer: LearnedVerifierScorer,
    hypotheses: list[str],
    references: list[str],
    sources: list[str | None],
    overlong_max_words: int | None = None,
    overlong_ratio_threshold: float = 0.0,
    overlong_penalty_weight: float = 1.0,
) -> list[float]:
    verifier_scores = scorer.score_many(sources, references, hypotheses)
    rewards: list[float] = []
    for verifier_score, hypothesis, reference, source in zip(verifier_scores, hypotheses, references, sources, strict=True):
        chrf = sentence_chrfpp(hypothesis, reference)
        bleu = sentence_bleu(hypothesis, reference)
        f1 = token_f1(hypothesis, reference)
        length_score = length_ratio_score(hypothesis, reference)
        guard_score = baseline_reward_score(hypothesis, reference, source, chrf, bleu, f1, length_score)
        guard_penalty = (
            0.35 * source_copy_ratio(hypothesis, source)
            + 0.45 * spanish_leakage_penalty(hypothesis)
            + 0.20 * repetition_penalty(hypothesis)
            + 0.60 * chat_artifact_penalty(hypothesis)
            + overlong_penalty_weight
            * overlong_completion_penalty(
                hypothesis,
                reference,
                max_words=overlong_max_words,
                ratio_threshold=overlong_ratio_threshold,
            )
        )
        if exact_source_copy(hypothesis, source):
            guard_penalty += 0.50
        rewards.append((0.72 * verifier_score) + (0.28 * guard_score) - guard_penalty)
    return rewards


def learned_verifier_bleu_margin_rewards(
    scorer: LearnedVerifierScorer,
    hypotheses: list[str],
    references: list[str],
    sources: list[str | None],
    overlong_max_words: int | None = None,
    overlong_ratio_threshold: float = 0.0,
    overlong_penalty_weight: float = 1.0,
) -> list[float]:
    verifier_scores = scorer.score_many(sources, references, hypotheses)
    rewards: list[float] = []
    for verifier_score, hypothesis, reference, source in zip(verifier_scores, hypotheses, references, sources, strict=True):
        chrf = sentence_chrfpp(hypothesis, reference)
        bleu = sentence_bleu(hypothesis, reference)
        f1 = token_f1(hypothesis, reference)
        precision = token_precision(hypothesis, reference)
        length_score = length_ratio_score(hypothesis, reference)
        semantic_guard = (0.38 * chrf) + (0.26 * f1) + (0.18 * bleu)
        shape_guard = (0.10 * length_score) + (0.08 * precision)
        guard_penalty = (
            0.32 * source_copy_ratio(hypothesis, source)
            + 0.55 * spanish_leakage_penalty(hypothesis)
            + 0.18 * repetition_penalty(hypothesis)
            + 0.70 * chat_artifact_penalty(hypothesis)
            + overlong_penalty_weight
            * overlong_completion_penalty(
                hypothesis,
                reference,
                max_words=overlong_max_words,
                ratio_threshold=overlong_ratio_threshold,
            )
        )
        if exact_source_copy(hypothesis, source):
            guard_penalty += 0.55
        rewards.append((0.58 * verifier_score) + (0.42 * (semantic_guard + shape_guard)) - guard_penalty)
    return rewards


def learned_verifier_ensemble_rewards(
    primary_scorer: LearnedVerifierScorer,
    secondary_scorer: LearnedVerifierScorer,
    hypotheses: list[str],
    references: list[str],
    sources: list[str | None],
    overlong_max_words: int | None = None,
    overlong_ratio_threshold: float = 0.0,
    overlong_penalty_weight: float = 1.0,
) -> list[float]:
    primary_scores = primary_scorer.score_many(sources, references, hypotheses)
    secondary_scores = secondary_scorer.score_many(sources, references, hypotheses)
    rewards: list[float] = []
    for primary_score, secondary_score, hypothesis, reference, source in zip(
        primary_scores,
        secondary_scores,
        hypotheses,
        references,
        sources,
        strict=True,
    ):
        chrf = sentence_chrfpp(hypothesis, reference)
        bleu = sentence_bleu(hypothesis, reference)
        f1 = token_f1(hypothesis, reference)
        length_score = length_ratio_score(hypothesis, reference)
        guard_score = baseline_reward_score(hypothesis, reference, source, chrf, bleu, f1, length_score)
        guard_penalty = (
            0.35 * source_copy_ratio(hypothesis, source)
            + 0.45 * spanish_leakage_penalty(hypothesis)
            + 0.20 * repetition_penalty(hypothesis)
            + 0.60 * chat_artifact_penalty(hypothesis)
            + overlong_penalty_weight
            * overlong_completion_penalty(
                hypothesis,
                reference,
                max_words=overlong_max_words,
                ratio_threshold=overlong_ratio_threshold,
            )
        )
        if exact_source_copy(hypothesis, source):
            guard_penalty += 0.50
        verifier_score = (0.56 * primary_score) + (0.44 * secondary_score)
        rewards.append((0.70 * verifier_score) + (0.30 * guard_score) - guard_penalty)
    return rewards


def add_vibethinker_diversity_bonus(rewards: list[float], hypotheses: list[str], sources: list[str | None]) -> list[float]:
    grouped: dict[str | None, list[int]] = {}
    for index, source in enumerate(sources):
        grouped.setdefault(source, []).append(index)
    adjusted = rewards[:]
    for indices in grouped.values():
        normalized = [normalize_text(hypotheses[index]).lower() for index in indices]
        counts = {text: normalized.count(text) for text in set(normalized)}
        unique_count = len(counts)
        for index, text in zip(indices, normalized, strict=True):
            duplicate_penalty = 0.04 * max(0, counts[text] - 1)
            diversity_bonus = 0.03 * (unique_count - 1) / max(1, len(indices) - 1)
            adjusted[index] += diversity_bonus - duplicate_penalty
    return adjusted


def chanka_reward(
    completions: list,
    target: list[str],
    source: list[str] | None = None,
    profile: str | None = None,
    overlong_max_words: int | None = None,
    overlong_ratio_threshold: float = 0.0,
    overlong_penalty_weight: float = 1.0,
    **_: object,
) -> list[float]:
    active_profile = profile or REWARD_PROFILE
    rewards: list[float] = []
    hypotheses: list[str] = []
    sources = list(source) if source is not None else [None] * len(target)
    for completion, reference, source_text in zip(completions, target, sources, strict=True):
        hypothesis = completion_text(completion)
        hypotheses.append(hypothesis)
        rewards.append(
            reward_score(
                hypothesis,
                reference,
                source_text,
                active_profile,
                overlong_max_words=overlong_max_words,
                overlong_ratio_threshold=overlong_ratio_threshold,
                overlong_penalty_weight=overlong_penalty_weight,
            )
        )
    if active_profile in {"vibethinker_2511", "mix_verifier_vibe", "reference_rerank_vibe_v1"}:
        rewards = add_vibethinker_diversity_bonus(rewards, hypotheses, sources)
    return rewards


def make_reward_fn(
    profile: str,
    verifier_adapter_path: Path | None = None,
    secondary_verifier_adapter_path: Path | None = None,
    verifier_max_seq_length: int = 512,
    verifier_max_new_tokens: int = 96,
    verifier_batch_size: int = 4,
    overlong_max_words: int | None = None,
    overlong_ratio_threshold: float = 0.0,
    overlong_penalty_weight: float = 1.0,
):
    verifier_scorer = None
    secondary_verifier_scorer = None
    learned_profiles = {
        "learned_verifier_2511",
        "learned_verifier_vibe_2511",
        "learned_verifier_ensemble_vibe_2511",
        "learned_verifier_bleu_margin_vibe_2511",
    }
    if profile in learned_profiles:
        if verifier_adapter_path is None:
            raise ValueError(
                "--verifier-adapter-path is required for learned-verifier reward profiles"
            )
        verifier_scorer = LearnedVerifierScorer(
            verifier_adapter_path,
            max_seq_length=verifier_max_seq_length,
            max_new_tokens=verifier_max_new_tokens,
            batch_size=verifier_batch_size,
        )
    if profile == "learned_verifier_ensemble_vibe_2511":
        if secondary_verifier_adapter_path is None:
            raise ValueError(
                "--secondary-verifier-adapter-path is required for learned_verifier_ensemble_vibe_2511"
            )
        secondary_verifier_scorer = LearnedVerifierScorer(
            secondary_verifier_adapter_path,
            max_seq_length=verifier_max_seq_length,
            max_new_tokens=verifier_max_new_tokens,
            batch_size=verifier_batch_size,
        )

    def reward_fn(completions: list, target: list[str], source: list[str] | None = None, **kwargs: object) -> list[float]:
        if profile in {"learned_verifier_2511", "learned_verifier_vibe_2511"}:
            assert verifier_scorer is not None
            sources = list(source) if source is not None else [None] * len(target)
            hypotheses = [completion_text(completion) for completion in completions]
            rewards = learned_verifier_rewards(
                verifier_scorer,
                hypotheses,
                list(target),
                sources,
                overlong_max_words=overlong_max_words,
                overlong_ratio_threshold=overlong_ratio_threshold,
                overlong_penalty_weight=overlong_penalty_weight,
            )
            if profile == "learned_verifier_vibe_2511":
                rewards = add_vibethinker_diversity_bonus(rewards, hypotheses, sources)
            return rewards
        if profile == "learned_verifier_bleu_margin_vibe_2511":
            assert verifier_scorer is not None
            sources = list(source) if source is not None else [None] * len(target)
            hypotheses = [completion_text(completion) for completion in completions]
            rewards = learned_verifier_bleu_margin_rewards(
                verifier_scorer,
                hypotheses,
                list(target),
                sources,
                overlong_max_words=overlong_max_words,
                overlong_ratio_threshold=overlong_ratio_threshold,
                overlong_penalty_weight=overlong_penalty_weight,
            )
            return add_vibethinker_diversity_bonus(rewards, hypotheses, sources)
        if profile == "learned_verifier_ensemble_vibe_2511":
            assert verifier_scorer is not None
            assert secondary_verifier_scorer is not None
            sources = list(source) if source is not None else [None] * len(target)
            hypotheses = [completion_text(completion) for completion in completions]
            rewards = learned_verifier_ensemble_rewards(
                verifier_scorer,
                secondary_verifier_scorer,
                hypotheses,
                list(target),
                sources,
                overlong_max_words=overlong_max_words,
                overlong_ratio_threshold=overlong_ratio_threshold,
                overlong_penalty_weight=overlong_penalty_weight,
            )
            return add_vibethinker_diversity_bonus(rewards, hypotheses, sources)
        return chanka_reward(
            completions,
            target,
            source=source,
            profile=profile,
            overlong_max_words=overlong_max_words,
            overlong_ratio_threshold=overlong_ratio_threshold,
            overlong_penalty_weight=overlong_penalty_weight,
            **kwargs,
        )

    reward_fn.__name__ = f"chanka_reward_{profile}"
    return reward_fn


def corpus_metrics(predictions: list[str], references: list[str], sources: list[str] | None = None) -> dict[str, float]:
    sacrebleu = load_sacrebleu()
    cleaned_predictions = [normalize_text(prediction) for prediction in predictions]
    cleaned_references = [normalize_text(reference) for reference in references]
    metrics = {
        "chrf2": sacrebleu.metrics.CHRF(word_order=0).corpus_score(
            cleaned_predictions, [cleaned_references]
        ).score,
        "chrf++": sacrebleu.metrics.CHRF(word_order=2).corpus_score(
            cleaned_predictions, [cleaned_references]
        ).score,
        "bleu": sacrebleu.metrics.BLEU(effective_order=True).corpus_score(
            cleaned_predictions, [cleaned_references]
        ).score,
        "ter": sacrebleu.metrics.TER().corpus_score(cleaned_predictions, [cleaned_references]).score,
        "token_f1": 100.0
        * sum(token_f1(prediction, reference) for prediction, reference in zip(cleaned_predictions, cleaned_references))
        / max(1, len(cleaned_predictions)),
        "length_ratio_score": 100.0
        * sum(
            length_ratio_score(prediction, reference)
            for prediction, reference in zip(cleaned_predictions, cleaned_references)
        )
        / max(1, len(cleaned_predictions)),
        "spanish_leakage_penalty": 100.0
        * sum(spanish_leakage_penalty(prediction) for prediction in cleaned_predictions)
        / max(1, len(cleaned_predictions)),
        "chat_artifact_penalty": 100.0
        * sum(chat_artifact_penalty(prediction) for prediction in cleaned_predictions)
        / max(1, len(cleaned_predictions)),
        "avg_prediction_words": sum(len(prediction.split()) for prediction in cleaned_predictions)
        / max(1, len(cleaned_predictions)),
        "avg_reference_words": sum(len(reference.split()) for reference in cleaned_references)
        / max(1, len(cleaned_references)),
    }
    if sources:
        copied = 0
        copy_ratios = []
        for prediction, source in zip(cleaned_predictions, sources, strict=True):
            copied += int(prediction.lower() == normalize_text(source).lower())
            copy_ratios.append(source_copy_ratio(prediction, source))
        metrics["exact_source_copy_rate"] = 100.0 * copied / max(1, len(cleaned_predictions))
        metrics["source_copy_ratio"] = 100.0 * sum(copy_ratios) / max(1, len(copy_ratios))
    return metrics


def generate_predictions(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    max_completion_length: int,
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> list[str]:
    import torch

    predictions: list[str] = []
    model.eval()
    for row in rows:
        terminology = (
            select_terminology(row["source"], terminology_entries or [], terminology_top_k)
            if terminology_entries and terminology_top_k > 0
            else None
        )
        prompt = apply_chat_template_no_thinking(
            tokenizer,
            prompt_messages(row["source"], terminology),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        predictions.append(normalize_text(tokenizer.decode(completion_ids, skip_special_tokens=True)))
    return predictions


def latest_eval_metrics(log_history: list[dict[str, float]]) -> dict[str, float]:
    for record in reversed(log_history):
        if "eval_reward" in record:
            return record
    return {}


def make_jsonl_log_callback(path: Path):
    from transformers import TrainerCallback

    class JsonlLogCallback(TrainerCallback):
        def __init__(self, log_path: Path) -> None:
            self.log_path = log_path
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")

        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, control, kwargs
            if not logs:
                return
            payload = {"step": state.global_step, **logs}
            with self.log_path.open("a") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    return JsonlLogCallback(path)


def main() -> None:
    args = parse_args()
    validate_grpo_batching(args)
    global REWARD_PROFILE
    REWARD_PROFILE = args.reward_profile

    from unsloth import FastLanguageModel
    import torch
    import trl.import_utils as trl_import_utils

    # TRL imports optional callbacks and judges while exposing GRPO. Keep those
    # optional integrations disabled so the working Unsloth stack does not need
    # unrelated extras such as MergeKit or LLM-Blender.
    trl_import_utils._mergekit_available = False
    trl_import_utils._llm_blender_available = False
    trl_import_utils._weave_available = False
    from trl import GRPOConfig, GRPOTrainer

    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project

    rows = load_chanka_rows(args.dataset_repo, args.dataset_file)
    train_rows, eval_rows = split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    terminology_entries = (
        load_terminology_entries(args.dataset_repo, args.terminology_file, args.terminology_min_source_chars)
        if args.terminology_file
        else []
    )
    train_term_rows = (
        sum(
            1
            for row in train_rows
            if select_terminology(row["source"], terminology_entries, args.terminology_top_k)
        )
        if terminology_entries
        else 0
    )
    eval_term_rows = (
        sum(
            1
            for row in eval_rows
            if select_terminology(row["source"], terminology_entries, args.terminology_top_k)
        )
        if terminology_entries
        else 0
    )
    configure_step_schedule(args, len(train_rows))

    run_dir = args.output_dir / "chanka_gspo"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_name_or_adapter = str(args.adapter_path) if args.adapter_path else args.model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if args.attach_lora:
        if hasattr(model, "peft_config"):
            raise ValueError("--attach-lora expects a base/full model, but the loaded model already has PEFT config")
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            max_seq_length=args.max_seq_length,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    train_dataset = build_dataset(train_rows, terminology_entries, args.terminology_top_k)
    eval_dataset = build_dataset(eval_rows, terminology_entries, args.terminology_top_k)

    print(f"Loaded Chanka rows: {len(rows):,}")
    print(f"Train rows: {len(train_dataset):,}")
    print(f"Validation rows: {len(eval_dataset):,}")
    if args.terminology_file:
        print(f"Terminology file: {args.terminology_file}")
        print(f"Terminology entries: {len(terminology_entries):,}")
        print(f"Terminology-matched train rows: {train_term_rows:,}")
        print(f"Terminology-matched validation rows: {eval_term_rows:,}")
    print(f"Model or adapter: {model_name_or_adapter}")
    if args.attach_lora:
        print(f"Attached fresh LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    print('GSPO mode: GRPO with importance_sampling_level=\"sequence\"')
    print(f"Reward profile: {args.reward_profile}")
    print(f"Validation: every {args.eval_steps} steps")
    print(f"Saving: every {args.save_steps} steps")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_fn(
            args.reward_profile,
            verifier_adapter_path=args.verifier_adapter_path,
            secondary_verifier_adapter_path=args.secondary_verifier_adapter_path,
            verifier_max_seq_length=args.verifier_max_seq_length,
            verifier_max_new_tokens=args.verifier_max_new_tokens,
            verifier_batch_size=args.verifier_batch_size,
            overlong_max_words=args.overlong_max_words,
            overlong_ratio_threshold=args.overlong_ratio_threshold,
            overlong_penalty_weight=args.overlong_penalty_weight,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[make_jsonl_log_callback(run_dir / "scalar_logs.jsonl")],
        args=GRPOConfig(
            output_dir=str(run_dir),
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit or None,
            beta=args.beta,
            epsilon=args.epsilon,
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            generation_kwargs={
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
            },
            loss_type=args.loss_type,
            scale_rewards=args.scale_rewards,
            importance_sampling_level="sequence",
            mask_truncated_completions=True,
            log_completions=args.log_completions,
            num_completions_to_print=4,
            optim="adamw_8bit",
            seed=args.seed,
            report_to="wandb" if args.wandb_project else "none",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
        ),
    )

    trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None
    )
    trainer_metrics = latest_eval_metrics(trainer.state.log_history)
    print(trainer_metrics)

    final_dir = run_dir / "final_gspo_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    predictions = generate_predictions(
        model,
        tokenizer,
        eval_rows,
        args.max_completion_length,
        terminology_entries,
        args.terminology_top_k,
    )
    references = [row["target"] for row in eval_rows]
    sources = [row["source"] for row in eval_rows]
    metrics = corpus_metrics(predictions, references, sources)
    metrics["eval_rows"] = len(eval_rows)
    metrics["trainer_eval_reward"] = float(trainer_metrics.get("eval_reward", 0.0))
    metrics["final_adapter"] = str(final_dir)
    metrics["reward_profile"] = args.reward_profile
    metrics["overlong_max_words"] = args.overlong_max_words
    metrics["overlong_ratio_threshold"] = args.overlong_ratio_threshold
    metrics["overlong_penalty_weight"] = args.overlong_penalty_weight
    if args.overlong_max_words or args.overlong_ratio_threshold > 0:
        metrics["overlong_completion_penalty"] = 100.0 * sum(
            overlong_completion_penalty(
                prediction,
                reference,
                max_words=args.overlong_max_words,
                ratio_threshold=args.overlong_ratio_threshold,
            )
            for prediction, reference in zip(predictions, references, strict=True)
        ) / max(1, len(predictions))
    if args.terminology_file:
        metrics["terminology_file"] = args.terminology_file
        metrics["terminology_entries"] = len(terminology_entries)
        metrics["terminology_top_k"] = args.terminology_top_k
        metrics["terminology_train_matched_rows"] = train_term_rows
        metrics["terminology_eval_matched_rows"] = eval_term_rows
    print(json.dumps(metrics, indent=2, sort_keys=True))

    metrics_path = args.metrics_json or (run_dir / "final_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
