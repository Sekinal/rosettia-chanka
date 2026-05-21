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
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
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
    parser.add_argument("--evals-per-epoch", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--wandb-project", default=None)
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
    train_global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    eval_global_batch = args.per_device_eval_batch_size
    if train_global_batch % args.num_generations != 0:
        raise ValueError(
            "GRPO/GSPO requires train batch * gradient accumulation "
            f"({train_global_batch}) to be divisible by num_generations "
            f"({args.num_generations})."
        )
    if eval_global_batch % args.num_generations != 0:
        raise ValueError(
            "GRPO/GSPO requires eval batch "
            f"({eval_global_batch}) to be divisible by num_generations "
            f"({args.num_generations})."
        )


def prompt_messages(source: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Eres un traductor profesional español-quechua chanka."},
        {
            "role": "user",
            "content": (
                "Traduce del español al quechua chanka. Usa una traducción directa, "
                "fiel y apropiada para contexto judicial.\n\n"
                f"Español: {source}"
            ),
        },
    ]


def build_dataset(rows: Iterable[dict[str, str]]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": prompt_messages(row["source"]),
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
    hyp_tokens = normalize_text(hypothesis).lower().split()
    ref_tokens = normalize_text(reference).lower().split()
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


def length_ratio_score(hypothesis: str, reference: str) -> float:
    hyp_len = max(1, len(normalize_text(hypothesis).split()))
    ref_len = max(1, len(normalize_text(reference).split()))
    ratio = hyp_len / ref_len
    return max(0.0, 1.0 - abs(math.log(ratio)))


def spanish_leakage_penalty(hypothesis: str) -> float:
    tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", hypothesis.lower())
    if not tokens:
        return 0.0
    leaked = sum(1 for token in tokens if token in SPANISH_STOPWORDS)
    return min(0.25, leaked / max(4, len(tokens)))


def chanka_reward(completions: list, target: list[str], source: list[str] | None = None, **_: object) -> list[float]:
    rewards: list[float] = []
    for completion, reference in zip(completions, target, strict=True):
        hypothesis = completion_text(completion)
        chrf = sentence_chrfpp(hypothesis, reference)
        bleu = sentence_bleu(hypothesis, reference)
        f1 = token_f1(hypothesis, reference)
        length_score = length_ratio_score(hypothesis, reference)
        copy_penalty = spanish_leakage_penalty(hypothesis)
        rewards.append((0.62 * chrf) + (0.18 * bleu) + (0.12 * f1) + (0.08 * length_score) - copy_penalty)
    return rewards


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
    }
    if sources:
        copied = 0
        for prediction, source in zip(cleaned_predictions, sources, strict=True):
            copied += int(prediction.lower() == normalize_text(source).lower())
        metrics["exact_source_copy_rate"] = 100.0 * copied / max(1, len(cleaned_predictions))
    return metrics


def generate_predictions(model, tokenizer, rows: list[dict[str, str]], max_completion_length: int) -> list[str]:
    import torch

    predictions: list[str] = []
    model.eval()
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            prompt_messages(row["source"]),
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

    train_dataset = build_dataset(train_rows)
    eval_dataset = build_dataset(eval_rows)

    print(f"Loaded Chanka rows: {len(rows):,}")
    print(f"Train rows: {len(train_dataset):,}")
    print(f"Validation rows: {len(eval_dataset):,}")
    print(f"Model or adapter: {model_name_or_adapter}")
    print('GSPO mode: GRPO with importance_sampling_level=\"sequence\"')
    print(f"Reward: chrF++ + BLEU + token-F1 + length score - Spanish leakage penalty")
    print(f"Validation: every {args.eval_steps} steps")
    print(f"Saving: every {args.save_steps} steps")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=chanka_reward,
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
            save_total_limit=3,
            beta=args.beta,
            epsilon=args.epsilon,
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
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

    trainer.train()
    trainer_metrics = latest_eval_metrics(trainer.state.log_history)
    print(trainer_metrics)

    final_dir = run_dir / "final_gspo_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    predictions = generate_predictions(model, tokenizer, eval_rows, args.max_completion_length)
    references = [row["target"] for row in eval_rows]
    sources = [row["source"] for row in eval_rows]
    metrics = corpus_metrics(predictions, references, sources)
    metrics["eval_rows"] = len(eval_rows)
    metrics["trainer_eval_reward"] = float(trainer_metrics.get("eval_reward", 0.0))
    metrics["final_adapter"] = str(final_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    metrics_path = args.metrics_json or (run_dir / "final_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
