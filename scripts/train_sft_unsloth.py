"""SFT train RosettIA translation adapters with Unsloth.

This script intentionally uses 16-bit LoRA, not QLoRA. Unsloth's Qwen3.5 guide
does not recommend QLoRA for Qwen3.5 because quantization differences are larger
than usual. Every run creates a validation split and monitors eval loss.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
from datasets import Dataset
from huggingface_hub import hf_hub_download


DATASET_REPO = "Thermostatic/rosettia-chanka-data"
DEFAULT_MODEL_ID = "unsloth/Qwen3.5-2B"

BROAD_FILES = [
    "broad_quechua/somosnlp_spanish_to_quechua_high_quality_sft.parquet",
    "americasnlp/americasnlp_quechua_spanish_high_quality_real_sft.parquet",
]
CHANKA_FILE = "clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["broad", "chanka"],
        required=True,
        help="SFT data tier: broad SomosNLP/AmericasNLP data, or the clean reviewed Chanka corpus.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--prompt-style",
        choices=["generic", "hymt2"],
        default="generic",
        help="Chat prompt protocol for SFT examples. Hy-MT2 uses a user-only translation prompt.",
    )
    parser.add_argument("--target-language-name", default=None)
    parser.add_argument(
        "--adapter-method",
        choices=["lora", "dora", "rslora"],
        default="lora",
        help="Adapter variant to train when starting from the base model.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["lora", "full"],
        default="lora",
        help="Train a PEFT adapter, or full-finetune all model weights with Unsloth FFT.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional local/HF LoRA adapter to continue training from. Only valid with --training-mode lora.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use Unsloth 4-bit loading for huge non-Qwen model-family canaries. Keep off for Qwen3.5 runs.",
    )
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument(
        "--terminology-file",
        default=None,
        help="Optional dataset-repo parquet glossary for terminology-conditioned SFT prompts.",
    )
    parser.add_argument("--terminology-top-k", type=int, default=6)
    parser.add_argument("--terminology-min-source-chars", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--optim",
        default="adamw_8bit",
        help=(
            "Trainer optimizer. Use paged_adamw_8bit for large full-finetuning "
            "smokes when regular adamw_8bit runs out of GPU memory."
        ),
    )
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument(
        "--save-only-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save only model/tokenizer files at checkpoints. Defaults to true for full fine-tuning "
            "because optimizer states can exceed model checkpoint size."
        ),
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep. Use 0 to disable checkpoint pruning.",
    )
    parser.add_argument(
        "--evals-per-epoch",
        type=int,
        default=None,
        help="If eval/save steps are not set, evaluate this many times inside each epoch.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument(
        "--train-on-responses-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask user/prompt tokens. Disable if chat-template markers change and masking fails.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo id for the final trained artifact.")
    return parser.parse_args(argv)


def stage_defaults(args: argparse.Namespace) -> None:
    training_mode = getattr(args, "training_mode", "lora")
    if args.max_seq_length is None:
        args.max_seq_length = 512 if args.stage == "broad" else 128
    if args.validation_fraction is None:
        args.validation_fraction = 0.02 if args.stage == "broad" else 0.15
    if args.num_train_epochs is None:
        args.num_train_epochs = 1.0 if args.stage == "broad" else 8.0
    if args.learning_rate is None:
        if training_mode == "full":
            args.learning_rate = 2.0e-6 if args.stage == "broad" else 1.0e-6
        else:
            args.learning_rate = 1.0e-4 if args.stage == "broad" else 2.0e-5
    if args.per_device_train_batch_size is None:
        args.per_device_train_batch_size = 1 if training_mode == "full" else 8
    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = 2 if training_mode == "full" else args.per_device_train_batch_size
    if args.gradient_accumulation_steps is None:
        if training_mode == "full":
            args.gradient_accumulation_steps = 8
        else:
            args.gradient_accumulation_steps = 2 if args.stage == "broad" else 1
    if args.lora_r is None:
        args.lora_r = 64
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2
    if args.evals_per_epoch is None:
        args.evals_per_epoch = 4 if args.stage == "broad" else 8
    if getattr(args, "save_only_model", None) is None:
        args.save_only_model = training_mode == "full"


def adapter_flags(adapter_method: str) -> dict[str, bool]:
    return {
        "use_dora": adapter_method == "dora",
        "use_rslora": adapter_method == "rslora",
    }


def validate_training_mode_args(args: argparse.Namespace) -> None:
    if args.training_mode == "full" and args.adapter_path is not None:
        raise ValueError("--adapter-path is only valid with --training-mode lora")
    if args.training_mode == "full" and args.load_in_4bit:
        raise ValueError("--load-in-4bit is only valid with --training-mode lora")
    if args.load_in_4bit and args.adapter_path is not None:
        raise ValueError("--load-in-4bit is only supported when creating a new LoRA adapter")


def final_artifact_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.training_mode == "full":
        return run_dir / "final_full_model"
    return run_dir / f"final_{args.adapter_method}"


def download_parquet(repo_id: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))


def rows_from_frame(frame: pl.DataFrame, stage: str, source_name: str) -> list[dict[str, str]]:
    if stage == "chanka":
        spanish_col = "reviewed_spanish"
        quechua_col = "reviewed_chanka_quechua"
        variant = "quy/chanka"
    else:
        spanish_col = "es"
        quechua_col = "quy" if "quy" in frame.columns else "qu"
        variant = "quy_or_broad_quechua"

    rows = []
    for row in frame.select([spanish_col, quechua_col]).iter_rows(named=True):
        source = str(row[spanish_col]).strip()
        target = str(row[quechua_col]).strip()
        if not source or not target:
            continue
        rows.append(
            {
                "source": source,
                "target": target,
                "source_name": source_name,
                "variant": variant,
            }
        )
    return rows


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    filenames = BROAD_FILES if args.stage == "broad" else [CHANKA_FILE]
    rows: list[dict[str, str]] = []
    for filename in filenames:
        path = download_parquet(args.dataset_repo, filename)
        frame = pl.read_parquet(path)
        rows.extend(rows_from_frame(frame, args.stage, filename))
    if not rows:
        raise RuntimeError(f"No rows loaded for stage={args.stage}")
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


def optimizer_steps_per_epoch(
    train_row_count: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    effective_batch_size = max(1, per_device_train_batch_size) * max(1, gradient_accumulation_steps)
    return max(1, math.ceil(train_row_count / effective_batch_size))


def configure_step_schedule(args: argparse.Namespace, train_row_count: int) -> None:
    if args.eval_steps is not None and args.save_steps is not None:
        return

    if args.max_steps and args.max_steps > 0:
        fallback_steps = max(1, args.max_steps // max(1, args.evals_per_epoch))
    else:
        steps_per_epoch = optimizer_steps_per_epoch(
            train_row_count,
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
        )
        fallback_steps = max(1, steps_per_epoch // max(1, args.evals_per_epoch))

    if args.eval_steps is None:
        args.eval_steps = fallback_steps
    if args.save_steps is None:
        args.save_steps = args.eval_steps


def target_language_name_for(stage: str, override: str | None) -> str:
    if override:
        return override
    return "Quechua Chanka" if stage == "chanka" else "Quechua"


def instruction_for(stage: str) -> str:
    if stage == "chanka":
        return (
            "Traduce del español al quechua chanka. Usa una traducción directa, "
            "fiel y apropiada para contexto judicial."
        )
    return "Traduce del español al quechua. Conserva el significado y no copies el español."


def terminology_block(terminology: Sequence[tuple[str, str]] | None) -> str:
    if not terminology:
        return ""
    lines = [
        "",
        "Glosario sugerido (úsalo solo cuando encaje con el contexto; no fuerces términos incorrectos):",
    ]
    lines.extend(f"- {source_term} = {target_text}" for source_term, target_text in terminology)
    return "\n".join(lines)


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


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def source_contains_term(source: str, source_term: str) -> bool:
    import re

    source_norm = normalize_text(source).lower()
    term_norm = normalize_text(source_term).lower()
    if not term_norm:
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


def terminology_for_row(
    row: dict[str, str],
    terminology_entries: Sequence[tuple[str, str]] | None,
    terminology_top_k: int,
) -> list[tuple[str, str]] | None:
    if not terminology_entries or terminology_top_k <= 0:
        return None
    selected = select_terminology(row["source"], terminology_entries, terminology_top_k)
    return selected or None


def hymt2_user_prompt(
    source: str,
    target_language_name: str,
    terminology: Sequence[tuple[str, str]] | None = None,
) -> str:
    prompt = (
        f"Translate the following text into {target_language_name}. "
        "Note that you should only output the translated result without any additional explanation:\n\n"
        f"{source}"
    )
    if terminology:
        prompt += "\n" + terminology_block(terminology)
    return prompt


def messages_for_example(
    args: argparse.Namespace,
    row: dict[str, str],
    terminology: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    if args.prompt_style == "hymt2":
        return [
            {
                "role": "user",
                "content": hymt2_user_prompt(
                    row["source"],
                    target_language_name_for(args.stage, args.target_language_name),
                    terminology,
                ),
            },
            {"role": "assistant", "content": row["target"]},
        ]
    return [
        {"role": "system", "content": "Eres un traductor profesional español-quechua."},
        {
            "role": "user",
            "content": f"{instruction_for(args.stage)}{terminology_block(terminology)}\n\nEspañol: {row['source']}",
        },
        {"role": "assistant", "content": row["target"]},
    ]


def apply_chat_template_no_thinking(tokenizer, messages: list[dict[str, str]], **kwargs):
    """Disable reasoning traces for chat templates that expose enable_thinking."""
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def format_example(
    tokenizer,
    args: argparse.Namespace,
    row: dict[str, str],
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> dict[str, str]:
    terminology = terminology_for_row(row, terminology_entries, terminology_top_k)
    return {
        "text": apply_chat_template_no_thinking(
            tokenizer,
            messages_for_example(args, row, terminology),
            tokenize=False,
            add_generation_prompt=False,
        ),
        "source_name": row["source_name"],
        "variant": row["variant"],
    }


def response_marker_parts(tokenizer) -> tuple[str, str]:
    probe = apply_chat_template_no_thinking(
        tokenizer,
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "assistant"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if "<｜hy_User｜>" in probe and "<｜hy_Assistant｜>" in probe:
        return "<｜hy_User｜>", "<｜hy_Assistant｜>"
    if "<|extra_4|>" in probe and "<|extra_0|>" in probe:
        return "<|extra_4|>", "<|extra_0|>"
    if "<|turn>user\n" in probe and "<|turn>model\n" in probe:
        return "<|turn>user\n", "<|turn>model\n"
    if "<|im_start|>user\n" in probe and "<|im_start|>assistant\n" in probe:
        return "<|im_start|>user\n", "<|im_start|>assistant\n"
    raise ValueError("Unsupported chat template for response-only SFT masking")


def build_dataset(
    tokenizer,
    args: argparse.Namespace,
    rows: Iterable[dict[str, str]],
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> Dataset:
    return Dataset.from_list(
        [format_example(tokenizer, args, row, terminology_entries, terminology_top_k) for row in rows]
    )


def main() -> None:
    args = parse_args()
    validate_training_mode_args(args)
    stage_defaults(args)

    from unsloth import FastLanguageModel
    import torch
    from trl import SFTConfig, SFTTrainer

    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project

    rows = load_rows(args)
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
        sum(1 for row in train_rows if terminology_for_row(row, terminology_entries, args.terminology_top_k))
        if terminology_entries
        else 0
    )
    eval_term_rows = (
        sum(1 for row in eval_rows if terminology_for_row(row, terminology_entries, args.terminology_top_k))
        if terminology_entries
        else 0
    )

    run_dir = args.output_dir / args.stage
    run_dir.mkdir(parents=True, exist_ok=True)

    model_name_or_adapter = str(args.adapter_path) if args.adapter_path else args.model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=args.training_mode == "lora" and not args.load_in_4bit,
        full_finetuning=args.training_mode == "full",
    )
    if args.training_mode == "lora" and args.adapter_path is None:
        peft_kwargs = {
            "r": args.lora_r,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": args.seed,
            "max_seq_length": args.max_seq_length,
            **adapter_flags(args.adapter_method),
        }
        model = FastLanguageModel.get_peft_model(
            model,
            **peft_kwargs,
        )

    train_dataset = build_dataset(tokenizer, args, train_rows, terminology_entries, args.terminology_top_k)
    eval_dataset = build_dataset(tokenizer, args, eval_rows, terminology_entries, args.terminology_top_k)
    configure_step_schedule(args, len(train_dataset))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            max_length=args.max_seq_length,
            dataset_text_field="text",
            packing=args.packing,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit or None,
            save_only_model=args.save_only_model,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir=str(run_dir),
            optim=args.optim,
            seed=args.seed,
            dataset_num_proc=args.dataset_num_proc,
            report_to="wandb" if args.wandb_project else "none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )

    if args.train_on_responses_only:
        from unsloth.chat_templates import train_on_responses_only

        instruction_part, response_part = response_marker_parts(tokenizer)
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

    print(f"Loaded rows: {len(rows):,}")
    print(f"Train rows: {len(train_dataset):,}")
    print(f"Validation rows: {len(eval_dataset):,}")
    print(f"Stage: {args.stage}")
    print(f"Prompt style: {args.prompt_style}")
    if args.terminology_file:
        print(f"Terminology file: {args.terminology_file}")
        print(f"Terminology entries: {len(terminology_entries):,}")
        print(f"Terminology-matched train rows: {train_term_rows:,}")
        print(f"Terminology-matched validation rows: {eval_term_rows:,}")
    print(f"Training mode: {args.training_mode}")
    print(f"Load in 4-bit: {args.load_in_4bit}")
    if args.training_mode == "lora":
        print(f"Adapter method: {args.adapter_method}")
    print(f"Model or adapter: {model_name_or_adapter}")
    if args.training_mode == "lora":
        print(f"LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    print(f"Validation: every {args.eval_steps} steps, best checkpoint by eval_loss")
    print(f"Saving: every {args.save_steps} steps, keeping {args.save_total_limit} checkpoints")
    print(f"Save only model: {args.save_only_model}")
    print(f"Optimizer: {args.optim}")

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    final_dir = final_artifact_dir(args, run_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
