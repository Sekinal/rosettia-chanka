"""SFT train from local JSONL translation rows with Unsloth.

This is intended for pseudo-label/self-training experiments such as MBR-selected
candidate translations. It deliberately keeps validation enabled and never
downloads or writes training data outside the user-provided JSONL files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_gspo_chanka_unsloth as gspo
from scripts import train_sft_unsloth as train_sft


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, action="append", required=True, help="Input JSONL file. Repeatable.")
    parser.add_argument("--source-field", default="source")
    parser.add_argument(
        "--target-field",
        default="prediction",
        help="Field to use as the assistant target. MBR files use 'prediction'.",
    )
    parser.add_argument("--reference-field", default="reference", help="Optional reference metadata field.")
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument(
        "--terminology-file",
        default=None,
        help="Optional dataset-repo parquet glossary for terminology-conditioned SFT prompts.",
    )
    parser.add_argument("--terminology-top-k", type=int, default=6)
    parser.add_argument("--terminology-min-source-chars", type=int, default=3)
    parser.add_argument("--model-id", default=train_sft.DEFAULT_MODEL_ID)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--evals-per-epoch", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument(
        "--train-on-responses-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask user/prompt tokens. Disable if chat-template markers change and masking fails.",
    )
    parser.add_argument(
        "--reject-exact-source-copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop pseudo-targets that exactly copy the Spanish source.",
    )
    parser.add_argument(
        "--dedupe-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dedupe rows by normalized source and target. Disable for deliberate oversampling experiments.",
    )
    parser.add_argument(
        "--max-source-copy-ratio",
        type=float,
        default=0.80,
        help="Drop rows above this source-copy ratio. Set above 1 to disable.",
    )
    parser.add_argument(
        "--max-spanish-leakage-penalty",
        type=float,
        default=0.50,
        help="Drop rows above this Spanish leakage penalty. Set above 1 to disable.",
    )
    parser.add_argument(
        "--max-chat-artifact-penalty",
        type=float,
        default=0.0,
        help="Drop rows above this chat-artifact penalty. Set above 1 to disable.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo id for the final trained artifact.")
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


def pseudo_target_passes_quality_filters(source: str, target: str, args: argparse.Namespace) -> bool:
    if args.reject_exact_source_copy and gspo.exact_source_copy(target, source):
        return False
    if args.max_source_copy_ratio <= 1.0 and gspo.source_copy_ratio(target, source) > args.max_source_copy_ratio:
        return False
    if args.max_spanish_leakage_penalty <= 1.0 and gspo.spanish_leakage_penalty(target) > args.max_spanish_leakage_penalty:
        return False
    if args.max_chat_artifact_penalty <= 1.0 and gspo.chat_artifact_penalty(target) > args.max_chat_artifact_penalty:
        return False
    return True


def load_jsonl_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in args.jsonl:
        for record in iter_jsonl(path):
            source = gspo.normalize_text(str(record.get(args.source_field, "")))
            target = gspo.normalize_text(str(record.get(args.target_field, "")))
            if not source or not target:
                continue
            if not pseudo_target_passes_quality_filters(source, target, args):
                continue
            if args.dedupe_rows:
                key = (source.lower(), target.lower())
                if key in seen:
                    continue
                seen.add(key)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "reference": gspo.normalize_text(str(record.get(args.reference_field, "")))
                    if args.reference_field
                    else "",
                    "source_name": str(record.get("source_name") or path.name),
                    "variant": str(record.get("variant") or "quy/chanka_pseudo"),
                    "target_field": args.target_field,
                }
            )
    if not rows:
        raise RuntimeError("No JSONL rows survived loading and quality filtering.")
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


def configure_step_schedule(args: argparse.Namespace, train_row_count: int) -> None:
    if args.eval_steps is not None and args.save_steps is not None:
        return

    if args.max_steps and args.max_steps > 0:
        fallback_steps = max(1, args.max_steps // max(1, args.evals_per_epoch))
    else:
        effective_batch_size = max(1, args.per_device_train_batch_size) * max(1, args.gradient_accumulation_steps)
        steps_per_epoch = max(1, math.ceil(train_row_count / effective_batch_size))
        fallback_steps = max(1, steps_per_epoch // max(1, args.evals_per_epoch))

    if args.eval_steps is None:
        args.eval_steps = fallback_steps
    if args.save_steps is None:
        args.save_steps = args.eval_steps


def terminology_for_row(
    row: dict[str, str],
    terminology_entries: Sequence[tuple[str, str]] | None,
    terminology_top_k: int,
) -> list[tuple[str, str]] | None:
    if not terminology_entries or terminology_top_k <= 0:
        return None
    selected = gspo.select_terminology(row["source"], terminology_entries, terminology_top_k)
    return selected or None


def format_example(
    tokenizer,
    row: dict[str, str],
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> dict[str, str]:
    messages = [
        *gspo.prompt_messages(row["source"], terminology_for_row(row, terminology_entries, terminology_top_k)),
        {"role": "assistant", "content": row["target"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
        "source_name": row["source_name"],
        "variant": row["variant"],
        "target_field": row["target_field"],
    }


def build_dataset(
    tokenizer,
    rows: Iterable[dict[str, str]],
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> Dataset:
    return Dataset.from_list(
        [format_example(tokenizer, row, terminology_entries, terminology_top_k) for row in rows]
    )


def main() -> None:
    args = parse_args()
    train_sft.validate_training_mode_args(args)

    from unsloth import FastLanguageModel
    import torch
    from trl import SFTConfig, SFTTrainer

    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project

    rows = load_jsonl_rows(args)
    train_rows, eval_rows = split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    terminology_entries = (
        gspo.load_terminology_entries(args.dataset_repo, args.terminology_file, args.terminology_min_source_chars)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_name_or_adapter = str(args.adapter_path) if args.adapter_path else args.model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=args.training_mode == "lora",
        full_finetuning=args.training_mode == "full",
    )
    if args.training_mode == "lora" and args.adapter_path is None:
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
            **train_sft.adapter_flags(args.adapter_method),
        )

    train_dataset = build_dataset(tokenizer, train_rows, terminology_entries, args.terminology_top_k)
    eval_dataset = build_dataset(tokenizer, eval_rows, terminology_entries, args.terminology_top_k)
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
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir=str(args.output_dir),
            optim="adamw_8bit",
            seed=args.seed,
            dataset_num_proc=args.dataset_num_proc,
            report_to="wandb" if args.wandb_project else "none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )

    if args.train_on_responses_only:
        from unsloth.chat_templates import train_on_responses_only

        instruction_part, response_part = train_sft.response_marker_parts(tokenizer)
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

    print(f"Loaded JSONL rows after filtering: {len(rows):,}")
    print(f"Train rows: {len(train_dataset):,}")
    print(f"Validation rows: {len(eval_dataset):,}")
    print(f"Target field: {args.target_field}")
    if args.terminology_file:
        print(f"Terminology file: {args.terminology_file}")
        print(f"Terminology entries: {len(terminology_entries):,}")
        print(f"Terminology-matched train rows: {train_term_rows:,}")
        print(f"Terminology-matched validation rows: {eval_term_rows:,}")
    print(f"Training mode: {args.training_mode}")
    print(f"Model or adapter: {model_name_or_adapter}")
    if args.training_mode == "lora":
        print(f"LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    print(f"Validation: every {args.eval_steps} steps, best checkpoint by eval_loss")
    print(f"Saving: every {args.save_steps} steps, keeping the 3 most recent checkpoints")

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    final_dir = train_sft.final_artifact_dir(args, args.output_dir)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
