"""SFT train RosettIA translation adapters with Unsloth.

This script intentionally uses 16-bit LoRA, not QLoRA. Unsloth's Qwen3.5 guide
does not recommend QLoRA for Qwen3.5 because quantization differences are larger
than usual. Every run creates a validation split and monitors eval loss.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import polars as pl
from datasets import Dataset
from huggingface_hub import hf_hub_download


DATASET_REPO = "Thermostatic/rosettia-chanka-data"
DEFAULT_MODEL_ID = "unsloth/Qwen3.5-4B"

BROAD_FILES = [
    "broad_quechua/somosnlp_spanish_to_quechua_high_quality_sft.parquet",
    "americasnlp/americasnlp_quechua_spanish_high_quality_real_sft.parquet",
]
CHANKA_FILE = "clean_chanka/manual_quechua_chanka_parallel_training_ready_augmented.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["broad", "chanka"], required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional local/HF LoRA adapter to continue training from, for example the broad SFT final_lora.",
    )
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
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
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo id for the final LoRA adapter.")
    return parser.parse_args()


def stage_defaults(args: argparse.Namespace) -> None:
    if args.validation_fraction is None:
        args.validation_fraction = 0.02 if args.stage == "broad" else 0.15
    if args.num_train_epochs is None:
        args.num_train_epochs = 1.0 if args.stage == "broad" else 8.0
    if args.learning_rate is None:
        args.learning_rate = 1.0e-4 if args.stage == "broad" else 2.0e-5
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 16 if args.stage == "broad" else 8
    if args.lora_r is None:
        args.lora_r = 64 if args.stage == "broad" else 128
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2


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


def instruction_for(stage: str) -> str:
    if stage == "chanka":
        return (
            "Traduce del español al quechua chanka. Usa una traducción directa, "
            "fiel y apropiada para contexto judicial."
        )
    return "Traduce del español al quechua. Conserva el significado y no copies el español."


def format_example(tokenizer, stage: str, row: dict[str, str]) -> dict[str, str]:
    messages = [
        {"role": "system", "content": "Eres un traductor profesional español-quechua."},
        {"role": "user", "content": f"{instruction_for(stage)}\n\nEspañol: {row['source']}"},
        {"role": "assistant", "content": row["target"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ),
        "source_name": row["source_name"],
        "variant": row["variant"],
    }


def build_dataset(tokenizer, stage: str, rows: Iterable[dict[str, str]]) -> Dataset:
    return Dataset.from_list([format_example(tokenizer, stage, row) for row in rows])


def main() -> None:
    args = parse_args()
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

    run_dir = args.output_dir / args.stage
    run_dir.mkdir(parents=True, exist_ok=True)

    model_name_or_adapter = str(args.adapter_path) if args.adapter_path else args.model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if args.adapter_path is None:
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

    train_dataset = build_dataset(tokenizer, args.stage, train_rows)
    eval_dataset = build_dataset(tokenizer, args.stage, eval_rows)

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
            output_dir=str(run_dir),
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

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    print(f"Loaded rows: {len(rows):,}")
    print(f"Train rows: {len(train_dataset):,}")
    print(f"Validation rows: {len(eval_dataset):,}")
    print(f"Stage: {args.stage}")
    print(f"Model or adapter: {model_name_or_adapter}")
    print(f"LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    print(f"Validation: every {args.eval_steps} steps, best checkpoint by eval_loss")

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    final_dir = run_dir / "final_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
