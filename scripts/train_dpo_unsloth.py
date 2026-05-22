"""DPO train a Chanka translator from local preference-pair JSONL."""

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
from scripts import train_jsonl_sft_unsloth as train_jsonl_sft


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, action="append", required=True, help="Preference-pair JSONL. Repeatable.")
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--chosen-field", default="chosen")
    parser.add_argument("--rejected-field", default="rejected")
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--terminology-file", default=None)
    parser.add_argument("--terminology-top-k", type=int, default=1)
    parser.add_argument("--terminology-min-source-chars", type=int, default=3)
    parser.add_argument("--model-id", default=train_sft.DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=None, help="Policy adapter to continue training.")
    parser.add_argument(
        "--reference-adapter-path",
        type=Path,
        default=None,
        help="Frozen reference adapter. Defaults to --adapter-path when provided.",
    )
    parser.add_argument(
        "--no-reference-model",
        action="store_true",
        help="Use TRL's reference-free/implicit reference behavior instead of loading a frozen reference model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=192)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-7)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--loss-type", action="append", default=None, help="TRL DPO loss type. Repeatable.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--evals-per-epoch", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument(
        "--reject-exact-source-copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop chosen outputs that exactly copy the Spanish source.",
    )
    parser.add_argument("--max-source-copy-ratio", type=float, default=0.80)
    parser.add_argument("--max-spanish-leakage-penalty", type=float, default=0.50)
    parser.add_argument("--max-chat-artifact-penalty", type=float, default=0.0)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--push-to-hub", default=None)
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


def load_preference_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in args.jsonl:
        for record in iter_jsonl(path):
            source = gspo.normalize_text(str(record.get(args.source_field, "")))
            chosen = gspo.normalize_text(str(record.get(args.chosen_field, "")))
            rejected = gspo.normalize_text(str(record.get(args.rejected_field, "")))
            if not source or not chosen or not rejected or chosen == rejected:
                continue
            if not train_jsonl_sft.pseudo_target_passes_quality_filters(source, chosen, args):
                continue
            key = (source.lower(), chosen.lower(), rejected.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source": source,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source_name": str(record.get("source_name") or path.name),
                    "variant": str(record.get("variant") or "quy/chanka_preference"),
                }
            )
    if not rows:
        raise RuntimeError("No preference rows survived loading.")
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


def format_preference_row(
    row: dict[str, str],
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> dict[str, Any]:
    terminology = train_jsonl_sft.terminology_for_row(row, terminology_entries, terminology_top_k)
    return {
        "prompt": gspo.prompt_messages(row["source"], terminology),
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
        "source_name": row["source_name"],
        "variant": row["variant"],
    }


def build_dataset(
    rows: Iterable[dict[str, str]],
    terminology_entries: Sequence[tuple[str, str]] | None = None,
    terminology_top_k: int = 0,
) -> Dataset:
    return Dataset.from_list([format_preference_row(row, terminology_entries, terminology_top_k) for row in rows])


def disable_trl_optional_imports() -> None:
    import trl.import_utils as trl_import_utils

    trl_import_utils._mergekit_available = False
    trl_import_utils._llm_blender_available = False
    trl_import_utils._weave_available = False


def main() -> None:
    args = parse_args()
    if args.adapter_path is None:
        raise ValueError("DPO training currently requires --adapter-path to avoid accidental full-model training.")

    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project

    from unsloth import FastLanguageModel
    import torch

    disable_trl_optional_imports()
    from trl import DPOConfig, DPOTrainer

    rows = load_preference_rows(args)
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
        sum(1 for row in train_rows if train_jsonl_sft.terminology_for_row(row, terminology_entries, args.terminology_top_k))
        if terminology_entries
        else 0
    )
    eval_term_rows = (
        sum(1 for row in eval_rows if train_jsonl_sft.terminology_for_row(row, terminology_entries, args.terminology_top_k))
        if terminology_entries
        else 0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy_path = str(args.adapter_path) if args.adapter_path else args.model_id
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=policy_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    ref_model = None
    if not args.no_reference_model:
        reference_path = str(args.reference_adapter_path or args.adapter_path or args.model_id)
        ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=reference_path,
            max_seq_length=args.max_seq_length,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        ref_model.eval()
        for parameter in ref_model.parameters():
            parameter.requires_grad_(False)

    train_dataset = build_dataset(train_rows, terminology_entries, args.terminology_top_k)
    eval_dataset = build_dataset(eval_rows, terminology_entries, args.terminology_top_k)
    configure_step_schedule(args, len(train_dataset))

    print(f"Loaded preference rows after filtering: {len(rows):,}")
    print(f"Train rows: {len(train_dataset):,}")
    print(f"Validation rows: {len(eval_dataset):,}")
    print(f"Policy model or adapter: {policy_path}")
    print(f"Reference model: {'none' if ref_model is None else str(args.reference_adapter_path or args.adapter_path or args.model_id)}")
    if args.terminology_file:
        print(f"Terminology file: {args.terminology_file}")
        print(f"Terminology entries: {len(terminology_entries):,}")
        print(f"Terminology-matched train rows: {train_term_rows:,}")
        print(f"Terminology-matched validation rows: {eval_term_rows:,}")
    print(f"DPO beta/loss: {args.beta}/{args.loss_type or ['sigmoid']}")
    print(f"Validation: every {args.eval_steps} steps, best checkpoint by eval_loss")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=DPOConfig(
            max_length=args.max_seq_length,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            beta=args.beta,
            loss_type=args.loss_type or ["sigmoid"],
            label_smoothing=args.label_smoothing,
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
            remove_unused_columns=False,
        ),
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    final_dir = args.output_dir / "final_dpo_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
