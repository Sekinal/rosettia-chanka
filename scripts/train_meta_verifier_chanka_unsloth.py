"""Train a DeepSeekMath-style meta-verifier for Chanka translation analyses.

The translation verifier scores candidate translations. The meta-verifier
scores the verifier/self-verifier analysis itself: did it identify real issues,
avoid hallucinated issues, and justify the score? This is the missing
DeepSeekMath-V2 component between a scalar verifier and a self-verifying
generator.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_self_verifiable_translation_data as self_data
from scripts import train_gspo_chanka_unsloth as gspo


DEFAULT_MODEL_ID = "unsloth/Qwen3.5-2B"
DEFAULT_OUTPUT_DIR = Path("outputs/chanka_translation_meta_verifier")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--dataset-file", default=gspo.CHANKA_FILE)
    parser.add_argument(
        "--meta-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Optional prebuilt translation_meta_verifier_cold_start.jsonl. If omitted, rows are generated from the clean corpus.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--evals-per-epoch", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--train-on-responses-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo id for the final meta-verifier LoRA.")
    return parser.parse_args(argv)


def load_meta_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.meta_jsonl:
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str, str, str, str]] = set()
        for path in args.meta_jsonl:
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    row = {
                        "source": str(payload["source"]),
                        "reference": str(payload["reference"]),
                        "candidate": str(payload["candidate"]),
                        "analysis": str(payload["analysis"]),
                        "label": str(payload["label"]),
                    }
                    key = (
                        gspo.normalize_text(row["source"]),
                        gspo.normalize_text(row["reference"]),
                        gspo.normalize_text(row["candidate"]),
                        gspo.normalize_text(row["analysis"]),
                        row["label"],
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        return rows

    clean_rows = gspo.load_chanka_rows(args.dataset_repo, args.dataset_file)
    if args.max_rows is not None:
        clean_rows = clean_rows[: args.max_rows]
    _, meta_records, _ = self_data.build_records(clean_rows, args.seed)
    return [
        {
            "source": str(record["source"]),
            "reference": str(record["reference"]),
            "candidate": str(record["candidate"]),
            "analysis": str(record["analysis"]),
            "label": str(record["label"]),
        }
        for record in meta_records
    ]


def optimizer_steps_per_epoch(row_count: int, batch_size: int, gradient_accumulation_steps: int) -> int:
    effective_batch_size = max(1, batch_size) * max(1, gradient_accumulation_steps)
    return max(1, math.ceil(row_count / effective_batch_size))


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


def meta_verifier_prompt_messages(
    source: str,
    reference: str,
    candidate: str,
    analysis: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Eres un meta-verificador de analisis de traducciones español a quechua chanka.",
        },
        {
            "role": "user",
            "content": (
                "Evalua si el analisis identifica problemas reales de la traduccion candidata "
                "y si el puntaje se justifica. Penaliza analisis que inventan errores, ocultan "
                "errores, o dan confianza falsa. Devuelve solo JSON compacto con score entre "
                "0 y 1, severity y rationale.\n\n"
                f"Español: {source}\n"
                f"Referencia chanka: {reference}\n"
                f"Candidata: {candidate}\n"
                f"Analisis: {analysis}"
            ),
        },
    ]


def format_example(tokenizer, row: dict[str, str]) -> dict[str, str]:
    messages = [
        *meta_verifier_prompt_messages(
            row["source"],
            row["reference"],
            row["candidate"],
            row["analysis"],
        ),
        {"role": "assistant", "content": row["label"]},
    ]
    return {
        "text": gspo.apply_chat_template_no_thinking(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def build_dataset(tokenizer, rows: Iterable[dict[str, str]]) -> Dataset:
    return Dataset.from_list([format_example(tokenizer, row) for row in rows])


def main() -> None:
    args = parse_args()

    from unsloth import FastLanguageModel
    import torch
    from trl import SFTConfig, SFTTrainer

    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project

    meta_rows = load_meta_rows(args)
    rng = random.Random(args.seed)
    rng.shuffle(meta_rows)
    train_rows, eval_rows = gspo.split_rows(
        meta_rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    configure_step_schedule(args, len(train_rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    gspo.force_tokenizer_no_thinking_template(tokenizer)
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

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=build_dataset(tokenizer, train_rows),
        eval_dataset=build_dataset(tokenizer, eval_rows),
        args=SFTConfig(
            max_length=args.max_seq_length,
            dataset_text_field="text",
            packing=False,
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

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    print(f"Meta-verifier examples: {len(meta_rows):,}")
    print(f"Train examples: {len(train_rows):,}")
    print(f"Validation examples: {len(eval_rows):,}")
    print(f"Validation: every {args.eval_steps} steps, best checkpoint by eval_loss")

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    final_dir = args.output_dir / "final_meta_verifier_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
