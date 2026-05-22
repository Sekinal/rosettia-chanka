"""Train a pairwise Chanka candidate reranker with hidden oracle labels.

The model sees a Spanish source and two Chanka candidate translations, but not
the reference. References are used only to choose the training winner. This is a
pairwise/listwise follow-up to the scalar source-only reranker, which learned a
global plausibility score but failed to rank similar candidates within a group.
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

from scripts import rerank_candidate_predictions as candidates_io
from scripts import train_gspo_chanka_unsloth as gspo


DEFAULT_MODEL_ID = "unsloth/Qwen3.5-2B"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--candidate-jsonl", action="append", type=Path, required=True)
    parser.add_argument("--candidate-max-examples", type=int, default=None)
    parser.add_argument("--min-score-margin", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--evals-per-epoch", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--train-on-responses-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default=None)
    return parser.parse_args(argv)


def pairwise_prompt_messages(source: str, candidate_a: str, candidate_b: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Eres un evaluador de traducciones español a quechua chanka.",
        },
        {
            "role": "user",
            "content": (
                "Elige la mejor traduccion candidata. Considera fidelidad al español, "
                "fluidez en quechua chanka, evitar copiar español, ausencia de artefactos, "
                "y longitud razonable. No expliques. Devuelve solo JSON compacto "
                'con el formato {"winner":"A"} o {"winner":"B"}.\n\n'
                f"Español: {source}\n"
                f"Candidata A: {candidate_a}\n"
                f"Candidata B: {candidate_b}"
            ),
        },
    ]


def winner_target(winner: str) -> str:
    if winner not in {"A", "B"}:
        raise ValueError(f"winner must be A or B, got {winner!r}")
    return json.dumps({"winner": winner}, separators=(",", ":"))


def hidden_oracle_score(candidate: candidates_io.Candidate) -> float:
    return candidates_io.candidate_oracle_score(candidate)


def pairwise_training_rows(
    paths: Sequence[Path],
    max_examples: int | None,
    min_score_margin: float,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    groups = candidates_io.group_candidates(
        [candidate for path in paths for candidate in candidates_io.load_candidates(path)]
    )
    for group in groups:
        scored = [(candidate, hidden_oracle_score(candidate)) for candidate in group]
        for left_index in range(len(scored)):
            for right_index in range(left_index + 1, len(scored)):
                left, left_score = scored[left_index]
                right, right_score = scored[right_index]
                margin = abs(left_score - right_score)
                if margin < min_score_margin:
                    continue
                if gspo.normalize_text(left.prediction).lower() == gspo.normalize_text(right.prediction).lower():
                    continue
                pair_key = (
                    gspo.normalize_text(left.source).lower(),
                    gspo.normalize_text(left.prediction).lower(),
                    gspo.normalize_text(right.prediction).lower(),
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                winner_is_left = left_score > right_score
                if rng.random() < 0.5:
                    candidate_a, candidate_b = left, right
                    winner = "A" if winner_is_left else "B"
                else:
                    candidate_a, candidate_b = right, left
                    winner = "B" if winner_is_left else "A"
                rows.append(
                    {
                        "source": left.source,
                        "candidate_a": candidate_a.prediction,
                        "candidate_b": candidate_b.prediction,
                        "label": winner_target(winner),
                        "score_margin": f"{margin:.6f}",
                    }
                )

    rng.shuffle(rows)
    if max_examples is not None:
        rows = rows[:max_examples]
    return rows


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


def format_example(tokenizer, row: dict[str, str]) -> dict[str, str]:
    messages = [
        *pairwise_prompt_messages(row["source"], row["candidate_a"], row["candidate_b"]),
        {"role": "assistant", "content": row["label"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}


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

    reranker_rows = pairwise_training_rows(
        args.candidate_jsonl,
        max_examples=args.candidate_max_examples,
        min_score_margin=args.min_score_margin,
        seed=args.seed,
    )
    train_rows, eval_rows = gspo.split_rows(
        reranker_rows,
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

    print(f"Pairwise reranker examples: {len(reranker_rows):,}")
    print(f"Train examples: {len(train_rows):,}")
    print(f"Validation examples: {len(eval_rows):,}")
    print(f"Validation: every {args.eval_steps} steps, best checkpoint by eval_loss")

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    final_dir = args.output_dir / "final_pairwise_reranker_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
