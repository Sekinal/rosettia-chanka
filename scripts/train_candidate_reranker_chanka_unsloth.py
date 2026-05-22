"""Train a source-only Chanka candidate reranker with oracle-derived labels.

The reranker sees only the Spanish source and a candidate Chanka translation.
Training labels are computed from held-out references, but references are not
included in the prompt. This makes the scorer usable for inference-time
reranking of multiple sampled translations.
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
from scripts.train_verifier_chanka_unsloth import verifier_target


DEFAULT_MODEL_ID = "unsloth/Qwen3.5-2B"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--dataset-file", default=gspo.CHANKA_FILE)
    parser.add_argument("--candidate-jsonl", action="append", type=Path, required=True)
    parser.add_argument("--candidate-max-examples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--include-clean-anchors", action=argparse.BooleanOptionalAction, default=True)
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


def reranker_prompt_messages(source: str, candidate: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Eres un evaluador de traducciones español a quechua chanka.",
        },
        {
            "role": "user",
            "content": (
                "Evalua la traduccion candidata sin ver una referencia. Considera fidelidad al español, "
                "fluidez en quechua chanka, evitar copiar español, ausencia de artefactos, y longitud razonable. "
                "Devuelve solo JSON compacto con score entre 0 y 1, severity y rationale.\n\n"
                f"Español: {source}\n"
                f"Candidata chanka: {candidate}"
            ),
        },
    ]


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


def hidden_reference_score(source: str, reference: str, candidate: str) -> float:
    chrf = gspo.sentence_chrfpp(candidate, reference)
    bleu = gspo.sentence_bleu(candidate, reference)
    f1 = gspo.token_f1(candidate, reference)
    length = gspo.length_ratio_score(candidate, reference)
    return gspo.reference_rerank_metric_score(candidate, reference, source, chrf, bleu, f1, length)


def severity_for_score(score: float) -> str:
    if score >= 0.86:
        return "none"
    if score >= 0.68:
        return "minor"
    if score >= 0.40:
        return "major"
    return "critical"


def rationale_for_candidate(source: str, candidate: str, label_score: float) -> str:
    if gspo.exact_source_copy(candidate, source):
        return "copied_spanish_source"
    if gspo.chat_artifact_penalty(candidate) > 0.0:
        return "chat_template_artifact"
    if gspo.spanish_leakage_penalty(candidate) >= 0.08:
        return "spanish_leakage"
    if gspo.source_copy_ratio(candidate, source) >= 0.25:
        return "copies_source_terms"
    if gspo.repetition_penalty(candidate) > 0.0:
        return "repetition"
    if label_score < 0.40:
        return "semantic_or_fluency_failure"
    if label_score < 0.68:
        return "partial_translation"
    return "good_candidate"


def group_key(record: dict[str, str]) -> tuple[str, str, str | None, str | None]:
    return (
        gspo.normalize_text(record["source"]),
        gspo.normalize_text(record["reference"]),
        record.get("source_name"),
        record.get("variant"),
    )


def candidate_training_rows(paths: Sequence[Path], max_examples: int | None, seed: int) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str | None, str | None], list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for payload in iter_jsonl(path):
            source = str(payload.get("source") or "")
            reference = str(payload.get("reference") or payload.get("target") or "")
            candidate = str(payload.get("prediction") or payload.get("candidate") or payload.get("hypothesis") or "")
            if not source or not reference or not candidate:
                continue
            key = (gspo.normalize_text(source), gspo.normalize_text(reference), gspo.normalize_text(candidate))
            if key in seen:
                continue
            seen.add(key)
            raw = hidden_reference_score(source, reference, candidate)
            record: dict[str, Any] = {
                "source": source,
                "reference": reference,
                "candidate": candidate,
                "source_name": payload.get("source_name"),
                "variant": payload.get("variant"),
                "raw_score": raw,
            }
            grouped.setdefault(group_key(record), []).append(record)

    rows: list[dict[str, str]] = []
    for group in grouped.values():
        scores = [float(record["raw_score"]) for record in group]
        low = min(scores)
        high = max(scores)
        spread = max(1.0e-6, high - low)
        for record in group:
            raw_score = max(0.0, min(1.0, float(record["raw_score"])))
            relative = (float(record["raw_score"]) - low) / spread if len(group) > 1 else raw_score
            label_score = max(0.0, min(1.0, (0.65 * raw_score) + (0.35 * relative)))
            source = str(record["source"])
            candidate = str(record["candidate"])
            rows.append(
                {
                    "source": source,
                    "candidate": candidate,
                    "label": verifier_target(
                        label_score,
                        severity_for_score(label_score),
                        rationale_for_candidate(source, candidate, label_score),
                    ),
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_examples is not None:
        rows = rows[:max_examples]
    return rows


def clean_anchor_rows(dataset_rows: Sequence[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for row in dataset_rows:
        source = row["source"]
        target = row["target"]
        rows.append(
            {
                "source": source,
                "candidate": target,
                "label": verifier_target(0.98, "none", "clean_reference_anchor"),
            }
        )
        rows.append(
            {
                "source": source,
                "candidate": source,
                "label": verifier_target(0.02, "critical", "source_copy_anchor"),
            }
        )
    rng.shuffle(rows)
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
        *reranker_prompt_messages(row["source"], row["candidate"]),
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

    dataset_rows = gspo.load_chanka_rows(args.dataset_repo, args.dataset_file)
    reranker_rows = candidate_training_rows(args.candidate_jsonl, args.candidate_max_examples, args.seed)
    if args.include_clean_anchors:
        reranker_rows.extend(clean_anchor_rows(dataset_rows, args.seed + 1))
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

    print(f"Candidate reranker examples: {len(reranker_rows):,}")
    print(f"Train examples: {len(train_rows):,}")
    print(f"Validation examples: {len(eval_rows):,}")
    print(f"Validation: every {args.eval_steps} steps, best checkpoint by eval_loss")

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    final_dir = args.output_dir / "final_reranker_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
