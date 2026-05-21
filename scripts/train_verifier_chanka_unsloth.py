"""Train a Chanka translation verifier with Unsloth SFT.

This is the serious DeepSeek-style branch: train a separate model to score
candidate Spanish -> Chanka translations before using it to create preferences
or reward GSPO samples. The labels are bootstrapped from the clean Chanka
parallel corpus with synthetic corruptions, so human review is still the next
upgrade path.
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

from scripts import train_gspo_chanka_unsloth as gspo


DEFAULT_MODEL_ID = "unsloth/Qwen3.5-2B"
DEFAULT_OUTPUT_DIR = Path("outputs/chanka_translation_verifier")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--dataset-file", default=gspo.CHANKA_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--candidate-jsonl",
        action="append",
        type=Path,
        default=[],
        help="Optional prediction JSONL files with source/reference/prediction fields for real hard-negative verifier examples.",
    )
    parser.add_argument("--candidate-max-examples", type=int, default=None)
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
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo id for the final verifier LoRA.")
    return parser.parse_args(argv)


def verifier_prompt_messages(source: str, reference: str, candidate: str) -> list[dict[str, str]]:
    return [
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


def verifier_target(score: float, severity: str, rationale: str) -> str:
    return json.dumps(
        {
            "score": round(max(0.0, min(1.0, score)), 3),
            "severity": severity,
            "rationale": rationale,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def shuffled_words(text: str, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    shuffled = words[:]
    for _ in range(4):
        rng.shuffle(shuffled)
        if shuffled != words:
            break
    return " ".join(shuffled)


def truncate_translation(text: str) -> str:
    words = text.split()
    if len(words) <= 2:
        return words[0] if words else ""
    return " ".join(words[: max(1, len(words) // 2)])


def other_target(reference: str, distractors: Sequence[str], rng: random.Random) -> str | None:
    candidates = [target for target in distractors if target and target != reference]
    if not candidates:
        return None
    return rng.choice(candidates)


def mix_translations(reference: str, distractor: str) -> str:
    ref_words = reference.split()
    distractor_words = distractor.split()
    if len(ref_words) < 2 or len(distractor_words) < 2:
        return f"{reference} {distractor}".strip()
    ref_cut = max(1, len(ref_words) // 2)
    distractor_cut = max(1, len(distractor_words) // 2)
    return " ".join([*ref_words[:ref_cut], *distractor_words[distractor_cut:]])


def unsupported_chanka_addition(reference: str, distractor: str) -> str:
    addition = " ".join(distractor.split()[:4])
    if not addition:
        return reference
    return f"{reference} {addition}"


def repeat_translation(text: str) -> str:
    words = text.split()
    if len(words) <= 2:
        return f"{text} {text}".strip()
    repeated_tail = " ".join(words[-min(4, len(words)) :])
    return f"{text} {repeated_tail}"


def verifier_examples_for_row(
    row: dict[str, str],
    rng: random.Random,
    distractors: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    source = row["source"]
    reference = row["target"]
    examples = [
        {
            "source": source,
            "reference": reference,
            "candidate": reference,
            "label": verifier_target(0.98, "none", "faithful_reference_translation"),
        },
        {
            "source": source,
            "reference": reference,
            "candidate": source,
            "label": verifier_target(0.02, "critical", "copied_spanish_source"),
        },
        {
            "source": source,
            "reference": reference,
            "candidate": truncate_translation(reference),
            "label": verifier_target(0.38, "major", "incomplete_translation"),
        },
        {
            "source": source,
            "reference": reference,
            "candidate": f"{reference} de la autoridad",
            "label": verifier_target(0.58, "major", "spanish_leakage"),
        },
    ]
    shuffled = shuffled_words(reference, rng)
    if shuffled != reference:
        examples.append(
            {
                "source": source,
                "reference": reference,
                "candidate": shuffled,
                "label": verifier_target(0.48, "major", "word_order_or_fluency_damage"),
            }
        )
    examples.append(
        {
            "source": source,
            "reference": reference,
            "candidate": repeat_translation(reference),
            "label": verifier_target(0.72, "minor", "repetition_or_fluency_damage"),
        }
    )
    distractor = other_target(reference, distractors or (), rng)
    if distractor:
        examples.extend(
            [
                {
                    "source": source,
                    "reference": reference,
                    "candidate": distractor,
                    "label": verifier_target(0.16, "critical", "fluent_but_semantically_unrelated_chanka"),
                },
                {
                    "source": source,
                    "reference": reference,
                    "candidate": mix_translations(reference, distractor),
                    "label": verifier_target(0.42, "major", "mixed_translation_from_another_example"),
                },
                {
                    "source": source,
                    "reference": reference,
                    "candidate": unsupported_chanka_addition(reference, distractor),
                    "label": verifier_target(0.64, "minor", "unsupported_extra_chanka_content"),
                },
            ]
        )
    return examples


def build_verifier_rows(rows: Iterable[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    row_list = list(rows)
    distractors = [row["target"] for row in row_list]
    examples: list[dict[str, str]] = []
    for row in row_list:
        examples.extend(verifier_examples_for_row(row, rng, distractors=distractors))
    rng.shuffle(examples)
    return examples


def candidate_score(source: str, reference: str, candidate: str) -> float:
    chrf = gspo.sentence_chrfpp(candidate, reference)
    bleu = gspo.sentence_bleu(candidate, reference)
    f1 = gspo.token_f1(candidate, reference)
    length = gspo.length_ratio_score(candidate, reference)
    score = (0.42 * chrf) + (0.28 * f1) + (0.10 * bleu) + (0.20 * length)
    score -= 0.35 * gspo.source_copy_ratio(candidate, source)
    score -= 0.45 * gspo.spanish_leakage_penalty(candidate)
    score -= 0.20 * gspo.repetition_penalty(candidate)
    if gspo.exact_source_copy(candidate, source):
        score -= 0.50
    if gspo.normalize_text(candidate).lower() == gspo.normalize_text(reference).lower():
        score = max(score, 0.98)
    return max(0.0, min(0.98, score))


def severity_for_score(score: float) -> str:
    if score >= 0.90:
        return "none"
    if score >= 0.70:
        return "minor"
    if score >= 0.40:
        return "major"
    return "critical"


def rationale_for_candidate(source: str, reference: str, candidate: str, score: float) -> str:
    if gspo.exact_source_copy(candidate, source):
        return "model_output_copied_spanish_source"
    if gspo.spanish_leakage_penalty(candidate) >= 0.08:
        return "model_output_has_spanish_leakage"
    if gspo.source_copy_ratio(candidate, source) >= 0.25:
        return "model_output_copies_source_terms"
    if gspo.repetition_penalty(candidate) > 0.0:
        return "model_output_repetition"
    if score < 0.40:
        return "model_output_low_reference_agreement"
    if score < 0.70:
        return "model_output_partial_or_noisy_translation"
    return "model_output_near_reference_translation"


def load_candidate_verifier_rows(paths: Sequence[Path], max_examples: int | None, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                source = str(payload.get("source") or "")
                reference = str(payload.get("reference") or payload.get("target") or "")
                candidate = str(payload.get("prediction") or payload.get("candidate") or payload.get("hypothesis") or "")
                if not source or not reference or not candidate:
                    continue
                key = (gspo.normalize_text(source), gspo.normalize_text(reference), gspo.normalize_text(candidate))
                if key in seen:
                    continue
                seen.add(key)
                score = candidate_score(source, reference, candidate)
                rows.append(
                    {
                        "source": source,
                        "reference": reference,
                        "candidate": candidate,
                        "label": verifier_target(
                            score,
                            severity_for_score(score),
                            rationale_for_candidate(source, reference, candidate, score),
                        ),
                    }
                )
    rng = random.Random(seed)
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
        *verifier_prompt_messages(row["source"], row["reference"], row["candidate"]),
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

    rows = gspo.load_chanka_rows(args.dataset_repo, args.dataset_file)
    verifier_rows = build_verifier_rows(rows, args.seed)
    candidate_rows = load_candidate_verifier_rows(args.candidate_jsonl, args.candidate_max_examples, args.seed)
    verifier_rows.extend(candidate_rows)
    train_rows, eval_rows = gspo.split_rows(
        verifier_rows,
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

    print(f"Base rows: {len(rows):,}")
    print(f"Real candidate verifier examples: {len(candidate_rows):,}")
    print(f"Verifier examples: {len(verifier_rows):,}")
    print(f"Train examples: {len(train_rows):,}")
    print(f"Validation examples: {len(eval_rows):,}")
    print(f"Validation: every {args.eval_steps} steps, best checkpoint by eval_loss")

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    final_dir = args.output_dir / "final_verifier_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
