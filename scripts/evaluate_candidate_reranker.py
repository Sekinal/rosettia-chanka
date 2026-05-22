"""Evaluate a source-only candidate reranker on multi-candidate predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_gspo_chanka_unsloth as gspo
from scripts.train_candidate_reranker_chanka_unsloth import reranker_prompt_messages
from scripts.summarize_gspo_canaries import selection_score


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranker-adapter-path", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(argv)


class CandidateRerankerScorer:
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
        self.tokenizer.padding_side = "left"
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        FastLanguageModel.for_inference(self.model)
        self.model.eval()

    def score_many(self, candidates: Sequence[oracle_rerank.Candidate]) -> list[float]:
        prompts = [
            self.tokenizer.apply_chat_template(
                reranker_prompt_messages(candidate.source, candidate.prediction),
                tokenize=False,
                add_generation_prompt=True,
            )
            for candidate in candidates
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
                scores.append(gspo.parse_verifier_score(decoded))
        return scores


def select_learned(
    groups: Sequence[Sequence[oracle_rerank.Candidate]],
    scorer: CandidateRerankerScorer,
) -> tuple[list[oracle_rerank.Candidate], list[float]]:
    flat = [candidate for group in groups for candidate in group]
    flat_scores = scorer.score_many(flat)
    selected: list[oracle_rerank.Candidate] = []
    selected_scores: list[float] = []
    offset = 0
    for group in groups:
        group_scores = flat_scores[offset : offset + len(group)]
        offset += len(group)
        best_index = max(
            range(len(group)),
            key=lambda index: (
                group_scores[index],
                -group[index].candidate_index,
            ),
        )
        selected.append(group[best_index])
        selected_scores.append(group_scores[best_index])
    return selected, selected_scores


def metrics_for_selection(
    selected: Sequence[oracle_rerank.Candidate],
    method: str,
    predictions_jsonl: Path,
    total_candidates: int,
    selected_scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    metrics = oracle_rerank.metrics_for_selection(selected, method, predictions_jsonl, total_candidates)
    metrics["selection_score"] = selection_score(metrics)
    if selected_scores is not None:
        metrics["mean_reranker_score"] = sum(selected_scores) / max(1, len(selected_scores))
        non_first = [candidate for candidate in selected if candidate.candidate_index != 0]
        metrics["learned_non_first_rate"] = 100.0 * len(non_first) / max(1, len(selected))
        metrics["learned_mean_selected_index"] = sum(candidate.candidate_index for candidate in selected) / max(1, len(selected))
    return metrics


def write_summary(path: Path, records: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Learned Candidate Reranker Eval",
        "",
        "| Method | Selection | chrF++ | BLEU | token F1 | source copy % | exact copy % | leakage % | artifact % | TER | non-first % | mean score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        non_first = record.get("oracle_non_first_rate", record.get("learned_non_first_rate", 0.0))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["method"]),
                    f"{float(record.get('selection_score', 0.0)):.4f}",
                    f"{float(record.get('chrf++', 0.0)):.4f}",
                    f"{float(record.get('bleu', 0.0)):.4f}",
                    f"{float(record.get('token_f1', 0.0)):.4f}",
                    f"{float(record.get('source_copy_ratio', 0.0)):.4f}",
                    f"{float(record.get('exact_source_copy_rate', 0.0)):.4f}",
                    f"{float(record.get('spanish_leakage_penalty', 0.0)):.4f}",
                    f"{float(record.get('chat_artifact_penalty', 0.0)):.4f}",
                    f"{float(record.get('ter', 0.0)):.4f}",
                    f"{float(non_first):.4f}",
                    f"{float(record.get('mean_reranker_score', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    candidates = oracle_rerank.load_candidates(args.predictions_jsonl)
    groups = oracle_rerank.group_candidates(candidates)
    if not groups:
        raise ValueError(f"No candidates found in {args.predictions_jsonl}")

    scorer = CandidateRerankerScorer(
        args.reranker_adapter_path,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    first = oracle_rerank.select_first(groups)
    learned, learned_scores = select_learned(groups, scorer)
    oracle = oracle_rerank.select_oracle(groups)
    records = [
        metrics_for_selection(first, "first", args.predictions_jsonl, len(candidates)),
        metrics_for_selection(learned, "learned", args.predictions_jsonl, len(candidates), learned_scores),
        metrics_for_selection(oracle, "oracle", args.predictions_jsonl, len(candidates)),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for selected, record in [(first, records[0]), (learned, records[1]), (oracle, records[2])]:
        oracle_rerank.write_predictions(args.output_dir / f"{record['method']}_predictions.jsonl", selected)
        oracle_rerank.write_metrics(args.output_dir / f"{record['method']}_metrics.json", record)
    oracle_rerank.write_metrics(args.output_dir / "summary.json", {"records": records})
    write_summary(args.output_dir / "summary.md", records)
    print(json.dumps({"records": records}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
