"""Evaluate a pairwise Chanka candidate reranker on multi-candidate JSONL."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import mbr_candidate_predictions as mbr_rerank
from scripts import rerank_candidate_predictions as oracle_rerank
from scripts.summarize_gspo_canaries import selection_score
from scripts.train_pairwise_candidate_reranker_chanka_unsloth import pairwise_prompt_messages


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranker-adapter-path", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def parse_winner(text: str) -> str | None:
    match = re.search(r'"winner"\s*:\s*"([AB])"', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    loose = re.search(r"\b([AB])\b", text.strip(), flags=re.IGNORECASE)
    if loose:
        return loose.group(1).upper()
    return None


class PairwiseRerankerScorer:
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

    def predict_many(self, pairs: Sequence[tuple[oracle_rerank.Candidate, oracle_rerank.Candidate]]) -> list[str | None]:
        prompts = [
            self.tokenizer.apply_chat_template(
                pairwise_prompt_messages(left.source, left.prediction, right.prediction),
                tokenize=False,
                add_generation_prompt=True,
            )
            for left, right in pairs
        ]
        winners: list[str | None] = []
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
                winners.append(parse_winner(decoded))
        return winners


def select_pairwise(
    groups: Sequence[Sequence[oracle_rerank.Candidate]],
    scorer: PairwiseRerankerScorer,
) -> tuple[list[oracle_rerank.Candidate], dict[str, float]]:
    selected: list[oracle_rerank.Candidate] = []
    total_pairs = 0
    parsed_pairs = 0
    non_first = 0
    for group in groups:
        if not group:
            continue
        if len(group) == 1:
            selected.append(group[0])
            continue
        pairs: list[tuple[oracle_rerank.Candidate, oracle_rerank.Candidate]] = []
        pair_indices: list[tuple[int, int]] = []
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                pairs.append((group[left_index], group[right_index]))
                pair_indices.append((left_index, right_index))
        winners = scorer.predict_many(pairs)
        votes = [0.0] * len(group)
        for (left_index, right_index), winner in zip(pair_indices, winners, strict=True):
            total_pairs += 1
            if winner == "A":
                votes[left_index] += 1.0
                parsed_pairs += 1
            elif winner == "B":
                votes[right_index] += 1.0
                parsed_pairs += 1
        best_index = max(
            range(len(group)),
            key=lambda index: (
                votes[index],
                -group[index].candidate_index,
            ),
        )
        chosen = group[best_index]
        selected.append(chosen)
        non_first += int(chosen.candidate_index != 0)
    stats = {
        "pairwise_total_pairs": float(total_pairs),
        "pairwise_parsed_pairs": float(parsed_pairs),
        "pairwise_parse_rate": 100.0 * parsed_pairs / max(1, total_pairs),
        "pairwise_non_first_rate": 100.0 * non_first / max(1, len(selected)),
        "pairwise_mean_selected_index": sum(candidate.candidate_index for candidate in selected) / max(1, len(selected)),
    }
    return selected, stats


def metrics_for_selection(
    selected: Sequence[oracle_rerank.Candidate],
    method: str,
    predictions_jsonl: Path,
    total_candidates: int,
    extra: dict[str, float] | None = None,
) -> dict[str, Any]:
    metrics = oracle_rerank.metrics_for_selection(selected, method, predictions_jsonl, total_candidates)
    metrics["selection_score"] = selection_score(metrics)
    if extra:
        metrics.update(extra)
    return metrics


def write_summary(path: Path, records: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Pairwise Candidate Reranker Eval",
        "",
        "| Method | Selection | chrF++ | BLEU | token F1 | source copy % | exact copy % | leakage % | artifact % | TER | non-first % | parse % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        non_first = record.get(
            "pairwise_non_first_rate",
            record.get("oracle_non_first_rate", record.get("mbr_non_first_rate", 0.0)),
        )
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
                    f"{float(record.get('pairwise_parse_rate', 0.0)):.4f}",
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

    scorer = PairwiseRerankerScorer(
        args.reranker_adapter_path,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    first = oracle_rerank.select_first(groups)
    pairwise, pairwise_stats = select_pairwise(groups, scorer)
    mbr = mbr_rerank.select_mbr(groups)
    oracle = oracle_rerank.select_oracle(groups)
    mbr_metrics = mbr_rerank.metrics_for_selection(mbr, "mbr", args.predictions_jsonl, len(candidates))
    records = [
        metrics_for_selection(first, "first", args.predictions_jsonl, len(candidates)),
        metrics_for_selection(pairwise, "pairwise", args.predictions_jsonl, len(candidates), pairwise_stats),
        mbr_metrics,
        metrics_for_selection(oracle, "oracle", args.predictions_jsonl, len(candidates)),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for selected, record in [(first, records[0]), (pairwise, records[1]), (mbr, records[2]), (oracle, records[3])]:
        oracle_rerank.write_predictions(args.output_dir / f"{record['method']}_predictions.jsonl", selected)
        oracle_rerank.write_metrics(args.output_dir / f"{record['method']}_metrics.json", record)
    oracle_rerank.write_metrics(args.output_dir / "summary.json", {"records": records})
    write_summary(args.output_dir / "summary.md", records)
    print(json.dumps({"records": records}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
