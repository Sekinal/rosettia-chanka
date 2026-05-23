#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-4b-full-sft-candidate-rerank)}"

FULL_SFT_SUMMARY="${FULL_SFT_SUMMARY:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/summary.json}"
FULL_SFT_EVAL_ROOT="${FULL_SFT_EVAL_ROOT:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/checkpoint_eval}"
FULL_SFT_RUN_ROOT="${FULL_SFT_RUN_ROOT:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups}"
FULL_SFT_CHECKPOINT="${FULL_SFT_CHECKPOINT:-}"

BASE_EVAL_POOL="${BASE_EVAL_POOL:-outputs/rerank_candidate_evals/20260522-current-k32-plus-qwen35-4b-full-k16-term/candidates_predictions.jsonl}"
BASE_TRAIN_POOL="${BASE_TRAIN_POOL:-outputs/verifier_candidate_mining/20260522-train-current-k32-plus-qwen35-4b-full-k16-term/train_current_k32_plus_4bfull_k16_predictions.jsonl}"

WORK_DIR="${WORK_DIR:-outputs/qwen35_4b_full_sft_candidate_rerank/${STAMP}}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"

NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-16}"
TEMPERATURE="${TEMPERATURE:-0.65}"
TOP_P="${TOP_P:-0.90}"
TOP_K="${TOP_K:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-80}"
PROGRESS_EVERY="${PROGRESS_EVERY:-64}"

TEXT_EPOCHS="${TEXT_EPOCHS:-8}"
TEXT_LEARNING_RATE="${TEXT_LEARNING_RATE:-0.05}"
TEXT_MAX_NEGATIVES="${TEXT_MAX_NEGATIVES:-16}"
TEXT_LISTWISE_TEMPERATURE="${TEXT_LISTWISE_TEMPERATURE:-0.06}"

mkdir -p "$WORK_DIR"

if [[ -n "$FULL_SFT_CHECKPOINT" ]]; then
  if [[ ! -d "$FULL_SFT_CHECKPOINT" ]]; then
    echo "Selected full-SFT checkpoint does not exist: $FULL_SFT_CHECKPOINT" >&2
    exit 1
  fi
  BEST_CHECKPOINT="$FULL_SFT_CHECKPOINT"
else
  while [[ ! -f "$FULL_SFT_SUMMARY" ]]; do
    echo "$(date -u +%FT%TZ) waiting for full-SFT summary: $FULL_SFT_SUMMARY"
    sleep 300
  done

  BEST_CHECKPOINT="$(
    "$PYTHON" - <<'PY' "$FULL_SFT_SUMMARY" "$FULL_SFT_EVAL_ROOT" "$FULL_SFT_RUN_ROOT"
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
eval_root = Path(sys.argv[2])
run_root = Path(sys.argv[3])
records = json.loads(summary_path.read_text()).get("records", [])
records = [row for row in records if row.get("selection_score") is not None and row.get("checkpoint")]
if not records:
    raise SystemExit(f"No scored records found in {summary_path}")
best = max(records, key=lambda row: (float(row["selection_score"]), float(row.get("bleu") or 0.0)))
metrics_path = Path(best["metrics_json"])
try:
    metrics_rel = metrics_path.relative_to(eval_root)
except ValueError:
    raise SystemExit(f"Best metrics path is outside eval root: {metrics_path}")
lr_label = metrics_rel.parts[0]
checkpoint_name = str(best["checkpoint"])
run_label = lr_label.removeprefix("lr_")
checkpoint = run_root / f"lr_{run_label}_48steps" / "chanka" / checkpoint_name
if not checkpoint.exists():
    raise SystemExit(f"Selected checkpoint does not exist: {checkpoint}")
print(checkpoint)
PY
  )"
fi

echo "$(date -u +%FT%TZ) selected 4B full-SFT checkpoint: $BEST_CHECKPOINT"
echo "$BEST_CHECKPOINT" > "${WORK_DIR}/selected_checkpoint.txt"

EVAL_FULL_POOL="${WORK_DIR}/eval_qwen35_4b_full_sft_k${NUM_RETURN_SEQUENCES}_predictions.jsonl"
TRAIN_FULL_POOL="${WORK_DIR}/train_qwen35_4b_full_sft_k${NUM_RETURN_SEQUENCES}_predictions.jsonl"
MERGED_EVAL_POOL="${WORK_DIR}/eval_current_k32_4b_old_k16_4b_new_k${NUM_RETURN_SEQUENCES}.jsonl"
MERGED_TRAIN_POOL="${WORK_DIR}/train_current_k32_4b_old_k16_4b_new_k${NUM_RETURN_SEQUENCES}.jsonl"

if [[ ! -f "$EVAL_FULL_POOL" ]]; then
  "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$BEST_CHECKPOINT" \
    --output-json "${WORK_DIR}/eval_qwen35_4b_full_sft_k${NUM_RETURN_SEQUENCES}_metrics.json" \
    --predictions-jsonl "$EVAL_FULL_POOL" \
    --batch-size "$BATCH_SIZE" \
    --max-completion-length "$MAX_COMPLETION_LENGTH" \
    --num-return-sequences "$NUM_RETURN_SEQUENCES" \
    --do-sample \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --strip-chat-artifacts \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "$TERMINOLOGY_TOP_K" \
    --progress-every "$PROGRESS_EVERY"
fi

if [[ ! -f "$TRAIN_FULL_POOL" ]]; then
  "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$BEST_CHECKPOINT" \
    --split train \
    --output-json "${WORK_DIR}/train_qwen35_4b_full_sft_k${NUM_RETURN_SEQUENCES}_metrics.json" \
    --predictions-jsonl "$TRAIN_FULL_POOL" \
    --batch-size "$BATCH_SIZE" \
    --max-completion-length "$MAX_COMPLETION_LENGTH" \
    --num-return-sequences "$NUM_RETURN_SEQUENCES" \
    --do-sample \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --strip-chat-artifacts \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "$TERMINOLOGY_TOP_K" \
    --progress-every "$PROGRESS_EVERY"
fi

if [[ ! -f "$MERGED_EVAL_POOL" ]]; then
  "$PYTHON" scripts/merge_candidate_prediction_pools.py \
    --predictions-jsonl "$BASE_EVAL_POOL" \
    --predictions-jsonl "$EVAL_FULL_POOL" \
    --output-jsonl "$MERGED_EVAL_POOL" \
    | tee "${WORK_DIR}/merge_eval_summary.json"
fi

if [[ ! -f "$MERGED_TRAIN_POOL" ]]; then
  "$PYTHON" scripts/merge_candidate_prediction_pools.py \
    --predictions-jsonl "$BASE_TRAIN_POOL" \
    --predictions-jsonl "$TRAIN_FULL_POOL" \
    --output-jsonl "$MERGED_TRAIN_POOL" \
    | tee "${WORK_DIR}/merge_train_summary.json"
fi

TEXT_DIR="${WORK_DIR}/text_ranker_listwise"
"$PYTHON" scripts/train_text_candidate_reranker.py \
  --train-jsonl "$MERGED_TRAIN_POOL" \
  --eval-jsonl "$MERGED_EVAL_POOL" \
  --output-dir "$TEXT_DIR" \
  --prefix text_current_4b_old_4b_new \
  --training-objective listwise \
  --epochs "$TEXT_EPOCHS" \
  --learning-rate "$TEXT_LEARNING_RATE" \
  --max-negatives-per-group "$TEXT_MAX_NEGATIVES" \
  --listwise-temperature "$TEXT_LISTWISE_TEMPERATURE"
