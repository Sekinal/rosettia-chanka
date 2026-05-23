#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-9b-candidate-rerank)}"

RUN_ROOT="${RUN_ROOT:-outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64}"
CHANKA_RUN_DIR="${CHANKA_RUN_DIR:-${RUN_ROOT}/chanka/chanka}"
CHECKPOINT_EVAL_DIR="${CHECKPOINT_EVAL_DIR:-outputs/gspo_checkpoint_evals/20260522-qwen35-curriculum-eval}"

BASE_EVAL_POOL="${BASE_EVAL_POOL:-outputs/rerank_candidate_evals/20260522-current-k32-plus-qwen35-4b-full-k16-term/candidates_predictions.jsonl}"
BASE_TRAIN_POOL="${BASE_TRAIN_POOL:-outputs/verifier_candidate_mining/20260522-train-current-k32-plus-qwen35-4b-full-k16-term/train_current_k32_plus_4bfull_k16_predictions.jsonl}"

WORK_DIR="${WORK_DIR:-outputs/qwen35_9b_candidate_rerank/${STAMP}}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"
FEW_SHOT_TOP_K="${FEW_SHOT_TOP_K:-0}"
FEW_SHOT_MAX_CANDIDATES="${FEW_SHOT_MAX_CANDIDATES:-128}"

NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-16}"
TEMPERATURE="${TEMPERATURE:-0.65}"
TOP_P="${TOP_P:-0.90}"
TOP_K="${TOP_K:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-80}"
PROGRESS_EVERY="${PROGRESS_EVERY:-64}"

TEXT_EPOCHS="${TEXT_EPOCHS:-5}"
TEXT_LEARNING_RATE="${TEXT_LEARNING_RATE:-0.08}"
TEXT_MAX_NEGATIVES="${TEXT_MAX_NEGATIVES:-16}"
TEXT_TRAINING_OBJECTIVE="${TEXT_TRAINING_OBJECTIVE:-listwise}"
TEXT_LISTWISE_TEMPERATURE="${TEXT_LISTWISE_TEMPERATURE:-0.06}"
FEATURE_EPOCHS="${FEATURE_EPOCHS:-120}"
ENSEMBLE_SEARCH_ITERATIONS="${ENSEMBLE_SEARCH_ITERATIONS:-8000}"

mkdir -p "$WORK_DIR"

FEW_SHOT_ARGS=()
if [[ "$FEW_SHOT_TOP_K" -gt 0 ]]; then
  FEW_SHOT_ARGS=(
    --few-shot-top-k "$FEW_SHOT_TOP_K"
    --few-shot-max-candidates "$FEW_SHOT_MAX_CANDIDATES"
  )
fi

while [[ ! -f "${CHECKPOINT_EVAL_DIR}/summary.json" ]]; do
  echo "$(date -u +%FT%TZ) waiting for checkpoint eval summary: ${CHECKPOINT_EVAL_DIR}/summary.json"
  sleep 300
done

BEST_CHECKPOINT="$(
  "$PYTHON" - <<'PY' "$CHECKPOINT_EVAL_DIR" "$CHANKA_RUN_DIR"
import json
import sys
from pathlib import Path

from scripts.summarize_gspo_canaries import selection_score

eval_dir = Path(sys.argv[1])
chanka_dir = Path(sys.argv[2])
summary = json.loads((eval_dir / "summary.json").read_text())
records = []
for row in summary.get("records", []):
    score = row.get("selection_score")
    if score is None and row.get("metrics_json"):
        score = selection_score(json.loads(Path(row["metrics_json"]).read_text()))
        row = {**row, "selection_score": score}
    if score is not None:
        records.append(row)
if not records:
    raise SystemExit("No scored checkpoint records found")
best = max(records, key=lambda row: row["selection_score"])
checkpoint = chanka_dir / best["checkpoint"]
if not checkpoint.exists():
    raise SystemExit(f"Selected checkpoint does not exist: {checkpoint}")
print(checkpoint)
PY
)"

echo "$(date -u +%FT%TZ) selected 9B checkpoint: $BEST_CHECKPOINT"
echo "$BEST_CHECKPOINT" > "${WORK_DIR}/selected_checkpoint.txt"

EVAL_9B_POOL="${WORK_DIR}/eval_qwen35_9b_k${NUM_RETURN_SEQUENCES}_predictions.jsonl"
TRAIN_9B_POOL="${WORK_DIR}/train_qwen35_9b_k${NUM_RETURN_SEQUENCES}_predictions.jsonl"
MERGED_EVAL_POOL="${WORK_DIR}/eval_current_k32_4b_k16_9b_k${NUM_RETURN_SEQUENCES}.jsonl"
MERGED_TRAIN_POOL="${WORK_DIR}/train_current_k32_4b_k16_9b_k${NUM_RETURN_SEQUENCES}.jsonl"

if [[ ! -f "$EVAL_9B_POOL" ]]; then
  "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$BEST_CHECKPOINT" \
    --output-json "${WORK_DIR}/eval_qwen35_9b_k${NUM_RETURN_SEQUENCES}_metrics.json" \
    --predictions-jsonl "$EVAL_9B_POOL" \
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
    "${FEW_SHOT_ARGS[@]}" \
    --progress-every "$PROGRESS_EVERY"
fi

if [[ ! -f "$TRAIN_9B_POOL" ]]; then
  "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$BEST_CHECKPOINT" \
    --split train \
    --output-json "${WORK_DIR}/train_qwen35_9b_k${NUM_RETURN_SEQUENCES}_metrics.json" \
    --predictions-jsonl "$TRAIN_9B_POOL" \
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
    "${FEW_SHOT_ARGS[@]}" \
    --progress-every "$PROGRESS_EVERY"
fi

if [[ ! -f "$MERGED_EVAL_POOL" ]]; then
  "$PYTHON" scripts/merge_candidate_prediction_pools.py \
    --predictions-jsonl "$BASE_EVAL_POOL" \
    --predictions-jsonl "$EVAL_9B_POOL" \
    --output-jsonl "$MERGED_EVAL_POOL" \
    | tee "${WORK_DIR}/merge_eval_summary.json"
fi

if [[ ! -f "$MERGED_TRAIN_POOL" ]]; then
  "$PYTHON" scripts/merge_candidate_prediction_pools.py \
    --predictions-jsonl "$BASE_TRAIN_POOL" \
    --predictions-jsonl "$TRAIN_9B_POOL" \
    --output-jsonl "$MERGED_TRAIN_POOL" \
    | tee "${WORK_DIR}/merge_train_summary.json"
fi

FEATURE_DIR="${WORK_DIR}/feature_listwise"
TEXT_DIR="${WORK_DIR}/text_ranker"
ENSEMBLE_DIR="${WORK_DIR}/score_ensemble"

"$PYTHON" scripts/train_feature_candidate_reranker.py \
  --train-jsonl "$MERGED_TRAIN_POOL" \
  --eval-jsonl "$MERGED_EVAL_POOL" \
  --output-dir "$FEATURE_DIR" \
  --prefix feature_current_4b_9b \
  --training-objective listwise \
  --listwise-epochs "$FEATURE_EPOCHS" \
  --listwise-learning-rate 0.03 \
  --listwise-temperature 0.04

TEXT_ARGS=(
  --train-jsonl "$MERGED_TRAIN_POOL"
  --eval-jsonl "$MERGED_EVAL_POOL"
  --output-dir "$TEXT_DIR"
  --prefix text_current_4b_9b
  --training-objective "$TEXT_TRAINING_OBJECTIVE"
  --epochs "$TEXT_EPOCHS"
  --learning-rate "$TEXT_LEARNING_RATE"
  --max-negatives-per-group "$TEXT_MAX_NEGATIVES"
)
if [[ "$TEXT_TRAINING_OBJECTIVE" == "listwise" ]]; then
  TEXT_ARGS+=(--listwise-temperature "$TEXT_LISTWISE_TEMPERATURE")
fi
"$PYTHON" scripts/train_text_candidate_reranker.py "${TEXT_ARGS[@]}"

"$PYTHON" scripts/train_score_ensemble_reranker.py \
  --train-jsonl "$MERGED_TRAIN_POOL" \
  --eval-jsonl "$MERGED_EVAL_POOL" \
  --feature-weights-json "${FEATURE_DIR}/feature_current_4b_9b_weights.json" \
  --text-model-json "${TEXT_DIR}/text_current_4b_9b_model.json" \
  --output-dir "$ENSEMBLE_DIR" \
  --prefix ensemble_current_4b_9b \
  --search-iterations "$ENSEMBLE_SEARCH_ITERATIONS"
