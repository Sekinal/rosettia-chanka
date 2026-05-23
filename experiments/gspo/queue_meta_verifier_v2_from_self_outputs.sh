#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-meta-v2-self)}"

SELF_ADAPTER="${SELF_ADAPTER:-outputs/gspo_paper_profiles/2511_self_verifiable_translation_20260523-self-verifiable-fixed/chanka_gspo/final_gspo_lora}"
WAIT_PATTERN="${WAIT_PATTERN:-train_gspo_chanka_unsloth.py.*self-verifiable-fixed}"
WORK_DIR="${WORK_DIR:-outputs/self_verification_mining/${STAMP}}"
DATA_DIR="${DATA_DIR:-outputs/self_verifiable_translation_data_${STAMP}}"
META_OUTPUT_DIR="${META_OUTPUT_DIR:-outputs/chanka_translation_meta_verifier_v2_${STAMP}}"
META_ADAPTER="${META_ADAPTER:-${META_OUTPUT_DIR}/final_meta_verifier_lora}"
GSPO_OUTPUT_DIR="${GSPO_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_translation_meta_v2_${STAMP}}"

cd "$ROOT_DIR"
mkdir -p "$WORK_DIR"

active_wait_matches() {
  ps -eo pid=,cmd= \
    | grep -E "$WAIT_PATTERN" \
    | grep -v grep \
    | grep -v "queue_meta_verifier_v2_from_self_outputs.sh" \
    || true
}

while [[ -n "$(active_wait_matches)" ]]; do
  echo "Waiting for active self-verifiable run matching: $WAIT_PATTERN"
  sleep "${WAIT_SECONDS:-60}"
done

if [[ ! -d "$SELF_ADAPTER" ]]; then
  echo "Missing self-verifiable adapter: $SELF_ADAPTER" >&2
  exit 1
fi

"$PYTHON" scripts/evaluate_gspo_checkpoint.py \
  --adapter-path "$SELF_ADAPTER" \
  --output-json "$WORK_DIR/self_outputs_metrics.json" \
  --predictions-jsonl "$WORK_DIR/self_outputs_predictions.jsonl" \
  --split "${MINE_SPLIT:-all}" \
  --max-train-samples "${MINE_MAX_TRAIN_SAMPLES:-256}" \
  --max-eval-samples "${MINE_MAX_EVAL_SAMPLES:-64}" \
  --batch-size "${MINE_BATCH_SIZE:-1}" \
  --max-seq-length "${MINE_MAX_SEQ_LENGTH:-192}" \
  --max-completion-length "${MINE_MAX_COMPLETION_LENGTH:-96}" \
  --self-verification-output \
  --strip-chat-artifacts \
  --terminology-file "${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}" \
  --terminology-top-k "${TERMINOLOGY_TOP_K:-1}" \
  --progress-every "${MINE_PROGRESS_EVERY:-32}"

"$PYTHON" scripts/build_self_verifiable_translation_data.py \
  --output-dir "$DATA_DIR" \
  --max-rows "${COLD_START_MAX_ROWS:-256}"

REAL_META_ARGS=()
if [[ -n "${REAL_META_MAX_RECORDS:-}" ]]; then
  REAL_META_ARGS+=(--max-records "$REAL_META_MAX_RECORDS")
fi

"$PYTHON" scripts/build_meta_verifier_from_self_outputs.py \
  --self-predictions-jsonl "$WORK_DIR/self_outputs_predictions.jsonl" \
  --output-jsonl "$WORK_DIR/meta_v2_real_self_outputs.jsonl" \
  --summary-json "$WORK_DIR/meta_v2_real_self_outputs_summary.json" \
  --min-quality-gap "${MIN_QUALITY_GAP:-0.20}" \
  "${REAL_META_ARGS[@]}"

"$PYTHON" scripts/train_meta_verifier_chanka_unsloth.py \
  --meta-jsonl "$DATA_DIR/translation_meta_verifier_cold_start.jsonl" \
  --meta-jsonl "$WORK_DIR/meta_v2_real_self_outputs.jsonl" \
  --output-dir "$META_OUTPUT_DIR" \
  --max-seq-length "${META_MAX_SEQ_LENGTH:-768}" \
  --max-steps "${META_MAX_STEPS:-128}" \
  --eval-steps "${META_EVAL_STEPS:-32}" \
  --save-steps "${META_SAVE_STEPS:-32}" \
  --per-device-train-batch-size "${META_TRAIN_BATCH_SIZE:-4}" \
  --per-device-eval-batch-size "${META_EVAL_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${META_GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning-rate "${META_LEARNING_RATE:-2e-5}" \
  --lora-r "${META_LORA_R:-64}" \
  --lora-alpha "${META_LORA_ALPHA:-128}" \
  --logging-steps "${META_LOGGING_STEPS:-8}"

META_VERIFIER_ADAPTER="$META_ADAPTER" \
OUTPUT_DIR="$GSPO_OUTPUT_DIR" \
MAX_STEPS="${GSPO_MAX_STEPS:-16}" \
EVAL_STEPS="${GSPO_EVAL_STEPS:-8}" \
SAVE_STEPS="${GSPO_SAVE_STEPS:-8}" \
MAX_TRAIN_SAMPLES="${GSPO_MAX_TRAIN_SAMPLES:-256}" \
MAX_EVAL_SAMPLES="${GSPO_MAX_EVAL_SAMPLES:-64}" \
experiments/gspo/run_2511_self_verifiable_translation.sh
