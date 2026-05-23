#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-meta-verifier-self-gspo)}"

DATA_DIR="${DATA_DIR:-outputs/self_verifiable_translation_data_${STAMP}}"
META_OUTPUT_DIR="${META_OUTPUT_DIR:-outputs/chanka_translation_meta_verifier_${STAMP}}"
META_ADAPTER="${META_ADAPTER:-${META_OUTPUT_DIR}/final_meta_verifier_lora}"
GSPO_OUTPUT_DIR="${GSPO_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_translation_meta_${STAMP}}"

cd "$ROOT_DIR"

DATA_ARGS=()
if [[ -n "${MAX_META_SOURCE_ROWS:-}" ]]; then
  DATA_ARGS+=(--max-rows "$MAX_META_SOURCE_ROWS")
fi

"$PYTHON" scripts/build_self_verifiable_translation_data.py \
  --output-dir "$DATA_DIR" \
  "${DATA_ARGS[@]}"

"$PYTHON" scripts/train_meta_verifier_chanka_unsloth.py \
  --meta-jsonl "$DATA_DIR/translation_meta_verifier_cold_start.jsonl" \
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
experiments/gspo/run_2511_self_verifiable_translation.sh
