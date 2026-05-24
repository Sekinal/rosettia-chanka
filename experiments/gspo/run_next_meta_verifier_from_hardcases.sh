#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-next-meta-verifier)}"

SFT_META_JSONL="${SFT_META_JSONL:-}"
GSPO_META_JSONL="${GSPO_META_JSONL:-}"
EXTRA_META_JSONLS="${EXTRA_META_JSONLS:-}"
META_DATA_DIR="${META_DATA_DIR:-outputs/self_verifiable_meta_data_${STAMP}}"
META_OUTPUT_DIR="${META_OUTPUT_DIR:-outputs/chanka_translation_meta_verifier_iter_${STAMP}}"
MIN_META_HARDCASES_FOR_TRAIN="${MIN_META_HARDCASES_FOR_TRAIN:-32}"

cd "$ROOT_DIR"

META_JSONLS=()
if [[ -n "$SFT_META_JSONL" ]]; then
  META_JSONLS+=("$SFT_META_JSONL")
fi
if [[ -n "$GSPO_META_JSONL" ]]; then
  META_JSONLS+=("$GSPO_META_JSONL")
fi
if [[ -n "$EXTRA_META_JSONLS" ]]; then
  IFS=':' read -r -a EXTRA_META_JSONL_ARRAY <<< "$EXTRA_META_JSONLS"
  for path in "${EXTRA_META_JSONL_ARRAY[@]}"; do
    if [[ -n "$path" ]]; then
      META_JSONLS+=("$path")
    fi
  done
fi

if [[ "${#META_JSONLS[@]}" -eq 0 ]]; then
  echo "No meta hardcase JSONLs provided. Set SFT_META_JSONL, GSPO_META_JSONL, or EXTRA_META_JSONLS." >&2
  exit 1
fi

CHECK_ARGS=()
TRAIN_ARGS=()
for path in "${META_JSONLS[@]}"; do
  CHECK_ARGS+=(--jsonl "$path")
  TRAIN_ARGS+=(--meta-jsonl "$path")
done

"$PYTHON" scripts/check_meta_hardcase_data.py \
  "${CHECK_ARGS[@]}" \
  --min-records "$MIN_META_HARDCASES_FOR_TRAIN"

"$PYTHON" scripts/build_self_verifiable_translation_data.py \
  --output-dir "$META_DATA_DIR" \
  --max-rows "${META_COLD_START_MAX_ROWS:-256}"

"$PYTHON" scripts/train_meta_verifier_chanka_unsloth.py \
  --model-id "${META_VERIFIER_MODEL_ID:-unsloth/Qwen3.5-2B}" \
  --meta-jsonl "${META_DATA_DIR}/translation_meta_verifier_cold_start.jsonl" \
  "${TRAIN_ARGS[@]}" \
  --output-dir "$META_OUTPUT_DIR" \
  --max-seq-length "${META_VERIFIER_MAX_SEQ_LENGTH:-768}" \
  --max-steps "${META_VERIFIER_MAX_STEPS:-64}" \
  --eval-steps "${META_VERIFIER_EVAL_STEPS:-16}" \
  --save-steps "${META_VERIFIER_SAVE_STEPS:-16}" \
  --per-device-train-batch-size "${META_VERIFIER_TRAIN_BATCH_SIZE:-4}" \
  --per-device-eval-batch-size "${META_VERIFIER_EVAL_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${META_VERIFIER_GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning-rate "${META_VERIFIER_LEARNING_RATE:-2e-5}" \
  --lora-r "${META_VERIFIER_LORA_R:-64}" \
  --lora-alpha "${META_VERIFIER_LORA_ALPHA:-128}" \
  --logging-steps "${META_VERIFIER_LOGGING_STEPS:-8}"

echo "Next-iteration meta-verifier adapter: ${META_OUTPUT_DIR}/final_meta_verifier_lora"
