#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-9b-direct-full-sft-smoke)}"

BEST_CHECKPOINT="${BEST_CHECKPOINT:-outputs/qwen35_9b_curriculum/20260522-broad256-chanka192-r64/chanka/chanka/checkpoint-192}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-outputs/merged_full_models/${STAMP}-merged16}"
SMOKE_DIR="${SMOKE_DIR:-outputs/qwen35_9b_full_sft/${STAMP}/smoke_full_chanka_lr5e-7_2steps}"
FAIL_MARKER="${FAIL_MARKER:-outputs/qwen35_9b_full_sft/${STAMP}/full_sft_smoke_failed.txt}"

TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}"
OPTIM="${OPTIM:-paged_adamw_8bit}"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$FAIL_MARKER")"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -f "${SMOKE_DIR}/chanka/final_full_model/config.json" || -f "$FAIL_MARKER" ]]; then
  echo "$(date -u +%FT%TZ) 9B direct full-SFT smoke already has success or failure marker"
  exit 0
fi

if [[ ! -f "${MERGED_MODEL_DIR}/config.json" ]]; then
  "$PYTHON" scripts/export_unsloth_merged_model.py \
    --adapter-path "$BEST_CHECKPOINT" \
    --output-dir "$MERGED_MODEL_DIR" \
    --max-seq-length 128 \
    --save-method merged_16bit
fi

set +e
"$PYTHON" scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id "$MERGED_MODEL_DIR" \
  --training-mode full \
  --output-dir "$SMOKE_DIR" \
  --max-seq-length 128 \
  --max-train-samples 32 \
  --max-eval-samples 8 \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 2 \
  --save-total-limit 1 \
  --logging-steps 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate "${LEARNING_RATE:-5e-7}" \
  --warmup-ratio "${WARMUP_RATIO:-0.05}" \
  --terminology-file "$TERMINOLOGY_FILE" \
  --terminology-top-k "$TERMINOLOGY_TOP_K" \
  --optim "$OPTIM"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  {
    echo "$(date -u +%FT%TZ) 9B direct full-SFT smoke failed with exit status $status"
    echo "Optimizer: $OPTIM"
    echo "Merged model: $MERGED_MODEL_DIR"
  } | tee "$FAIL_MARKER"
  exit 0
fi

rm -rf "${SMOKE_DIR}/chanka/checkpoint-"* || true
df -h /root
echo "$(date -u +%FT%TZ) 9B direct full-SFT smoke completed: ${SMOKE_DIR}/chanka/final_full_model"
