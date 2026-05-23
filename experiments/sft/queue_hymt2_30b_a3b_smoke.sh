#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-hymt2-30b-a3b-smoke)}"

WAIT_FOR_FILE="${WAIT_FOR_FILE:-outputs/contextual_terminology_sft/${STAMP_CONTEXT_WAIT:-20260523-contextual-terminology-sft}/summary.json}"
WORK_DIR="${WORK_DIR:-outputs/hymt2_30b_a3b_canaries/${STAMP}}"
SMOKE_DIR="${SMOKE_DIR:-${WORK_DIR}/smoke_chanka_lora_r16_4bit_2steps}"
FAIL_MARKER="${FAIL_MARKER:-${WORK_DIR}/smoke_failed.txt}"

MODEL_ID="${MODEL_ID:-tencent/Hy-MT2-30B-A3B}"
TARGET_LANGUAGE_NAME="${TARGET_LANGUAGE_NAME:-Quechua Chanka}"

mkdir -p "$WORK_DIR"

while [[ -n "$WAIT_FOR_FILE" && ! -f "$WAIT_FOR_FILE" ]]; do
  echo "$(date -u +%FT%TZ) waiting before Hy-MT2 30B-A3B smoke: $WAIT_FOR_FILE"
  sleep 300
done

if [[ -f "${SMOKE_DIR}/chanka/final_lora/adapter_config.json" || -f "$FAIL_MARKER" ]]; then
  echo "$(date -u +%FT%TZ) Hy-MT2 30B-A3B smoke already has success or failure marker"
  exit 0
fi

set +e
"$PYTHON" scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id "$MODEL_ID" \
  --prompt-style hymt2 \
  --target-language-name "$TARGET_LANGUAGE_NAME" \
  --load-in-4bit \
  --max-train-samples 16 \
  --max-eval-samples 4 \
  --max-steps 2 \
  --eval-steps 1 \
  --save-steps 1 \
  --logging-steps 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-seq-length 128 \
  --learning-rate 2e-5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --output-dir "$SMOKE_DIR"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  {
    echo "$(date -u +%FT%TZ) Hy-MT2 30B-A3B 4-bit Unsloth smoke failed with exit status $status"
    echo "This is expected on smaller GPUs if the MoE checkpoint is unsupported or still too large."
  } | tee "$FAIL_MARKER"
  exit 0
fi

echo "$(date -u +%FT%TZ) Hy-MT2 30B-A3B smoke completed: ${SMOKE_DIR}/chanka/final_lora" | tee "${WORK_DIR}/smoke_success.txt"
