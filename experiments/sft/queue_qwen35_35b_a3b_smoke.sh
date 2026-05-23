#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-35b-a3b-smoke)}"

WAIT_FOR_FILE="${WAIT_FOR_FILE:-AGENTS.MD}"
WORK_DIR="${WORK_DIR:-outputs/qwen35_35b_a3b_canaries/${STAMP}}"
SMOKE_DIR="${SMOKE_DIR:-${WORK_DIR}/smoke_chanka_lora_r16_4bit_2steps}"
FAIL_MARKER="${FAIL_MARKER:-${WORK_DIR}/smoke_failed.txt}"

MODEL_ID="${MODEL_ID:-unsloth/Qwen3.5-35B-A3B}"

mkdir -p "$WORK_DIR"

while [[ -n "$WAIT_FOR_FILE" && ! -f "$WAIT_FOR_FILE" ]]; do
  echo "$(date -u +%FT%TZ) waiting before Qwen3.5 35B-A3B smoke: $WAIT_FOR_FILE"
  sleep 300
done

if [[ -f "${SMOKE_DIR}/chanka/final_lora/adapter_config.json" || -f "$FAIL_MARKER" ]]; then
  echo "$(date -u +%FT%TZ) Qwen3.5 35B-A3B smoke already has success or failure marker"
  exit 0
fi

set +e
"$PYTHON" scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id "$MODEL_ID" \
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
    echo "$(date -u +%FT%TZ) Qwen3.5 35B-A3B 4-bit Unsloth smoke failed with exit status $status"
    echo "This is a compatibility/memory probe only. QLoRA remains disfavored for serious Qwen3.5 training."
  } | tee "$FAIL_MARKER"
  exit 0
fi

echo "$(date -u +%FT%TZ) Qwen3.5 35B-A3B smoke completed: ${SMOKE_DIR}/chanka/final_lora" | tee "${WORK_DIR}/smoke_success.txt"
