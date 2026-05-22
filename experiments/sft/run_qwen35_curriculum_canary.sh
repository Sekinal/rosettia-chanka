#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_ID="${MODEL_ID:-unsloth/Qwen3.5-4B}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-curriculum)}"
ROOT_OUTPUT_DIR="${ROOT_OUTPUT_DIR:-outputs/qwen35_curriculum/${STAMP}}"

BROAD_OUTPUT_DIR="${BROAD_OUTPUT_DIR:-${ROOT_OUTPUT_DIR}/broad}"
CHANKA_OUTPUT_DIR="${CHANKA_OUTPUT_DIR:-${ROOT_OUTPUT_DIR}/chanka}"

BROAD_MAX_SEQ_LENGTH="${BROAD_MAX_SEQ_LENGTH:-256}"
BROAD_MAX_STEPS="${BROAD_MAX_STEPS:-512}"
BROAD_EVAL_STEPS="${BROAD_EVAL_STEPS:-128}"
BROAD_SAVE_STEPS="${BROAD_SAVE_STEPS:-128}"
BROAD_TRAIN_BATCH_SIZE="${BROAD_TRAIN_BATCH_SIZE:-4}"
BROAD_EVAL_BATCH_SIZE="${BROAD_EVAL_BATCH_SIZE:-4}"
BROAD_GRADIENT_ACCUMULATION_STEPS="${BROAD_GRADIENT_ACCUMULATION_STEPS:-4}"
BROAD_LEARNING_RATE="${BROAD_LEARNING_RATE:-5e-5}"

CHANKA_MAX_SEQ_LENGTH="${CHANKA_MAX_SEQ_LENGTH:-128}"
CHANKA_MAX_STEPS="${CHANKA_MAX_STEPS:-256}"
CHANKA_EVAL_STEPS="${CHANKA_EVAL_STEPS:-32}"
CHANKA_SAVE_STEPS="${CHANKA_SAVE_STEPS:-32}"
CHANKA_TRAIN_BATCH_SIZE="${CHANKA_TRAIN_BATCH_SIZE:-4}"
CHANKA_EVAL_BATCH_SIZE="${CHANKA_EVAL_BATCH_SIZE:-4}"
CHANKA_GRADIENT_ACCUMULATION_STEPS="${CHANKA_GRADIENT_ACCUMULATION_STEPS:-2}"
CHANKA_LEARNING_RATE="${CHANKA_LEARNING_RATE:-2e-5}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-16}"

mkdir -p "$ROOT_OUTPUT_DIR"

"$PYTHON" scripts/train_sft_unsloth.py \
  --stage broad \
  --model-id "$MODEL_ID" \
  --output-dir "$BROAD_OUTPUT_DIR" \
  --max-seq-length "$BROAD_MAX_SEQ_LENGTH" \
  --max-steps "$BROAD_MAX_STEPS" \
  --eval-steps "$BROAD_EVAL_STEPS" \
  --save-steps "$BROAD_SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --per-device-train-batch-size "$BROAD_TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$BROAD_EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$BROAD_GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$BROAD_LEARNING_RATE" \
  --warmup-ratio "$WARMUP_RATIO" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA"

BASE_ADAPTER="${BASE_ADAPTER:-${BROAD_OUTPUT_DIR}/broad/final_lora}"

"$PYTHON" scripts/train_sft_unsloth.py \
  --stage chanka \
  --model-id "$MODEL_ID" \
  --adapter-path "$BASE_ADAPTER" \
  --output-dir "$CHANKA_OUTPUT_DIR" \
  --max-seq-length "$CHANKA_MAX_SEQ_LENGTH" \
  --max-steps "$CHANKA_MAX_STEPS" \
  --eval-steps "$CHANKA_EVAL_STEPS" \
  --save-steps "$CHANKA_SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --per-device-train-batch-size "$CHANKA_TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$CHANKA_EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$CHANKA_GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$CHANKA_LEARNING_RATE" \
  --warmup-ratio "$WARMUP_RATIO" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA"
