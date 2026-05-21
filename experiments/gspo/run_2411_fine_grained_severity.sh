#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
cd "$ROOT_DIR"
source .venv/bin/activate

python scripts/train_gspo_chanka_unsloth.py \
  --reward-profile fg_severity_2411 \
  --adapter-path "${ADAPTER_PATH:-outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400}" \
  --output-dir "${OUTPUT_DIR:-outputs/gspo_paper_profiles/2411_fine_grained_severity}" \
  --learning-rate "${LEARNING_RATE:-2e-6}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --eval-steps "${EVAL_STEPS:-112}" \
  --save-steps "${SAVE_STEPS:-112}" \
  --logging-steps "${LOGGING_STEPS:-5}" \
  ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
  ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"} \
  ${MAX_EVAL_SAMPLES:+--max-eval-samples "$MAX_EVAL_SAMPLES"}
