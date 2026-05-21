#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
cd "$ROOT_DIR"
source .venv/bin/activate

python scripts/train_verifier_chanka_unsloth.py \
  --output-dir "${VERIFIER_OUTPUT_DIR:-outputs/chanka_translation_verifier}" \
  --learning-rate "${VERIFIER_LEARNING_RATE:-5e-5}" \
  --num-train-epochs "${VERIFIER_EPOCHS:-3}" \
  --per-device-train-batch-size "${VERIFIER_TRAIN_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${VERIFIER_GRADIENT_ACCUMULATION_STEPS:-4}" \
  --per-device-eval-batch-size "${VERIFIER_EVAL_BATCH_SIZE:-4}" \
  --eval-steps "${VERIFIER_EVAL_STEPS:-64}" \
  --save-steps "${VERIFIER_SAVE_STEPS:-64}" \
  --logging-steps "${LOGGING_STEPS:-5}" \
  ${VERIFIER_MAX_STEPS:+--max-steps "$VERIFIER_MAX_STEPS"} \
  ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"} \
  ${MAX_EVAL_SAMPLES:+--max-eval-samples "$MAX_EVAL_SAMPLES"}

# The GSPO step still uses the verifier rubric reward in-process. The trained
# verifier LoRA above is meant for preference generation and the next learned
# reward integration pass.
python scripts/train_gspo_chanka_unsloth.py \
  --reward-profile self_verifier_2511 \
  --adapter-path "${ADAPTER_PATH:-outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400}" \
  --output-dir "${GSPO_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifier_chained}" \
  --learning-rate "${GSPO_LEARNING_RATE:-2e-6}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --eval-steps "${EVAL_STEPS:-112}" \
  --save-steps "${SAVE_STEPS:-112}" \
  --logging-steps "${LOGGING_STEPS:-5}" \
  ${GSPO_MAX_STEPS:+--max-steps "$GSPO_MAX_STEPS"} \
  ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"} \
  ${MAX_EVAL_SAMPLES:+--max-eval-samples "$MAX_EVAL_SAMPLES"}
