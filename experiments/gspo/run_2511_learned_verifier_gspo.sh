#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
cd "$ROOT_DIR"
source .venv/bin/activate

VERIFIER_OUTPUT_DIR="${VERIFIER_OUTPUT_DIR:-outputs/chanka_translation_verifier_full}"
VERIFIER_ADAPTER_PATH="${VERIFIER_ADAPTER_PATH:-$VERIFIER_OUTPUT_DIR/final_verifier_lora}"

if [[ ! -d "$VERIFIER_ADAPTER_PATH" ]]; then
  python scripts/train_verifier_chanka_unsloth.py \
    --output-dir "$VERIFIER_OUTPUT_DIR" \
    --num-train-epochs "${VERIFIER_NUM_TRAIN_EPOCHS:-5}" \
    --max-steps "${VERIFIER_MAX_STEPS:--1}" \
    --learning-rate "${VERIFIER_LEARNING_RATE:-3e-5}" \
    --lora-r "${VERIFIER_LORA_R:-128}" \
    --lora-alpha "${VERIFIER_LORA_ALPHA:-256}" \
    --per-device-train-batch-size "${VERIFIER_TRAIN_BATCH_SIZE:-4}" \
    --per-device-eval-batch-size "${VERIFIER_EVAL_BATCH_SIZE:-4}" \
    --gradient-accumulation-steps "${VERIFIER_GRADIENT_ACCUMULATION_STEPS:-4}" \
    --evals-per-epoch "${VERIFIER_EVALS_PER_EPOCH:-8}" \
    --logging-steps "${VERIFIER_LOGGING_STEPS:-10}" \
    ${VERIFIER_MAX_TRAIN_SAMPLES:+--max-train-samples "$VERIFIER_MAX_TRAIN_SAMPLES"} \
    ${VERIFIER_MAX_EVAL_SAMPLES:+--max-eval-samples "$VERIFIER_MAX_EVAL_SAMPLES"}
fi

python scripts/train_gspo_chanka_unsloth.py \
  --reward-profile learned_verifier_2511 \
  --verifier-adapter-path "$VERIFIER_ADAPTER_PATH" \
  --verifier-batch-size "${VERIFIER_REWARD_BATCH_SIZE:-4}" \
  --adapter-path "${SFT_ADAPTER_PATH:-outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400}" \
  --output-dir "${GSPO_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_learned_verifier}" \
  --learning-rate "${LEARNING_RATE:-1e-6}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-2}" \
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE:-4}" \
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --logging-steps "${LOGGING_STEPS:-5}" \
  --temperature "${TEMPERATURE:-0.75}" \
  --top-p "${TOP_P:-0.90}" \
  ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
  ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"} \
  ${MAX_EVAL_SAMPLES:+--max-eval-samples "$MAX_EVAL_SAMPLES"} \
  ${EVAL_STEPS:+--eval-steps "$EVAL_STEPS"} \
  ${SAVE_STEPS:+--save-steps "$SAVE_STEPS"}
