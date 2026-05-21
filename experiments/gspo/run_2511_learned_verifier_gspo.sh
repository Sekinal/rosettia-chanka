#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
cd "$ROOT_DIR"
source .venv/bin/activate

VERIFIER_OUTPUT_DIR="${VERIFIER_OUTPUT_DIR:-outputs/chanka_translation_verifier_full}"
VERIFIER_ADAPTER_PATH="${VERIFIER_ADAPTER_PATH:-$VERIFIER_OUTPUT_DIR/final_verifier_lora}"

if [[ ! -d "$VERIFIER_ADAPTER_PATH" ]]; then
  verifier_args=(
    scripts/train_verifier_chanka_unsloth.py
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
    --logging-steps "${VERIFIER_LOGGING_STEPS:-10}"
  )
  [[ -n "${VERIFIER_MAX_TRAIN_SAMPLES:-}" ]] && verifier_args+=(--max-train-samples "$VERIFIER_MAX_TRAIN_SAMPLES")
  [[ -n "${VERIFIER_MAX_EVAL_SAMPLES:-}" ]] && verifier_args+=(--max-eval-samples "$VERIFIER_MAX_EVAL_SAMPLES")
  python "${verifier_args[@]}"
fi

gspo_args=(
  scripts/train_gspo_chanka_unsloth.py
  --reward-profile "${GSPO_REWARD_PROFILE:-learned_verifier_2511}" \
  --verifier-adapter-path "$VERIFIER_ADAPTER_PATH" \
  --verifier-batch-size "${VERIFIER_REWARD_BATCH_SIZE:-4}" \
  --adapter-path "${SFT_ADAPTER_PATH:-outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400}" \
  --output-dir "${GSPO_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_learned_verifier}" \
  --learning-rate "${LEARNING_RATE:-1e-6}" \
  --warmup-ratio "${WARMUP_RATIO:-0.05}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-2}" \
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE:-4}" \
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --logging-steps "${LOGGING_STEPS:-10}" \
  --no-log-completions \
  --temperature "${TEMPERATURE:-0.75}" \
  --top-p "${TOP_P:-0.90}"
)

[[ -n "${MAX_STEPS:-}" ]] && gspo_args+=(--max-steps "$MAX_STEPS")
[[ -n "${RESUME_FROM_CHECKPOINT:-}" ]] && gspo_args+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
[[ -n "${MAX_TRAIN_SAMPLES:-}" ]] && gspo_args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
[[ -n "${MAX_EVAL_SAMPLES:-}" ]] && gspo_args+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
[[ -n "${EVAL_STEPS:-}" ]] && gspo_args+=(--eval-steps "$EVAL_STEPS")
[[ -n "${SAVE_STEPS:-}" ]] && gspo_args+=(--save-steps "$SAVE_STEPS")
[[ -n "${SECONDARY_VERIFIER_ADAPTER_PATH:-}" ]] && gspo_args+=(--secondary-verifier-adapter-path "$SECONDARY_VERIFIER_ADAPTER_PATH")

python "${gspo_args[@]}"
