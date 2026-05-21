#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
SWEEP_DIR="${SWEEP_DIR:-outputs/gspo_canary_sweeps/$(date -u +%Y%m%d-%H%M%S)}"
PROFILES="${PROFILES:-baseline fg_severity_2411 severity_proxy_2310 self_verifier_2511 vibethinker_2511 mix_severity_verifier mix_verifier_vibe mix_all_strict rosettia_guard_v1 rosettia_guard_v2}"

cd "$ROOT_DIR"
source .venv/bin/activate
mkdir -p "$SWEEP_DIR"

common_args=(
  --adapter-path "${ADAPTER_PATH:-outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400}"
  --learning-rate "${LEARNING_RATE:-2e-6}"
  --max-steps "${MAX_STEPS:-24}"
  --max-train-samples "${MAX_TRAIN_SAMPLES:-256}"
  --max-eval-samples "${MAX_EVAL_SAMPLES:-64}"
  --eval-steps "${EVAL_STEPS:-8}"
  --save-steps "${SAVE_STEPS:-8}"
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE:-8}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE:-8}"
  --logging-steps "${LOGGING_STEPS:-2}"
  --temperature "${TEMPERATURE:-0.75}"
  --top-p "${TOP_P:-0.90}"
)

for profile in $PROFILES; do
  run_dir="$SWEEP_DIR/$profile"
  generations="${NUM_GENERATIONS:-4}"
  if [[ "$profile" == "vibethinker_2511" || "$profile" == "mix_verifier_vibe" ]]; then
    generations="${VIBE_NUM_GENERATIONS:-8}"
  fi

  echo "=== $profile num_generations=$generations ==="
  python scripts/train_gspo_chanka_unsloth.py \
    --reward-profile "$profile" \
    --num-generations "$generations" \
    --output-dir "$run_dir" \
    --metrics-json "$run_dir/chanka_gspo/final_metrics.json" \
    "${common_args[@]}" \
    2>&1 | tee "$run_dir.log"

  python scripts/summarize_gspo_canaries.py "$SWEEP_DIR"
done

python scripts/summarize_gspo_canaries.py "$SWEEP_DIR"
