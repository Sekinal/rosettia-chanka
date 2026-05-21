#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
SWEEP_DIR="${SWEEP_DIR:?Set SWEEP_DIR to a completed canary sweep directory}"

cd "$ROOT_DIR"
source .venv/bin/activate

python scripts/summarize_gspo_canaries.py "$SWEEP_DIR"

PROFILE="${PROFILE:-$(python - "$SWEEP_DIR/summary.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    first = handle.readline()
if not first:
    raise SystemExit("No canary records found")
print(json.loads(first)["reward_profile"])
PY
)}"

num_generations="${NUM_GENERATIONS:-4}"
train_batch="${TRAIN_BATCH_SIZE:-8}"
eval_batch="${EVAL_BATCH_SIZE:-8}"
if [[ "$PROFILE" == "vibethinker_2511" || "$PROFILE" == "mix_verifier_vibe" ]]; then
  num_generations="${NUM_GENERATIONS:-8}"
  train_batch="${TRAIN_BATCH_SIZE:-8}"
  eval_batch="${EVAL_BATCH_SIZE:-8}"
fi

run_stamp="$(date -u +%Y%m%d-%H%M%S)"
output_dir="${OUTPUT_DIR:-outputs/gspo_full_contenders/${PROFILE}_${run_stamp}}"

echo "Launching full contender profile=$PROFILE output_dir=$output_dir"

python scripts/train_gspo_chanka_unsloth.py \
  --reward-profile "$PROFILE" \
  --num-generations "$num_generations" \
  --adapter-path "${ADAPTER_PATH:-outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400}" \
  --output-dir "$output_dir" \
  --learning-rate "${LEARNING_RATE:-1e-6}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-2}" \
  --per-device-train-batch-size "$train_batch" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --per-device-eval-batch-size "$eval_batch" \
  --eval-steps "${EVAL_STEPS:-28}" \
  --save-steps "${SAVE_STEPS:-28}" \
  --logging-steps "${LOGGING_STEPS:-10}" \
  --no-log-completions \
  --temperature "${TEMPERATURE:-0.75}" \
  --top-p "${TOP_P:-0.90}" \
  ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
  ${MAX_TRAIN_SAMPLES:+--max-train-samples "$MAX_TRAIN_SAMPLES"} \
  ${MAX_EVAL_SAMPLES:+--max-eval-samples "$MAX_EVAL_SAMPLES"}
