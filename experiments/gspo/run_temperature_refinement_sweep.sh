#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M)}"
BEST_ADAPTER="${BEST_ADAPTER:-outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora}"
VERIFIER_ADAPTER_PATH="${VERIFIER_ADAPTER_PATH:-outputs/chanka_translation_verifier_hard_r128/checkpoint-1368}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/gspo_temperature_refinement_sweeps/${STAMP}}"
EVAL_DIR="${EVAL_DIR:-outputs/gspo_checkpoint_evals/${STAMP}-temperature-refine-sweep}"

cd "$ROOT_DIR"
source .venv/bin/activate

mkdir -p "$SWEEP_ROOT" "$EVAL_DIR"

run_variant() {
  local name="$1"
  local lr="$2"
  local temperature="$3"
  local top_p="$4"
  local max_steps="$5"
  local output_dir="$SWEEP_ROOT/$name"

  env \
    ROOT_DIR="$ROOT_DIR" \
    GSPO_REWARD_PROFILE=learned_verifier_vibe_2511 \
    VERIFIER_ADAPTER_PATH="$VERIFIER_ADAPTER_PATH" \
    VERIFIER_REWARD_BATCH_SIZE="${VERIFIER_REWARD_BATCH_SIZE:-2}" \
    SFT_ADAPTER_PATH="$BEST_ADAPTER" \
    GSPO_OUTPUT_DIR="$output_dir" \
    LEARNING_RATE="$lr" \
    WARMUP_RATIO="${GSPO_WARMUP_RATIO:-0.0}" \
    NUM_TRAIN_EPOCHS="${GSPO_NUM_TRAIN_EPOCHS:-1}" \
    TRAIN_BATCH_SIZE="${GSPO_TRAIN_BATCH_SIZE:-4}" \
    EVAL_BATCH_SIZE="${GSPO_EVAL_BATCH_SIZE:-4}" \
    GRADIENT_ACCUMULATION_STEPS="${GSPO_GRADIENT_ACCUMULATION_STEPS:-2}" \
    NUM_GENERATIONS="${GSPO_NUM_GENERATIONS:-4}" \
    LOGGING_STEPS="${GSPO_LOGGING_STEPS:-4}" \
    TEMPERATURE="$temperature" \
    TOP_P="$top_p" \
    MAX_STEPS="$max_steps" \
    MAX_TRAIN_SAMPLES="${GSPO_MAX_TRAIN_SAMPLES:-256}" \
    MAX_EVAL_SAMPLES="${GSPO_MAX_EVAL_SAMPLES:-64}" \
    EVAL_STEPS="${GSPO_EVAL_STEPS:-8}" \
    SAVE_STEPS="${GSPO_SAVE_STEPS:-8}" \
    bash experiments/gspo/run_2511_learned_verifier_gspo.sh

  local run_root="$output_dir/chanka_gspo"
  for item in checkpoint-8 checkpoint-16 final_gspo_lora; do
    local adapter_path="$run_root/$item"
    [[ -d "$adapter_path" ]] || continue
    local label="${name}_${item//-/_}"
    python scripts/evaluate_gspo_checkpoint.py \
      --adapter-path "$adapter_path" \
      --output-json "$EVAL_DIR/${label}_metrics.json" \
      --predictions-jsonl "$EVAL_DIR/${label}_predictions.jsonl" \
      --batch-size 1
  done
}

run_variant conservative_t060_top085_lr5e7 "${CONSERVATIVE_LR:-5e-7}" "${CONSERVATIVE_TEMPERATURE:-0.60}" "${CONSERVATIVE_TOP_P:-0.85}" "${CONSERVATIVE_MAX_STEPS:-16}"
run_variant exploratory_t090_top095_lr5e7 "${EXPLORATORY_LR:-5e-7}" "${EXPLORATORY_TEMPERATURE:-0.90}" "${EXPLORATORY_TOP_P:-0.95}" "${EXPLORATORY_MAX_STEPS:-16}"

python scripts/summarize_gspo_checkpoint_evals.py "$EVAL_DIR"
cat "$EVAL_DIR/summary.md"
