#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M)}"

cd "$ROOT_DIR"
source .venv/bin/activate

while pgrep -f "python scripts/train_gspo_chanka_unsloth.py" >/dev/null; do
  date -u
  pgrep -af "python scripts/train_gspo_chanka_unsloth.py" || true
  sleep "$POLL_SECONDS"
done

evaluate_checkpoint() {
  local adapter_path="$1"
  local label="$2"
  if [[ ! -d "$adapter_path" ]]; then
    return
  fi
  local output_dir="outputs/gspo_checkpoint_evals/${STAMP}"
  mkdir -p "$output_dir"
  python scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$adapter_path" \
    --output-json "$output_dir/${label}_metrics.json" \
    --predictions-jsonl "$output_dir/${label}_predictions.jsonl"
}

evaluate_checkpoint \
  "outputs/gspo_selected_checkpoints/vibethinker_phase2_checkpoint_896" \
  "vibethinker_phase2_checkpoint_896_preserved"

evaluate_checkpoint \
  "outputs/gspo_selected_checkpoints/learned_verifier_on_vibe112_checkpoint_56" \
  "learned_verifier_on_vibe112_checkpoint_56_preserved"

for checkpoint in 896 1008 1120 1232 1344 1456 1486; do
  evaluate_checkpoint \
    "outputs/gspo_full_contenders/vibethinker_2511_phase2_from112_eval112_20260521-0909/chanka_gspo/checkpoint-${checkpoint}" \
    "vibethinker_phase2_checkpoint_${checkpoint}"
done

for checkpoint in 56 112 168 224; do
  evaluate_checkpoint \
    "outputs/gspo_full_contenders/learned_verifier_on_vibe112_full_20260521-1016/chanka_gspo/checkpoint-${checkpoint}" \
    "learned_verifier_on_vibe112_checkpoint_${checkpoint}"
done

common_env=(
  ROOT_DIR="$ROOT_DIR"
  GSPO_REWARD_PROFILE=learned_verifier_vibe_2511
  VERIFIER_ADAPTER_PATH="${VERIFIER_ADAPTER_PATH:-outputs/chanka_translation_verifier_hard_r128/checkpoint-1368}"
  SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-outputs/gspo_full_contenders/vibethinker_2511_continued_no_tables_20260521-083900/chanka_gspo/checkpoint-112}"
  VERIFIER_REWARD_BATCH_SIZE="${VERIFIER_REWARD_BATCH_SIZE:-2}"
  MAX_STEPS="${MAX_STEPS:-16}"
  MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-256}"
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-64}"
  EVAL_STEPS="${EVAL_STEPS:-8}"
  SAVE_STEPS="${SAVE_STEPS:-8}"
  LEARNING_RATE="${LEARNING_RATE:-7e-7}"
  WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
  LOGGING_STEPS="${LOGGING_STEPS:-4}"
)

env "${common_env[@]}" \
  NUM_GENERATIONS=4 \
  TRAIN_BATCH_SIZE=4 \
  EVAL_BATCH_SIZE=4 \
  GRADIENT_ACCUMULATION_STEPS=2 \
  GSPO_OUTPUT_DIR="outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe112_4gen_canary_${STAMP}" \
  experiments/gspo/run_2511_learned_verifier_gspo.sh

env "${common_env[@]}" \
  NUM_GENERATIONS=8 \
  TRAIN_BATCH_SIZE=8 \
  EVAL_BATCH_SIZE=8 \
  GRADIENT_ACCUMULATION_STEPS=1 \
  GSPO_OUTPUT_DIR="outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe112_8gen_canary_${STAMP}" \
  experiments/gspo/run_2511_learned_verifier_gspo.sh
