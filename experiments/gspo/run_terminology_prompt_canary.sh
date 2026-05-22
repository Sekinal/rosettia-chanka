#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M)}"
BEST_ADAPTER="${BEST_ADAPTER:-outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora}"
VERIFIER_ADAPTER_PATH="${VERIFIER_ADAPTER_PATH:-outputs/chanka_translation_verifier_hard_r128/checkpoint-1368}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
CANARY_OUTPUT_DIR="${CANARY_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_terminology_prompt_current_best_${STAMP}}"
EVAL_DIR="${EVAL_DIR:-outputs/gspo_checkpoint_evals/${STAMP}-terminology-prompt-canary}"

cd "$ROOT_DIR"
source .venv/bin/activate

env \
  ROOT_DIR="$ROOT_DIR" \
  GSPO_REWARD_PROFILE=learned_verifier_vibe_2511 \
  VERIFIER_ADAPTER_PATH="$VERIFIER_ADAPTER_PATH" \
  VERIFIER_REWARD_BATCH_SIZE="${VERIFIER_REWARD_BATCH_SIZE:-2}" \
  SFT_ADAPTER_PATH="$BEST_ADAPTER" \
  GSPO_OUTPUT_DIR="$CANARY_OUTPUT_DIR" \
  TERMINOLOGY_FILE="$TERMINOLOGY_FILE" \
  TERMINOLOGY_TOP_K="${TERMINOLOGY_TOP_K:-1}" \
  LEARNING_RATE="${GSPO_LEARNING_RATE:-3e-7}" \
  WARMUP_RATIO="${GSPO_WARMUP_RATIO:-0.0}" \
  NUM_TRAIN_EPOCHS="${GSPO_NUM_TRAIN_EPOCHS:-1}" \
  TRAIN_BATCH_SIZE="${GSPO_TRAIN_BATCH_SIZE:-4}" \
  EVAL_BATCH_SIZE="${GSPO_EVAL_BATCH_SIZE:-4}" \
  GRADIENT_ACCUMULATION_STEPS="${GSPO_GRADIENT_ACCUMULATION_STEPS:-2}" \
  NUM_GENERATIONS="${GSPO_NUM_GENERATIONS:-4}" \
  LOGGING_STEPS="${GSPO_LOGGING_STEPS:-4}" \
  TEMPERATURE="${GSPO_TEMPERATURE:-0.70}" \
  TOP_P="${GSPO_TOP_P:-0.90}" \
  MAX_STEPS="${GSPO_MAX_STEPS:-24}" \
  MAX_TRAIN_SAMPLES="${GSPO_MAX_TRAIN_SAMPLES:-384}" \
  MAX_EVAL_SAMPLES="${GSPO_MAX_EVAL_SAMPLES:-64}" \
  EVAL_STEPS="${GSPO_EVAL_STEPS:-8}" \
  SAVE_STEPS="${GSPO_SAVE_STEPS:-8}" \
  bash experiments/gspo/run_2511_learned_verifier_gspo.sh

mkdir -p "$EVAL_DIR"
run_root="$CANARY_OUTPUT_DIR/chanka_gspo"
for item in checkpoint-8 checkpoint-16 checkpoint-24 final_gspo_lora; do
  adapter_path="$run_root/$item"
  [[ -d "$adapter_path" ]] || continue
  label="terminology_prompt_${item//-/_}"
  python scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$adapter_path" \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "${TERMINOLOGY_TOP_K:-1}" \
    --output-json "$EVAL_DIR/${label}_metrics.json" \
    --predictions-jsonl "$EVAL_DIR/${label}_predictions.jsonl" \
    --batch-size 1
done

python scripts/summarize_gspo_checkpoint_evals.py "$EVAL_DIR"
cat "$EVAL_DIR/summary.md"
