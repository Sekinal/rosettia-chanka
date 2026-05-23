#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-thinking-generator-gspo)}"

DATA_DIR="${DATA_DIR:-outputs/self_verifiable_translation_data_${STAMP}}"
BASE_ADAPTER="${BASE_ADAPTER:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36}"
THINKING_SFT_OUTPUT_DIR="${THINKING_SFT_OUTPUT_DIR:-outputs/self_verifiable_thinking_generator_sft_${STAMP}}"
THINKING_SFT_ADAPTER="${THINKING_SFT_ADAPTER:-${THINKING_SFT_OUTPUT_DIR}/final_lora}"
GSPO_OUTPUT_DIR="${GSPO_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_sft_seeded_${STAMP}}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"

cd "$ROOT_DIR"

if [[ -z "${META_VERIFIER_ADAPTER:-}" ]]; then
  META_VERIFIER_V3="outputs/chanka_translation_meta_verifier_v3_thinking_20260523-meta-v3-thinking/final_meta_verifier_lora"
  META_VERIFIER_V2="outputs/chanka_translation_meta_verifier_v2_20260523-meta-v2-self/final_meta_verifier_lora"
  if [[ -d "$META_VERIFIER_V3" ]]; then
    META_VERIFIER_ADAPTER="$META_VERIFIER_V3"
  else
    META_VERIFIER_ADAPTER="$META_VERIFIER_V2"
  fi
fi

DATA_ARGS=()
if [[ -n "${MAX_SOURCE_ROWS:-}" ]]; then
  DATA_ARGS+=(--max-rows "$MAX_SOURCE_ROWS")
fi

"$PYTHON" scripts/build_self_verifiable_translation_data.py \
  --output-dir "$DATA_DIR" \
  "${DATA_ARGS[@]}"

"$PYTHON" scripts/train_jsonl_sft_unsloth.py \
  --jsonl "$DATA_DIR/self_verifiable_thinking_generator_sft.jsonl" \
  --source-field source \
  --target-field target \
  --reference-field reference \
  --adapter-path "$BASE_ADAPTER" \
  --output-dir "$THINKING_SFT_OUTPUT_DIR" \
  --max-seq-length "${SFT_MAX_SEQ_LENGTH:-384}" \
  --max-steps "${SFT_MAX_STEPS:-48}" \
  --eval-steps "${SFT_EVAL_STEPS:-12}" \
  --save-steps "${SFT_SAVE_STEPS:-12}" \
  --per-device-train-batch-size "${SFT_TRAIN_BATCH_SIZE:-2}" \
  --per-device-eval-batch-size "${SFT_EVAL_BATCH_SIZE:-2}" \
  --gradient-accumulation-steps "${SFT_GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning-rate "${SFT_LEARNING_RATE:-2e-6}" \
  --lora-r "${SFT_LORA_R:-64}" \
  --lora-alpha "${SFT_LORA_ALPHA:-128}" \
  --prompt-self-verification-thinking \
  --max-spanish-leakage-penalty 1.1 \
  --max-chat-artifact-penalty 1.1 \
  --terminology-file "$TERMINOLOGY_FILE" \
  --terminology-top-k "${TERMINOLOGY_TOP_K:-1}" \
  --logging-steps "${SFT_LOGGING_STEPS:-4}"

BASE_MODEL="$THINKING_SFT_ADAPTER" \
META_VERIFIER_ADAPTER="$META_VERIFIER_ADAPTER" \
OUTPUT_DIR="$GSPO_OUTPUT_DIR" \
MAX_STEPS="${GSPO_MAX_STEPS:-16}" \
EVAL_STEPS="${GSPO_EVAL_STEPS:-8}" \
SAVE_STEPS="${GSPO_SAVE_STEPS:-8}" \
MAX_TRAIN_SAMPLES="${GSPO_MAX_TRAIN_SAMPLES:-256}" \
MAX_EVAL_SAMPLES="${GSPO_MAX_EVAL_SAMPLES:-64}" \
TRAINER_EVAL="${GSPO_TRAINER_EVAL:-true}" \
FINAL_METRICS_MAX_SAMPLES="${GSPO_FINAL_METRICS_MAX_SAMPLES:-}" \
FINAL_GENERATION_BATCH_SIZE="${GSPO_FINAL_GENERATION_BATCH_SIZE:-}" \
experiments/gspo/run_2511_self_verifiable_thinking_translation.sh
