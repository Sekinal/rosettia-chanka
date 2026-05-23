#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-deepseek-v4-pro-thinking)}"

BASE_ADAPTER="${BASE_ADAPTER:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36}"
DATA_DIR="${DATA_DIR:-outputs/frontier_thinking_data_${STAMP}}"
FRONTIER_JSONL="${FRONTIER_JSONL:-${DATA_DIR}/deepseek_v4_pro_thinking_sft.jsonl}"
THINKING_SFT_OUTPUT_DIR="${THINKING_SFT_OUTPUT_DIR:-outputs/deepseek_v4_pro_thinking_sft_${STAMP}}"
THINKING_SFT_ADAPTER="${THINKING_SFT_ADAPTER:-${THINKING_SFT_OUTPUT_DIR}/final_lora}"
GSPO_OUTPUT_DIR="${GSPO_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_deepseek_seeded_${STAMP}}"
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

"$PYTHON" scripts/build_frontier_thinking_sft_jsonl.py \
  --output-jsonl "$FRONTIER_JSONL" \
  --model "${FRONTIER_MODEL:-deepseek-v4-pro}" \
  --reasoning-effort "${FRONTIER_REASONING_EFFORT:-max}" \
  --max-rows "${FRONTIER_MAX_ROWS:-128}" \
  --offset "${FRONTIER_OFFSET:-0}" \
  --sleep-seconds "${FRONTIER_SLEEP_SECONDS:-0}" \
  --max-retries "${FRONTIER_MAX_RETRIES:-3}" \
  --max-output-tokens "${FRONTIER_MAX_OUTPUT_TOKENS:-512}"

"$PYTHON" scripts/train_jsonl_sft_unsloth.py \
  --jsonl "$FRONTIER_JSONL" \
  --source-field source \
  --target-field target \
  --reference-field reference \
  --adapter-path "$BASE_ADAPTER" \
  --output-dir "$THINKING_SFT_OUTPUT_DIR" \
  --max-seq-length "${SFT_MAX_SEQ_LENGTH:-384}" \
  --max-steps "${SFT_MAX_STEPS:-32}" \
  --eval-steps "${SFT_EVAL_STEPS:-8}" \
  --save-steps "${SFT_SAVE_STEPS:-8}" \
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
MAX_STEPS="${GSPO_MAX_STEPS:-8}" \
EVAL_STEPS="${GSPO_EVAL_STEPS:-8}" \
SAVE_STEPS="${GSPO_SAVE_STEPS:-8}" \
MAX_TRAIN_SAMPLES="${GSPO_MAX_TRAIN_SAMPLES:-256}" \
MAX_EVAL_SAMPLES="${GSPO_MAX_EVAL_SAMPLES:-32}" \
TRAINER_EVAL="${GSPO_TRAINER_EVAL:-false}" \
FINAL_METRICS_MAX_SAMPLES="${GSPO_FINAL_METRICS_MAX_SAMPLES:-16}" \
FINAL_GENERATION_BATCH_SIZE="${GSPO_FINAL_GENERATION_BATCH_SIZE:-8}" \
experiments/gspo/run_2511_self_verifiable_thinking_translation.sh
