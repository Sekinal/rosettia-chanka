#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-self-verifiable-thinking)}"

BASE_MODEL="${BASE_MODEL:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_${STAMP}}"
METRICS_JSON="${METRICS_JSON:-${OUTPUT_DIR}/chanka_gspo/final_metrics.json}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
META_VERIFIER_ADAPTER="${META_VERIFIER_ADAPTER:-}"

MAX_STEPS="${MAX_STEPS:-16}"
EVAL_STEPS="${EVAL_STEPS:-8}"
SAVE_STEPS="${SAVE_STEPS:-8}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-144}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-7}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"

META_ARGS=()
if [[ -n "$META_VERIFIER_ADAPTER" ]]; then
  META_ARGS+=(--meta-verifier-adapter-path "$META_VERIFIER_ADAPTER")
fi

"$PYTHON" scripts/train_gspo_chanka_unsloth.py \
  --adapter-path "$BASE_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --reward-profile self_verifiable_thinking_translation_2511 \
  --attach-lora \
  --lora-r "${LORA_R:-32}" \
  --lora-alpha "${LORA_ALPHA:-64}" \
  --max-seq-length 256 \
  --max-prompt-length 176 \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  --max-steps "$MAX_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging-steps "${LOGGING_STEPS:-4}" \
  --no-log-completions \
  --per-device-train-batch-size "$TRAIN_BATCH_SIZE" \
  --per-device-eval-batch-size "$EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --num-generations "$NUM_GENERATIONS" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-ratio "${WARMUP_RATIO:-0.03}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.9}" \
  --top-k "${TOP_K:-50}" \
  --repetition-penalty "${REPETITION_PENALTY:-1.05}" \
  --terminology-file "$TERMINOLOGY_FILE" \
  --terminology-top-k "${TERMINOLOGY_TOP_K:-1}" \
  --overlong-ratio-threshold "${OVERLONG_RATIO_THRESHOLD:-3.0}" \
  --overlong-penalty-weight "${OVERLONG_PENALTY_WEIGHT:-0.7}" \
  "${META_ARGS[@]}" \
  --meta-verifier-max-seq-length "${META_VERIFIER_MAX_SEQ_LENGTH:-768}" \
  --meta-verifier-max-new-tokens "${META_VERIFIER_MAX_NEW_TOKENS:-96}" \
  --meta-verifier-batch-size "${META_VERIFIER_BATCH_SIZE:-2}" \
  --metrics-json "$METRICS_JSON"

cat "$METRICS_JSON"
