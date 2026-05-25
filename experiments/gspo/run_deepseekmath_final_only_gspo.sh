#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-deepseekmath-final-only)}"

BASE_MODEL="${BASE_MODEL:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36}"
VERIFIER_ADAPTER_PATH="${VERIFIER_ADAPTER_PATH:-outputs/chanka_translation_verifier_hard_r128/checkpoint-1368}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_deepseekmath_final_only_${STAMP}}"
METRICS_JSON="${METRICS_JSON:-${OUTPUT_DIR}/chanka_gspo/final_metrics.json}"
PREDICTIONS_JSONL="${PREDICTIONS_JSONL:-${OUTPUT_DIR}/chanka_gspo/final_predictions.jsonl}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"

MAX_STEPS="${MAX_STEPS:-16}"
EVAL_STEPS="${EVAL_STEPS:-8}"
SAVE_STEPS="${SAVE_STEPS:-8}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-48}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-7}"
TRAINER_EVAL="${TRAINER_EVAL:-true}"
ATTACH_LORA="${ATTACH_LORA:-true}"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"

SAMPLE_ARGS=()
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  SAMPLE_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "${MAX_EVAL_SAMPLES:-}" ]]; then
  SAMPLE_ARGS+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
fi

SPEED_ARGS=()
if [[ "$TRAINER_EVAL" == "false" || "$TRAINER_EVAL" == "0" || "$TRAINER_EVAL" == "no" ]]; then
  SPEED_ARGS+=(--no-trainer-eval)
fi
if [[ -n "${FINAL_METRICS_MAX_SAMPLES:-}" ]]; then
  SPEED_ARGS+=(--final-metrics-max-samples "$FINAL_METRICS_MAX_SAMPLES")
fi
if [[ -n "${FINAL_GENERATION_BATCH_SIZE:-}" ]]; then
  SPEED_ARGS+=(--final-generation-batch-size "$FINAL_GENERATION_BATCH_SIZE")
fi

ATTACH_ARGS=()
if [[ "$ATTACH_LORA" == "true" || "$ATTACH_LORA" == "1" || "$ATTACH_LORA" == "yes" ]]; then
  ATTACH_ARGS+=(--attach-lora)
fi

"$PYTHON" scripts/train_gspo_chanka_unsloth.py \
  --adapter-path "$BASE_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --reward-profile deepseekmath_final_verifier_2511 \
  --verifier-adapter-path "$VERIFIER_ADAPTER_PATH" \
  --verifier-batch-size "${VERIFIER_REWARD_BATCH_SIZE:-4}" \
  --verifier-max-seq-length "${VERIFIER_MAX_SEQ_LENGTH:-512}" \
  --verifier-max-new-tokens "${VERIFIER_MAX_NEW_TOKENS:-96}" \
  "${ATTACH_ARGS[@]}" \
  --lora-r "${LORA_R:-32}" \
  --lora-alpha "${LORA_ALPHA:-64}" \
  --max-seq-length 192 \
  --max-prompt-length 144 \
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
  "${SAMPLE_ARGS[@]}" \
  "${SPEED_ARGS[@]}" \
  --overlong-ratio-threshold "${OVERLONG_RATIO_THRESHOLD:-3.0}" \
  --overlong-penalty-weight "${OVERLONG_PENALTY_WEIGHT:-0.7}" \
  --metrics-json "$METRICS_JSON" \
  --predictions-jsonl "$PREDICTIONS_JSONL"

cat "$METRICS_JSON"
