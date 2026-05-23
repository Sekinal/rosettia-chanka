#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-4b-full-sft-gspo-canary)}"

WAIT_FOR_PID="${WAIT_FOR_PID:-}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_qwen35_4b_full_sft_lora_gspo_${STAMP}}"
EVAL_DIR="${EVAL_DIR:-outputs/gspo_checkpoint_evals/${STAMP}-qwen35-4b-full-sft-lora-gspo}"
VERIFIER_ADAPTER_PATH="${VERIFIER_ADAPTER_PATH:-outputs/chanka_translation_verifier_hard_r128/checkpoint-1368}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"

cd "$ROOT_DIR"

if [[ -n "$WAIT_FOR_PID" ]]; then
  while ps -p "$WAIT_FOR_PID" >/dev/null 2>&1; do
    echo "$(date -u +%FT%TZ) waiting for PID $WAIT_FOR_PID before 4B full-SFT GSPO canary"
    sleep 300
  done
fi

"$PYTHON" scripts/train_gspo_chanka_unsloth.py \
  --adapter-path "$BASE_CHECKPOINT" \
  --attach-lora \
  --lora-r "${LORA_R:-32}" \
  --lora-alpha "${LORA_ALPHA:-64}" \
  --lora-dropout "${LORA_DROPOUT:-0.0}" \
  --output-dir "$OUTPUT_DIR" \
  --reward-profile "${REWARD_PROFILE:-learned_verifier_vibe_2511}" \
  --verifier-adapter-path "$VERIFIER_ADAPTER_PATH" \
  --verifier-batch-size "${VERIFIER_BATCH_SIZE:-2}" \
  --terminology-file "$TERMINOLOGY_FILE" \
  --terminology-top-k "${TERMINOLOGY_TOP_K:-1}" \
  --learning-rate "${LEARNING_RATE:-2e-7}" \
  --warmup-ratio "${WARMUP_RATIO:-0.0}" \
  --max-steps "${MAX_STEPS:-16}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES:-256}" \
  --max-eval-samples "${MAX_EVAL_SAMPLES:-64}" \
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE:-2}" \
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE:-2}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --num-generations "${NUM_GENERATIONS:-4}" \
  --temperature "${TEMPERATURE:-0.70}" \
  --top-p "${TOP_P:-0.90}" \
  --eval-steps "${EVAL_STEPS:-8}" \
  --save-steps "${SAVE_STEPS:-8}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-0}" \
  --logging-steps "${LOGGING_STEPS:-4}" \
  --no-log-completions

mkdir -p "$EVAL_DIR"
run_root="$OUTPUT_DIR/chanka_gspo"
for item in checkpoint-8 checkpoint-16 final_gspo_lora; do
  adapter_path="$run_root/$item"
  [[ -d "$adapter_path" ]] || continue
  label="qwen35_4b_full_sft_gspo_${item//-/_}"
  "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$adapter_path" \
    --output-json "$EVAL_DIR/${label}_metrics.json" \
    --predictions-jsonl "$EVAL_DIR/${label}_predictions.jsonl" \
    --batch-size 1 \
    --max-completion-length 80 \
    --strip-chat-artifacts \
    --terminology-file "$TERMINOLOGY_FILE" \
    --terminology-top-k "${TERMINOLOGY_TOP_K:-1}"
done

"$PYTHON" scripts/write_nested_metrics_summary.py "$EVAL_DIR" --metric-field source_copy_ratio
