#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-followup-gspo)}"

BASE_MODEL="${BASE_MODEL:-}"
META_VERIFIER_ADAPTER="${META_VERIFIER_ADAPTER:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_followup_${STAMP}}"
PREDICTIONS_JSONL="${PREDICTIONS_JSONL:-${OUTPUT_DIR}/chanka_gspo/final_predictions.jsonl}"
GSPO_META_JSONL="${GSPO_META_JSONL:-${OUTPUT_DIR}/chanka_gspo/meta_hardcases_from_followup_gspo_eval.jsonl}"
GSPO_META_SUMMARY_JSON="${GSPO_META_SUMMARY_JSON:-${OUTPUT_DIR}/chanka_gspo/meta_hardcases_from_followup_gspo_eval.summary.json}"
MINE_GSPO_META="${MINE_GSPO_META:-true}"

cd "$ROOT_DIR"

is_truthy() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" ]]
}

if [[ -z "$BASE_MODEL" ]]; then
  echo "BASE_MODEL is required. Use the frontier thinking SFT adapter or previous policy adapter." >&2
  exit 1
fi

if [[ -z "$META_VERIFIER_ADAPTER" ]]; then
  echo "META_VERIFIER_ADAPTER is required. Use run_next_meta_verifier_from_hardcases.sh first." >&2
  exit 1
fi

BASE_MODEL="$BASE_MODEL" \
META_VERIFIER_ADAPTER="$META_VERIFIER_ADAPTER" \
OUTPUT_DIR="$OUTPUT_DIR" \
PREDICTIONS_JSONL="$PREDICTIONS_JSONL" \
MAX_STEPS="${GSPO_MAX_STEPS:-8}" \
EVAL_STEPS="${GSPO_EVAL_STEPS:-8}" \
SAVE_STEPS="${GSPO_SAVE_STEPS:-8}" \
MAX_TRAIN_SAMPLES="${GSPO_MAX_TRAIN_SAMPLES:-256}" \
MAX_EVAL_SAMPLES="${GSPO_MAX_EVAL_SAMPLES:-32}" \
TRAINER_EVAL="${GSPO_TRAINER_EVAL:-false}" \
FINAL_METRICS_MAX_SAMPLES="${GSPO_FINAL_METRICS_MAX_SAMPLES:-16}" \
FINAL_GENERATION_BATCH_SIZE="${GSPO_FINAL_GENERATION_BATCH_SIZE:-8}" \
experiments/gspo/run_2511_self_verifiable_thinking_translation.sh

if is_truthy "$MINE_GSPO_META" && [[ -f "$PREDICTIONS_JSONL" ]]; then
  GSPO_META_ARGS=()
  if [[ -n "${GSPO_META_MAX_RECORDS:-}" ]]; then
    GSPO_META_ARGS+=(--max-records "$GSPO_META_MAX_RECORDS")
  fi
  "$PYTHON" scripts/build_meta_verifier_from_self_outputs.py \
    --self-predictions-jsonl "$PREDICTIONS_JSONL" \
    --output-jsonl "$GSPO_META_JSONL" \
    --summary-json "$GSPO_META_SUMMARY_JSON" \
    --min-quality-gap "${GSPO_META_MIN_QUALITY_GAP:-0.20}" \
    "${GSPO_META_ARGS[@]}"
elif is_truthy "$MINE_GSPO_META"; then
  echo "No follow-up GSPO predictions JSONL found at $PREDICTIONS_JSONL; skipping meta hardcase mining."
fi

echo "Follow-up GSPO output: $OUTPUT_DIR"
echo "Follow-up GSPO meta hardcases: $GSPO_META_JSONL"
