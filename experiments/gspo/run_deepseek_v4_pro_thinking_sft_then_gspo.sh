#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-deepseek-v4-pro-thinking)}"

BASE_ADAPTER="${BASE_ADAPTER:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36}"
DATA_DIR="${DATA_DIR:-outputs/frontier_thinking_data_${STAMP}}"
FRONTIER_JSONL="${FRONTIER_JSONL:-${DATA_DIR}/deepseek_v4_pro_thinking_sft.jsonl}"
FRONTIER_FAILURES_JSONL="${FRONTIER_FAILURES_JSONL:-${DATA_DIR}/deepseek_v4_pro_thinking_failures.jsonl}"
FRONTIER_SUMMARY_JSON="${FRONTIER_SUMMARY_JSON:-${DATA_DIR}/deepseek_v4_pro_thinking_sft.summary.json}"
THINKING_SFT_OUTPUT_DIR="${THINKING_SFT_OUTPUT_DIR:-outputs/deepseek_v4_pro_thinking_sft_${STAMP}}"
THINKING_SFT_ADAPTER="${THINKING_SFT_ADAPTER:-${THINKING_SFT_OUTPUT_DIR}/final_lora}"
GSPO_OUTPUT_DIR="${GSPO_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_deepseek_seeded_${STAMP}}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"
SFT_EVAL_DIR="${SFT_EVAL_DIR:-${THINKING_SFT_OUTPUT_DIR}/sft_only_eval}"
SFT_EVAL_JSON="${SFT_EVAL_JSON:-${SFT_EVAL_DIR}/metrics.json}"
SFT_EVAL_PREDICTIONS="${SFT_EVAL_PREDICTIONS:-${SFT_EVAL_DIR}/predictions.jsonl}"
SFT_META_JSONL="${SFT_META_JSONL:-${SFT_EVAL_DIR}/meta_hardcases_from_sft_eval.jsonl}"
SFT_META_SUMMARY_JSON="${SFT_META_SUMMARY_JSON:-${SFT_EVAL_DIR}/meta_hardcases_from_sft_eval.summary.json}"
MINE_SFT_META="${MINE_SFT_META:-true}"
RUN_GSPO="${RUN_GSPO:-true}"
MIN_SFT_CHRF_FOR_GSPO="${MIN_SFT_CHRF_FOR_GSPO:-35}"
MIN_SFT_FORMAT_FOR_GSPO="${MIN_SFT_FORMAT_FOR_GSPO:-60}"
MIN_FRONTIER_ROWS_FOR_SFT="${MIN_FRONTIER_ROWS_FOR_SFT:-64}"
MIN_FRONTIER_ACCEPT_RATE="${MIN_FRONTIER_ACCEPT_RATE:-0.50}"

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

FRONTIER_ARGS=()
if [[ "${FRONTIER_AUDIT:-true}" == "true" || "${FRONTIER_AUDIT:-true}" == "1" || "${FRONTIER_AUDIT:-true}" == "yes" ]]; then
  FRONTIER_ARGS+=(--audit)
fi
if [[ -n "${FRONTIER_AUDIT_MODEL:-}" ]]; then
  FRONTIER_ARGS+=(--audit-model "$FRONTIER_AUDIT_MODEL")
fi
if [[ -n "${FRONTIER_AUDIT_MIN_SCORE:-}" ]]; then
  FRONTIER_ARGS+=(--audit-min-score "$FRONTIER_AUDIT_MIN_SCORE")
fi
if [[ "${FRONTIER_RESUME:-true}" == "false" || "${FRONTIER_RESUME:-true}" == "0" || "${FRONTIER_RESUME:-true}" == "no" ]]; then
  FRONTIER_ARGS+=(--no-resume)
fi
if [[ "${FRONTIER_RETRY_FAILURES:-false}" == "true" || "${FRONTIER_RETRY_FAILURES:-false}" == "1" || "${FRONTIER_RETRY_FAILURES:-false}" == "yes" ]]; then
  FRONTIER_ARGS+=(--retry-failures)
fi

"$PYTHON" scripts/build_frontier_thinking_sft_jsonl.py \
  --output-jsonl "$FRONTIER_JSONL" \
  --failures-jsonl "$FRONTIER_FAILURES_JSONL" \
  --summary-json "$FRONTIER_SUMMARY_JSON" \
  --model "${FRONTIER_MODEL:-deepseek-v4-pro}" \
  --reasoning-effort "${FRONTIER_REASONING_EFFORT:-max}" \
  --max-rows "${FRONTIER_MAX_ROWS:-128}" \
  --offset "${FRONTIER_OFFSET:-0}" \
  --sleep-seconds "${FRONTIER_SLEEP_SECONDS:-0}" \
  --max-retries "${FRONTIER_MAX_RETRIES:-3}" \
  --max-output-tokens "${FRONTIER_MAX_OUTPUT_TOKENS:-512}" \
  "${FRONTIER_ARGS[@]}"

"$PYTHON" scripts/check_frontier_thinking_data.py \
  --summary-json "$FRONTIER_SUMMARY_JSON" \
  --output-jsonl "$FRONTIER_JSONL" \
  --failures-jsonl "$FRONTIER_FAILURES_JSONL" \
  --min-written-rows "$MIN_FRONTIER_ROWS_FOR_SFT" \
  --min-accept-rate "$MIN_FRONTIER_ACCEPT_RATE"

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

"$PYTHON" scripts/evaluate_gspo_checkpoint.py \
  --adapter-path "$THINKING_SFT_ADAPTER" \
  --output-json "$SFT_EVAL_JSON" \
  --predictions-jsonl "$SFT_EVAL_PREDICTIONS" \
  --max-seq-length "${SFT_EVAL_MAX_SEQ_LENGTH:-384}" \
  --max-completion-length "${SFT_EVAL_MAX_COMPLETION_LENGTH:-112}" \
  --batch-size "${SFT_EVAL_BATCH_SIZE:-8}" \
  --max-eval-samples "${SFT_EVAL_MAX_ROWS:-64}" \
  --self-verification-thinking-output \
  --terminology-file "$TERMINOLOGY_FILE" \
  --terminology-top-k "${TERMINOLOGY_TOP_K:-1}" \
  --progress-every "${SFT_EVAL_PROGRESS_EVERY:-16}"

if [[ "$MINE_SFT_META" == "true" || "$MINE_SFT_META" == "1" || "$MINE_SFT_META" == "yes" ]]; then
  SFT_META_ARGS=()
  if [[ -n "${SFT_META_MAX_RECORDS:-}" ]]; then
    SFT_META_ARGS+=(--max-records "$SFT_META_MAX_RECORDS")
  fi
  "$PYTHON" scripts/build_meta_verifier_from_self_outputs.py \
    --self-predictions-jsonl "$SFT_EVAL_PREDICTIONS" \
    --output-jsonl "$SFT_META_JSONL" \
    --summary-json "$SFT_META_SUMMARY_JSON" \
    --min-quality-gap "${SFT_META_MIN_QUALITY_GAP:-0.20}" \
    "${SFT_META_ARGS[@]}"
fi

if [[ "$RUN_GSPO" == "false" || "$RUN_GSPO" == "0" || "$RUN_GSPO" == "no" ]]; then
  echo "RUN_GSPO=$RUN_GSPO; stopping after SFT-only evaluation at $SFT_EVAL_JSON"
  exit 0
fi

if ! SFT_EVAL_JSON="$SFT_EVAL_JSON" \
  MIN_SFT_CHRF_FOR_GSPO="$MIN_SFT_CHRF_FOR_GSPO" \
  MIN_SFT_FORMAT_FOR_GSPO="$MIN_SFT_FORMAT_FOR_GSPO" \
  "$PYTHON" - <<'PY'
import json
import os
import sys
from pathlib import Path

metrics = json.loads(Path(os.environ["SFT_EVAL_JSON"]).read_text())
chrf = float(metrics.get("chrf++", 0.0))
format_rate = float(metrics.get("self_verification_required_format_rate", 0.0))
min_chrf = float(os.environ["MIN_SFT_CHRF_FOR_GSPO"])
min_format = float(os.environ["MIN_SFT_FORMAT_FOR_GSPO"])
print(
    f"SFT-only gate: chrF++={chrf:.4f} required>={min_chrf:.4f}; "
    f"format={format_rate:.2f}% required>={min_format:.2f}%"
)
if chrf < min_chrf or format_rate < min_format:
    sys.exit(1)
PY
then
  echo "SFT-only gate failed; skipping GSPO to avoid another translation collapse."
  exit 0
fi

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
