#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-deepseek-v4-pro-sft-seed-gspo)}"

THINKING_SFT_OUTPUT_DIR="${THINKING_SFT_OUTPUT_DIR:-}"
if [[ -z "$THINKING_SFT_OUTPUT_DIR" ]]; then
  echo "THINKING_SFT_OUTPUT_DIR is required. Point it at the SFT output from run_deepseek_v4_pro_sft_from_frontier_data.sh." >&2
  exit 2
fi

THINKING_SFT_ADAPTER="${THINKING_SFT_ADAPTER:-${THINKING_SFT_OUTPUT_DIR}/final_lora}"
SFT_EVAL_DIR="${SFT_EVAL_DIR:-${THINKING_SFT_OUTPUT_DIR}/sft_only_eval}"
SFT_EVAL_JSON="${SFT_EVAL_JSON:-${SFT_EVAL_DIR}/metrics.json}"
SFT_META_JSONL="${SFT_META_JSONL:-${SFT_EVAL_DIR}/meta_hardcases_from_sft_eval.jsonl}"
SFT_META_SUMMARY_JSON="${SFT_META_SUMMARY_JSON:-${SFT_EVAL_DIR}/meta_hardcases_from_sft_eval.summary.json}"
SFT_CYCLE_MANIFEST_JSON="${SFT_CYCLE_MANIFEST_JSON:-${THINKING_SFT_OUTPUT_DIR}/cycle_manifest.json}"
SFT_MANIFEST_GATE_JSON="${SFT_MANIFEST_GATE_JSON:-${THINKING_SFT_OUTPUT_DIR}/sft_seed_manifest_gate.json}"

META_VERIFIER_V3="${META_VERIFIER_V3:-outputs/chanka_translation_meta_verifier_v3_thinking_20260523-meta-v3-thinking/final_meta_verifier_lora}"
META_VERIFIER_V2="${META_VERIFIER_V2:-outputs/chanka_translation_meta_verifier_v2_20260523-meta-v2-self/final_meta_verifier_lora}"
META_VERIFIER_ADAPTER="${META_VERIFIER_ADAPTER:-}"
SFT_META_DATA_DIR="${SFT_META_DATA_DIR:-${SFT_EVAL_DIR}/self_verifiable_meta_data}"
SFT_META_OUTPUT_DIR="${SFT_META_OUTPUT_DIR:-outputs/chanka_translation_meta_verifier_deepseek_sft_${STAMP}}"
SFT_META_ADAPTER="${SFT_META_ADAPTER:-${SFT_META_OUTPUT_DIR}/final_meta_verifier_lora}"
TRAIN_SFT_META_VERIFIER="${TRAIN_SFT_META_VERIFIER:-true}"
MIN_SFT_META_RECORDS_FOR_TRAIN="${MIN_SFT_META_RECORDS_FOR_TRAIN:-32}"

GSPO_OUTPUT_DIR="${GSPO_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_deepseek_seeded_${STAMP}}"
GSPO_METRICS_JSON="${GSPO_METRICS_JSON:-${GSPO_OUTPUT_DIR}/chanka_gspo/final_metrics.json}"
GSPO_PREDICTIONS_JSONL="${GSPO_PREDICTIONS_JSONL:-${GSPO_OUTPUT_DIR}/chanka_gspo/final_predictions.jsonl}"
GSPO_PROMOTION_JSON="${GSPO_PROMOTION_JSON:-${GSPO_OUTPUT_DIR}/chanka_gspo/promotion_gate.json}"
GSPO_CYCLE_MANIFEST_JSON="${GSPO_CYCLE_MANIFEST_JSON:-${GSPO_OUTPUT_DIR}/cycle_manifest.json}"
GSPO_META_JSONL="${GSPO_META_JSONL:-${GSPO_OUTPUT_DIR}/chanka_gspo/meta_hardcases_from_gspo_eval.jsonl}"
GSPO_META_SUMMARY_JSON="${GSPO_META_SUMMARY_JSON:-${GSPO_OUTPUT_DIR}/chanka_gspo/meta_hardcases_from_gspo_eval.summary.json}"
FRONTIER_DATA_DIR="${FRONTIER_DATA_DIR:-${DATA_DIR:-}}"
STAGED_STATUS_JSON="${STAGED_STATUS_JSON:-${GSPO_OUTPUT_DIR}/deepseekmath_staged_status.json}"
STAGED_STATUS_MD="${STAGED_STATUS_MD:-${GSPO_OUTPUT_DIR}/deepseekmath_staged_status.md}"

MIN_SFT_CHRF_FOR_GSPO="${MIN_SFT_CHRF_FOR_GSPO:-35}"
MIN_SFT_FORMAT_FOR_GSPO="${MIN_SFT_FORMAT_FOR_GSPO:-60}"
RUN_SFT_GATE="${RUN_SFT_GATE:-true}"
RUN_SFT_MANIFEST_GATE="${RUN_SFT_MANIFEST_GATE:-true}"
RUN_GSPO_PROMOTION_GATE="${RUN_GSPO_PROMOTION_GATE:-true}"
REQUIRE_GSPO_PROMOTION="${REQUIRE_GSPO_PROMOTION:-false}"
MINE_GSPO_META="${MINE_GSPO_META:-true}"

cd "$ROOT_DIR"

is_truthy() {
  [[ "$1" == "true" || "$1" == "1" || "$1" == "yes" ]]
}

write_staged_status() {
  local exit_status=$?
  set +e
  STAGED_FRONTIER_ARGS=()
  if [[ -n "$FRONTIER_DATA_DIR" ]]; then
    STAGED_FRONTIER_ARGS+=(--frontier-dir "$FRONTIER_DATA_DIR")
  fi
  "$PYTHON" scripts/summarize_deepseekmath_staged_run.py \
    "${STAGED_FRONTIER_ARGS[@]}" \
    --sft-dir "$THINKING_SFT_OUTPUT_DIR" \
    --gspo-dir "$GSPO_OUTPUT_DIR" \
    --no-discover \
    --output-json "$STAGED_STATUS_JSON" \
    --output-md "$STAGED_STATUS_MD" \
    >/dev/null
  if [[ -f "$STAGED_STATUS_MD" ]]; then
    echo "Staged status: $STAGED_STATUS_MD"
  fi
  return "$exit_status"
}
trap write_staged_status EXIT

if [[ ! -d "$THINKING_SFT_ADAPTER" ]]; then
  echo "SFT adapter not found: $THINKING_SFT_ADAPTER" >&2
  exit 2
fi
if [[ ! -f "$SFT_EVAL_JSON" ]]; then
  echo "SFT metrics not found: $SFT_EVAL_JSON" >&2
  exit 2
fi
if is_truthy "$RUN_SFT_MANIFEST_GATE"; then
  "$PYTHON" scripts/check_deepseekmath_cycle_manifest.py \
    --manifest-json "$SFT_CYCLE_MANIFEST_JSON" \
    --output-json "$SFT_MANIFEST_GATE_JSON" \
    --no-require-promoted \
    --expected-stage sft_seed \
    --expected-policy-adapter "$THINKING_SFT_ADAPTER" \
    --min-chrf "$MIN_SFT_CHRF_FOR_GSPO" \
    --min-bleu "${MIN_SFT_BLEU_FOR_GSPO:-0}" \
    --min-token-f1 "${MIN_SFT_TOKEN_F1_FOR_GSPO:-0}"
fi

if [[ -z "$META_VERIFIER_ADAPTER" ]]; then
  if [[ -d "$META_VERIFIER_V3" ]]; then
    META_VERIFIER_ADAPTER="$META_VERIFIER_V3"
  elif [[ -d "$META_VERIFIER_V2" ]]; then
    META_VERIFIER_ADAPTER="$META_VERIFIER_V2"
  else
    echo "No default meta-verifier found. Set META_VERIFIER_ADAPTER explicitly." >&2
    exit 2
  fi
fi

if is_truthy "$TRAIN_SFT_META_VERIFIER" && [[ -f "$SFT_META_SUMMARY_JSON" ]]; then
  if SFT_META_SUMMARY_JSON="$SFT_META_SUMMARY_JSON" \
    MIN_SFT_META_RECORDS_FOR_TRAIN="$MIN_SFT_META_RECORDS_FOR_TRAIN" \
    "$PYTHON" - <<'PY'
import json
import os
import sys
from pathlib import Path

summary = json.loads(Path(os.environ["SFT_META_SUMMARY_JSON"]).read_text())
records = int(summary.get("records", 0))
minimum = int(os.environ["MIN_SFT_META_RECORDS_FOR_TRAIN"])
print(f"SFT meta-verifier refresh gate: hardcases={records} required>={minimum}")
if records < minimum:
    sys.exit(1)
PY
  then
    "$PYTHON" scripts/build_self_verifiable_translation_data.py \
      --output-dir "$SFT_META_DATA_DIR" \
      --max-rows "${SFT_META_COLD_START_MAX_ROWS:-256}"

    "$PYTHON" scripts/train_meta_verifier_chanka_unsloth.py \
      --meta-jsonl "${SFT_META_DATA_DIR}/translation_meta_verifier_cold_start.jsonl" \
      --meta-jsonl "$SFT_META_JSONL" \
      --output-dir "$SFT_META_OUTPUT_DIR" \
      --max-seq-length "${SFT_META_VERIFIER_MAX_SEQ_LENGTH:-768}" \
      --max-steps "${SFT_META_VERIFIER_MAX_STEPS:-64}" \
      --eval-steps "${SFT_META_VERIFIER_EVAL_STEPS:-16}" \
      --save-steps "${SFT_META_VERIFIER_SAVE_STEPS:-16}" \
      --per-device-train-batch-size "${SFT_META_VERIFIER_TRAIN_BATCH_SIZE:-4}" \
      --per-device-eval-batch-size "${SFT_META_VERIFIER_EVAL_BATCH_SIZE:-4}" \
      --gradient-accumulation-steps "${SFT_META_VERIFIER_GRADIENT_ACCUMULATION_STEPS:-4}" \
      --learning-rate "${SFT_META_VERIFIER_LEARNING_RATE:-2e-5}" \
      --lora-r "${SFT_META_VERIFIER_LORA_R:-64}" \
      --lora-alpha "${SFT_META_VERIFIER_LORA_ALPHA:-128}" \
      --logging-steps "${SFT_META_VERIFIER_LOGGING_STEPS:-8}"

    META_VERIFIER_ADAPTER="$SFT_META_ADAPTER"
    echo "Using refreshed SFT hardcase meta-verifier: $META_VERIFIER_ADAPTER"
  else
    echo "SFT meta-verifier refresh gate failed; using existing meta-verifier: $META_VERIFIER_ADAPTER"
  fi
fi

if is_truthy "$RUN_SFT_GATE"; then
  SFT_EVAL_JSON="$SFT_EVAL_JSON" \
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
fi

BASE_MODEL="$THINKING_SFT_ADAPTER" \
META_VERIFIER_ADAPTER="$META_VERIFIER_ADAPTER" \
OUTPUT_DIR="$GSPO_OUTPUT_DIR" \
METRICS_JSON="$GSPO_METRICS_JSON" \
PREDICTIONS_JSONL="$GSPO_PREDICTIONS_JSONL" \
ATTACH_LORA=false \
MAX_STEPS="${GSPO_MAX_STEPS:-8}" \
EVAL_STEPS="${GSPO_EVAL_STEPS:-8}" \
SAVE_STEPS="${GSPO_SAVE_STEPS:-8}" \
MAX_TRAIN_SAMPLES="${GSPO_MAX_TRAIN_SAMPLES:-256}" \
MAX_EVAL_SAMPLES="${GSPO_MAX_EVAL_SAMPLES:-32}" \
TRAINER_EVAL="${GSPO_TRAINER_EVAL:-false}" \
FINAL_METRICS_MAX_SAMPLES="${GSPO_FINAL_METRICS_MAX_SAMPLES:-16}" \
FINAL_GENERATION_BATCH_SIZE="${GSPO_FINAL_GENERATION_BATCH_SIZE:-8}" \
experiments/gspo/run_2511_self_verifiable_thinking_translation.sh

GSPO_PROMOTION_FAILED=0
if is_truthy "$RUN_GSPO_PROMOTION_GATE"; then
  if "$PYTHON" scripts/check_policy_iteration_metrics.py \
    --candidate-json "$GSPO_METRICS_JSON" \
    --baseline-json "$SFT_EVAL_JSON" \
    --output-json "$GSPO_PROMOTION_JSON" \
    --min-chrf "${PROMOTION_MIN_CHRF:-35}" \
    --min-bleu "${PROMOTION_MIN_BLEU:-8}" \
    --min-token-f1 "${PROMOTION_MIN_TOKEN_F1:-15}" \
    --max-ter "${PROMOTION_MAX_TER:-120}" \
    --min-format-rate "${PROMOTION_MIN_FORMAT_RATE:-50}" \
    --max-false-confidence-rate "${PROMOTION_MAX_FALSE_CONFIDENCE_RATE:-95}" \
    --max-missing-score-rate "${PROMOTION_MAX_MISSING_SCORE_RATE:-50}" \
    --min-chrf-delta "${PROMOTION_MIN_CHRF_DELTA:--1}" \
    --min-bleu-delta "${PROMOTION_MIN_BLEU_DELTA:--1}" \
    --min-token-f1-delta "${PROMOTION_MIN_TOKEN_F1_DELTA:--1}" \
    --max-false-confidence-delta "${PROMOTION_MAX_FALSE_CONFIDENCE_DELTA:-5}"
  then
    echo "Initial GSPO promotion gate passed: $GSPO_PROMOTION_JSON"
  else
    GSPO_PROMOTION_FAILED=1
    echo "Initial GSPO promotion gate failed: $GSPO_PROMOTION_JSON"
  fi
fi

if is_truthy "$MINE_GSPO_META" && [[ -f "$GSPO_PREDICTIONS_JSONL" ]]; then
  GSPO_META_ARGS=()
  if [[ -n "${GSPO_META_MAX_RECORDS:-}" ]]; then
    GSPO_META_ARGS+=(--max-records "$GSPO_META_MAX_RECORDS")
  fi
  "$PYTHON" scripts/build_meta_verifier_from_self_outputs.py \
    --self-predictions-jsonl "$GSPO_PREDICTIONS_JSONL" \
    --output-jsonl "$GSPO_META_JSONL" \
    --summary-json "$GSPO_META_SUMMARY_JSON" \
    --min-quality-gap "${GSPO_META_MIN_QUALITY_GAP:-0.20}" \
    "${GSPO_META_ARGS[@]}"
fi

MANIFEST_INPUT_ARGS=()
if [[ -f "$SFT_META_JSONL" ]]; then
  MANIFEST_INPUT_ARGS+=(--input-hardcase-jsonl "$SFT_META_JSONL")
fi

"$PYTHON" scripts/write_deepseekmath_cycle_manifest.py \
  --output-json "$GSPO_CYCLE_MANIFEST_JSON" \
  --stamp "$STAMP" \
  --stage initial_gspo \
  --base-model "$THINKING_SFT_ADAPTER" \
  --policy-adapter "${GSPO_OUTPUT_DIR}/chanka_gspo/final_gspo_lora" \
  --meta-verifier-adapter "$META_VERIFIER_ADAPTER" \
  --meta-output-dir "$SFT_META_OUTPUT_DIR" \
  --followup-output-dir "$GSPO_OUTPUT_DIR" \
  --metrics-json "$GSPO_METRICS_JSON" \
  --promotion-json "$GSPO_PROMOTION_JSON" \
  --predictions-jsonl "$GSPO_PREDICTIONS_JSONL" \
  --output-hardcase-jsonl "$GSPO_META_JSONL" \
  --baseline-metrics-json "$SFT_EVAL_JSON" \
  "${MANIFEST_INPUT_ARGS[@]}"

echo "Initial GSPO output: $GSPO_OUTPUT_DIR"
echo "Initial GSPO metrics: $GSPO_METRICS_JSON"
echo "Initial GSPO meta hardcases: $GSPO_META_JSONL"
echo "Initial GSPO cycle manifest: $GSPO_CYCLE_MANIFEST_JSON"
echo "Staged status report: $STAGED_STATUS_MD"

if [[ "$GSPO_PROMOTION_FAILED" -eq 1 ]] && is_truthy "$REQUIRE_GSPO_PROMOTION"; then
  exit 1
fi
