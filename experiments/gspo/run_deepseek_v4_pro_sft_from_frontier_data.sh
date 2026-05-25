#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-deepseek-v4-pro-frontier-sft)}"

BASE_ADAPTER="${BASE_ADAPTER:-outputs/full_sft_sweeps/20260523-qwen35-4b-full-sft-lr-followups/lr_2em6_48steps/chanka/checkpoint-36}"
DATA_DIR="${DATA_DIR:-}"
if [[ -z "$DATA_DIR" && -z "${FRONTIER_JSONL:-}" ]]; then
  echo "Set DATA_DIR to a paid-smoke frontier data directory, or set FRONTIER_JSONL directly." >&2
  exit 2
fi

if [[ -n "$DATA_DIR" ]]; then
  FRONTIER_JSONL="${FRONTIER_JSONL:-${DATA_DIR}/deepseek_v4_pro_thinking_sft.jsonl}"
  FRONTIER_FAILURES_JSONL="${FRONTIER_FAILURES_JSONL:-${DATA_DIR}/deepseek_v4_pro_thinking_failures.jsonl}"
  FRONTIER_SUMMARY_JSON="${FRONTIER_SUMMARY_JSON:-${DATA_DIR}/deepseek_v4_pro_thinking_sft.summary.json}"
  FRONTIER_REPORT_JSON="${FRONTIER_REPORT_JSON:-${DATA_DIR}/deepseek_v4_pro_thinking_report.json}"
  FRONTIER_REPORT_MD="${FRONTIER_REPORT_MD:-${DATA_DIR}/deepseek_v4_pro_thinking_report.md}"
  FRONTIER_PAID_GATE_JSON="${FRONTIER_PAID_GATE_JSON:-${DATA_DIR}/deepseek_v4_pro_paid_smoke_gate.json}"
else
  FRONTIER_FAILURES_JSONL="${FRONTIER_FAILURES_JSONL:-}"
  FRONTIER_SUMMARY_JSON="${FRONTIER_SUMMARY_JSON:-}"
  FRONTIER_REPORT_JSON="${FRONTIER_REPORT_JSON:-outputs/frontier_thinking_data_${STAMP}/deepseek_v4_pro_thinking_report.json}"
  FRONTIER_REPORT_MD="${FRONTIER_REPORT_MD:-outputs/frontier_thinking_data_${STAMP}/deepseek_v4_pro_thinking_report.md}"
  FRONTIER_PAID_GATE_JSON="${FRONTIER_PAID_GATE_JSON:-}"
fi

THINKING_SFT_OUTPUT_DIR="${THINKING_SFT_OUTPUT_DIR:-outputs/deepseek_v4_pro_thinking_sft_${STAMP}}"
THINKING_SFT_ADAPTER="${THINKING_SFT_ADAPTER:-${THINKING_SFT_OUTPUT_DIR}/final_lora}"
TERMINOLOGY_FILE="${TERMINOLOGY_FILE:-clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet}"

SFT_EVAL_DIR="${SFT_EVAL_DIR:-${THINKING_SFT_OUTPUT_DIR}/sft_only_eval}"
SFT_EVAL_JSON="${SFT_EVAL_JSON:-${SFT_EVAL_DIR}/metrics.json}"
SFT_EVAL_PREDICTIONS="${SFT_EVAL_PREDICTIONS:-${SFT_EVAL_DIR}/predictions.jsonl}"
SFT_META_JSONL="${SFT_META_JSONL:-${SFT_EVAL_DIR}/meta_hardcases_from_sft_eval.jsonl}"
SFT_META_SUMMARY_JSON="${SFT_META_SUMMARY_JSON:-${SFT_EVAL_DIR}/meta_hardcases_from_sft_eval.summary.json}"
SFT_PROMOTION_JSON="${SFT_PROMOTION_JSON:-${SFT_EVAL_DIR}/promotion_gate.json}"
SFT_CYCLE_MANIFEST_JSON="${SFT_CYCLE_MANIFEST_JSON:-${THINKING_SFT_OUTPUT_DIR}/cycle_manifest.json}"
SFT_BASELINE_EVAL_DIR="${SFT_BASELINE_EVAL_DIR:-${THINKING_SFT_OUTPUT_DIR}/base_adapter_thinking_eval}"
SFT_BASELINE_METRICS_JSON="${SFT_BASELINE_METRICS_JSON:-${SFT_BASELINE_EVAL_DIR}/metrics.json}"
SFT_BASELINE_PREDICTIONS="${SFT_BASELINE_PREDICTIONS:-${SFT_BASELINE_EVAL_DIR}/predictions.jsonl}"
STAGED_STATUS_JSON="${STAGED_STATUS_JSON:-${THINKING_SFT_OUTPUT_DIR}/deepseekmath_staged_status.json}"
STAGED_STATUS_MD="${STAGED_STATUS_MD:-${THINKING_SFT_OUTPUT_DIR}/deepseekmath_staged_status.md}"
PAID_FRONTIER_GATE_CHECK_JSON="${PAID_FRONTIER_GATE_CHECK_JSON:-${THINKING_SFT_OUTPUT_DIR}/paid_frontier_gate_check.json}"

MIN_FRONTIER_ROWS_FOR_SFT="${MIN_FRONTIER_ROWS_FOR_SFT:-8}"
MIN_FRONTIER_ACCEPT_RATE="${MIN_FRONTIER_ACCEPT_RATE:-0.50}"
MIN_FRONTIER_PRIMITIVE_TAGS_PER_ROW="${MIN_FRONTIER_PRIMITIVE_TAGS_PER_ROW:-2}"
MIN_FRONTIER_PRIMITIVE_ROW_RATE="${MIN_FRONTIER_PRIMITIVE_ROW_RATE:-0.90}"
MIN_FRONTIER_DISTINCT_PRIMITIVES="${MIN_FRONTIER_DISTINCT_PRIMITIVES:-4}"
MIN_FRONTIER_EXPECTED_PRIMITIVE_COVERAGE="${MIN_FRONTIER_EXPECTED_PRIMITIVE_COVERAGE:-0.95}"
MIN_FRONTIER_ANALYSIS_WORDS="${MIN_FRONTIER_ANALYSIS_WORDS:-6}"
MIN_FRONTIER_ANALYSIS_WORD_ROW_RATE="${MIN_FRONTIER_ANALYSIS_WORD_ROW_RATE:-0.75}"
MIN_FRONTIER_SPECIFIC_ANALYSIS_RATE="${MIN_FRONTIER_SPECIFIC_ANALYSIS_RATE:-0.75}"
MIN_FRONTIER_AUDITED_ROW_RATE="${MIN_FRONTIER_AUDITED_ROW_RATE:-1.0}"
MIN_FRONTIER_AUDIT_PASS_RATE="${MIN_FRONTIER_AUDIT_PASS_RATE:-1.0}"
MIN_FRONTIER_AVG_AUDIT_SCORE="${MIN_FRONTIER_AVG_AUDIT_SCORE:-0.75}"
MIN_FRONTIER_REFERENCE_FINAL_MATCH_RATE="${MIN_FRONTIER_REFERENCE_FINAL_MATCH_RATE:-1.0}"

RUN_FRONTIER_REPORT="${RUN_FRONTIER_REPORT:-true}"
REQUIRE_PAID_FRONTIER_GATE="${REQUIRE_PAID_FRONTIER_GATE:-true}"
RUN_SFT_TRAINING="${RUN_SFT_TRAINING:-true}"
RUN_SFT_BASELINE_EVAL="${RUN_SFT_BASELINE_EVAL:-true}"
RUN_SFT_PROMOTION_GATE="${RUN_SFT_PROMOTION_GATE:-true}"
MINE_SFT_META="${MINE_SFT_META:-true}"
REQUIRE_SFT_PROMOTION="${REQUIRE_SFT_PROMOTION:-false}"

cd "$ROOT_DIR"

write_staged_status() {
  local exit_status=$?
  set +e
  STAGED_FRONTIER_ARGS=()
  if [[ -n "$DATA_DIR" ]]; then
    STAGED_FRONTIER_ARGS+=(--frontier-dir "$DATA_DIR")
  fi
  "$PYTHON" scripts/summarize_deepseekmath_staged_run.py \
    "${STAGED_FRONTIER_ARGS[@]}" \
    --sft-dir "$THINKING_SFT_OUTPUT_DIR" \
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

if [[ ! -f "$FRONTIER_JSONL" ]]; then
  echo "Frontier JSONL not found: $FRONTIER_JSONL" >&2
  exit 2
fi

REPORT_ARGS=()
if [[ -n "$FRONTIER_FAILURES_JSONL" ]]; then
  REPORT_ARGS+=(--failures-jsonl "$FRONTIER_FAILURES_JSONL")
fi
if [[ -n "$FRONTIER_SUMMARY_JSON" ]]; then
  REPORT_ARGS+=(--summary-json "$FRONTIER_SUMMARY_JSON")
fi

if [[ "$RUN_FRONTIER_REPORT" == "true" || "$RUN_FRONTIER_REPORT" == "1" || "$RUN_FRONTIER_REPORT" == "yes" ]]; then
  "$PYTHON" scripts/report_frontier_thinking_data.py \
    --output-jsonl "$FRONTIER_JSONL" \
    --report-json "$FRONTIER_REPORT_JSON" \
    --report-md "$FRONTIER_REPORT_MD" \
    "${REPORT_ARGS[@]}"
fi

if [[ "$REQUIRE_PAID_FRONTIER_GATE" == "true" || "$REQUIRE_PAID_FRONTIER_GATE" == "1" || "$REQUIRE_PAID_FRONTIER_GATE" == "yes" ]]; then
  if [[ -z "$FRONTIER_PAID_GATE_JSON" ]]; then
    echo "REQUIRE_PAID_FRONTIER_GATE=true but FRONTIER_PAID_GATE_JSON is unset. Use DATA_DIR from a paid smoke run or set REQUIRE_PAID_FRONTIER_GATE=false for explicit debugging." >&2
    exit 2
  fi
  "$PYTHON" scripts/check_deepseek_frontier_paid_gate.py \
    --paid-gate-json "$FRONTIER_PAID_GATE_JSON" \
    --frontier-report-json "$FRONTIER_REPORT_JSON" \
    --output-json "$PAID_FRONTIER_GATE_CHECK_JSON"
fi

GATE_ARGS=()
if [[ -n "$FRONTIER_FAILURES_JSONL" ]]; then
  GATE_ARGS+=(--failures-jsonl "$FRONTIER_FAILURES_JSONL")
fi
if [[ -n "$FRONTIER_SUMMARY_JSON" ]]; then
  GATE_ARGS+=(--summary-json "$FRONTIER_SUMMARY_JSON")
fi

"$PYTHON" scripts/check_frontier_thinking_data.py \
  --output-jsonl "$FRONTIER_JSONL" \
  "${GATE_ARGS[@]}" \
  --min-written-rows "$MIN_FRONTIER_ROWS_FOR_SFT" \
  --min-accept-rate "$MIN_FRONTIER_ACCEPT_RATE" \
  --min-primitive-tags-per-row "$MIN_FRONTIER_PRIMITIVE_TAGS_PER_ROW" \
  --min-primitive-row-rate "$MIN_FRONTIER_PRIMITIVE_ROW_RATE" \
  --min-distinct-primitives "$MIN_FRONTIER_DISTINCT_PRIMITIVES" \
  --min-expected-primitive-coverage "$MIN_FRONTIER_EXPECTED_PRIMITIVE_COVERAGE" \
  --min-analysis-words "$MIN_FRONTIER_ANALYSIS_WORDS" \
  --min-analysis-word-row-rate "$MIN_FRONTIER_ANALYSIS_WORD_ROW_RATE" \
  --min-specific-analysis-rate "$MIN_FRONTIER_SPECIFIC_ANALYSIS_RATE" \
  --min-audited-row-rate "$MIN_FRONTIER_AUDITED_ROW_RATE" \
  --min-audit-pass-rate "$MIN_FRONTIER_AUDIT_PASS_RATE" \
  --min-avg-audit-score "$MIN_FRONTIER_AVG_AUDIT_SCORE" \
  --min-reference-final-match-rate "$MIN_FRONTIER_REFERENCE_FINAL_MATCH_RATE"

if [[ "$RUN_SFT_TRAINING" == "false" || "$RUN_SFT_TRAINING" == "0" || "$RUN_SFT_TRAINING" == "no" ]]; then
  echo "RUN_SFT_TRAINING=$RUN_SFT_TRAINING; stopping after frontier report and data gate."
  echo "Frontier report: $FRONTIER_REPORT_MD"
  echo "Staged status report: $STAGED_STATUS_MD"
  exit 0
fi

if [[ "$RUN_SFT_BASELINE_EVAL" == "true" || "$RUN_SFT_BASELINE_EVAL" == "1" || "$RUN_SFT_BASELINE_EVAL" == "yes" ]]; then
  if [[ ! -f "$SFT_BASELINE_METRICS_JSON" || "${SFT_BASELINE_FORCE_EVAL:-false}" == "true" ]]; then
    "$PYTHON" scripts/evaluate_gspo_checkpoint.py \
      --adapter-path "$BASE_ADAPTER" \
      --output-json "$SFT_BASELINE_METRICS_JSON" \
      --predictions-jsonl "$SFT_BASELINE_PREDICTIONS" \
      --max-seq-length "${SFT_EVAL_MAX_SEQ_LENGTH:-384}" \
      --max-completion-length "${SFT_EVAL_MAX_COMPLETION_LENGTH:-112}" \
      --batch-size "${SFT_EVAL_BATCH_SIZE:-8}" \
      --max-eval-samples "${SFT_EVAL_MAX_ROWS:-64}" \
      --self-verification-thinking-output \
      --terminology-file "$TERMINOLOGY_FILE" \
      --terminology-top-k "${TERMINOLOGY_TOP_K:-1}" \
      --progress-every "${SFT_EVAL_PROGRESS_EVERY:-16}"
  else
    echo "Reusing existing SFT baseline metrics: $SFT_BASELINE_METRICS_JSON"
  fi
fi

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

SFT_PROMOTION_BASELINE_ARGS=()
if [[ -f "$SFT_BASELINE_METRICS_JSON" ]]; then
  SFT_PROMOTION_BASELINE_ARGS+=(--baseline-json "$SFT_BASELINE_METRICS_JSON")
fi
SFT_MANIFEST_BASELINE_ARGS=()
if [[ -f "$SFT_BASELINE_METRICS_JSON" ]]; then
  SFT_MANIFEST_BASELINE_ARGS+=(--baseline-metrics-json "$SFT_BASELINE_METRICS_JSON")
fi

SFT_PROMOTION_FAILED=0
if [[ "$RUN_SFT_PROMOTION_GATE" == "true" || "$RUN_SFT_PROMOTION_GATE" == "1" || "$RUN_SFT_PROMOTION_GATE" == "yes" ]]; then
  if "$PYTHON" scripts/check_policy_iteration_metrics.py \
    --candidate-json "$SFT_EVAL_JSON" \
    --output-json "$SFT_PROMOTION_JSON" \
    --min-chrf "${SFT_PROMOTION_MIN_CHRF:-35}" \
    --min-bleu "${SFT_PROMOTION_MIN_BLEU:-8}" \
    --min-token-f1 "${SFT_PROMOTION_MIN_TOKEN_F1:-15}" \
    --max-ter "${SFT_PROMOTION_MAX_TER:-120}" \
    --min-format-rate "${SFT_PROMOTION_MIN_FORMAT_RATE:-50}" \
    --max-false-confidence-rate "${SFT_PROMOTION_MAX_FALSE_CONFIDENCE_RATE:-95}" \
    --max-missing-score-rate "${SFT_PROMOTION_MAX_MISSING_SCORE_RATE:-50}" \
    --min-chrf-delta "${SFT_PROMOTION_MIN_CHRF_DELTA:--1}" \
    --min-bleu-delta "${SFT_PROMOTION_MIN_BLEU_DELTA:--1}" \
    --min-token-f1-delta "${SFT_PROMOTION_MIN_TOKEN_F1_DELTA:--1}" \
    --max-false-confidence-delta "${SFT_PROMOTION_MAX_FALSE_CONFIDENCE_DELTA:-5}" \
    "${SFT_PROMOTION_BASELINE_ARGS[@]}"
  then
    echo "SFT seed promotion gate passed: $SFT_PROMOTION_JSON"
  else
    SFT_PROMOTION_FAILED=1
    echo "SFT seed promotion gate failed: $SFT_PROMOTION_JSON"
  fi
fi

"$PYTHON" scripts/write_deepseekmath_cycle_manifest.py \
  --output-json "$SFT_CYCLE_MANIFEST_JSON" \
  --stamp "${STAMP}-sft-seed" \
  --stage sft_seed \
  --base-model "$BASE_ADAPTER" \
  --policy-adapter "$THINKING_SFT_ADAPTER" \
  --meta-verifier-adapter "${META_VERIFIER_ADAPTER:-none}" \
  --meta-output-dir "${SFT_META_OUTPUT_DIR:-none}" \
  --followup-output-dir "$THINKING_SFT_OUTPUT_DIR" \
  --metrics-json "$SFT_EVAL_JSON" \
  --promotion-json "$SFT_PROMOTION_JSON" \
  --predictions-jsonl "$SFT_EVAL_PREDICTIONS" \
  --output-hardcase-jsonl "$SFT_META_JSONL" \
  "${SFT_MANIFEST_BASELINE_ARGS[@]}"

echo "SFT metrics: $SFT_EVAL_JSON"
echo "SFT predictions: $SFT_EVAL_PREDICTIONS"
echo "SFT meta hardcases: $SFT_META_JSONL"
echo "SFT cycle manifest: $SFT_CYCLE_MANIFEST_JSON"
echo "Staged status report: $STAGED_STATUS_MD"

if [[ "$SFT_PROMOTION_FAILED" -eq 1 && ("$REQUIRE_SFT_PROMOTION" == "true" || "$REQUIRE_SFT_PROMOTION" == "1" || "$REQUIRE_SFT_PROMOTION" == "yes") ]]; then
  exit 1
fi
