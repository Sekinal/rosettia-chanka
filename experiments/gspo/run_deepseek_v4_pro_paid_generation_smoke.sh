#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-deepseek-v4-pro-paid-smoke)}"

FRONTIER_MAX_ROWS="${FRONTIER_MAX_ROWS:-8}"
FRONTIER_MAX_API_REQUESTS="${FRONTIER_MAX_API_REQUESTS:-16}"
FRONTIER_AUDIT="${FRONTIER_AUDIT:-true}"
FRONTIER_BASE_URL="${FRONTIER_BASE_URL:-https://api.deepseek.com}"
FRONTIER_MODEL="${FRONTIER_MODEL:-deepseek-v4-pro}"
FRONTIER_REASONING_EFFORT="${FRONTIER_REASONING_EFFORT:-max}"
FRONTIER_API_KEY_ENV="${FRONTIER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
READ_FRONTIER_API_KEY="${READ_FRONTIER_API_KEY:-true}"

MIN_FRONTIER_ROWS_FOR_SFT="${MIN_FRONTIER_ROWS_FOR_SFT:-$FRONTIER_MAX_ROWS}"
MIN_FRONTIER_SELECTION_ROWS="${MIN_FRONTIER_SELECTION_ROWS:-$FRONTIER_MAX_ROWS}"
MAX_FRONTIER_SELECTION_REQUESTS="${MAX_FRONTIER_SELECTION_REQUESTS:-$FRONTIER_MAX_API_REQUESTS}"

MIN_PAID_SMOKE_ACCEPTED_ROWS="${MIN_PAID_SMOKE_ACCEPTED_ROWS:-6}"
MIN_PAID_SMOKE_ACCEPT_RATE="${MIN_PAID_SMOKE_ACCEPT_RATE:-0.50}"
MIN_PAID_SMOKE_PRIMITIVE_TAGS_PER_ROW="${MIN_PAID_SMOKE_PRIMITIVE_TAGS_PER_ROW:-2}"
MIN_PAID_SMOKE_PRIMITIVE_ROW_RATE="${MIN_PAID_SMOKE_PRIMITIVE_ROW_RATE:-0.90}"
MIN_PAID_SMOKE_DISTINCT_PRIMITIVES="${MIN_PAID_SMOKE_DISTINCT_PRIMITIVES:-4}"
MIN_PAID_SMOKE_EXPECTED_PRIMITIVE_COVERAGE="${MIN_PAID_SMOKE_EXPECTED_PRIMITIVE_COVERAGE:-0.95}"

DATA_DIR="${DATA_DIR:-outputs/frontier_thinking_data_${STAMP}}"
FRONTIER_JSONL="${FRONTIER_JSONL:-${DATA_DIR}/deepseek_v4_pro_thinking_sft.jsonl}"
FRONTIER_FAILURES_JSONL="${FRONTIER_FAILURES_JSONL:-${DATA_DIR}/deepseek_v4_pro_thinking_failures.jsonl}"
FRONTIER_SUMMARY_JSON="${FRONTIER_SUMMARY_JSON:-${DATA_DIR}/deepseek_v4_pro_thinking_sft.summary.json}"
FRONTIER_REPORT_MD="${FRONTIER_REPORT_MD:-${DATA_DIR}/deepseek_v4_pro_thinking_report.md}"
PAID_SMOKE_GATE_JSON="${PAID_SMOKE_GATE_JSON:-${DATA_DIR}/deepseek_v4_pro_paid_smoke_gate.json}"

if [[ -z "${!FRONTIER_API_KEY_ENV:-}" ]]; then
  if [[ "$READ_FRONTIER_API_KEY" == "true" || "$READ_FRONTIER_API_KEY" == "1" || "$READ_FRONTIER_API_KEY" == "yes" ]]; then
    if [[ ! -t 0 ]]; then
      echo "${FRONTIER_API_KEY_ENV} is unset and stdin is not an interactive terminal." >&2
      echo "Run from a TTY or export ${FRONTIER_API_KEY_ENV} in the shell environment; do not write it to files or command history." >&2
      exit 2
    fi
    read -rsp "${FRONTIER_API_KEY_ENV}: " frontier_api_key
    echo
    if [[ -z "$frontier_api_key" ]]; then
      echo "${FRONTIER_API_KEY_ENV} was empty." >&2
      exit 2
    fi
    export "${FRONTIER_API_KEY_ENV}=${frontier_api_key}"
    unset frontier_api_key
  else
    echo "${FRONTIER_API_KEY_ENV} is unset. Export it in the shell environment; do not write it to files or command history." >&2
    exit 2
  fi
fi

if [[ "$FRONTIER_BASE_URL" != "https://api.deepseek.com" && "${ALLOW_NON_DEEPSEEK_BASE_URL:-false}" != "true" ]]; then
  echo "Refusing paid smoke against non-DeepSeek FRONTIER_BASE_URL=$FRONTIER_BASE_URL." >&2
  echo "Set ALLOW_NON_DEEPSEEK_BASE_URL=true only for explicit endpoint/stub debugging." >&2
  exit 2
fi

if [[ "$FRONTIER_AUDIT" != "true" && "$FRONTIER_AUDIT" != "1" && "$FRONTIER_AUDIT" != "yes" ]]; then
  echo "Refusing paid smoke with FRONTIER_AUDIT=$FRONTIER_AUDIT. Keep audit enabled for first real data quality checks." >&2
  exit 2
fi

cd "$ROOT_DIR"

echo "Starting DeepSeek V4 Pro paid generation smoke."
echo "Rows: $FRONTIER_MAX_ROWS; max API requests: $FRONTIER_MAX_API_REQUESTS; SFT disabled."
echo "Data directory: $DATA_DIR"

ROOT_DIR="$ROOT_DIR" \
PYTHON="$PYTHON" \
STAMP="$STAMP" \
DATA_DIR="$DATA_DIR" \
FRONTIER_JSONL="$FRONTIER_JSONL" \
FRONTIER_FAILURES_JSONL="$FRONTIER_FAILURES_JSONL" \
FRONTIER_SUMMARY_JSON="$FRONTIER_SUMMARY_JSON" \
FRONTIER_BASE_URL="$FRONTIER_BASE_URL" \
FRONTIER_MODEL="$FRONTIER_MODEL" \
FRONTIER_REASONING_EFFORT="$FRONTIER_REASONING_EFFORT" \
FRONTIER_API_KEY_ENV="$FRONTIER_API_KEY_ENV" \
FRONTIER_MAX_ROWS="$FRONTIER_MAX_ROWS" \
FRONTIER_MAX_API_REQUESTS="$FRONTIER_MAX_API_REQUESTS" \
FRONTIER_AUDIT="$FRONTIER_AUDIT" \
MIN_FRONTIER_ROWS_FOR_SFT="$MIN_FRONTIER_ROWS_FOR_SFT" \
MIN_FRONTIER_SELECTION_ROWS="$MIN_FRONTIER_SELECTION_ROWS" \
MAX_FRONTIER_SELECTION_REQUESTS="$MAX_FRONTIER_SELECTION_REQUESTS" \
RUN_THINKING_SFT=false \
RUN_GSPO=false \
experiments/gspo/run_deepseek_v4_pro_thinking_sft_then_gspo.sh

set +e
"$PYTHON" scripts/check_frontier_thinking_data.py \
  --summary-json "$FRONTIER_SUMMARY_JSON" \
  --output-jsonl "$FRONTIER_JSONL" \
  --failures-jsonl "$FRONTIER_FAILURES_JSONL" \
  --min-written-rows "$MIN_PAID_SMOKE_ACCEPTED_ROWS" \
  --min-accept-rate "$MIN_PAID_SMOKE_ACCEPT_RATE" \
  --min-primitive-tags-per-row "$MIN_PAID_SMOKE_PRIMITIVE_TAGS_PER_ROW" \
  --min-primitive-row-rate "$MIN_PAID_SMOKE_PRIMITIVE_ROW_RATE" \
  --min-distinct-primitives "$MIN_PAID_SMOKE_DISTINCT_PRIMITIVES" \
  --min-expected-primitive-coverage "$MIN_PAID_SMOKE_EXPECTED_PRIMITIVE_COVERAGE" \
  > "$PAID_SMOKE_GATE_JSON"
gate_status=$?
set -e

cat "$PAID_SMOKE_GATE_JSON"
echo "Frontier data report: $FRONTIER_REPORT_MD"
echo "Paid smoke gate: $PAID_SMOKE_GATE_JSON"

if [[ $gate_status -ne 0 ]]; then
  echo "Paid smoke generation completed, but the data gate failed. Inspect the report before spending GPU time." >&2
  exit "$gate_status"
fi

echo "Paid smoke generation passed the data gate. Inspect the report before enabling SFT."
