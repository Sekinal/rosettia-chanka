#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-hardcase-cycle)}"

BASE_MODEL="${BASE_MODEL:-}"
META_OUTPUT_DIR="${META_OUTPUT_DIR:-outputs/chanka_translation_meta_verifier_iter_${STAMP}}"
FOLLOWUP_OUTPUT_DIR="${FOLLOWUP_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_cycle_${STAMP}}"
BASELINE_METRICS_JSON="${BASELINE_METRICS_JSON:-}"
REQUIRE_PROMOTION="${REQUIRE_PROMOTION:-false}"
CYCLE_MANIFEST_JSON="${CYCLE_MANIFEST_JSON:-${FOLLOWUP_OUTPUT_DIR}/cycle_manifest.json}"

cd "$ROOT_DIR"

if [[ -z "$BASE_MODEL" ]]; then
  echo "BASE_MODEL is required. Use the frontier thinking SFT adapter or previous policy adapter." >&2
  exit 1
fi

META_OUTPUT_DIR="$META_OUTPUT_DIR" \
experiments/gspo/run_next_meta_verifier_from_hardcases.sh

NEXT_META_VERIFIER_ADAPTER="${META_OUTPUT_DIR}/final_meta_verifier_lora"

BASE_MODEL="$BASE_MODEL" \
META_VERIFIER_ADAPTER="$NEXT_META_VERIFIER_ADAPTER" \
OUTPUT_DIR="$FOLLOWUP_OUTPUT_DIR" \
BASELINE_METRICS_JSON="$BASELINE_METRICS_JSON" \
REQUIRE_PROMOTION="$REQUIRE_PROMOTION" \
experiments/gspo/run_followup_gspo_with_meta_verifier.sh

MANIFEST_HARDCASE_ARGS=()
if [[ -n "${SFT_META_JSONL:-}" ]]; then
  MANIFEST_HARDCASE_ARGS+=(--input-hardcase-jsonl "$SFT_META_JSONL")
fi
if [[ -n "${GSPO_META_JSONL:-}" ]]; then
  MANIFEST_HARDCASE_ARGS+=(--input-hardcase-jsonl "$GSPO_META_JSONL")
fi
if [[ -n "${EXTRA_META_JSONLS:-}" ]]; then
  IFS=':' read -r -a EXTRA_META_JSONL_ARRAY <<< "$EXTRA_META_JSONLS"
  for path in "${EXTRA_META_JSONL_ARRAY[@]}"; do
    if [[ -n "$path" ]]; then
      MANIFEST_HARDCASE_ARGS+=(--input-hardcase-jsonl "$path")
    fi
  done
fi
MANIFEST_BASELINE_ARGS=()
if [[ -n "$BASELINE_METRICS_JSON" ]]; then
  MANIFEST_BASELINE_ARGS+=(--baseline-metrics-json "$BASELINE_METRICS_JSON")
fi

"$PYTHON" scripts/write_deepseekmath_cycle_manifest.py \
  --output-json "$CYCLE_MANIFEST_JSON" \
  --stamp "$STAMP" \
  --base-model "$BASE_MODEL" \
  --meta-verifier-adapter "$NEXT_META_VERIFIER_ADAPTER" \
  --meta-output-dir "$META_OUTPUT_DIR" \
  --followup-output-dir "$FOLLOWUP_OUTPUT_DIR" \
  --metrics-json "${FOLLOWUP_OUTPUT_DIR}/chanka_gspo/final_metrics.json" \
  --promotion-json "${FOLLOWUP_OUTPUT_DIR}/chanka_gspo/promotion_gate.json" \
  --predictions-jsonl "${FOLLOWUP_OUTPUT_DIR}/chanka_gspo/final_predictions.jsonl" \
  --output-hardcase-jsonl "${FOLLOWUP_OUTPUT_DIR}/chanka_gspo/meta_hardcases_from_followup_gspo_eval.jsonl" \
  "${MANIFEST_HARDCASE_ARGS[@]}" \
  "${MANIFEST_BASELINE_ARGS[@]}"

echo "Cycle meta-verifier adapter: $NEXT_META_VERIFIER_ADAPTER"
echo "Cycle follow-up output: $FOLLOWUP_OUTPUT_DIR"
echo "Cycle manifest: $CYCLE_MANIFEST_JSON"
