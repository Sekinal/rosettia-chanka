#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
STAMP="${STAMP:-$(date -u +%Y%m%d-hardcase-cycle)}"

BASE_MODEL="${BASE_MODEL:-}"
META_OUTPUT_DIR="${META_OUTPUT_DIR:-outputs/chanka_translation_meta_verifier_iter_${STAMP}}"
FOLLOWUP_OUTPUT_DIR="${FOLLOWUP_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_cycle_${STAMP}}"
BASELINE_METRICS_JSON="${BASELINE_METRICS_JSON:-}"
REQUIRE_PROMOTION="${REQUIRE_PROMOTION:-false}"

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

echo "Cycle meta-verifier adapter: $NEXT_META_VERIFIER_ADAPTER"
echo "Cycle follow-up output: $FOLLOWUP_OUTPUT_DIR"
