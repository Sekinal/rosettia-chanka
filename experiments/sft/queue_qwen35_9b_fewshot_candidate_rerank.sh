#!/usr/bin/env bash
set -euo pipefail

WAIT_FOR_FILE="${WAIT_FOR_FILE:-outputs/qwen35_9b_candidate_rerank/20260523-qwen35-9b-candidate-rerank/score_ensemble/ensemble_current_4b_9b_summary.json}"
STAMP="${STAMP:-$(date -u +%Y%m%d-qwen35-9b-fewshot2-candidate-rerank)}"
FEW_SHOT_TOP_K="${FEW_SHOT_TOP_K:-2}"
FEW_SHOT_MAX_CANDIDATES="${FEW_SHOT_MAX_CANDIDATES:-128}"

while [[ ! -f "$WAIT_FOR_FILE" ]]; do
  echo "$(date -u +%FT%TZ) waiting for no-few-shot rerank summary: $WAIT_FOR_FILE"
  sleep 300
done

echo "$(date -u +%FT%TZ) launching few-shot 9B candidate rerank after $WAIT_FOR_FILE"
STAMP="$STAMP" \
FEW_SHOT_TOP_K="$FEW_SHOT_TOP_K" \
FEW_SHOT_MAX_CANDIDATES="$FEW_SHOT_MAX_CANDIDATES" \
bash experiments/sft/queue_qwen35_9b_candidate_rerank.sh
