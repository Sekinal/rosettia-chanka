#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
WAIT_PATTERN="${WAIT_PATTERN:-evaluate_gspo_checkpoint|train_meta_verifier_chanka_unsloth|train_gspo_chanka_unsloth|train_jsonl_sft_unsloth}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$ROOT_DIR"

active_wait_matches() {
  ps -eo pid=,cmd= \
    | grep -E "$WAIT_PATTERN" \
    | grep -v grep \
    | grep -v "queue_2511_thinking_generator_sft_after_active.sh" \
    || true
}

while [[ -n "$(active_wait_matches)" ]]; do
  echo "Waiting for active training/eval process matching: $WAIT_PATTERN"
  active_wait_matches
  sleep "$WAIT_SECONDS"
done

STAMP="${STAMP:-20260523-thinking-generator-seeded}" \
MAX_SOURCE_ROWS="${MAX_SOURCE_ROWS:-256}" \
SFT_MAX_STEPS="${SFT_MAX_STEPS:-32}" \
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-8}" \
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-8}" \
GSPO_MAX_STEPS="${GSPO_MAX_STEPS:-8}" \
GSPO_EVAL_STEPS="${GSPO_EVAL_STEPS:-8}" \
GSPO_SAVE_STEPS="${GSPO_SAVE_STEPS:-8}" \
GSPO_MAX_TRAIN_SAMPLES="${GSPO_MAX_TRAIN_SAMPLES:-256}" \
GSPO_MAX_EVAL_SAMPLES="${GSPO_MAX_EVAL_SAMPLES:-32}" \
experiments/gspo/run_2511_train_thinking_generator_then_gspo.sh
