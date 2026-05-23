#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
WAIT_PATTERN="${WAIT_PATTERN:-scripts/train_gspo_chanka_unsloth.py}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$ROOT_DIR"

active_wait_matches() {
  ps -eo pid=,cmd= \
    | grep -E "$WAIT_PATTERN" \
    | grep -v grep \
    | grep -v "queue_2511_self_verifiable_thinking_after_current_gspo.sh" \
    || true
}

while [[ -n "$(active_wait_matches)" ]]; do
  echo "Waiting for active GSPO process matching: $WAIT_PATTERN"
  active_wait_matches
  sleep "$WAIT_SECONDS"
done

META_VERIFIER_ADAPTER="${META_VERIFIER_ADAPTER:-outputs/chanka_translation_meta_verifier_v2_20260523-meta-v2-self/final_meta_verifier_lora}" \
STAMP="${STAMP:-20260523-thinking-meta-v2}" \
MAX_STEPS="${MAX_STEPS:-8}" \
EVAL_STEPS="${EVAL_STEPS:-4}" \
SAVE_STEPS="${SAVE_STEPS:-4}" \
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-256}" \
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-64}" \
experiments/gspo/run_2511_self_verifiable_thinking_translation.sh
