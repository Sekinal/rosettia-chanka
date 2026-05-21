#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M)}"
RUN_DIR="${RUN_DIR:-outputs/gspo_full_contenders/learned_verifier_vibe_on_vibe896_full_20260521-135419}"
BASELINE_DIR="${BASELINE_DIR:-outputs/gspo_checkpoint_evals/20260521-eosfix-ranking}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

cd "$ROOT_DIR"
source .venv/bin/activate

while pgrep -f "python scripts/train_gspo_chanka_unsloth.py" >/dev/null; do
  date -u
  pgrep -af "python scripts/train_gspo_chanka_unsloth.py" || true
  sleep "$POLL_SECONDS"
done

output_dir="outputs/gspo_checkpoint_evals/${STAMP}"
mkdir -p "$output_dir" outputs/gspo_selected_checkpoints

evaluate_checkpoint() {
  local adapter_path="$1"
  local label="$2"
  if [[ ! -d "$adapter_path" ]]; then
    echo "Skipping missing adapter: $adapter_path"
    return
  fi
  python scripts/evaluate_gspo_checkpoint.py \
    --adapter-path "$adapter_path" \
    --output-json "$output_dir/${label}_metrics.json" \
    --predictions-jsonl "$output_dir/${label}_predictions.jsonl" \
    --batch-size "$EVAL_BATCH_SIZE"
}

preserve_checkpoint() {
  local checkpoint="$1"
  local source_path="$RUN_DIR/chanka_gspo/checkpoint-${checkpoint}"
  local target_path="outputs/gspo_selected_checkpoints/learned_verifier_vibe_on_vibe896_checkpoint_${checkpoint}"
  if [[ -d "$source_path" ]]; then
    rm -rf "$target_path"
    cp -a "$source_path" "$target_path"
  fi
}

for checkpoint in 56 112 168 224; do
  preserve_checkpoint "$checkpoint"
  evaluate_checkpoint \
    "outputs/gspo_selected_checkpoints/learned_verifier_vibe_on_vibe896_checkpoint_${checkpoint}" \
    "learned_verifier_vibe_on_vibe896_checkpoint_${checkpoint}"
done

evaluate_checkpoint \
  "$RUN_DIR/chanka_gspo/final_gspo_lora" \
  "learned_verifier_vibe_on_vibe896_final"

evaluate_checkpoint \
  "outputs/gspo_paper_profiles/2511_learned_verifier_vibe_on_vibe896_4gen_canary_20260521-133146/chanka_gspo/final_gspo_lora" \
  "learned_verifier_vibe_on_vibe896_canary"

evaluate_checkpoint \
  "outputs/gspo_selected_checkpoints/vibethinker_phase2_checkpoint_896" \
  "vibethinker_phase2_checkpoint_896"

python scripts/summarize_gspo_checkpoint_evals.py "$output_dir"

if [[ -d "$BASELINE_DIR" ]]; then
  python scripts/summarize_gspo_checkpoint_evals.py "$BASELINE_DIR" || true
fi
