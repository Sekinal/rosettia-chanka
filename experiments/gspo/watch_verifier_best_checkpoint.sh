#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
VERIFIER_OUTPUT_DIR="${VERIFIER_OUTPUT_DIR:-outputs/chanka_translation_verifier_real_candidates_v2_20260521-verifier-v2-real}"
VERIFIER_SELECTED_DIR="${VERIFIER_SELECTED_DIR:-outputs/verifier_selected_checkpoints}"
POLL_SECONDS="${POLL_SECONDS:-300}"

cd "$ROOT_DIR"
mkdir -p "$VERIFIER_SELECTED_DIR"

while pgrep -f "python scripts/train_verifier_chanka_unsloth.py" >/dev/null; do
  python - "$VERIFIER_OUTPUT_DIR" "$VERIFIER_SELECTED_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
selected = Path(sys.argv[2])
best_path = None
best_loss = float("inf")
for state_path in sorted(root.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[-1])):
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        continue
    for row in state.get("log_history", []):
        if "eval_loss" not in row:
            continue
        loss = float(row["eval_loss"])
        if loss < best_loss:
            best_loss = loss
            best_path = state_path.parent

if best_path is None:
    print("No verifier eval checkpoint found yet.", flush=True)
    raise SystemExit

target = selected / f"{root.name}_{best_path.name}"
stable = selected / f"{root.name}_best"
if not target.exists():
    shutil.copytree(best_path, target)
    print(f"Preserved {best_path} -> {target} eval_loss={best_loss}", flush=True)
if stable.exists() or stable.is_symlink():
    if stable.is_dir() and not stable.is_symlink():
        shutil.rmtree(stable)
    else:
        stable.unlink()
stable.symlink_to(target.resolve(), target_is_directory=True)
PY
  sleep "$POLL_SECONDS"
done

echo "Verifier trainer no longer running."
