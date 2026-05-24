#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/rosettia-chanka}"
PYTHON="${PYTHON:-.venv/bin/python}"
STAMP="${STAMP:-$(date -u +%Y%m%d-hardcase-cycle)}"

BASE_MODEL="${BASE_MODEL:-}"
BASE_CYCLE_MANIFEST="${BASE_CYCLE_MANIFEST:-}"
BASE_CYCLE_GATE_JSON="${BASE_CYCLE_GATE_JSON:-}"
BASE_CYCLE_ROOT="${BASE_CYCLE_ROOT:-}"
BASE_CYCLE_SELECTION_JSON="${BASE_CYCLE_SELECTION_JSON:-}"
META_OUTPUT_DIR="${META_OUTPUT_DIR:-outputs/chanka_translation_meta_verifier_iter_${STAMP}}"
FOLLOWUP_OUTPUT_DIR="${FOLLOWUP_OUTPUT_DIR:-outputs/gspo_paper_profiles/2511_self_verifiable_thinking_translation_cycle_${STAMP}}"
BASELINE_METRICS_JSON="${BASELINE_METRICS_JSON:-}"
REQUIRE_PROMOTION="${REQUIRE_PROMOTION:-false}"
CYCLE_MANIFEST_JSON="${CYCLE_MANIFEST_JSON:-${FOLLOWUP_OUTPUT_DIR}/cycle_manifest.json}"

cd "$ROOT_DIR"

if [[ -n "$BASE_CYCLE_ROOT" && -z "$BASE_CYCLE_MANIFEST" ]]; then
  if [[ -z "$BASE_CYCLE_SELECTION_JSON" ]]; then
    BASE_CYCLE_SELECTION_JSON="${FOLLOWUP_OUTPUT_DIR}/base_cycle_selection.json"
  fi
  "$PYTHON" scripts/select_deepseekmath_cycle.py "$BASE_CYCLE_ROOT" \
    --output-json "$BASE_CYCLE_SELECTION_JSON" \
    --min-chrf "${BASE_CYCLE_MIN_CHRF:-35}" \
    --min-bleu "${BASE_CYCLE_MIN_BLEU:-8}" \
    --min-token-f1 "${BASE_CYCLE_MIN_TOKEN_F1:-15}"
  BASE_CYCLE_MANIFEST="$("$PYTHON" - <<'PY' "$BASE_CYCLE_SELECTION_JSON"
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
print(selection["selected"]["manifest_json"])
PY
)"
  if [[ -z "$BASELINE_METRICS_JSON" ]]; then
    BASELINE_METRICS_JSON="$("$PYTHON" - <<'PY' "$BASE_CYCLE_SELECTION_JSON"
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
print(selection["selected"].get("baseline_metrics_json") or "")
PY
)"
  fi
fi

if [[ -n "$BASE_CYCLE_MANIFEST" ]]; then
  if [[ -z "$BASE_CYCLE_GATE_JSON" ]]; then
    BASE_CYCLE_GATE_JSON="${FOLLOWUP_OUTPUT_DIR}/base_cycle_gate.json"
  fi
  "$PYTHON" scripts/check_deepseekmath_cycle_manifest.py \
    --manifest-json "$BASE_CYCLE_MANIFEST" \
    --output-json "$BASE_CYCLE_GATE_JSON" \
    --min-chrf "${BASE_CYCLE_MIN_CHRF:-35}" \
    --min-bleu "${BASE_CYCLE_MIN_BLEU:-8}" \
    --min-token-f1 "${BASE_CYCLE_MIN_TOKEN_F1:-15}"
  BASE_MODEL="$("$PYTHON" - <<'PY' "$BASE_CYCLE_GATE_JSON"
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text())
print(gate["policy_adapter"])
PY
)"
  if [[ -z "$BASELINE_METRICS_JSON" ]]; then
    BASELINE_METRICS_JSON="$("$PYTHON" - <<'PY' "$BASE_CYCLE_MANIFEST"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
artifact = (manifest.get("artifacts") or {}).get("metrics") or {}
print(artifact.get("path") or "")
PY
)"
  fi
fi

if [[ -z "$BASE_MODEL" ]]; then
  echo "BASE_MODEL, BASE_CYCLE_MANIFEST, or BASE_CYCLE_ROOT is required. Use a promoted cycle manifest/root or a previous policy adapter." >&2
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
  --stage hardcase_followup_gspo \
  --base-model "$BASE_MODEL" \
  --policy-adapter "${FOLLOWUP_OUTPUT_DIR}/chanka_gspo/final_gspo_lora" \
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
echo "Cycle base model: $BASE_MODEL"
if [[ -n "$BASE_CYCLE_MANIFEST" ]]; then
  echo "Cycle base manifest: $BASE_CYCLE_MANIFEST"
  echo "Cycle base gate: $BASE_CYCLE_GATE_JSON"
fi
if [[ -n "$BASE_CYCLE_SELECTION_JSON" ]]; then
  echo "Cycle base selection: $BASE_CYCLE_SELECTION_JSON"
fi
echo "Cycle follow-up output: $FOLLOWUP_OUTPUT_DIR"
echo "Cycle manifest: $CYCLE_MANIFEST_JSON"
