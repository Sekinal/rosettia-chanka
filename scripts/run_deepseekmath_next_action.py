"""Run the allowlisted next action from a staged DeepSeekMath status JSON.

The staged status already decides what should happen next. This runner parses
that recommendation without invoking a shell, validates it against the known
DeepSeekMath-language launchers, and either prints the resolved command or
executes it when --execute is set.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence


ALLOWED_ACTIONS = {
    "frontier_generation": {
        "scripts": {"experiments/gspo/run_deepseek_v4_pro_paid_generation_smoke.sh"},
        "env": {"DATA_DIR"},
    },
    "sft_seed": {
        "scripts": {"experiments/gspo/run_deepseek_v4_pro_sft_from_frontier_data.sh"},
        "env": {"DATA_DIR"},
    },
    "initial_gspo": {
        "scripts": {"experiments/gspo/run_deepseek_v4_pro_gspo_from_sft_seed.sh"},
        "env": {"THINKING_SFT_OUTPUT_DIR"},
    },
    "hardcase_iteration": {
        "scripts": {"experiments/gspo/run_next_meta_verifier_from_hardcases.sh"},
        "env": {"SFT_META_JSONL", "GSPO_META_JSONL", "EXTRA_META_JSONLS"},
    },
    "promoted_policy": {
        "scripts": {"experiments/gspo/run_hardcase_meta_then_followup_gspo_cycle.sh"},
        "env": {"BASE_CYCLE_MANIFEST", "BASE_CYCLE_ROOT"},
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true", help="Actually run the approved command.")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def load_status(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"status JSON is not an object: {path}")
    return payload


def failure_report(reason: str) -> dict[str, Any]:
    return {
        "approved": False,
        "reasons": [reason],
        "stage": None,
        "command": None,
        "script": None,
        "script_path": None,
        "env": {},
        "argv": [],
        "execute": False,
    }


def split_recommendation(command: str) -> tuple[dict[str, str], str | None, list[str]]:
    env: dict[str, str] = {}
    script: str | None = None
    extras: list[str] = []
    for token in shlex.split(command):
        if script is None and "=" in token and not token.startswith("="):
            name, value = token.split("=", 1)
            env[name] = value
            continue
        if script is None:
            script = token
        else:
            extras.append(token)
    return env, script, extras


def validate_action(status: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    next_action = status.get("next_action")
    if not isinstance(next_action, dict):
        return {"approved": False, "reasons": ["missing next_action object"]}

    stage = str(next_action.get("stage") or "")
    command = str(next_action.get("command") or "")
    env, script, extras = split_recommendation(command)
    reasons: list[str] = []
    spec = ALLOWED_ACTIONS.get(stage)
    if spec is None:
        reasons.append(f"stage is not runnable: {stage}")
    if script is None:
        reasons.append("next command has no script")
    elif spec is not None and script not in spec["scripts"]:
        reasons.append(f"script not allowlisted for {stage}: {script}")
    if extras:
        reasons.append(f"unexpected command arguments: {' '.join(extras)}")
    if spec is not None:
        unknown_env = sorted(set(env) - set(spec["env"]))
        if unknown_env:
            reasons.append(f"env vars not allowlisted for {stage}: {', '.join(unknown_env)}")
    for name, value in env.items():
        if "\n" in value or "\r" in value:
            reasons.append(f"env var contains a newline: {name}")
    script_path = repo_root / script if script else None
    if script_path is not None and not script_path.exists():
        reasons.append(f"script does not exist: {script_path}")

    return {
        "approved": not reasons,
        "reasons": reasons,
        "stage": stage,
        "command": command,
        "script": script,
        "script_path": str(script_path) if script_path else None,
        "env": env,
        "argv": [str(script_path)] if script_path else [],
        "execute": False,
    }


def run_action(report: dict[str, Any], repo_root: Path) -> int:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in report["env"].items()})
    completed = subprocess.run(report["argv"], cwd=repo_root, env=env, check=False)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        status = load_status(args.status_json)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        report = failure_report(f"status JSON missing or invalid: {exc}")
    else:
        report = validate_action(status, repo_root)
    if args.execute and report["approved"]:
        report["execute"] = True
        exit_code = run_action(report, repo_root)
        report["exit_code"] = exit_code
    else:
        exit_code = 0 if report["approved"] else 1
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
