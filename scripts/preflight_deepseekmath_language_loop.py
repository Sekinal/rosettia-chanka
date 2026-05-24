"""Preflight checks for the DeepSeekMath-style Chanka translation loop."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence


DEFAULT_REQUIRED_PATHS = (
    "scripts/build_frontier_thinking_sft_jsonl.py",
    "scripts/check_frontier_selection_report.py",
    "scripts/check_frontier_prompt_preview.py",
    "scripts/report_frontier_preapi_readiness.py",
    "scripts/check_frontier_thinking_data.py",
    "scripts/train_jsonl_sft_unsloth.py",
    "scripts/evaluate_gspo_checkpoint.py",
    "scripts/build_meta_verifier_from_self_outputs.py",
    "scripts/train_meta_verifier_chanka_unsloth.py",
    "scripts/train_gspo_chanka_unsloth.py",
    "scripts/write_deepseekmath_cycle_manifest.py",
    "scripts/check_deepseekmath_cycle_manifest.py",
    "scripts/summarize_deepseekmath_cycles.py",
    "experiments/gspo/run_2511_self_verifiable_thinking_translation.sh",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-adapter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--require-api-key", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--required-path", type=Path, action="append", default=[])
    parser.add_argument("--default-required-paths", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def disk_free_gb(path: Path) -> float:
    existing = nearest_existing_parent(path)
    usage = shutil.disk_usage(existing)
    return usage.free / (1024**3)


def adapter_health(path: Path) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    if not path.exists():
        return False, warnings
    if not path.is_dir():
        warnings.append("base adapter path exists but is not a directory")
        return True, warnings
    marker_names = {"adapter_config.json", "config.json"}
    if not any((path / marker).exists() for marker in marker_names):
        warnings.append("base adapter directory has no adapter_config.json or config.json marker")
    return True, warnings


def build_report(args: argparse.Namespace, root: Path | None = None) -> dict[str, Any]:
    root = root or Path.cwd()
    required_paths = (list(DEFAULT_REQUIRED_PATHS) if args.default_required_paths else []) + [
        str(path) for path in args.required_path
    ]
    failures: list[str] = []
    warnings: list[str] = []

    api_key_set = bool(os.environ.get(args.api_key_env))
    if args.require_api_key and not api_key_set:
        failures.append(f"missing API key environment variable: {args.api_key_env}")

    base_exists, adapter_warnings = adapter_health(args.base_adapter)
    warnings.extend(adapter_warnings)
    if not base_exists:
        failures.append(f"missing base adapter: {args.base_adapter}")

    missing_paths = [path for path in required_paths if not (root / path).exists()]
    for path in missing_paths:
        failures.append(f"missing required path: {path}")

    free_gb = disk_free_gb(args.output_root)
    if free_gb < args.min_free_gb:
        failures.append(f"free disk {free_gb:.2f} GB < required {args.min_free_gb:.2f} GB at {args.output_root}")

    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "api_key_env": args.api_key_env,
        "api_key_set": api_key_set,
        "base_adapter": str(args.base_adapter),
        "base_adapter_exists": base_exists,
        "output_root": str(args.output_root),
        "free_disk_gb": free_gb,
        "min_free_gb": args.min_free_gb,
        "required_paths": required_paths,
        "missing_required_paths": missing_paths,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
