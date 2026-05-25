"""Gate frontier prompt previews before any paid frontier API call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


SECRET_MARKERS = (
    "Authorization",
    "api_key",
    "DEEPSEEK_API_KEY",
    "Bearer ",
)
ANTI_VACUOUS_MARKERS = (
    "Source terms to consider:",
    "Reference terms to consider:",
    "Do not write generic checks",
    "Each tag must mention a concrete",
    "at least six non-tag words",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-preview-jsonl", type=Path, required=True)
    parser.add_argument("--selection-jsonl", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--expected-model", default="deepseek-v4-pro")
    parser.add_argument("--expected-reasoning-effort", choices=("high", "max"), default=None)
    parser.add_argument("--min-preview-rows", type=int, default=1)
    parser.add_argument(
        "--require-all-selected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --selection-jsonl is provided, require one prompt preview per selected row.",
    )
    parser.add_argument(
        "--require-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require payload thinking.enabled for frontier synthetic reasoning generation.",
    )
    parser.add_argument(
        "--require-json-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require response_format json_object in every generation payload.",
    )
    parser.add_argument(
        "--require-anti-vacuous-instructions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the prompt to include source/reference term hints and non-vacuous analysis instructions.",
    )
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def contains_secret_marker(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(marker in str(key) for marker in SECRET_MARKERS):
                return True
            if contains_secret_marker(nested):
                return True
        return False
    if isinstance(value, list):
        return any(contains_secret_marker(item) for item in value)
    if isinstance(value, str):
        return any(marker in value for marker in SECRET_MARKERS)
    return False


def user_message_content(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    user_contents = [
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return "\n".join(user_contents)


def required_primitive_line(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Required primitive tags for this row:"):
            return line
    return ""


def selected_row_count(path: Path | None) -> int | None:
    if path is None:
        return None
    return sum(1 for _ in iter_jsonl(path))


def gate_prompt_preview(
    records: Sequence[dict[str, Any]],
    expected_model: str,
    min_preview_rows: int,
    selected_rows: int | None = None,
    require_all_selected: bool = True,
    require_thinking: bool = True,
    require_json_response: bool = True,
    require_anti_vacuous_instructions: bool = True,
    expected_reasoning_effort: str | None = None,
) -> tuple[dict[str, Any], bool, list[str]]:
    reasons: list[str] = []
    row_keys: set[str] = set()
    missing_expected_tags = 0
    current_row_few_shot_leaks = 0
    secret_marker_records = 0
    bad_model_records = 0
    bad_thinking_records = 0
    bad_json_response_records = 0
    bad_reasoning_effort_records = 0
    bad_payload_shape_records = 0
    missing_anti_vacuous_instruction_records = 0

    if len(records) < min_preview_rows:
        reasons.append(f"preview_rows {len(records)} < {min_preview_rows}")
    if selected_rows is not None and require_all_selected and len(records) != selected_rows:
        reasons.append(f"preview_rows {len(records)} != selected_rows {selected_rows}")

    for index, record in enumerate(records, start=1):
        row_key = str(record.get("row_key") or "")
        if row_key:
            row_keys.add(row_key)
        if contains_secret_marker(record):
            secret_marker_records += 1
            reasons.append(f"record {index} contains secret/auth marker")

        payload = record.get("generation_payload")
        if not isinstance(payload, dict):
            bad_payload_shape_records += 1
            reasons.append(f"record {index} has no generation_payload object")
            continue
        if payload.get("model") != expected_model:
            bad_model_records += 1
            reasons.append(f"record {index} model {payload.get('model')!r} != {expected_model!r}")

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            bad_payload_shape_records += 1
            reasons.append(f"record {index} has empty or invalid messages")
        elif not all(isinstance(message, dict) and "role" in message and "content" in message for message in messages):
            bad_payload_shape_records += 1
            reasons.append(f"record {index} has malformed messages")

        if require_thinking:
            thinking = payload.get("thinking")
            if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
                bad_thinking_records += 1
                reasons.append(f"record {index} does not enable thinking")
        if expected_reasoning_effort is not None and payload.get("reasoning_effort") != expected_reasoning_effort:
            bad_reasoning_effort_records += 1
            reasons.append(
                f"record {index} reasoning_effort {payload.get('reasoning_effort')!r} "
                f"!= {expected_reasoning_effort!r}"
            )
        if require_json_response and payload.get("response_format") != {"type": "json_object"}:
            bad_json_response_records += 1
            reasons.append(f"record {index} does not require JSON response_format")

        prompt = user_message_content(payload)
        required_line = required_primitive_line(prompt)
        expected_primitives = [str(tag) for tag in record.get("expected_primitives") or []]
        if not expected_primitives:
            missing_expected_tags += 1
            reasons.append(f"record {index} has no expected_primitives")
        missing = [tag for tag in expected_primitives if tag not in required_line]
        if missing:
            missing_expected_tags += 1
            reasons.append(f"record {index} required primitive line missing expected tags: {', '.join(missing)}")
        if not required_line:
            missing_expected_tags += 1
            reasons.append(f"record {index} prompt lacks required primitive tag instruction")
        if require_anti_vacuous_instructions:
            missing_markers = [marker for marker in ANTI_VACUOUS_MARKERS if marker not in prompt]
            if missing_markers:
                missing_anti_vacuous_instruction_records += 1
                reasons.append(
                    f"record {index} prompt missing anti-vacuous instructions: {', '.join(missing_markers)}"
                )

        few_shot_row_keys = {str(key) for key in record.get("few_shot_row_keys") or []}
        if row_key and row_key in few_shot_row_keys:
            current_row_few_shot_leaks += 1
            reasons.append(f"record {index} uses current row as a few-shot example")

    if len(row_keys) < len(records):
        reasons.append(f"unique_row_keys {len(row_keys)} < preview_rows {len(records)}")

    metrics = {
        "preview_rows": len(records),
        "selected_rows": selected_rows,
        "unique_row_keys": len(row_keys),
        "expected_model": expected_model,
        "min_preview_rows": min_preview_rows,
        "require_all_selected": require_all_selected,
        "require_thinking": require_thinking,
        "require_json_response": require_json_response,
        "expected_reasoning_effort": expected_reasoning_effort,
        "missing_expected_tag_records": missing_expected_tags,
        "current_row_few_shot_leak_records": current_row_few_shot_leaks,
        "secret_marker_records": secret_marker_records,
        "bad_model_records": bad_model_records,
        "bad_thinking_records": bad_thinking_records,
        "bad_json_response_records": bad_json_response_records,
        "bad_reasoning_effort_records": bad_reasoning_effort_records,
        "bad_payload_shape_records": bad_payload_shape_records,
        "require_anti_vacuous_instructions": require_anti_vacuous_instructions,
        "missing_anti_vacuous_instruction_records": missing_anti_vacuous_instruction_records,
    }
    return metrics, not reasons, reasons


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = list(iter_jsonl(args.prompt_preview_jsonl))
    selected_rows = selected_row_count(args.selection_jsonl)
    metrics, passed, reasons = gate_prompt_preview(
        records,
        expected_model=args.expected_model,
        min_preview_rows=args.min_preview_rows,
        selected_rows=selected_rows,
        require_all_selected=args.require_all_selected,
        require_thinking=args.require_thinking,
        require_json_response=args.require_json_response,
        require_anti_vacuous_instructions=args.require_anti_vacuous_instructions,
        expected_reasoning_effort=args.expected_reasoning_effort,
    )
    payload = {
        **metrics,
        "passed": passed,
        "reasons": reasons,
        "prompt_preview_jsonl": str(args.prompt_preview_jsonl),
        "selection_jsonl": str(args.selection_jsonl) if args.selection_jsonl is not None else None,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
