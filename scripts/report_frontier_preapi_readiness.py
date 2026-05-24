"""Summarize frontier source selection and prompt previews before API calls."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_frontier_prompt_preview as prompt_gate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report-json", type=Path, required=True)
    parser.add_argument("--selection-gate-json", type=Path, default=None)
    parser.add_argument("--selection-jsonl", type=Path, required=True)
    parser.add_argument("--prompt-preview-jsonl", type=Path, required=True)
    parser.add_argument("--prompt-preview-gate-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=5)
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def payload_stats(previews: Sequence[dict[str, Any]]) -> dict[str, Any]:
    models: collections.Counter[str] = collections.Counter()
    reasoning: collections.Counter[str] = collections.Counter()
    max_tokens: collections.Counter[str] = collections.Counter()
    few_shot_counts: list[int] = []
    prompt_chars: list[int] = []
    required_line_missing = 0
    for preview in previews:
        payload = preview.get("generation_payload")
        if not isinstance(payload, dict):
            continue
        models[str(payload.get("model"))] += 1
        reasoning[str(payload.get("reasoning_effort"))] += 1
        max_tokens[str(payload.get("max_tokens"))] += 1
        few_shot_counts.append(len(preview.get("few_shot_row_keys") or []))
        prompt = prompt_gate.user_message_content(payload)
        prompt_chars.append(len(prompt))
        if not prompt_gate.required_primitive_line(prompt):
            required_line_missing += 1
    return {
        "models": dict(models.most_common()),
        "reasoning_efforts": dict(reasoning.most_common()),
        "max_tokens": dict(max_tokens.most_common()),
        "avg_few_shots": sum(few_shot_counts) / max(1, len(few_shot_counts)),
        "min_few_shots": min(few_shot_counts) if few_shot_counts else 0,
        "max_few_shots": max(few_shot_counts) if few_shot_counts else 0,
        "avg_prompt_chars": sum(prompt_chars) / max(1, len(prompt_chars)),
        "min_prompt_chars": min(prompt_chars) if prompt_chars else 0,
        "max_prompt_chars": max(prompt_chars) if prompt_chars else 0,
        "required_line_missing": required_line_missing,
    }


def selected_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sources = collections.Counter(str(record.get("source_name") or "unknown") for record in rows)
    variants = collections.Counter(str(record.get("variant") or "unknown") for record in rows)
    statuses = collections.Counter(str(record.get("resume_status") or "unknown") for record in rows)
    return {
        "source_names": dict(sources.most_common(20)),
        "variants": dict(variants.most_common(20)),
        "resume_statuses": dict(statuses.most_common()),
    }


def sample_rows(selected: Sequence[dict[str, Any]], previews: Sequence[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    preview_by_key = {str(record.get("row_key")): record for record in previews}
    samples: list[dict[str, Any]] = []
    for record in selected[: max(0, sample_size)]:
        key = str(record.get("row_key") or "")
        preview = preview_by_key.get(key, {})
        payload = preview.get("generation_payload") if isinstance(preview.get("generation_payload"), dict) else {}
        prompt = prompt_gate.user_message_content(payload)
        samples.append(
            {
                "source": record.get("source"),
                "reference": record.get("reference"),
                "expected_primitives": record.get("expected_primitives"),
                "resume_status": record.get("resume_status"),
                "few_shot_sources": preview.get("few_shot_sources"),
                "required_primitive_line": prompt_gate.required_primitive_line(prompt),
            }
        )
    return samples


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    selection_report = load_json(args.selection_report_json) or {}
    selection_gate = load_json(args.selection_gate_json)
    prompt_gate_report = load_json(args.prompt_preview_gate_json)
    selected = list(iter_jsonl(args.selection_jsonl))
    previews = list(iter_jsonl(args.prompt_preview_jsonl))
    selection_rows = int(selection_report.get("selected_rows", len(selected)) or 0)
    prompt_rows = len(previews)
    selection_gate_passed = None if selection_gate is None else bool(selection_gate.get("passed"))
    prompt_gate_passed = None if prompt_gate_report is None else bool(prompt_gate_report.get("passed"))
    ready_for_api = (
        selection_rows > 0
        and len(selected) == selection_rows
        and prompt_rows == selection_rows
        and selection_gate_passed is True
        and prompt_gate_passed is True
    )
    return {
        "ready_for_api": ready_for_api,
        "selection_report": selection_report,
        "selection_gate": selection_gate,
        "prompt_preview_gate": prompt_gate_report,
        "selected_rows_jsonl_rows": len(selected),
        "prompt_preview_rows": prompt_rows,
        "payload_stats": payload_stats(previews),
        "selected_stats": selected_stats(selected),
        "samples": sample_rows(selected, previews, args.sample_size),
        "artifacts": {
            "selection_report_json": str(args.selection_report_json),
            "selection_gate_json": str(args.selection_gate_json) if args.selection_gate_json else None,
            "selection_jsonl": str(args.selection_jsonl),
            "prompt_preview_jsonl": str(args.prompt_preview_jsonl),
            "prompt_preview_gate_json": str(args.prompt_preview_gate_json) if args.prompt_preview_gate_json else None,
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selection = report["selection_report"]
    payload_stats_report = report["payload_stats"]
    selection_gate = report.get("selection_gate") or {}
    prompt_gate_report = report.get("prompt_preview_gate") or {}
    lines = [
        "# Frontier Pre-API Readiness Report",
        "",
        f"- ready for API: {report['ready_for_api']}",
        f"- selected rows: {selection.get('selected_rows', report['selected_rows_jsonl_rows'])}",
        f"- prompt preview rows: {report['prompt_preview_rows']}",
        f"- estimated max frontier requests: {selection.get('estimated_max_frontier_requests')}",
        f"- selection gate passed: {selection_gate.get('passed')}",
        f"- prompt preview gate passed: {prompt_gate_report.get('passed')}",
        f"- models: {payload_stats_report['models']}",
        f"- reasoning efforts: {payload_stats_report['reasoning_efforts']}",
        f"- max tokens: {payload_stats_report['max_tokens']}",
        f"- avg few-shots: {payload_stats_report['avg_few_shots']:.4f}",
        f"- avg prompt chars: {payload_stats_report['avg_prompt_chars']:.1f}",
        "",
        "## Expected Primitive Counts",
        "",
    ]
    for tag, count in sorted((selection.get("expected_primitive_counts") or {}).items()):
        lines.append(f"- `{tag}`: {count}")
    lines.extend(["", "## Source Names", ""])
    for name, count in report["selected_stats"]["source_names"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Samples", ""])
    for sample in report["samples"]:
        tags = ", ".join(sample.get("expected_primitives") or [])
        lines.append(f"- `{sample.get('source')}` -> `{tags}`")
        if sample.get("required_primitive_line"):
            lines.append(f"  - required line: `{sample['required_primitive_line']}`")
        few_shots = sample.get("few_shot_sources") or []
        if few_shots:
            lines.append(f"  - few-shots: {', '.join(f'`{source}`' for source in few_shots)}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    if args.output_md is not None:
        write_markdown(report, args.output_md)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["ready_for_api"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
