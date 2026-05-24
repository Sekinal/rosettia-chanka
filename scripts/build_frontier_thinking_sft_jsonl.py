"""Build synthetic primitive-thinking SFT rows with a frontier model.

The safest first use is to synthesize concise translation checks for reviewed
Spanish -> Chanka pairs while keeping the reviewed reference as the final
translation. This gives the student model a DeepSeekMath-style cold start
without letting synthetic translations replace human-reviewed labels.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import train_gspo_chanka_unsloth as gspo

PRIMITIVES = (
    "[SIGNIFICADO]",
    "[GRAMATICA]",
    "[ENTIDADES]",
    "[TERMINOLOGIA]",
    "[ANTI_COPIA]",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", default=gspo.DATASET_REPO)
    parser.add_argument("--dataset-file", default=gspo.CHANKA_FILE)
    parser.add_argument("--source-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--selection-report-json", type=Path, default=None)
    parser.add_argument("--selection-report-md", type=Path, default=None)
    parser.add_argument("--selection-jsonl", type=Path, default=None)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--audit-model", default=None)
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default="max")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--stratify-primitives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select rows greedily to balance expected primitive tags before calling the frontier model.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--min-primitive-tags", type=int, default=2)
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=2,
        help="Include this many reviewed examples with synthetic primitive-check targets in the frontier prompt.",
    )
    parser.add_argument(
        "--require-expected-primitives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the generated analysis to include row-specific expected primitive tags.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows already present in output/failure JSONL and append new results.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="With --resume, retry rows from the failures JSONL instead of skipping them.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run a second frontier pass that accepts or rejects the generated thinking row.",
    )
    parser.add_argument("--audit-min-score", type=float, default=0.75)
    parser.add_argument(
        "--allow-model-translation",
        action="store_true",
        help="Use the frontier model's translation instead of the reviewed reference. Off by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first prompt payload shape without calling the API.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write the selected source-row report and exit without requiring an API key.",
    )
    return parser.parse_args(argv)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.source_jsonl is not None:
        rows = []
        for record in iter_jsonl(args.source_jsonl):
            source = str(record.get("source") or "").strip()
            target = str(record.get("target") or record.get("reference") or record.get("prediction") or "").strip()
            if source and target:
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "source_name": str(record.get("source_name") or args.source_jsonl.name),
                        "variant": str(record.get("variant") or "quy/chanka"),
                    }
                )
        return rows
    return gspo.load_chanka_rows(args.dataset_repo, args.dataset_file)


def dedupe_rows(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    duplicate_count = 0
    for row in rows:
        key = row_key(row["source"], row["target"])
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, duplicate_count


def balance_rows_by_expected_primitives(rows: Sequence[dict[str, str]], max_rows: int) -> list[dict[str, str]]:
    """Greedily select rows that cover underrepresented expected primitive tags."""
    remaining = list(rows)
    selected: list[dict[str, str]] = []
    counts = {tag: 0 for tag in PRIMITIVES}
    while remaining and len(selected) < max_rows:
        best_index = max(
            range(len(remaining)),
            key=lambda index: sum(
                1.0 / (1.0 + counts[tag])
                for tag in expected_primitives_for_row(remaining[index]["source"], remaining[index]["target"])
            ),
        )
        row = remaining.pop(best_index)
        selected.append(row)
        for tag in expected_primitives_for_row(row["source"], row["target"]):
            counts[tag] += 1
    return selected


def select_rows(
    rows: Sequence[dict[str, str]],
    offset: int,
    max_rows: int | None,
    seed: int,
    stratify_primitives: bool = True,
) -> list[dict[str, str]]:
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    selected, _ = dedupe_rows(selected)
    if offset:
        selected = selected[offset:]
    if max_rows is not None:
        if stratify_primitives:
            selected = balance_rows_by_expected_primitives(selected, max_rows)
        else:
            selected = selected[:max_rows]
    return selected


def row_key(source: str, reference: str) -> str:
    return f"{gspo.normalize_text(source).lower()}\t{gspo.normalize_text(reference).lower()}"


def few_shot_target(reference: str, index: int) -> dict[str, Any]:
    tag_pairs = (
        ("[SIGNIFICADO]", "[ANTI_COPIA]", "conserva el sentido central", "evita copiar palabras del espanol"),
        ("[GRAMATICA]", "[SIGNIFICADO]", "mantiene una forma chanka natural", "preserva la intencion del enunciado"),
        ("[ENTIDADES]", "[TERMINOLOGIA]", "conserva nombres o numeros", "usa vocabulario compatible con la referencia"),
    )
    first_tag, second_tag, first_note, second_note = tag_pairs[index % len(tag_pairs)]
    return {
        "analysis": f"{first_tag} {first_note}; {second_tag} {second_note}.",
        "translation": gspo.normalize_text(reference),
        "self_evaluation": "Riesgo bajo porque la traduccion esta anclada a la referencia revisada.",
        "score": 0.98,
    }


def select_few_shots(
    rows: Sequence[dict[str, str]],
    current: dict[str, str],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    current_key = row_key(current["source"], current["target"])
    examples: list[dict[str, Any]] = []
    for row in rows:
        if row_key(row["source"], row["target"]) == current_key:
            continue
        examples.append(
            {
                "source": row["source"],
                "reference": row["target"],
                "target": few_shot_target(row["target"], len(examples)),
            }
        )
        if len(examples) >= count:
            break
    return examples


def render_few_shots(few_shots: Sequence[dict[str, Any]]) -> str:
    if not few_shots:
        return ""
    blocks = ["Good examples of the desired concise JSON style:"]
    for index, example in enumerate(few_shots, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Example {index} Spanish source: {example['source']}",
                    f"Example {index} reviewed Chanka reference: {example['reference']}",
                    f"Example {index} JSON target: {json.dumps(example['target'], ensure_ascii=False, sort_keys=True)}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n\n"


def has_entity_signal(text: str) -> bool:
    if re.search(r"\d", text):
        return True
    tokens = re.findall(r"\b[^\W\d_][\wÁÉÍÓÚÜÑáéíóúüñ'-]*\b", text, flags=re.UNICODE)
    for index, token in enumerate(tokens):
        if index == 0:
            continue
        if token[:1].isupper() and token[1:].islower():
            return True
    return False


def has_terminology_signal(source: str, reference: str) -> bool:
    legal_or_domain_terms = {
        "acta",
        "articulo",
        "artículo",
        "autoridad",
        "comunidad",
        "constitucion",
        "constitución",
        "contrato",
        "derecho",
        "expediente",
        "fiscal",
        "juez",
        "judicial",
        "ley",
        "ministerio",
        "municipalidad",
        "penal",
        "proceso",
        "resolucion",
        "resolución",
        "sentencia",
        "tribunal",
    }
    combined = f"{source} {reference}".lower()
    return any(term in combined for term in legal_or_domain_terms)


def has_copy_risk_signal(source: str, reference: str) -> bool:
    source_terms = {
        token.lower()
        for token in re.findall(r"\b[^\W\d_]{4,}\b", source, flags=re.UNICODE)
        if token.lower() not in {"para", "como", "esta", "este", "estos", "estas", "donde", "cuando"}
    }
    reference_terms = {token.lower() for token in re.findall(r"\b[^\W\d_]{4,}\b", reference, flags=re.UNICODE)}
    return bool(source_terms.intersection(reference_terms))


def expected_primitives_for_row(source: str, reference: str, max_tags: int = 3) -> list[str]:
    expected = ["[SIGNIFICADO]", "[GRAMATICA]"]
    if has_entity_signal(source) or has_entity_signal(reference):
        expected.append("[ENTIDADES]")
    if len(expected) < max_tags and has_terminology_signal(source, reference):
        expected.append("[TERMINOLOGIA]")
    if len(expected) < max_tags and has_copy_risk_signal(source, reference):
        expected.append("[ANTI_COPIA]")
    if len(expected) < max_tags:
        expected.append("[ANTI_COPIA]")
    return expected[:max_tags]


def prompt_messages(
    source: str,
    reference: str,
    few_shots: Sequence[dict[str, Any]] | None = None,
    expected_primitives: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    few_shot_block = render_few_shots(few_shots or [])
    expected = list(expected_primitives or expected_primitives_for_row(source, reference))
    expected_block = ", ".join(expected)
    return [
        {
            "role": "system",
            "content": (
                "You are helping build training data for a Spanish to Quechua Chanka translator. "
                "Return only valid JSON. Do not include private chain-of-thought. "
                "Use short, auditable translation checks."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create a concise supervised thinking target for this reviewed translation.\n\n"
                f"{few_shot_block}"
                f"Spanish source:\n{source}\n\n"
                f"Reviewed Quechua Chanka reference:\n{reference}\n\n"
                f"Required primitive tags for this row: {expected_block}.\n"
                "The analysis must include every required primitive tag, and may add at most one extra useful tag.\n\n"
                "Return a JSON object with exactly these keys:\n"
                "- analysis: one sentence, maximum 35 words, using 2 to 4 tags from "
                "[SIGNIFICADO], [GRAMATICA], [ENTIDADES], [TERMINOLOGIA], [ANTI_COPIA].\n"
                "- translation: the best Quechua Chanka final translation. Prefer the reviewed reference.\n"
                "- self_evaluation: one short sentence about remaining risk.\n"
                "- score: a calibrated number from 0.0 to 1.0.\n\n"
                "The analysis must be useful for teaching grammar/meaning checks, not a long derivation."
            ),
        },
    ]


def audit_messages(source: str, reference: str, parsed: dict[str, Any]) -> list[dict[str, str]]:
    expected = parsed.get("expected_primitives")
    expected_line = ""
    if isinstance(expected, list) and expected:
        expected_line = f"\nRequired primitive tags: {', '.join(str(tag) for tag in expected)}\n"
    return [
        {
            "role": "system",
            "content": (
                "You audit synthetic training data for Spanish to Quechua Chanka translation. "
                "Return only valid JSON. Do not include private chain-of-thought."
            ),
        },
        {
            "role": "user",
            "content": (
                "Decide whether this synthetic thinking target is safe to train on.\n\n"
                f"Spanish source:\n{source}\n\n"
                f"Reviewed Quechua Chanka reference:\n{reference}\n\n"
                f"Analysis:\n{parsed['analysis']}\n\n"
                f"Proposed translation:\n{parsed['translation']}\n\n"
                f"Self evaluation:\n{parsed['self_evaluation']}\n\n"
                f"Score:\n{parsed['score']}\n\n"
                f"{expected_line}"
                "Accept only if the analysis is concise, uses the primitive tags correctly, "
                "includes the required primitive tags when provided, does not invent unsupported grammar claims, "
                "and the final translation is compatible with the reference.\n"
                "Return a JSON object with exactly: pass (boolean), score (0.0 to 1.0), reason (short sentence)."
            ),
        },
    ]


def chat_payload(
    args: argparse.Namespace,
    model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    if args.disable_thinking:
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = args.reasoning_effort
    return payload


def request_payload(args: argparse.Namespace, source: str, reference: str) -> dict[str, Any]:
    return chat_payload(
        args,
        args.model,
        prompt_messages(source, reference, expected_primitives=expected_primitives_for_row(source, reference)),
        args.max_output_tokens,
    )


def request_payload_with_few_shots(
    args: argparse.Namespace,
    source: str,
    reference: str,
    few_shots: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return chat_payload(
        args,
        args.model,
        prompt_messages(source, reference, few_shots, expected_primitives_for_row(source, reference)),
        args.max_output_tokens,
    )


def call_chat_payload(args: argparse.Namespace, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(args.max_retries):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise
        except urllib.error.URLError as exc:
            last_error = exc
        time.sleep(min(2.0**attempt, 30.0))
    raise RuntimeError(f"DeepSeek request failed after {args.max_retries} attempts: {last_error}")


def call_chat_completion(
    args: argparse.Namespace,
    api_key: str,
    source: str,
    reference: str,
    few_shots: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return call_chat_payload(
        args,
        api_key,
        request_payload_with_few_shots(args, source, reference, few_shots or []),
    )


def response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("Response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Response has empty content")
    return content.strip()


def parse_frontier_json(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(content[start : end + 1])
    return {
        "analysis": gspo.normalize_text(str(payload.get("analysis") or "")),
        "translation": gspo.normalize_text(str(payload.get("translation") or "")),
        "self_evaluation": gspo.normalize_text(str(payload.get("self_evaluation") or "")),
        "score": float(payload.get("score", 0.0)),
    }


def parse_audit_json(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(content[start : end + 1])
    return {
        "pass": bool(payload.get("pass", False)),
        "score": float(payload.get("score", 0.0)),
        "reason": gspo.normalize_text(str(payload.get("reason") or "")),
    }


def primitive_count(text: str) -> int:
    return sum(1 for tag in PRIMITIVES if tag in text)


def build_target(parsed: dict[str, Any], reference: str, allow_model_translation: bool) -> str:
    translation = parsed["translation"] if allow_model_translation and parsed["translation"] else reference
    score = min(1.0, max(0.0, float(parsed["score"])))
    return (
        f"Analisis de traduccion: {parsed['analysis']}\n"
        f"Traduccion final: {gspo.normalize_text(translation)}\n"
        f"Autoevaluacion: {parsed['self_evaluation']}\n"
        f"Puntaje: \\boxed{{{score:.2f}}}"
    )


def record_passes(
    parsed: dict[str, Any],
    min_primitive_tags: int,
    required_primitives: Sequence[str] | None = None,
) -> bool:
    if not parsed["analysis"] or not parsed["self_evaluation"]:
        return False
    if primitive_count(parsed["analysis"]) < min_primitive_tags:
        return False
    if required_primitives and any(tag not in parsed["analysis"] for tag in required_primitives):
        return False
    if len(parsed["analysis"].split()) > 45:
        return False
    return 0.0 <= float(parsed["score"]) <= 1.0


def audit_passes(audit: dict[str, Any], min_score: float) -> bool:
    return bool(audit["pass"]) and float(audit["score"]) >= min_score


def audit_record(args: argparse.Namespace, api_key: str, source: str, reference: str, parsed: dict[str, Any]) -> dict[str, Any]:
    payload = chat_payload(
        args,
        args.audit_model or args.model,
        audit_messages(source, reference, parsed),
        min(args.max_output_tokens, 256),
    )
    response = call_chat_payload(args, api_key, payload)
    return parse_audit_json(response_content(response))


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for record in iter_jsonl(path):
        key = record.get("row_key")
        if isinstance(key, str) and key:
            keys.add(key)
            continue
        source = str(record.get("source") or "")
        reference = str(record.get("reference") or record.get("target") or "")
        if source and reference:
            keys.add(row_key(source, reference))
    return keys


def failure_record(index: int, row: dict[str, str], reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "index": index,
        "row_key": row_key(row["source"], row["target"]),
        "source": row["source"],
        "reference": row["target"],
        "reason": reason,
        **extra,
    }


def selected_row_resume_accounting(
    rows: Sequence[dict[str, str]],
    output_jsonl: Path,
    failures_jsonl: Path,
    resume: bool,
    retry_failures: bool,
    audit: bool,
    max_retries: int,
) -> dict[str, Any]:
    selected_keys = [row_key(row["source"], row["target"]) for row in rows]
    if not resume:
        accepted_keys: set[str] = set()
        failed_keys: set[str] = set()
    else:
        accepted_keys = existing_keys(output_jsonl)
        failed_keys = set() if retry_failures else existing_keys(failures_jsonl)
    completed = accepted_keys.union(failed_keys)
    selected_accepted = sum(1 for key in selected_keys if key in accepted_keys)
    selected_failed = sum(1 for key in selected_keys if key in failed_keys)
    pending = sum(1 for key in selected_keys if key not in completed)
    generation_requests = pending
    max_audit_requests = pending if audit else 0
    max_frontier_requests = generation_requests + max_audit_requests
    return {
        "resume": resume,
        "retry_failures": retry_failures,
        "existing_accepted_rows": len(accepted_keys),
        "existing_failed_rows": len(failed_keys),
        "selected_existing_accepted_rows": selected_accepted,
        "selected_existing_failed_rows": selected_failed,
        "pending_rows": pending,
        "estimated_generation_requests": generation_requests,
        "estimated_max_audit_requests": max_audit_requests,
        "estimated_max_frontier_requests": max_frontier_requests,
        "estimated_max_http_attempts": max_frontier_requests * max(1, max_retries),
    }


def selected_row_records(
    rows: Sequence[dict[str, str]],
    output_jsonl: Path,
    failures_jsonl: Path,
    resume: bool,
    retry_failures: bool,
) -> list[dict[str, Any]]:
    if not resume:
        accepted_keys: set[str] = set()
        failed_keys: set[str] = set()
    else:
        accepted_keys = existing_keys(output_jsonl)
        failed_keys = set() if retry_failures else existing_keys(failures_jsonl)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        key = row_key(row["source"], row["target"])
        if key in accepted_keys:
            status = "existing_accepted"
        elif key in failed_keys:
            status = "existing_failed"
        else:
            status = "pending"
        records.append(
            {
                "index": index,
                "row_key": key,
                "source": row["source"],
                "reference": row["target"],
                "expected_primitives": expected_primitives_for_row(row["source"], row["target"]),
                "source_name": row.get("source_name"),
                "variant": row.get("variant"),
                "resume_status": status,
            }
        )
    return records


def selection_report(
    rows: Sequence[dict[str, str]],
    args: argparse.Namespace,
    input_rows: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    raw_rows = list(input_rows) if input_rows is not None else load_rows(args)
    _, duplicate_input_rows = dedupe_rows(raw_rows)
    failures_path = args.failures_jsonl or args.output_jsonl.with_suffix(".failures.jsonl")
    resume_accounting = selected_row_resume_accounting(
        rows,
        args.output_jsonl,
        failures_path,
        args.resume,
        args.retry_failures,
        args.audit,
        args.max_retries,
    )
    primitive_counts = {tag: 0 for tag in PRIMITIVES}
    row_tag_counts: list[int] = []
    samples: list[dict[str, Any]] = []
    for row in rows:
        expected = expected_primitives_for_row(row["source"], row["target"])
        row_tag_counts.append(len(expected))
        for tag in expected:
            primitive_counts[tag] += 1
        if len(samples) < 10:
            samples.append(
                {
                    "source": row["source"],
                    "reference": row["target"],
                    "expected_primitives": expected,
                    "source_name": row.get("source_name"),
                    "variant": row.get("variant"),
                }
            )
    distinct = sum(1 for count in primitive_counts.values() if count > 0)
    return {
        "model": args.model,
        "input_rows": len(raw_rows),
        "duplicate_input_rows": duplicate_input_rows,
        "selected_rows": len(rows),
        "max_rows": args.max_rows,
        "offset": args.offset,
        "seed": args.seed,
        "stratify_primitives": args.stratify_primitives,
        "expected_primitive_counts": primitive_counts,
        "distinct_expected_primitives": distinct,
        "avg_expected_primitives_per_row": sum(row_tag_counts) / max(1, len(row_tag_counts)),
        **resume_accounting,
        "samples": samples,
    }


def write_selection_report(report: dict[str, Any], json_path: Path | None, md_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Frontier Source Selection Report",
            "",
            f"- input rows: {report['input_rows']}",
            f"- duplicate input rows skipped: {report['duplicate_input_rows']}",
            f"- selected rows: {report['selected_rows']}",
            f"- pending rows: {report['pending_rows']}",
            f"- estimated max frontier requests: {report['estimated_max_frontier_requests']}",
            f"- stratify primitives: {report['stratify_primitives']}",
            f"- distinct expected primitives: {report['distinct_expected_primitives']}",
            f"- avg expected primitives per row: {report['avg_expected_primitives_per_row']:.4f}",
            "",
            "## Expected Primitive Counts",
            "",
        ]
        for tag, count in sorted(report["expected_primitive_counts"].items()):
            lines.append(f"- `{tag}`: {count}")
        lines.extend(["", "## Sample Rows", ""])
        for sample in report["samples"]:
            tags = ", ".join(sample["expected_primitives"])
            lines.append(f"- `{sample['source']}` -> `{tags}`")
        md_path.write_text("\n".join(lines) + "\n")


def write_selected_rows_jsonl(records: Sequence[dict[str, Any]], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main_from_args(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_rows = load_rows(args)
    rows = select_rows(input_rows, args.offset, args.max_rows, args.seed, args.stratify_primitives)
    selected_report = selection_report(rows, args, input_rows)
    write_selection_report(selected_report, args.selection_report_json, args.selection_report_md)
    failures_path = args.failures_jsonl or args.output_jsonl.with_suffix(".failures.jsonl")
    write_selected_rows_jsonl(
        selected_row_records(rows, args.output_jsonl, failures_path, args.resume, args.retry_failures),
        args.selection_jsonl,
    )
    if args.selection_only:
        print(json.dumps(selected_report, indent=2, ensure_ascii=False, sort_keys=True))
        return
    if args.dry_run:
        expected_primitives = expected_primitives_for_row(rows[0]["source"], rows[0]["target"])
        dry_payload = request_payload_with_few_shots(
            args,
            rows[0]["source"],
            rows[0]["target"],
            select_few_shots(rows, rows[0], args.few_shot_count),
        )
        if args.audit:
            dry_payload = {
                "generation": dry_payload,
                "audit": chat_payload(
                    args,
                    args.audit_model or args.model,
                    audit_messages(
                        rows[0]["source"],
                        rows[0]["target"],
                        {
                            "analysis": "[SIGNIFICADO] conserva sentido; [ANTI_COPIA] evita copiar espanol.",
                            "translation": rows[0]["target"],
                            "self_evaluation": "Riesgo bajo.",
                            "score": 0.95,
                            "expected_primitives": expected_primitives,
                        },
                    ),
                    min(args.max_output_tokens, 256),
                ),
            }
        print(json.dumps(dry_payload, indent=2, ensure_ascii=False))
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set {args.api_key_env} in the environment.")

    failures_path = args.failures_jsonl or args.output_jsonl.with_suffix(".failures.jsonl")
    accepted_keys = existing_keys(args.output_jsonl) if args.resume else set()
    failed_keys = existing_keys(failures_path) if args.resume and not args.retry_failures else set()
    completed_keys = accepted_keys.union(failed_keys)
    if not args.resume:
        for path in (args.output_jsonl, failures_path):
            if path.exists():
                path.unlink()

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped_rows = 0
    for index, row in enumerate(rows, start=1):
        key = row_key(row["source"], row["target"])
        if key in completed_keys:
            skipped_rows += 1
            continue
        try:
            expected_primitives = expected_primitives_for_row(row["source"], row["target"])
            few_shots = select_few_shots(rows, row, args.few_shot_count)
            response = call_chat_completion(args, api_key, row["source"], row["target"], few_shots)
            parsed = parse_frontier_json(response_content(response))
            required_primitives = expected_primitives if args.require_expected_primitives else []
            if not record_passes(parsed, args.min_primitive_tags, required_primitives):
                failure = failure_record(
                    index,
                    row,
                    "failed_quality_filter",
                    parsed=parsed,
                    expected_primitives=expected_primitives,
                )
                failures.append(failure)
                append_jsonl(failures_path, failure)
                completed_keys.add(key)
                continue
            audit: dict[str, Any] | None = None
            if args.audit:
                audit_payload = {**parsed, "expected_primitives": expected_primitives}
                audit = audit_record(args, api_key, row["source"], row["target"], audit_payload)
                if not audit_passes(audit, args.audit_min_score):
                    failure = failure_record(
                        index,
                        row,
                        "failed_frontier_audit",
                        parsed=parsed,
                        audit=audit,
                        expected_primitives=expected_primitives,
                    )
                    failures.append(failure)
                    append_jsonl(failures_path, failure)
                    completed_keys.add(key)
                    continue
            record = {
                "row_key": key,
                "source": row["source"],
                "reference": row["target"],
                "target": build_target(parsed, row["target"], args.allow_model_translation),
                "frontier_model": args.model,
                "frontier_analysis": parsed["analysis"],
                "frontier_translation": parsed["translation"],
                "frontier_score": parsed["score"],
                "frontier_audit": audit,
                "expected_primitives": expected_primitives,
                "source_name": row.get("source_name"),
                "variant": row.get("variant"),
                "task": "frontier_synthetic_thinking_translation_generation",
            }
            records.append(record)
            append_jsonl(args.output_jsonl, record)
            completed_keys.add(key)
        except Exception as exc:  # noqa: BLE001 - keep batch generation moving.
            failure = failure_record(index, row, repr(exc))
            failures.append(failure)
            append_jsonl(failures_path, failure)
            completed_keys.add(key)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary = {
        "model": args.model,
        "requested_rows": len(rows),
        "new_written_rows": len(records),
        "new_failed_rows": len(failures),
        "skipped_rows": skipped_rows,
        "existing_accepted_rows": len(accepted_keys),
        "existing_failed_rows": len(failed_keys),
        "total_written_rows": len(existing_keys(args.output_jsonl)),
        "total_failed_rows": len(existing_keys(failures_path)),
        "audit": args.audit,
        "audit_model": (args.audit_model or args.model) if args.audit else None,
        "audit_min_score": args.audit_min_score if args.audit else None,
        "allow_model_translation": args.allow_model_translation,
        "require_expected_primitives": args.require_expected_primitives,
        "stratify_primitives": args.stratify_primitives,
        "resume": args.resume,
        "retry_failures": args.retry_failures,
        "output_jsonl": str(args.output_jsonl),
        "failures_jsonl": str(failures_path),
    }
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"summary": summary, "failures": failures}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main_from_args()
