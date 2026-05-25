"""Extract hardcase JSONL paths from a DeepSeekMath cycle manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument(
        "--include-input",
        action="store_true",
        help="Also include input_hardcases artifacts from the manifest.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Exit successfully even when no usable hardcase JSONL paths are found.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return payload


def artifact_path(artifact: Any) -> str | None:
    if not isinstance(artifact, dict):
        return None
    path = artifact.get("path")
    if not path:
        return None
    if artifact.get("exists") is False:
        return None
    return str(path)


def valid_record_count(record: Any) -> int | None:
    if not isinstance(record, dict):
        return None
    value = record.get("valid_records")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def hardcase_paths(manifest: dict[str, Any], include_input: bool = False) -> list[str]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    paths: list[str] = []

    output_count = valid_record_count(manifest.get("output_hardcases"))
    output_path = artifact_path(artifacts.get("output_hardcases"))
    if output_path and (output_count is None or output_count > 0):
        paths.append(output_path)

    if include_input:
        input_artifacts = artifacts.get("input_hardcases")
        input_counts = manifest.get("input_hardcases")
        if isinstance(input_artifacts, list):
            for index, artifact in enumerate(input_artifacts):
                path = artifact_path(artifact)
                if not path:
                    continue
                count = None
                if isinstance(input_counts, dict):
                    by_file = input_counts.get("files")
                    if isinstance(by_file, list) and index < len(by_file):
                        count = valid_record_count(by_file[index])
                if count is None or count > 0:
                    paths.append(path)

    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def report_for(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest_json)
    paths = hardcase_paths(manifest, include_input=args.include_input)
    return {
        "passed": bool(paths) or args.allow_empty,
        "manifest_json": str(args.manifest_json),
        "include_input": bool(args.include_input),
        "paths": paths,
        "colon_joined": ":".join(paths),
        "path_count": len(paths),
        "reasons": [] if paths or args.allow_empty else ["no usable hardcase JSONL paths found"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = report_for(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report["colon_joined"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
