from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import extract_deepseekmath_manifest_hardcases as extract


def manifest_payload(output_path: Path, input_path: Path | None = None) -> dict:
    payload = {
        "artifacts": {
            "output_hardcases": {"path": str(output_path), "exists": True, "is_file": True},
            "input_hardcases": [],
        },
        "output_hardcases": {"valid_records": 7},
        "input_hardcases": {"valid_records": 0, "files": []},
    }
    if input_path is not None:
        payload["artifacts"]["input_hardcases"].append(
            {"path": str(input_path), "exists": True, "is_file": True}
        )
        payload["input_hardcases"] = {
            "valid_records": 3,
            "files": [{"path": str(input_path), "valid_records": 3}],
        }
    return payload


class ExtractDeepSeekMathManifestHardcasesTests(unittest.TestCase):
    def test_extracts_output_hardcases_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "output.jsonl"
            input_path = root / "input.jsonl"
            manifest = root / "cycle_manifest.json"
            manifest.write_text(json.dumps(manifest_payload(output, input_path)))

            report = extract.report_for(
                extract.parse_args(["--manifest-json", str(manifest)])
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["paths"], [str(output)])
        self.assertEqual(report["colon_joined"], str(output))

    def test_can_include_input_hardcases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "output.jsonl"
            input_path = root / "input.jsonl"
            manifest = root / "cycle_manifest.json"
            manifest.write_text(json.dumps(manifest_payload(output, input_path)))

            report = extract.report_for(
                extract.parse_args(["--manifest-json", str(manifest), "--include-input"])
            )

        self.assertEqual(report["paths"], [str(output), str(input_path)])
        self.assertEqual(report["colon_joined"], f"{output}:{input_path}")

    def test_zero_output_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "output.jsonl"
            manifest = root / "cycle_manifest.json"
            payload = manifest_payload(output)
            payload["output_hardcases"] = {"valid_records": 0}
            manifest.write_text(json.dumps(payload))

            report = extract.report_for(
                extract.parse_args(["--manifest-json", str(manifest)])
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["paths"], [])
        self.assertIn("no usable hardcase", report["reasons"][0])

    def test_missing_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "output.jsonl"
            manifest = root / "cycle_manifest.json"
            payload = manifest_payload(output)
            payload["artifacts"]["output_hardcases"]["exists"] = False
            manifest.write_text(json.dumps(payload))

            report = extract.report_for(
                extract.parse_args(["--manifest-json", str(manifest)])
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["paths"], [])


if __name__ == "__main__":
    unittest.main()
