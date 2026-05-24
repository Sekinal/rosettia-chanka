from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import report_frontier_preapi_readiness as reporter


def preview_record(row_key: str = "hola\trimaykullayki") -> dict:
    return {
        "row_key": row_key,
        "base_url": "https://api.deepseek.com",
        "source": "Hola.",
        "reference": "Rimaykullayki.",
        "expected_primitives": ["[SIGNIFICADO]", "[GRAMATICA]", "[ANTI_COPIA]"],
        "few_shot_sources": ["Gracias."],
        "few_shot_row_keys": ["gracias\tanay"],
        "generation_payload": {
            "model": "deepseek-v4-pro",
            "max_tokens": 512,
            "reasoning_effort": "max",
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
            "messages": [
                {"role": "system", "content": "Return JSON."},
                {
                    "role": "user",
                    "content": (
                        "Required primitive tags for this row: "
                        "[SIGNIFICADO], [GRAMATICA], [ANTI_COPIA]."
                    ),
                },
            ],
        },
    }


class ReportFrontierPreapiReadinessTests(unittest.TestCase):
    def test_build_report_marks_ready_when_selection_and_prompt_gates_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection_report = root / "selection_report.json"
            selection_gate = root / "selection_gate.json"
            selection_jsonl = root / "selection.jsonl"
            preview_jsonl = root / "preview.jsonl"
            prompt_gate = root / "prompt_gate.json"
            selection_report.write_text(
                json.dumps(
                    {
                        "selected_rows": 1,
                        "estimated_max_frontier_requests": 2,
                        "expected_primitive_counts": {
                            "[SIGNIFICADO]": 1,
                            "[GRAMATICA]": 1,
                            "[ANTI_COPIA]": 1,
                        },
                    }
                )
            )
            selection_gate.write_text(json.dumps({"passed": True}))
            prompt_gate.write_text(json.dumps({"passed": True}))
            selection_jsonl.write_text(
                json.dumps(
                    {
                        "row_key": "hola\trimaykullayki",
                        "source": "Hola.",
                        "reference": "Rimaykullayki.",
                        "expected_primitives": ["[SIGNIFICADO]", "[GRAMATICA]", "[ANTI_COPIA]"],
                        "source_name": "rows.jsonl",
                        "variant": "quy/chanka",
                        "resume_status": "pending",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            preview_jsonl.write_text(json.dumps(preview_record(), ensure_ascii=False) + "\n")
            args = reporter.parse_args(
                [
                    "--selection-report-json",
                    str(selection_report),
                    "--selection-gate-json",
                    str(selection_gate),
                    "--selection-jsonl",
                    str(selection_jsonl),
                    "--prompt-preview-jsonl",
                    str(preview_jsonl),
                    "--prompt-preview-gate-json",
                    str(prompt_gate),
                    "--output-json",
                    str(root / "report.json"),
                    "--output-md",
                    str(root / "report.md"),
                ]
            )

            report = reporter.build_report(args)
            status = reporter.main(
                [
                    "--selection-report-json",
                    str(selection_report),
                    "--selection-gate-json",
                    str(selection_gate),
                    "--selection-jsonl",
                    str(selection_jsonl),
                    "--prompt-preview-jsonl",
                    str(preview_jsonl),
                    "--prompt-preview-gate-json",
                    str(prompt_gate),
                    "--output-json",
                    str(root / "report.json"),
                    "--output-md",
                    str(root / "report.md"),
                ]
            )
            markdown = (root / "report.md").read_text()

        self.assertTrue(report["ready_for_api"])
        self.assertEqual(status, 0)
        self.assertEqual(report["payload_stats"]["models"], {"deepseek-v4-pro": 1})
        self.assertEqual(report["payload_stats"]["base_urls"], {"https://api.deepseek.com": 1})
        self.assertEqual(report["payload_stats"]["reasoning_efforts"], {"max": 1})
        self.assertIn("ready for API: True", markdown)
        self.assertIn("base URLs", markdown)
        self.assertIn("Required primitive tags for this row", markdown)

    def test_report_not_ready_when_preview_count_mismatches_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection_report = root / "selection_report.json"
            selection_jsonl = root / "selection.jsonl"
            preview_jsonl = root / "preview.jsonl"
            selection_report.write_text(json.dumps({"selected_rows": 2}))
            selection_jsonl.write_text(json.dumps({"row_key": "a\tb"}) + "\n" + json.dumps({"row_key": "c\td"}) + "\n")
            preview_jsonl.write_text(json.dumps(preview_record("a\tb"), ensure_ascii=False) + "\n")
            args = reporter.parse_args(
                [
                    "--selection-report-json",
                    str(selection_report),
                    "--selection-jsonl",
                    str(selection_jsonl),
                    "--prompt-preview-jsonl",
                    str(preview_jsonl),
                    "--output-json",
                    str(root / "report.json"),
                ]
            )

            report = reporter.build_report(args)

        self.assertFalse(report["ready_for_api"])
        self.assertEqual(report["prompt_preview_rows"], 1)

    def test_main_only_fails_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection_report = root / "selection_report.json"
            selection_jsonl = root / "selection.jsonl"
            preview_jsonl = root / "preview.jsonl"
            output_json = root / "report.json"
            selection_report.write_text(json.dumps({"selected_rows": 2}))
            selection_jsonl.write_text(json.dumps({"row_key": "a\tb"}) + "\n" + json.dumps({"row_key": "c\td"}) + "\n")
            preview_jsonl.write_text(json.dumps(preview_record("a\tb"), ensure_ascii=False) + "\n")
            argv = [
                "--selection-report-json",
                str(selection_report),
                "--selection-jsonl",
                str(selection_jsonl),
                "--prompt-preview-jsonl",
                str(preview_jsonl),
                "--output-json",
                str(output_json),
            ]

            default_status = reporter.main(argv)
            strict_status = reporter.main([*argv, "--fail-if-not-ready"])

        self.assertEqual(default_status, 0)
        self.assertEqual(strict_status, 1)


if __name__ == "__main__":
    unittest.main()
