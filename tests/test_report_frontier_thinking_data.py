from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import report_frontier_thinking_data as report


class ReportFrontierThinkingDataTests(unittest.TestCase):
    def test_report_summarizes_accepted_failures_primitives_and_audits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            accepted = root / "accepted.jsonl"
            failures = root / "failures.jsonl"
            summary = root / "summary.json"
            accepted.write_text(
                json.dumps(
                    {
                        "source": "Buenos dias.",
                        "reference": "Allin punchaw.",
                        "frontier_analysis": "[SIGNIFICADO] conserva; [ANTI_COPIA] evita copia.",
                        "expected_primitives": ["[SIGNIFICADO]", "[ANTI_COPIA]"],
                        "frontier_score": 0.94,
                        "frontier_audit": {"pass": True, "score": 0.88, "reason": "concise"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "source": "Hola.",
                        "reference": "Rimaykullayki.",
                        "frontier_analysis": "[GRAMATICA] natural; [ENTIDADES] sin nombres.",
                        "expected_primitives": ["[GRAMATICA]", "[ENTIDADES]", "[TERMINOLOGIA]"],
                        "frontier_score": 0.92,
                        "frontier_audit": {"pass": True, "score": 0.91, "reason": "grounded"},
                    }
                )
                + "\n"
            )
            failures.write_text(
                json.dumps(
                    {
                        "source": "Gracias.",
                        "reference": "Añay.",
                        "reason": "failed_frontier_audit",
                        "parsed": {"analysis": "[SIGNIFICADO] conserva."},
                        "audit": {"pass": False, "score": 0.2, "reason": "too vague"},
                    }
                )
                + "\n"
            )
            summary.write_text(json.dumps({"summary": {"total_written_rows": 2, "total_failed_rows": 1}}))
            args = report.parse_args(
                [
                    "--output-jsonl",
                    str(accepted),
                    "--failures-jsonl",
                    str(failures),
                    "--summary-json",
                    str(summary),
                    "--report-json",
                    str(root / "report.json"),
                ]
            )

            payload = report.build_report(args)

        self.assertEqual(payload["counts"]["written"], 2)
        self.assertEqual(payload["counts"]["failed"], 1)
        self.assertEqual(payload["primitive_tag_counts"]["[SIGNIFICADO]"], 1)
        self.assertEqual(payload["primitive_tag_counts"]["[GRAMATICA]"], 1)
        self.assertEqual(payload["audit"]["audited_rows"], 2)
        self.assertEqual(payload["expected_primitives"]["rows"], 2)
        self.assertEqual(payload["expected_primitives"]["covered_rows"], 1)
        self.assertEqual(payload["expected_primitives"]["missing_counts"]["[TERMINOLOGIA]"], 1)
        self.assertEqual(payload["failures"]["failure_reasons"]["failed_frontier_audit"], 1)
        self.assertEqual(payload["failures"]["audit_rejection_reasons"]["too vague"], 1)
        self.assertEqual(len(payload["accepted_samples"]), 2)

    def test_markdown_report_contains_core_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.write_markdown(
                {
                    "gate_metrics": {
                        "written_rows": 2.0,
                        "failed_rows": 1.0,
                        "accept_rate": 0.6667,
                        "avg_primitive_tags": 2.0,
                        "distinct_primitives": 4.0,
                        "expected_primitive_coverage": 1.0,
                    },
                    "primitive_tag_counts": {"[SIGNIFICADO]": 2},
                    "expected_primitives": {"missing_counts": {}},
                    "failures": {"failure_reasons": {"failed_quality_filter": 1}},
                    "accepted_samples": [{"source": "Hola.", "frontier_analysis": "[SIGNIFICADO] conserva."}],
                },
                path,
            )

            content = path.read_text()

        self.assertIn("Frontier Thinking Data Report", content)
        self.assertIn("Primitive Tags", content)
        self.assertIn("Missing Expected Primitives", content)
        self.assertIn("failed_quality_filter", content)


if __name__ == "__main__":
    unittest.main()
