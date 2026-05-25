from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import check_frontier_thinking_data as checker


class CheckFrontierThinkingDataTests(unittest.TestCase):
    def test_summary_gate_passes_when_enough_rows_and_accept_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "summary.json"
            summary.write_text(
                json.dumps({"summary": {"total_written_rows": 80, "total_failed_rows": 20, "requested_rows": 100}})
            )
            args = checker.parse_args(
                ["--summary-json", str(summary), "--min-written-rows", "64", "--min-accept-rate", "0.5"]
            )

            metrics = checker.gate_metrics(checker.load_counts(args))
            passed, reasons = checker.check_gate(metrics, args.min_written_rows, args.min_accept_rate)

        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertEqual(metrics["written_rows"], 80)
        self.assertEqual(metrics["accept_rate"], 0.8)

    def test_summary_gate_fails_when_rows_are_sparse(self):
        counts = {"written": 12, "failed": 28, "requested": 40}

        metrics = checker.gate_metrics(counts)
        passed, reasons = checker.check_gate(metrics, min_written_rows=32, min_accept_rate=0.5)

        self.assertFalse(passed)
        self.assertIn("written_rows 12 < 32", reasons)
        self.assertIn("accept_rate 0.3000 < 0.5000", reasons)

    def test_counts_jsonl_when_summary_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.jsonl"
            failures = Path(tmpdir) / "failures.jsonl"
            output.write_text('{"row_key":"a"}\n{"row_key":"b"}\n')
            failures.write_text('{"row_key":"c"}\n')
            args = checker.parse_args(["--output-jsonl", str(output), "--failures-jsonl", str(failures)])

            counts = checker.load_counts(args)

        self.assertEqual(counts, {"written": 2, "failed": 1, "requested": 3})

    def test_primitive_coverage_gate_passes_on_diverse_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "frontier_analysis": "[SIGNIFICADO] conserva; [GRAMATICA] revisa sufijo.",
                        "expected_primitives": ["[SIGNIFICADO]", "[GRAMATICA]"],
                        "row_key": "a",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "frontier_analysis": "[ENTIDADES] preserva nombre; [ANTI_COPIA] evita espanol.",
                        "expected_primitives": ["[ENTIDADES]", "[ANTI_COPIA]"],
                        "row_key": "b",
                    }
                )
                + "\n"
            )

            primitives = checker.primitive_counts(output)
            metrics = checker.gate_metrics(
                {"written": 2, "failed": 0, "requested": 2},
                primitives,
                min_tags_per_row=2,
            )
            passed, reasons = checker.check_gate(
                metrics,
                min_written_rows=2,
                min_accept_rate=0.5,
                min_primitive_tags_per_row=2,
                min_primitive_row_rate=1.0,
                min_distinct_primitives=4,
                min_expected_primitive_coverage=1.0,
            )

        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertEqual(metrics["avg_primitive_tags"], 2.0)
        self.assertEqual(metrics["distinct_primitives"], 4.0)
        self.assertEqual(metrics["primitive_row_rate"], 1.0)
        self.assertEqual(metrics["expected_primitive_coverage"], 1.0)

    def test_primitive_coverage_gate_fails_on_one_note_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.jsonl"
            output.write_text(
                json.dumps({"target": "Analisis de traduccion: [SIGNIFICADO] conserva.", "row_key": "a"}) + "\n"
            )

            metrics = checker.gate_metrics(
                {"written": 1, "failed": 0, "requested": 1},
                checker.primitive_counts(output),
                min_tags_per_row=2,
            )
            passed, reasons = checker.check_gate(
                metrics,
                min_written_rows=1,
                min_accept_rate=0.5,
                min_primitive_tags_per_row=2,
                min_primitive_row_rate=0.9,
                min_distinct_primitives=4,
            )

        self.assertFalse(passed)
        self.assertIn("avg_primitive_tags 1.0000 < 2.0000", reasons)
        self.assertIn("primitive_row_rate 0.0000 < 0.9000", reasons)
        self.assertIn("distinct_primitives 1 < 4", reasons)

    def test_expected_primitive_coverage_gate_fails_when_required_tags_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "target": "Analisis de traduccion: [SIGNIFICADO] conserva; [GRAMATICA] natural.",
                        "expected_primitives": ["[SIGNIFICADO]", "[GRAMATICA]", "[ENTIDADES]"],
                        "row_key": "a",
                    }
                )
                + "\n"
            )

            primitives = checker.primitive_counts(output)
            metrics = checker.gate_metrics({"written": 1, "failed": 0, "requested": 1}, primitives)
            passed, reasons = checker.check_gate(
                metrics,
                min_written_rows=1,
                min_accept_rate=0.5,
                min_expected_primitive_coverage=1.0,
            )

        self.assertFalse(passed)
        self.assertEqual(primitives["missing_expected_primitive_counts"]["[ENTIDADES]"], 1)
        self.assertIn("expected_primitive_coverage 0.0000 < 1.0000", reasons)

    def test_analysis_content_gate_passes_on_specific_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "source": "Hay 3 documentos judiciales.",
                        "reference": "Kimsa qillqa kasqan.",
                        "frontier_analysis": (
                            "[ENTIDADES] conserva el numero 3; "
                            "[GRAMATICA] revisa forma chanka natural y sufijo verbal."
                        ),
                        "expected_primitives": ["[ENTIDADES]", "[GRAMATICA]"],
                    }
                )
                + "\n"
            )

            primitives = checker.primitive_counts(output)
            metrics = checker.gate_metrics(
                {"written": 1, "failed": 0, "requested": 1},
                primitives,
                min_tags_per_row=2,
                min_analysis_words=6,
            )
            passed, reasons = checker.check_gate(
                metrics,
                min_written_rows=1,
                min_accept_rate=0.5,
                min_primitive_tags_per_row=2,
                min_primitive_row_rate=1.0,
                min_analysis_words=6,
                min_analysis_word_row_rate=1.0,
                min_specific_analysis_rate=1.0,
            )

        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertGreaterEqual(metrics["avg_analysis_words"], 6)
        self.assertEqual(metrics["analysis_word_row_rate"], 1.0)
        self.assertEqual(metrics["specific_analysis_rate"], 1.0)

    def test_analysis_content_gate_fails_on_vacuous_tag_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "source": "Hay 3 documentos judiciales.",
                        "reference": "Kimsa qillqa kasqan.",
                        "frontier_analysis": "[SIGNIFICADO] correcto; [GRAMATICA] correcto.",
                        "expected_primitives": ["[SIGNIFICADO]", "[GRAMATICA]"],
                    }
                )
                + "\n"
            )

            metrics = checker.gate_metrics(
                {"written": 1, "failed": 0, "requested": 1},
                checker.primitive_counts(output),
                min_tags_per_row=2,
                min_analysis_words=6,
            )
            passed, reasons = checker.check_gate(
                metrics,
                min_written_rows=1,
                min_accept_rate=0.5,
                min_analysis_words=6,
                min_analysis_word_row_rate=1.0,
                min_specific_analysis_rate=1.0,
            )

        self.assertFalse(passed)
        self.assertIn("avg_analysis_words 1.0000 < 6.0000", reasons)
        self.assertIn("analysis_word_row_rate 0.0000 < 1.0000", reasons)

    def test_audit_gate_passes_when_all_rows_are_audited_and_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "frontier_analysis": "[SIGNIFICADO] conserva documentos; [GRAMATICA] revisa forma chanka natural.",
                        "frontier_audit": {"pass": True, "score": 0.85, "reason": "specific"},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "frontier_analysis": "[ENTIDADES] conserva Quinua; [ANTI_COPIA] evita copiar calle literal.",
                        "frontier_audit": {"pass": True, "score": 0.95, "reason": "specific"},
                    }
                )
                + "\n"
            )

            primitives = checker.primitive_counts(output)
            metrics = checker.gate_metrics({"written": 2, "failed": 0, "requested": 2}, primitives)
            passed, reasons = checker.check_gate(
                metrics,
                min_written_rows=2,
                min_accept_rate=0.5,
                min_audited_row_rate=1.0,
                min_audit_pass_rate=1.0,
                min_avg_audit_score=0.75,
            )

        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertEqual(metrics["audited_row_rate"], 1.0)
        self.assertEqual(metrics["audit_pass_rate"], 1.0)
        self.assertAlmostEqual(metrics["avg_audit_score"], 0.9)

    def test_audit_gate_fails_when_rows_are_unaudited_or_low_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "frontier_analysis": "[SIGNIFICADO] conserva documentos; [GRAMATICA] revisa forma chanka natural.",
                        "frontier_audit": {"pass": True, "score": 0.60, "reason": "weak"},
                    }
                )
                + "\n"
                + json.dumps({"frontier_analysis": "[ENTIDADES] conserva Quinua."})
                + "\n"
            )

            metrics = checker.gate_metrics(
                {"written": 2, "failed": 0, "requested": 2},
                checker.primitive_counts(output),
            )
            passed, reasons = checker.check_gate(
                metrics,
                min_written_rows=2,
                min_accept_rate=0.5,
                min_audited_row_rate=1.0,
                min_audit_pass_rate=1.0,
                min_avg_audit_score=0.75,
            )

        self.assertFalse(passed)
        self.assertIn("audited_row_rate 0.5000 < 1.0000", reasons)
        self.assertIn("audit_pass_rate 0.5000 < 1.0000", reasons)
        self.assertIn("avg_audit_score 0.6000 < 0.7500", reasons)


if __name__ == "__main__":
    unittest.main()
