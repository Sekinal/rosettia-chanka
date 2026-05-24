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
                        "row_key": "a",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "frontier_analysis": "[ENTIDADES] preserva nombre; [ANTI_COPIA] evita espanol.",
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
            )

        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertEqual(metrics["avg_primitive_tags"], 2.0)
        self.assertEqual(metrics["distinct_primitives"], 4.0)
        self.assertEqual(metrics["primitive_row_rate"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
