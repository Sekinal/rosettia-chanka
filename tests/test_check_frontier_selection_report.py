from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import check_frontier_selection_report as checker


class CheckFrontierSelectionReportTests(unittest.TestCase):
    def test_gate_passes_on_balanced_selection(self):
        report = {
            "selected_rows": 128,
            "distinct_expected_primitives": 5,
            "avg_expected_primitives_per_row": 3.0,
            "expected_primitive_counts": {
                "[SIGNIFICADO]": 128,
                "[GRAMATICA]": 128,
                "[ENTIDADES]": 20,
                "[TERMINOLOGIA]": 16,
                "[ANTI_COPIA]": 92,
            },
        }

        metrics, passed, reasons = checker.gate_selection(
            report,
            min_selected_rows=64,
            min_distinct_expected_primitives=5,
            min_avg_expected_primitives=2.5,
        )

        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertEqual(metrics["selected_rows"], 128)

    def test_gate_fails_on_sparse_or_missing_primitives(self):
        report = {
            "selected_rows": 12,
            "distinct_expected_primitives": 3,
            "avg_expected_primitives_per_row": 1.5,
            "expected_primitive_counts": {
                "[SIGNIFICADO]": 12,
                "[GRAMATICA]": 12,
                "[ANTI_COPIA]": 12,
            },
        }

        _, passed, reasons = checker.gate_selection(
            report,
            min_selected_rows=64,
            min_distinct_expected_primitives=5,
            min_avg_expected_primitives=2.0,
        )

        self.assertFalse(passed)
        self.assertIn("selected_rows 12 < 64", reasons)
        self.assertIn("distinct_expected_primitives 3 < 5", reasons)
        self.assertIn("expected_primitive_counts[[ENTIDADES]] 0 < 1", reasons)
        self.assertIn("expected_primitive_counts[[TERMINOLOGIA]] 0 < 1", reasons)

    def test_main_writes_gate_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selection = root / "selection.json"
            output = root / "gate.json"
            selection.write_text(
                json.dumps(
                    {
                        "selected_rows": 2,
                        "distinct_expected_primitives": 2,
                        "avg_expected_primitives_per_row": 3.0,
                        "expected_primitive_counts": {
                            "[SIGNIFICADO]": 2,
                            "[GRAMATICA]": 2,
                        },
                    }
                )
            )

            status = checker.main(
                [
                    "--selection-report-json",
                    str(selection),
                    "--output-json",
                    str(output),
                    "--min-selected-rows",
                    "2",
                    "--min-distinct-expected-primitives",
                    "2",
                    "--required-primitive",
                    "[SIGNIFICADO]",
                    "--required-primitive",
                    "[GRAMATICA]",
                ]
            )
            payload = json.loads(output.read_text())

        self.assertEqual(status, 0)
        self.assertTrue(payload["passed"])


if __name__ == "__main__":
    unittest.main()
