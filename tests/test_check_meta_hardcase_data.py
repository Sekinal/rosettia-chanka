from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import check_meta_hardcase_data as checker


def hardcase(source: str, rationale: str = "false_confidence_self_score_too_high") -> dict[str, str]:
    return {
        "source": source,
        "reference": "Allin punchaw.",
        "candidate": "Buenos dias.",
        "analysis": "Puntaje: \\boxed{0.95}",
        "label": json.dumps({"score": 0.1, "severity": "critical", "rationale": rationale}),
    }


class CheckMetaHardcaseDataTests(unittest.TestCase):
    def test_counts_unique_valid_records_across_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "sft.jsonl"
            second = Path(tmpdir) / "gspo.jsonl"
            duplicate = hardcase("Buenos dias.")
            first.write_text(json.dumps(duplicate) + "\n" + json.dumps(hardcase("Buenas tardes.")) + "\n")
            second.write_text(json.dumps(duplicate) + "\n" + json.dumps(hardcase("Hola.")) + "\n")

            metrics = checker.count_records([first, second])
            passed, reasons = checker.check_gate(metrics, min_records=3)

        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertEqual(metrics["records"], 4)
        self.assertEqual(metrics["valid_records"], 3)
        self.assertEqual(metrics["label_rationales"], {"false_confidence_self_score_too_high": 3})

    def test_gate_fails_for_missing_files_and_too_few_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.jsonl"

            metrics = checker.count_records([missing])
            passed, reasons = checker.check_gate(metrics, min_records=1)

        self.assertFalse(passed)
        self.assertIn("missing_files:", reasons[0])
        self.assertIn("valid_records 0 < 1", reasons)

    def test_gate_fails_for_invalid_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text(json.dumps({"source": "missing fields"}) + "\n")

            metrics = checker.count_records([path])
            passed, reasons = checker.check_gate(metrics, min_records=1)

        self.assertFalse(passed)
        self.assertIn("invalid_records 1 > 0", reasons)


if __name__ == "__main__":
    unittest.main()
