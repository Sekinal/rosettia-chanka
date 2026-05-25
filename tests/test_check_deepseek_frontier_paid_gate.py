from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import check_deepseek_frontier_paid_gate as checker


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_frontier_report(path: Path, written_rows: int = 8) -> None:
    write_json(path, {"gate_metrics": {"written_rows": written_rows}})


class CheckDeepSeekFrontierPaidGateTests(unittest.TestCase):
    def test_passes_when_paid_gate_passed_and_report_has_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = root / "deepseek_v4_pro_paid_smoke_gate.json"
            report = root / "deepseek_v4_pro_thinking_report.json"
            write_json(gate, {"passed": True})
            write_frontier_report(report)

            result = checker.check_paid_gate(gate, report)

        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])

    def test_fails_when_paid_gate_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = root / "deepseek_v4_pro_thinking_report.json"
            write_frontier_report(report)

            result = checker.check_paid_gate(root / "missing.json", report)

        self.assertFalse(result["passed"])
        self.assertIn("paid gate JSON missing or invalid", result["reasons"][0])

    def test_fails_when_paid_gate_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = root / "deepseek_v4_pro_paid_smoke_gate.json"
            report = root / "deepseek_v4_pro_thinking_report.json"
            write_json(gate, {"passed": False})
            write_frontier_report(report)

            result = checker.check_paid_gate(gate, report)

        self.assertFalse(result["passed"])
        self.assertIn("paid gate did not pass", result["reasons"])

    def test_fails_when_report_has_no_accepted_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate = root / "deepseek_v4_pro_paid_smoke_gate.json"
            report = root / "deepseek_v4_pro_thinking_report.json"
            write_json(gate, {"passed": True})
            write_frontier_report(report, written_rows=0)

            result = checker.check_paid_gate(gate, report)

        self.assertFalse(result["passed"])
        self.assertIn("frontier report has no accepted rows", result["reasons"])


if __name__ == "__main__":
    unittest.main()
