from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_deepseekmath_cycles as summarize


def manifest(stamp: str, promoted: bool, chrf: float, bleu: float, output_hardcases: int = 0) -> dict:
    return {
        "stamp": stamp,
        "stage": "sft_seed" if not promoted else "initial_gspo",
        "promoted": promoted,
        "metrics": {
            "chrf++": chrf,
            "bleu": bleu,
            "token_f1": chrf / 2,
            "ter": 100 - chrf,
            "self_verification_required_format_rate": 80.0,
            "self_verification_false_confidence_rate": 20.0 if promoted else 90.0,
            "self_verification_missing_score_rate": 5.0,
        },
        "promotion": {"promoted": promoted, "reasons": [] if promoted else ["regressed"]},
        "artifacts": {
            "metrics": {"path": "metrics.json", "exists": True},
            "promotion": {"path": "promotion.json", "exists": True},
            "predictions": {"path": "predictions.jsonl", "exists": promoted},
            "baseline_metrics": {"path": "baseline.json", "exists": False},
        },
        "input_hardcases": {"valid_records": 32},
        "output_hardcases": {"valid_records": output_hardcases},
    }


class SummarizeDeepSeekMathCyclesTests(unittest.TestCase):
    def test_collect_manifests_prioritizes_promoted_cycles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, payload in [
                ("failed_high_chrf", manifest("failed", False, 50.0, 20.0, output_hardcases=200)),
                ("promoted_lower_chrf", manifest("promoted", True, 42.0, 12.0, output_hardcases=10)),
            ]:
                path = root / name / "cycle_manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload))

            records = summarize.collect_manifests(root)

        self.assertEqual(records[0]["stamp"], "promoted")
        self.assertTrue(records[0]["promoted"])
        self.assertIn("cycle_score", records[0])
        self.assertEqual(records[0]["artifact_missing_count"], 0)
        self.assertEqual(records[1]["missing_artifacts"], ["predictions"])

    def test_write_markdown_includes_hardcase_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.md"
            payload = manifest("cycle-a", True, 40.0, 10.0, output_hardcases=7)
            payload["promotion"]["deltas"] = {
                "chrf++": 1.25,
                "bleu": 0.5,
                "token_f1": 0.75,
                "self_verification_false_confidence_rate": -2.0,
            }
            payload["artifact_missing_count"] = 0
            summarize.write_markdown([payload], path)

            content = path.read_text()

        self.assertIn("cycle-a", content)
        self.assertIn("Stage", content)
        self.assertIn("initial_gspo", content)
        self.assertIn("delta chrF++", content)
        self.assertIn("1.2500", content)
        self.assertIn("missing artifacts", content)
        self.assertIn("output hardcases", content)
        self.assertIn("Promoted", content)


if __name__ == "__main__":
    unittest.main()
