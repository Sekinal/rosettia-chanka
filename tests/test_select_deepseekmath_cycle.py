from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import select_deepseekmath_cycle as select_cycle


def write_cycle(root: Path, name: str, *, promoted: bool, chrf: float, bleu: float, policy_exists: bool = True) -> Path:
    cycle_dir = root / name
    cycle_dir.mkdir(parents=True)
    policy = cycle_dir / "policy"
    if policy_exists:
        policy.mkdir()
    metrics = cycle_dir / "metrics.json"
    promotion = cycle_dir / "promotion.json"
    predictions = cycle_dir / "predictions.jsonl"
    metrics.write_text(json.dumps({"chrf++": chrf, "bleu": bleu, "token_f1": chrf / 2, "ter": 90.0}))
    promotion.write_text(json.dumps({"promoted": promoted, "reasons": [] if promoted else ["not promoted"]}))
    predictions.write_text("{}\n")
    manifest = cycle_dir / "cycle_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "stamp": name,
                "stage": "initial_gspo",
                "promoted": promoted,
                "policy_adapter": str(policy),
                "metrics": json.loads(metrics.read_text()),
                "promotion": json.loads(promotion.read_text()),
                "artifacts": {
                    "policy_adapter": {"path": str(policy), "exists": policy_exists, "is_dir": policy_exists},
                    "metrics": {"path": str(metrics), "exists": True, "is_file": True},
                    "promotion": {"path": str(promotion), "exists": True, "is_file": True},
                    "predictions": {"path": str(predictions), "exists": True, "is_file": True},
                },
            }
        )
    )
    return manifest


class SelectDeepSeekMathCycleTests(unittest.TestCase):
    def test_selects_best_passing_promoted_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_cycle(root, "weak", promoted=True, chrf=36.0, bleu=8.2)
            strong = write_cycle(root, "strong", promoted=True, chrf=44.0, bleu=11.0)
            write_cycle(root, "failed_high", promoted=False, chrf=60.0, bleu=20.0)
            args = select_cycle.parse_args([str(root)])

            report = select_cycle.selection_for(args)

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected"]["manifest_json"], str(strong))
        self.assertEqual(report["selected"]["policy_adapter"], str(root / "strong" / "policy"))
        self.assertEqual(report["selected"]["baseline_metrics_json"], str(root / "strong" / "metrics.json"))
        self.assertEqual(report["passing_count"], 2)

    def test_reports_failures_when_nothing_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_cycle(root, "missing_policy", promoted=True, chrf=40.0, bleu=9.0, policy_exists=False)
            write_cycle(root, "low_bleu", promoted=True, chrf=40.0, bleu=2.0)
            args = select_cycle.parse_args([str(root)])

            report = select_cycle.selection_for(args)

        self.assertFalse(report["passed"])
        self.assertIsNone(report["selected"])
        self.assertEqual(report["passing_count"], 0)
        self.assertEqual(report["failed_count"], 2)


if __name__ == "__main__":
    unittest.main()
