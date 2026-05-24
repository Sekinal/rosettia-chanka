from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import check_deepseekmath_cycle_manifest as check_cycle


def write_manifest(root: Path, *, promoted: bool, policy_exists: bool = True) -> Path:
    policy = root / "policy_adapter"
    if policy_exists:
        policy.mkdir()
    metrics = root / "metrics.json"
    promotion = root / "promotion.json"
    predictions = root / "predictions.jsonl"
    meta = root / "meta_adapter"
    followup = root / "followup"
    metrics.write_text(json.dumps({"chrf++": 42.0, "bleu": 9.0, "token_f1": 21.0, "ter": 88.0}))
    promotion.write_text(json.dumps({"promoted": promoted, "reasons": [] if promoted else ["regressed"]}))
    predictions.write_text("{}\n")
    meta.mkdir()
    followup.mkdir()
    path = root / "cycle_manifest.json"
    path.write_text(
        json.dumps(
            {
                "stamp": "cycle",
                "stage": "initial_gspo",
                "promoted": promoted,
                "policy_adapter": str(policy),
                "metrics": json.loads(metrics.read_text()),
                "promotion": json.loads(promotion.read_text()),
                "artifacts": {
                    "policy_adapter": {"path": str(policy), "exists": policy_exists, "is_dir": policy_exists},
                    "meta_verifier_adapter": {"path": str(meta), "exists": True, "is_dir": True},
                    "followup_output_dir": {"path": str(followup), "exists": True, "is_dir": True},
                    "metrics": {"path": str(metrics), "exists": True, "is_file": True},
                    "promotion": {"path": str(promotion), "exists": True, "is_file": True},
                    "predictions": {"path": str(predictions), "exists": True, "is_file": True},
                },
            }
        )
    )
    return path


class CheckDeepSeekMathCycleManifestTests(unittest.TestCase):
    def test_promoted_complete_manifest_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = write_manifest(root, promoted=True)
            args = check_cycle.parse_args(["--manifest-json", str(path)])
            record = check_cycle.summarize.load_manifest(path)
            report = check_cycle.report_for(record, args)

        self.assertTrue(report["passed"])
        self.assertEqual(report["policy_adapter"], str(root / "policy_adapter"))
        self.assertEqual(report["reasons"], [])

    def test_unpromoted_or_missing_policy_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = write_manifest(root, promoted=False, policy_exists=False)
            args = check_cycle.parse_args(["--manifest-json", str(path)])
            record = check_cycle.summarize.load_manifest(path)
            report = check_cycle.report_for(record, args)

        self.assertFalse(report["passed"])
        self.assertIn("cycle is not promoted", report["reasons"])
        self.assertIn("policy_adapter artifact is missing", report["reasons"])


if __name__ == "__main__":
    unittest.main()
