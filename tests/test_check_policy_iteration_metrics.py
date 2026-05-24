from __future__ import annotations

import argparse
import unittest

from scripts import check_policy_iteration_metrics as checker


def args(**overrides):
    defaults = {
        "min_chrf": 35.0,
        "min_bleu": 8.0,
        "min_token_f1": 15.0,
        "max_ter": 120.0,
        "min_format_rate": 50.0,
        "max_false_confidence_rate": 95.0,
        "max_missing_score_rate": 50.0,
        "min_chrf_delta": -1.0,
        "min_bleu_delta": -1.0,
        "min_token_f1_delta": -1.0,
        "max_false_confidence_delta": 5.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class CheckPolicyIterationMetricsTests(unittest.TestCase):
    def test_promotes_candidate_that_meets_absolute_and_delta_gates(self):
        candidate = {
            "chrf++": 42.0,
            "bleu": 11.0,
            "token_f1": 22.0,
            "ter": 90.0,
            "self_verification_required_format_rate": 75.0,
            "self_verification_false_confidence_rate": 40.0,
            "self_verification_missing_score_rate": 5.0,
        }
        baseline = {
            "chrf++": 41.5,
            "bleu": 10.5,
            "token_f1": 21.0,
            "self_verification_false_confidence_rate": 42.0,
        }

        report = checker.report_for(candidate, baseline, args())

        self.assertTrue(report["promoted"])
        self.assertEqual(report["reasons"], [])
        self.assertAlmostEqual(report["deltas"]["chrf++"], 0.5)
        self.assertAlmostEqual(report["deltas"]["self_verification_false_confidence_rate"], -2.0)

    def test_blocks_candidate_when_reward_style_run_collapses_translation(self):
        candidate = {
            "chrf++": 18.5,
            "bleu": 0.7,
            "token_f1": 6.1,
            "ter": 302.0,
            "self_verification_required_format_rate": 18.0,
            "self_verification_false_confidence_rate": 100.0,
            "self_verification_missing_score_rate": 81.0,
        }

        report = checker.report_for(candidate, None, args())

        self.assertFalse(report["promoted"])
        self.assertIn("chrf++ 18.5000 < 35.0000", report["reasons"])
        self.assertIn("bleu 0.7000 < 8.0000", report["reasons"])
        self.assertIn("ter 302.0000 > 120.0000", report["reasons"])
        self.assertIn("self_verification_missing_score_rate 81.0000 > 50.0000", report["reasons"])

    def test_blocks_candidate_with_large_regression_against_baseline(self):
        candidate = {
            "chrf++": 39.0,
            "bleu": 9.0,
            "token_f1": 19.0,
            "ter": 95.0,
            "self_verification_required_format_rate": 80.0,
            "self_verification_false_confidence_rate": 55.0,
            "self_verification_missing_score_rate": 5.0,
        }
        baseline = {
            "chrf++": 44.0,
            "bleu": 15.0,
            "token_f1": 25.0,
            "self_verification_false_confidence_rate": 45.0,
        }

        report = checker.report_for(candidate, baseline, args())

        self.assertFalse(report["promoted"])
        self.assertIn("chrf++ delta -5.0000 < -1.0000", report["reasons"])
        self.assertIn("bleu delta -6.0000 < -1.0000", report["reasons"])
        self.assertIn("self_verification_false_confidence_rate delta 10.0000 > 5.0000", report["reasons"])


if __name__ == "__main__":
    unittest.main()
