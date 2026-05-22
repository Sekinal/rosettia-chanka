from __future__ import annotations

import unittest
from unittest import mock

from scripts import mbr_candidate_predictions as mbr
from scripts import rerank_candidate_predictions as oracle_rerank


class MbrCandidatePredictionsTests(unittest.TestCase):
    def test_select_mbr_prefers_candidate_with_high_peer_utility(self):
        group = [
            oracle_rerank.Candidate("source", "reference", "alpha beta", candidate_index=0),
            oracle_rerank.Candidate("source", "reference", "alpha beta gamma", candidate_index=1),
            oracle_rerank.Candidate("source", "reference", "zzz", candidate_index=2),
        ]
        with mock.patch.object(
            mbr,
            "pairwise_utility",
            side_effect=lambda hyp, ref: 0.9 if hyp == "alpha beta gamma" and "alpha" in ref else 0.1,
        ):
            selected = mbr.select_mbr([group])

        self.assertEqual(selected[0].candidate_index, 1)

    def test_mbr_metrics_report_non_first_rate(self):
        selected = [
            oracle_rerank.Candidate("s1", "r1", "p1", candidate_index=0),
            oracle_rerank.Candidate("s2", "r2", "p2", candidate_index=2),
        ]
        with mock.patch.object(
            oracle_rerank.gspo,
            "corpus_metrics",
            return_value={
                "chrf++": 10.0,
                "bleu": 1.0,
                "token_f1": 5.0,
                "length_ratio_score": 90.0,
                "source_copy_ratio": 0.0,
                "exact_source_copy_rate": 0.0,
                "spanish_leakage_penalty": 0.0,
                "chat_artifact_penalty": 0.0,
                "ter": 80.0,
            },
        ):
            metrics = mbr.metrics_for_selection(selected, "mbr", predictions_jsonl=__file__, total_candidates=8)

        self.assertEqual(metrics["mbr_non_first_rate"], 50.0)
        self.assertEqual(metrics["mbr_mean_selected_index"], 1.0)


if __name__ == "__main__":
    unittest.main()
