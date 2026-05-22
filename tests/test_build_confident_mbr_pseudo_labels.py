from __future__ import annotations

import argparse
import unittest
from unittest import mock

from scripts import build_confident_mbr_pseudo_labels as build_confident
from scripts import rerank_candidate_predictions as oracle_rerank


class BuildConfidentMbrPseudoLabelsTests(unittest.TestCase):
    def test_rank_group_reports_top_candidate_margin_and_peer_utility(self):
        group = [
            oracle_rerank.Candidate("src", "ref", "one", candidate_index=0),
            oracle_rerank.Candidate("src", "ref", "two", candidate_index=1),
        ]
        with mock.patch.object(
            build_confident.mbr,
            "mbr_score",
            side_effect=lambda candidate, _: 0.7 if candidate.prediction == "two" else 0.2,
        ), mock.patch.object(build_confident, "mean_peer_utility", return_value=0.6):
            selection = build_confident.rank_group(group)

        self.assertEqual(selection.candidate.prediction, "two")
        self.assertAlmostEqual(selection.score, 0.7)
        self.assertAlmostEqual(selection.margin, 0.5)
        self.assertAlmostEqual(selection.mean_peer_utility, 0.6)
        self.assertEqual(selection.candidate_count, 2)

    def test_filters_reject_low_margin_and_copy(self):
        candidate = oracle_rerank.Candidate("Copiame", "ref", "Copiame", candidate_index=0)
        selection = build_confident.MbrSelection(candidate, score=0.8, margin=0.01, mean_peer_utility=0.8, candidate_count=8)
        args = argparse.Namespace(
            min_candidates=4,
            min_mbr_score=0.0,
            min_margin=0.02,
            min_mean_peer_utility=0.0,
            max_source_copy_ratio=0.60,
            max_spanish_leakage_penalty=0.25,
            max_chat_artifact_penalty=0.0,
            allow_exact_source_copy=False,
        )

        self.assertFalse(build_confident.passes_filters(selection, args))

    def test_pseudo_record_writes_target_and_diagnostics(self):
        candidate = oracle_rerank.Candidate("src", "ref", "pred", source_name="x", variant="quy", candidate_index=3)
        selection = build_confident.MbrSelection(candidate, score=0.8, margin=0.2, mean_peer_utility=0.7, candidate_count=16)

        record = build_confident.pseudo_record(selection)

        self.assertEqual(record["source"], "src")
        self.assertEqual(record["target"], "pred")
        self.assertEqual(record["prediction"], "pred")
        self.assertEqual(record["label_type"], "pseudo_mbr_confident")
        self.assertEqual(record["candidate_index"], 3)
        self.assertEqual(record["mbr_candidate_count"], 16)
        self.assertAlmostEqual(record["mbr_margin"], 0.2)


if __name__ == "__main__":
    unittest.main()
