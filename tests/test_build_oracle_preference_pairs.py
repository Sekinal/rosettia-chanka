from __future__ import annotations

import argparse
import unittest
from unittest import mock

from scripts import build_oracle_preference_pairs as build_preferences
from scripts import rerank_candidate_predictions as oracle_rerank


class BuildOraclePreferencePairsTests(unittest.TestCase):
    def test_build_pair_uses_top_scored_candidate_and_hard_rejected(self):
        group = [
            oracle_rerank.Candidate("src", "ref", "bad", candidate_index=0),
            oracle_rerank.Candidate("src", "ref", "best", candidate_index=1),
            oracle_rerank.Candidate("src", "ref", "hard", candidate_index=2),
        ]
        args = argparse.Namespace(
            min_candidates=3,
            min_margin=0.5,
            max_source_copy_ratio=1.0,
            max_spanish_leakage_penalty=1.0,
            max_chat_artifact_penalty=1.0,
            allow_exact_source_copy=True,
            rejected_strategy="hard",
        )

        with mock.patch.object(
            build_preferences.oracle_rerank,
            "candidate_oracle_score",
            side_effect=lambda candidate: {"bad": 0.1, "best": 2.0, "hard": 1.4}[candidate.prediction],
        ):
            pair = build_preferences.build_pair(group, args)

        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair.chosen, "best")
        self.assertEqual(pair.rejected, "hard")
        self.assertAlmostEqual(pair.margin, 0.6)

    def test_build_pair_rejects_small_margin(self):
        group = [
            oracle_rerank.Candidate("src", "ref", "a", candidate_index=0),
            oracle_rerank.Candidate("src", "ref", "b", candidate_index=1),
        ]
        args = argparse.Namespace(
            min_candidates=2,
            min_margin=0.5,
            max_source_copy_ratio=1.0,
            max_spanish_leakage_penalty=1.0,
            max_chat_artifact_penalty=1.0,
            allow_exact_source_copy=True,
            rejected_strategy="hard",
        )

        with mock.patch.object(
            build_preferences.oracle_rerank,
            "candidate_oracle_score",
            side_effect=lambda candidate: 1.0 if candidate.prediction == "a" else 0.8,
        ):
            pair = build_preferences.build_pair(group, args)

        self.assertIsNone(pair)

    def test_pair_to_record_preserves_preference_fields(self):
        pair = build_preferences.PreferencePair(
            source="src",
            reference="ref",
            chosen="allin",
            rejected="mana",
            chosen_score=2.0,
            rejected_score=0.5,
            chosen_index=3,
            rejected_index=1,
            candidate_count=8,
            source_name="mine",
            variant="quy",
        )

        record = build_preferences.pair_to_record(pair)

        self.assertEqual(record["source"], "src")
        self.assertEqual(record["chosen"], "allin")
        self.assertEqual(record["rejected"], "mana")
        self.assertEqual(record["score_margin"], 1.5)
        self.assertEqual(record["label_type"], "oracle_preference_pair")


if __name__ == "__main__":
    unittest.main()
