from __future__ import annotations

import unittest

from scripts import evaluate_pairwise_candidate_reranker as evaluate_pairwise
from scripts import rerank_candidate_predictions as oracle_rerank


class EvaluatePairwiseCandidateRerankerTests(unittest.TestCase):
    def test_parse_winner_accepts_json_and_loose_letter(self):
        self.assertEqual(evaluate_pairwise.parse_winner('{"winner":"A"}'), "A")
        self.assertEqual(evaluate_pairwise.parse_winner("B"), "B")
        self.assertIsNone(evaluate_pairwise.parse_winner("neither"))

    def test_select_pairwise_uses_tournament_votes(self):
        class FixedScorer:
            def predict_many(self, pairs):
                self.seen = pairs
                return ["B", "B", "A"]

        group = [
            oracle_rerank.Candidate("source", "reference", "first", candidate_index=0),
            oracle_rerank.Candidate("source", "reference", "second", candidate_index=1),
            oracle_rerank.Candidate("source", "reference", "third", candidate_index=2),
        ]

        selected, stats = evaluate_pairwise.select_pairwise([group], FixedScorer())

        self.assertEqual(selected[0].prediction, "second")
        self.assertEqual(stats["pairwise_total_pairs"], 3.0)
        self.assertEqual(stats["pairwise_parsed_pairs"], 3.0)
        self.assertEqual(stats["pairwise_non_first_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
