from __future__ import annotations

import unittest

from scripts import evaluate_candidate_reranker as evaluate_reranker
from scripts import rerank_candidate_predictions as oracle_rerank


class EvaluateCandidateRerankerTests(unittest.TestCase):
    def test_select_learned_uses_model_scores_without_reference(self):
        class FixedScorer:
            def score_many(self, candidates):
                self.seen = candidates
                return [0.1, 0.9]

        group = [
            oracle_rerank.Candidate("source", "reference", "first", candidate_index=0),
            oracle_rerank.Candidate("source", "reference", "second", candidate_index=1),
        ]
        scorer = FixedScorer()
        selected, scores = evaluate_reranker.select_learned([group], scorer)

        self.assertEqual(selected[0].prediction, "second")
        self.assertEqual(scores, [0.9])
        self.assertEqual([candidate.prediction for candidate in scorer.seen], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
