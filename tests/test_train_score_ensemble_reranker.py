from __future__ import annotations

import unittest

from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_feature_candidate_reranker as feature_reranker
from scripts import train_score_ensemble_reranker as ensemble_reranker


class TrainScoreEnsembleRerankerTests(unittest.TestCase):
    def test_zscore_handles_constant_values(self):
        self.assertEqual(ensemble_reranker.zscore([3.0, 3.0]), [0.0, 0.0])

    def test_select_ensemble_combines_signals(self):
        group = [
            ensemble_reranker.EnsembleRow(
                row=feature_reranker.CandidateFeatures(
                    oracle_rerank.Candidate("src", "ref", "a", candidate_index=0),
                    raw={},
                    oracle_score=0.0,
                ),
                signals={"feature_score": -1.0, "text_score": 1.0},
            ),
            ensemble_reranker.EnsembleRow(
                row=feature_reranker.CandidateFeatures(
                    oracle_rerank.Candidate("src", "ref", "b", candidate_index=1),
                    raw={},
                    oracle_score=1.0,
                ),
                signals={"feature_score": 1.0, "text_score": -1.0},
            ),
        ]
        model = ensemble_reranker.EnsembleModel(
            signal_names=["feature_score", "text_score"],
            weights={"feature_score": 1.0, "text_score": 0.0},
        )

        selected = ensemble_reranker.select_ensemble([group], model)

        self.assertEqual(selected[0].row.candidate.prediction, "b")

    def test_train_ensemble_model_can_improve_toward_oracle(self):
        group = [
            ensemble_reranker.EnsembleRow(
                row=feature_reranker.CandidateFeatures(
                    oracle_rerank.Candidate("src", "ref", "bad", candidate_index=0),
                    raw={},
                    oracle_score=0.0,
                ),
                signals={name: -1.0 for name in ensemble_reranker.SIGNAL_NAMES},
            ),
            ensemble_reranker.EnsembleRow(
                row=feature_reranker.CandidateFeatures(
                    oracle_rerank.Candidate("src", "ref", "good", candidate_index=1),
                    raw={},
                    oracle_score=1.0,
                ),
                signals={name: 1.0 for name in ensemble_reranker.SIGNAL_NAMES},
            ),
        ]

        model, diagnostics = ensemble_reranker.train_ensemble_model(
            [group],
            seed=1,
            search_iterations=10,
            initial_noise=0.1,
            min_noise=0.01,
        )

        self.assertEqual(diagnostics["training_objective"], "score_ensemble_hillclimb")
        self.assertEqual(ensemble_reranker.select_ensemble([group], model)[0].row.candidate.prediction, "good")


if __name__ == "__main__":
    unittest.main()
