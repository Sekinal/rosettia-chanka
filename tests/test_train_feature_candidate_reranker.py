from __future__ import annotations

import unittest
from unittest import mock

from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_feature_candidate_reranker as feature_reranker


class TrainFeatureCandidateRerankerTests(unittest.TestCase):
    def test_feature_rows_include_reference_free_signals(self):
        group = [
            oracle_rerank.Candidate("Buenos dias.", "Allin punchaw.", "Allin punchaw.", candidate_index=0),
            oracle_rerank.Candidate("Buenos dias.", "Allin punchaw.", "Buenos dias.", candidate_index=1),
        ]

        rows = feature_reranker.feature_rows_for_group(group)

        self.assertEqual(len(rows), 2)
        self.assertIn("mbr_score", rows[0].raw)
        self.assertIn("duplicate_rate", rows[0].raw)
        self.assertIn("source_root_copy_ratio", rows[0].raw)
        self.assertIn("terminology_target_coverage", rows[0].raw)
        self.assertEqual(rows[1].raw["exact_source_copy"], 1.0)
        self.assertGreaterEqual(rows[0].oracle_score, 0.0)

    def test_source_root_copy_ratio_detects_suffix_copying(self):
        ratio = feature_reranker.source_root_copy_ratio("En la fiesta", "Fiestapim")

        self.assertEqual(ratio, 1.0)

    def test_terminology_features_detect_target_coverage_and_source_root_leakage(self):
        terminology = [("fiesta", "Raymi")]

        covered = feature_reranker.terminology_features("En la fiesta", "Raymipim", terminology, 1)
        leaked = feature_reranker.terminology_features("En la fiesta", "Fiestapi", terminology, 1)

        self.assertEqual(covered, (1.0, 0.0))
        self.assertEqual(leaked, (0.0, 1.0))

    def test_select_feature_prefers_high_weighted_candidate(self):
        group = [
            oracle_rerank.Candidate("src", "ref", "a", candidate_index=0),
            oracle_rerank.Candidate("src", "ref", "b", candidate_index=1),
        ]
        with mock.patch.object(
            feature_reranker.mbr,
            "mbr_score",
            side_effect=lambda candidate, _: 0.1 if candidate.prediction == "a" else 0.9,
        ):
            rows = feature_reranker.featurize_groups([group])
        means, stds = feature_reranker.normalization_stats(rows)
        model = feature_reranker.FeatureModel(
            feature_names=["mbr_score"],
            means=means,
            stds=stds,
            weights={"mbr_score": 1.0},
        )

        selected = feature_reranker.select_feature(rows, model)

        self.assertEqual(selected[0].candidate.prediction, "b")

    def test_train_model_can_improve_toward_oracle_score(self):
        group = [
            oracle_rerank.Candidate("src", "ref", "short", candidate_index=0),
            oracle_rerank.Candidate("src", "ref", "longer better", candidate_index=1),
        ]
        with mock.patch.object(
            feature_reranker.oracle_rerank,
            "candidate_oracle_score",
            side_effect=lambda candidate: 1.0 if candidate.prediction == "longer better" else 0.0,
        ):
            rows = feature_reranker.featurize_groups([group])

        model, diagnostics = feature_reranker.train_model(
            rows,
            feature_names=feature_reranker.BASE_FEATURE_NAMES,
            seed=1,
            search_iterations=50,
            initial_noise=0.2,
            min_noise=0.01,
        )

        self.assertIn("best_objective", diagnostics)
        self.assertGreaterEqual(feature_reranker.mean_oracle_objective(rows, model), 1.0)


if __name__ == "__main__":
    unittest.main()
