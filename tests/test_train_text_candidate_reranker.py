from __future__ import annotations

import unittest
from argparse import Namespace
from unittest import mock

from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_feature_candidate_reranker as feature_reranker
from scripts import train_text_candidate_reranker as text_reranker


class TrainTextCandidateRerankerTests(unittest.TestCase):
    def test_sparse_features_include_candidate_and_cross_features(self):
        group = [
            oracle_rerank.Candidate(
                "En la fiesta",
                "Raymipim",
                "Raymipim",
                candidate_index=0,
                pool_path="outputs/qwen35_4b_full/eval.jsonl",
            ),
        ]
        rows = feature_reranker.featurize_groups([group])
        args = Namespace(
            hash_size=1024,
            char_ngram_min=3,
            char_ngram_max=4,
            word_ngram_max=2,
            max_source_tokens=4,
            max_candidate_tokens=6,
            include_manual_features=True,
            include_pool_origin_features=True,
            include_pool_token_cross_features=False,
        )
        model = text_reranker.model_shell(rows, args)

        features = text_reranker.sparse_features_for_row(rows[0][0], model)

        self.assertGreater(len(features), 0)
        self.assertIn(text_reranker.stable_hash("src_cand:fiesta->raymipim", 1024), features)
        self.assertIn(text_reranker.stable_hash("pool_origin:qwen35_4b_full", 1024), features)

    def test_train_text_ranker_can_learn_oracle_winner(self):
        group = [
            oracle_rerank.Candidate("En la fiesta", "Raymipim", "Fiestapi", candidate_index=0),
            oracle_rerank.Candidate("En la fiesta", "Raymipim", "Raymipim", candidate_index=1),
        ]
        with mock.patch.object(
            feature_reranker.oracle_rerank,
            "candidate_oracle_score",
            side_effect=lambda candidate: 1.0 if candidate.prediction == "Raymipim" else 0.0,
        ):
            rows = feature_reranker.featurize_groups([group])
        args = Namespace(
            hash_size=4096,
            epochs=4,
            learning_rate=0.2,
            l2=0.0,
            training_objective="pairwise",
            listwise_temperature=0.08,
            margin=0.0,
            max_negatives_per_group=8,
            char_ngram_min=3,
            char_ngram_max=5,
            word_ngram_max=2,
            max_source_tokens=4,
            max_candidate_tokens=6,
            include_manual_features=True,
            include_pool_origin_features=True,
            include_pool_token_cross_features=False,
            seed=1,
        )

        model, diagnostics = text_reranker.train_text_ranker(rows, args)
        sparse_groups = text_reranker.sparse_groups_for_model(rows, model)
        selected = text_reranker.select_sparse(sparse_groups, model)

        self.assertEqual(diagnostics["training_objective"], "pairwise_hashed_text_logistic")
        self.assertGreater(diagnostics["updates"], 0)
        self.assertEqual(selected[0].row.candidate.prediction, "Raymipim")

    def test_listwise_text_ranker_can_learn_oracle_winner(self):
        group = [
            oracle_rerank.Candidate("En la fiesta", "Raymipim", "Fiestapi", candidate_index=0),
            oracle_rerank.Candidate("En la fiesta", "Raymipim", "Raymipim", candidate_index=1),
            oracle_rerank.Candidate("En la fiesta", "Raymipim", "Raymikunapi", candidate_index=2),
        ]
        with mock.patch.object(
            feature_reranker.oracle_rerank,
            "candidate_oracle_score",
            side_effect=lambda candidate: 1.0 if candidate.prediction == "Raymipim" else 0.0,
        ):
            rows = feature_reranker.featurize_groups([group])
        args = Namespace(
            hash_size=4096,
            epochs=12,
            learning_rate=0.3,
            l2=0.0,
            training_objective="listwise",
            listwise_temperature=0.05,
            margin=0.0,
            max_negatives_per_group=8,
            char_ngram_min=3,
            char_ngram_max=5,
            word_ngram_max=2,
            max_source_tokens=4,
            max_candidate_tokens=6,
            include_manual_features=True,
            include_pool_origin_features=True,
            include_pool_token_cross_features=False,
            seed=1,
        )

        model, diagnostics = text_reranker.train_text_ranker(rows, args)
        sparse_groups = text_reranker.sparse_groups_for_model(rows, model)
        selected = text_reranker.select_sparse(sparse_groups, model)

        self.assertEqual(diagnostics["training_objective"], "listwise_hashed_text_logistic")
        self.assertGreater(diagnostics["updates"], 0)
        self.assertEqual(selected[0].row.candidate.prediction, "Raymipim")


if __name__ == "__main__":
    unittest.main()
