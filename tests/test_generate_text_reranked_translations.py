from __future__ import annotations

import unittest
from unittest import mock

from scripts import generate_text_reranked_translations as generate_text
from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_text_candidate_reranker as text_reranker


class GenerateTextRerankedTranslationsTests(unittest.TestCase):
    def test_select_translations_uses_text_ranker(self):
        groups = [
            [
                oracle_rerank.Candidate("En la fiesta", "", "Fiestapi", candidate_index=0),
                oracle_rerank.Candidate("En la fiesta", "", "Raymipim", candidate_index=1),
            ]
        ]
        model = text_reranker.TextRankerModel(
            hash_size=4096,
            feature_names=[],
            means={},
            stds={},
            weights={text_reranker.stable_hash("cand_tok:raymipim", 4096): 1.0},
            config={
                "char_ngram_min": 3,
                "char_ngram_max": 5,
                "word_ngram_max": 1,
                "max_source_tokens": 4,
                "max_candidate_tokens": 6,
                "include_manual_features": False,
            },
        )
        with mock.patch(
            "scripts.train_feature_candidate_reranker.oracle_rerank.candidate_oracle_score",
            return_value=0.0,
        ):
            selected = generate_text.select_translations(groups, model)

        self.assertEqual(selected[0].prediction, "Raymipim")


if __name__ == "__main__":
    unittest.main()
