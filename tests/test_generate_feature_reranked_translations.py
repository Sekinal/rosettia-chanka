from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import generate_feature_reranked_translations as generate
from scripts import rerank_candidate_predictions as oracle_rerank
from scripts import train_feature_candidate_reranker as feature_reranker


class GenerateFeatureRerankedTranslationsTests(unittest.TestCase):
    def test_load_source_rows_supports_inline_text_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.jsonl"
            path.write_text(json.dumps({"source": "Yo vivo en Quinua"}) + "\n")

            rows = generate.load_source_rows(path, ["Buenos dias"])

        self.assertEqual([row["source"] for row in rows], ["Buenos dias", "Yo vivo en Quinua"])
        self.assertEqual(rows[0]["source_name"], "inline")
        self.assertEqual(rows[1]["source_name"], "sources.jsonl")

    def test_candidate_groups_from_generation_uses_candidate_indices(self):
        generated_rows = [
            {"source": "A", "source_name": "x", "variant": "quy/chanka"},
            {"source": "A", "source_name": "x", "variant": "quy/chanka"},
            {"source": "B", "source_name": "x", "variant": "quy/chanka"},
            {"source": "B", "source_name": "x", "variant": "quy/chanka"},
        ]
        predictions = ["a0", "a1", "b0", "b1"]

        groups = generate.candidate_groups_from_generation(generated_rows, predictions, 2)

        self.assertEqual(len(groups), 2)
        self.assertEqual([candidate.candidate_index for candidate in groups[0]], [0, 1])
        self.assertEqual(groups[1][1].prediction, "b1")

    def test_select_translations_uses_feature_model_without_references(self):
        group = [
            oracle_rerank.Candidate("src", "", "short", candidate_index=0),
            oracle_rerank.Candidate("src", "", "longer output", candidate_index=1),
        ]
        rows = feature_reranker.featurize_groups([group])
        means, stds = feature_reranker.normalization_stats(rows, ["target_token_count"])
        model = feature_reranker.FeatureModel(
            feature_names=["target_token_count"],
            means=means,
            stds=stds,
            weights={"target_token_count": 1.0},
        )

        selected = generate.select_translations([group], model)

        self.assertEqual(selected[0].prediction, "longer output")


if __name__ == "__main__":
    unittest.main()
