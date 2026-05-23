from __future__ import annotations

import unittest
from pathlib import Path

from scripts import generate_multi_adapter_ensemble_reranked_translations as generate_multi
from scripts import rerank_candidate_predictions as oracle_rerank


class GenerateMultiAdapterEnsembleRerankedTranslationsTests(unittest.TestCase):
    def test_broadcast_accepts_single_value(self):
        self.assertEqual(generate_multi.broadcast([4], 3, 16), [4, 4, 4])

    def test_broadcast_rejects_mismatched_values(self):
        with self.assertRaises(ValueError):
            generate_multi.broadcast([4, 8], 3, 16)

    def test_parse_args_accepts_repeated_adapter_controls(self):
        args = generate_multi.parse_args(
            [
                "--adapter-path",
                "outputs/a",
                "--adapter-path",
                "outputs/b",
                "--feature-weights-json",
                "feature.json",
                "--text-model-json",
                "text.json",
                "--ensemble-json",
                "ensemble.json",
                "--output-jsonl",
                "predictions.jsonl",
                "--num-return-sequences",
                "32",
                "--num-return-sequences",
                "16",
                "--temperature",
                "0.65",
                "--temperature",
                "0.75",
                "--few-shot-top-k",
                "2",
            ]
        )

        self.assertEqual(args.adapter_path, [Path("outputs/a"), Path("outputs/b")])
        self.assertEqual(args.num_return_sequences, [32, 16])
        self.assertEqual(args.temperature, [0.65, 0.75])
        self.assertEqual(args.few_shot_top_k, 2)

    def test_merge_candidate_groups_dedupes_and_reindexes(self):
        groups = generate_multi.merge_candidate_groups(
            [
                [
                    [
                        oracle_rerank.Candidate("Hola", "", "Rimaykullayki", candidate_index=0),
                        oracle_rerank.Candidate("Hola", "", "Rimaykullayki", candidate_index=1),
                    ]
                ],
                [[oracle_rerank.Candidate("Hola", "", "Allinllachu", candidate_index=0)]],
            ]
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual([candidate.prediction for candidate in groups[0]], ["Rimaykullayki", "Allinllachu"])
        self.assertEqual([candidate.candidate_index for candidate in groups[0]], [0, 1])


if __name__ == "__main__":
    unittest.main()
