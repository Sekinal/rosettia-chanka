from __future__ import annotations

import unittest

from scripts import evaluate_gspo_checkpoint as evaluate


class EvaluateGspoCheckpointTests(unittest.TestCase):
    def test_requires_adapter_and_output_paths(self):
        args = evaluate.parse_args(
            [
                "--adapter-path",
                "outputs/run/chanka_gspo/checkpoint-56",
                "--output-json",
                "outputs/run/checkpoint-56_metrics.json",
            ]
        )

        self.assertEqual(str(args.adapter_path), "outputs/run/chanka_gspo/checkpoint-56")
        self.assertEqual(str(args.output_json), "outputs/run/checkpoint-56_metrics.json")
        self.assertEqual(args.seed, 3407)
        self.assertEqual(args.validation_fraction, 0.15)
        self.assertEqual(args.split, "eval")
        self.assertEqual(args.num_return_sequences, 1)
        self.assertIsNone(args.terminology_file)

    def test_candidate_generation_options_require_sampling_for_multiple_sequences(self):
        args = evaluate.parse_args(
            [
                "--adapter-path",
                "outputs/run/chanka_gspo/checkpoint-56",
                "--output-json",
                "outputs/run/candidates_metrics.json",
                "--predictions-jsonl",
                "outputs/run/candidates.jsonl",
                "--split",
                "train",
                "--max-train-samples",
                "128",
                "--num-return-sequences",
                "4",
                "--do-sample",
            ]
        )

        self.assertEqual(args.split, "train")
        self.assertEqual(args.max_train_samples, 128)
        self.assertEqual(args.num_return_sequences, 4)
        self.assertTrue(args.do_sample)

    def test_terminology_options_are_parsed(self):
        args = evaluate.parse_args(
            [
                "--adapter-path",
                "outputs/run/chanka_gspo/checkpoint-56",
                "--output-json",
                "outputs/run/term_metrics.json",
                "--terminology-file",
                "clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet",
                "--terminology-top-k",
                "4",
            ]
        )

        self.assertEqual(
            args.terminology_file,
            "clean_chanka/manual_quechua_chanka_glossary_simple_terms.parquet",
        )
        self.assertEqual(args.terminology_top_k, 4)

    def test_few_shot_options_are_parsed(self):
        args = evaluate.parse_args(
            [
                "--adapter-path",
                "outputs/run/chanka_gspo/checkpoint-56",
                "--output-json",
                "outputs/run/fewshot_metrics.json",
                "--few-shot-top-k",
                "3",
                "--few-shot-max-candidates",
                "64",
            ]
        )

        self.assertEqual(args.few_shot_top_k, 3)
        self.assertEqual(args.few_shot_max_candidates, 64)

    def test_select_terminology_prefers_longest_matches_and_dedupes_targets(self):
        terms = [
            ("madre abandonada", "saqisqa mama"),
            ("madre", "mama"),
            ("señora", "mama"),
            ("el", "dummy"),
        ]

        selected = evaluate.select_terminology(
            "La madre abandonada hablo con la señora.",
            terms,
            top_k=3,
        )

        self.assertEqual(selected, [("madre abandonada", "saqisqa mama"), ("madre", "mama")])

    def test_select_few_shot_examples_uses_overlap_and_excludes_same_source(self):
        examples = [
            {"source": "Yo vivo en Quinua.", "target": "Quinuapim tiyani."},
            {"source": "Yo vivo en Ayacucho.", "target": "Ayacuchopim tiyani."},
            {"source": "Tengo treinta años.", "target": "Kimsa chunka watayuqmi kani."},
        ]

        selected = evaluate.select_few_shot_examples(
            "Yo vivo en Quinua.",
            examples,
            top_k=2,
        )

        self.assertEqual(selected, [("Yo vivo en Ayacucho.", "Ayacuchopim tiyani.")])


if __name__ == "__main__":
    unittest.main()
