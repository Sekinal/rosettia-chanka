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


if __name__ == "__main__":
    unittest.main()
