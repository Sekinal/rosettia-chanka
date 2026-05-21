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


if __name__ == "__main__":
    unittest.main()
