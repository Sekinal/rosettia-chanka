from __future__ import annotations

import unittest

from scripts import evaluate_translation_baseline as baseline


class EvaluateTranslationBaselineTests(unittest.TestCase):
    def test_parse_args_accepts_causal_chat_backend(self):
        args = baseline.parse_args(
            [
                "--backend",
                "causal-chat",
                "--model-id",
                "tencent/Hy-MT2-1.8B",
                "--output-json",
                "outputs/baselines/smoke.json",
            ]
        )

        self.assertEqual(args.backend, "causal-chat")
        self.assertEqual(args.model_id, "tencent/Hy-MT2-1.8B")

    def test_causal_prompt_requests_only_chanka_translation(self):
        prompt = baseline.causal_translation_prompt("Buenos dias.")

        self.assertIn("Spanish", prompt)
        self.assertIn("Quechua Chanka", prompt)
        self.assertIn("Only output", prompt)
        self.assertIn("Buenos dias.", prompt)


if __name__ == "__main__":
    unittest.main()
