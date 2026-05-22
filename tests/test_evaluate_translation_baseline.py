from __future__ import annotations

import unittest
from pathlib import Path

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

    def test_parse_args_accepts_lora_adapter_path(self):
        args = baseline.parse_args(
            [
                "--backend",
                "causal-chat",
                "--model-id",
                "tencent/Hy-MT2-7B",
                "--adapter-path",
                "outputs/hymt2/chanka/final_lora",
                "--output-json",
                "outputs/baselines/smoke.json",
            ]
        )

        self.assertEqual(args.adapter_path, Path("outputs/hymt2/chanka/final_lora"))

    def test_parse_args_accepts_unsloth_adapter_loader_and_context_length(self):
        args = baseline.parse_args(
            [
                "--backend",
                "causal-chat",
                "--model-id",
                "google/gemma-4-E4B-it",
                "--adapter-path",
                "outputs/gemma4/chanka/final_lora",
                "--adapter-loader",
                "unsloth",
                "--max-seq-length",
                "128",
                "--output-json",
                "outputs/baselines/smoke.json",
            ]
        )

        self.assertEqual(args.adapter_loader, "unsloth")
        self.assertEqual(args.max_seq_length, 128)

    def test_gemma4_adapter_uses_unsloth_loader_by_default(self):
        args = baseline.parse_args(
            [
                "--backend",
                "causal-chat",
                "--model-id",
                "google/gemma-4-E4B-it",
                "--adapter-path",
                "outputs/gemma4/chanka/final_lora",
                "--output-json",
                "outputs/baselines/smoke.json",
            ]
        )

        self.assertTrue(baseline.should_load_adapter_with_unsloth(args))

    def test_non_gemma_adapter_uses_peft_loader_by_default(self):
        args = baseline.parse_args(
            [
                "--backend",
                "causal-chat",
                "--model-id",
                "tencent/Hy-MT2-7B",
                "--adapter-path",
                "outputs/hymt2/chanka/final_lora",
                "--output-json",
                "outputs/baselines/smoke.json",
            ]
        )

        self.assertFalse(baseline.should_load_adapter_with_unsloth(args))

    def test_explicit_peft_loader_overrides_gemma4_auto(self):
        args = baseline.parse_args(
            [
                "--backend",
                "causal-chat",
                "--model-id",
                "google/gemma-4-E4B-it",
                "--adapter-path",
                "outputs/gemma4/chanka/final_lora",
                "--adapter-loader",
                "peft",
                "--output-json",
                "outputs/baselines/smoke.json",
            ]
        )

        self.assertFalse(baseline.should_load_adapter_with_unsloth(args))

    def test_attach_adapter_noops_without_adapter(self):
        model = object()

        self.assertIs(baseline.attach_adapter_if_requested(model, None), model)

    def test_causal_prompt_requests_only_chanka_translation(self):
        prompt = baseline.causal_translation_prompt("Buenos dias.")

        self.assertIn("Spanish", prompt)
        self.assertIn("Quechua Chanka", prompt)
        self.assertIn("Only output", prompt)
        self.assertIn("Buenos dias.", prompt)

    def test_causal_prompt_accepts_target_language_name(self):
        prompt = baseline.causal_translation_prompt("Buenos dias.", target_language_name="Quechua Ayacuchano")

        self.assertIn("Quechua Ayacuchano", prompt)
        self.assertNotIn("Quechua Chanka", prompt)

    def test_hymt2_prompt_uses_model_card_style(self):
        prompt = baseline.hymt2_translation_prompt("Buenos dias.", "Quechua Chanka")

        self.assertIn("Translate the following text into Quechua Chanka", prompt)
        self.assertIn("only output the translated result", prompt)
        self.assertIn("Buenos dias.", prompt)

    def test_hymt2_messages_have_no_system_role(self):
        args = baseline.parse_args(
            [
                "--backend",
                "causal-chat",
                "--model-id",
                "tencent/Hy-MT2-7B",
                "--prompt-style",
                "hymt2",
                "--output-json",
                "outputs/baselines/smoke.json",
            ]
        )

        messages = baseline.causal_messages(args, "Buenos dias.")

        self.assertEqual([message["role"] for message in messages], ["user"])

    def test_generation_kwargs_records_sampling_overrides(self):
        args = baseline.parse_args(
            [
                "--backend",
                "causal-chat",
                "--model-id",
                "model",
                "--do-sample",
                "--temperature",
                "0.2",
                "--top-p",
                "0.9",
                "--top-k",
                "20",
                "--repetition-penalty",
                "1.05",
                "--output-json",
                "outputs/baselines/smoke.json",
            ]
        )

        self.assertEqual(
            baseline.generation_kwargs(args),
            {
                "max_new_tokens": 80,
                "do_sample": True,
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 20,
                "repetition_penalty": 1.05,
            },
        )


if __name__ == "__main__":
    unittest.main()
