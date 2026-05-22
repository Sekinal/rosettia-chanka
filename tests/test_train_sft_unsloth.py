from __future__ import annotations

import argparse
import unittest

from scripts import train_sft_unsloth as train_sft


class DummyTokenizer:
    def __init__(self, template: str | None = None):
        self.template = template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        if self.template is not None:
            return self.template
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class TrainSftUnslothTests(unittest.TestCase):
    def test_default_model_is_qwen35_2b(self):
        self.assertEqual(train_sft.DEFAULT_MODEL_ID, "unsloth/Qwen3.5-2B")

    def test_adapter_flags_are_mutually_exclusive(self):
        self.assertEqual(train_sft.adapter_flags("lora"), {"use_dora": False, "use_rslora": False})
        self.assertEqual(train_sft.adapter_flags("dora"), {"use_dora": True, "use_rslora": False})
        self.assertEqual(train_sft.adapter_flags("rslora"), {"use_dora": False, "use_rslora": True})

    def test_sft_cli_accepts_chanka_stage_for_model_family_canaries(self):
        args = train_sft.parse_args(["--stage", "chanka"])

        self.assertEqual(args.stage, "chanka")
        self.assertEqual(args.training_mode, "lora")

    def test_sft_cli_accepts_full_finetuning_mode(self):
        args = train_sft.parse_args(["--stage", "chanka", "--training-mode", "full"])

        self.assertEqual(args.training_mode, "full")

    def test_full_finetuning_rejects_adapter_path(self):
        args = train_sft.parse_args(
            ["--stage", "chanka", "--training-mode", "full", "--adapter-path", "outputs/run/final_lora"]
        )

        with self.assertRaisesRegex(ValueError, "--adapter-path"):
            train_sft.validate_training_mode_args(args)

    def test_sft_cli_accepts_hymt2_prompt_style(self):
        args = train_sft.parse_args(["--stage", "chanka", "--prompt-style", "hymt2"])

        self.assertEqual(args.prompt_style, "hymt2")

    def test_chat_template_helper_disables_thinking_when_supported(self):
        class ThinkingAwareTokenizer:
            def apply_chat_template(self, messages, enable_thinking=True, **kwargs):
                self.enable_thinking = enable_thinking
                self.kwargs = kwargs
                return "prompt"

        tokenizer = ThinkingAwareTokenizer()

        rendered = train_sft.apply_chat_template_no_thinking(
            tokenizer,
            [{"role": "user", "content": "Hola"}],
            tokenize=False,
            add_generation_prompt=False,
        )

        self.assertEqual(rendered, "prompt")
        self.assertFalse(tokenizer.enable_thinking)
        self.assertEqual(tokenizer.kwargs["tokenize"], False)

    def test_chanka_stage_defaults_use_measured_context_and_batch(self):
        args = argparse.Namespace(
            stage="chanka",
            training_mode="lora",
            max_seq_length=None,
            validation_fraction=None,
            num_train_epochs=None,
            learning_rate=None,
            per_device_train_batch_size=None,
            per_device_eval_batch_size=None,
            gradient_accumulation_steps=None,
            lora_r=None,
            lora_alpha=None,
            evals_per_epoch=None,
        )

        train_sft.stage_defaults(args)

        self.assertEqual(args.max_seq_length, 128)
        self.assertEqual(args.per_device_train_batch_size, 8)
        self.assertEqual(args.per_device_eval_batch_size, 8)
        self.assertEqual(args.gradient_accumulation_steps, 1)
        self.assertEqual(args.lora_r, 64)
        self.assertEqual(args.lora_alpha, 128)

    def test_full_finetuning_defaults_are_conservative(self):
        args = argparse.Namespace(
            stage="chanka",
            training_mode="full",
            max_seq_length=None,
            validation_fraction=None,
            num_train_epochs=None,
            learning_rate=None,
            per_device_train_batch_size=None,
            per_device_eval_batch_size=None,
            gradient_accumulation_steps=None,
            lora_r=None,
            lora_alpha=None,
            evals_per_epoch=None,
        )

        train_sft.stage_defaults(args)

        self.assertEqual(args.max_seq_length, 128)
        self.assertEqual(args.learning_rate, 1.0e-6)
        self.assertEqual(args.per_device_train_batch_size, 1)
        self.assertEqual(args.per_device_eval_batch_size, 2)
        self.assertEqual(args.gradient_accumulation_steps, 8)

    def test_broad_stage_defaults_keep_longer_context_and_accumulation(self):
        args = argparse.Namespace(
            stage="broad",
            training_mode="lora",
            max_seq_length=None,
            validation_fraction=None,
            num_train_epochs=None,
            learning_rate=None,
            per_device_train_batch_size=None,
            per_device_eval_batch_size=None,
            gradient_accumulation_steps=None,
            lora_r=None,
            lora_alpha=None,
            evals_per_epoch=None,
        )

        train_sft.stage_defaults(args)

        self.assertEqual(args.max_seq_length, 512)
        self.assertEqual(args.per_device_train_batch_size, 8)
        self.assertEqual(args.per_device_eval_batch_size, 8)
        self.assertEqual(args.gradient_accumulation_steps, 2)
        self.assertEqual(args.lora_r, 64)
        self.assertEqual(args.lora_alpha, 128)

    def test_step_schedule_evaluates_multiple_times_per_epoch(self):
        args = argparse.Namespace(
            eval_steps=None,
            save_steps=None,
            max_steps=-1,
            evals_per_epoch=8,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
        )

        train_sft.configure_step_schedule(args, train_row_count=1_055)

        self.assertEqual(args.eval_steps, 16)
        self.assertEqual(args.save_steps, 16)

    def test_explicit_step_schedule_is_preserved(self):
        args = argparse.Namespace(
            eval_steps=10,
            save_steps=20,
            max_steps=-1,
            evals_per_epoch=8,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
        )

        train_sft.configure_step_schedule(args, train_row_count=1_055)

        self.assertEqual(args.eval_steps, 10)
        self.assertEqual(args.save_steps, 20)

    def test_chanka_format_uses_translation_prompt_and_target(self):
        tokenizer = DummyTokenizer()
        row = {
            "source": "Buenos dias.",
            "target": "Allin punchaw.",
            "source_name": "manual",
            "variant": "quy/chanka",
        }

        args = train_sft.parse_args(["--stage", "chanka"])

        formatted = train_sft.format_example(tokenizer, args, row)

        self.assertIn("quechua chanka", formatted["text"])
        self.assertIn("Buenos dias.", formatted["text"])
        self.assertIn("Allin punchaw.", formatted["text"])
        self.assertEqual(formatted["variant"], "quy/chanka")
        self.assertFalse(tokenizer.tokenize)
        self.assertFalse(tokenizer.add_generation_prompt)

    def test_hymt2_format_uses_user_only_translation_prompt(self):
        tokenizer = DummyTokenizer()
        row = {
            "source": "Buenos dias.",
            "target": "Allin punchaw.",
            "source_name": "manual",
            "variant": "quy/chanka",
        }
        args = train_sft.parse_args(["--stage", "chanka", "--prompt-style", "hymt2"])

        formatted = train_sft.format_example(tokenizer, args, row)

        self.assertEqual([message["role"] for message in tokenizer.messages], ["user", "assistant"])
        self.assertIn("Translate the following text into Quechua Chanka", tokenizer.messages[0]["content"])
        self.assertIn("Buenos dias.", formatted["text"])
        self.assertIn("Allin punchaw.", formatted["text"])

    def test_response_marker_parts_supports_qwen_template(self):
        tokenizer = DummyTokenizer("<|im_start|>user\nusr<|im_start|>assistant\nast")

        self.assertEqual(
            train_sft.response_marker_parts(tokenizer),
            ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        )

    def test_response_marker_parts_supports_gemma4_template(self):
        tokenizer = DummyTokenizer("<|turn>user\nusr<turn|>\n<|turn>model\nast<turn|>")

        self.assertEqual(
            train_sft.response_marker_parts(tokenizer),
            ("<|turn>user\n", "<|turn>model\n"),
        )

    def test_response_marker_parts_supports_hymt2_template(self):
        tokenizer = DummyTokenizer("<｜hy_User｜>usr<｜hy_Assistant｜>ast<｜hy_place▁holder▁no▁2｜>")

        self.assertEqual(
            train_sft.response_marker_parts(tokenizer),
            ("<｜hy_User｜>", "<｜hy_Assistant｜>"),
        )

    def test_response_marker_parts_supports_hymt2_7b_template(self):
        tokenizer = DummyTokenizer("<|startoftext|>system<|extra_4|>user<|extra_0|>assistant<|eos|>")

        self.assertEqual(
            train_sft.response_marker_parts(tokenizer),
            ("<|extra_4|>", "<|extra_0|>"),
        )


if __name__ == "__main__":
    unittest.main()
