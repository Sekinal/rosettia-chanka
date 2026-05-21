from __future__ import annotations

import argparse
import unittest

from scripts import train_sft_unsloth as train_sft


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class TrainSftUnslothTests(unittest.TestCase):
    def test_default_model_is_qwen35_2b(self):
        self.assertEqual(train_sft.DEFAULT_MODEL_ID, "unsloth/Qwen3.5-2B")

    def test_adapter_flags_are_mutually_exclusive(self):
        self.assertEqual(train_sft.adapter_flags("lora"), {"use_dora": False, "use_rslora": False})
        self.assertEqual(train_sft.adapter_flags("dora"), {"use_dora": True, "use_rslora": False})
        self.assertEqual(train_sft.adapter_flags("rslora"), {"use_dora": False, "use_rslora": True})

    def test_sft_cli_rejects_chanka_stage(self):
        with self.assertRaises(SystemExit):
            train_sft.parse_args(["--stage", "chanka"])

    def test_chanka_stage_defaults_use_measured_context_and_batch(self):
        args = argparse.Namespace(
            stage="chanka",
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

    def test_broad_stage_defaults_keep_longer_context_and_accumulation(self):
        args = argparse.Namespace(
            stage="broad",
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

        formatted = train_sft.format_example(tokenizer, "chanka", row)

        self.assertIn("quechua chanka", formatted["text"])
        self.assertIn("Buenos dias.", formatted["text"])
        self.assertIn("Allin punchaw.", formatted["text"])
        self.assertEqual(formatted["variant"], "quy/chanka")
        self.assertFalse(tokenizer.tokenize)
        self.assertFalse(tokenizer.add_generation_prompt)


if __name__ == "__main__":
    unittest.main()
