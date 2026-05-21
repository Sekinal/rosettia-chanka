from __future__ import annotations

import argparse
import unittest
from unittest import mock

from scripts import train_gspo_chanka_unsloth as train_gspo


class TrainGspoChankaUnslothTests(unittest.TestCase):
    def test_default_checkpoint_is_current_broad_lora_checkpoint(self):
        args = train_gspo.parse_args([])

        self.assertEqual(
            str(args.adapter_path),
            "outputs/qwen35_2b_broad_lora_r64_a128_seq512_b16_ga1/broad/checkpoint-10400",
        )

    def test_prompt_is_general_chanka_translation(self):
        messages = train_gspo.prompt_messages("Buenos dias.")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("quechua chanka", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("natural y fiel", messages[1]["content"])
        self.assertIn("evita copiar", messages[1]["content"])
        self.assertNotIn("contexto judicial", messages[1]["content"])
        self.assertIn("Buenos dias.", messages[1]["content"])
        self.assertNotIn("assistant", {message["role"] for message in messages})

    def test_reward_profile_cli_accepts_paper_profiles(self):
        for profile in train_gspo.REWARD_PROFILES:
            args = train_gspo.parse_args(["--reward-profile", profile])
            self.assertEqual(args.reward_profile, profile)

        self.assertNotIn("xcomet_proxy_2310", train_gspo.REWARD_PROFILES)

    def test_build_dataset_keeps_targets_for_reward_not_sft_labels(self):
        dataset = train_gspo.build_dataset(
            [
                {
                    "source": "Buenos dias.",
                    "target": "Allin punchaw.",
                    "source_name": "manual",
                    "variant": "quy/chanka",
                }
            ]
        )

        row = dataset[0]
        self.assertIn("prompt", row)
        self.assertEqual(row["target"], "Allin punchaw.")
        self.assertEqual(row["variant"], "quy/chanka")
        self.assertNotIn("text", row)

    def test_step_schedule_evaluates_many_times_per_epoch(self):
        args = argparse.Namespace(
            eval_steps=None,
            save_steps=None,
            max_steps=-1,
            evals_per_epoch=8,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
        )

        train_gspo.configure_step_schedule(args, train_row_count=897)

        self.assertEqual(args.eval_steps, 56)
        self.assertEqual(args.save_steps, 56)

    def test_grpo_batching_must_be_divisible_by_generations(self):
        valid = argparse.Namespace(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=4,
            num_generations=4,
        )
        train_gspo.validate_grpo_batching(valid)

        invalid_train = argparse.Namespace(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=4,
            num_generations=4,
        )
        with self.assertRaises(ValueError):
            train_gspo.validate_grpo_batching(invalid_train)

        invalid_eval = argparse.Namespace(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=1,
            num_generations=4,
        )
        with self.assertRaises(ValueError):
            train_gspo.validate_grpo_batching(invalid_eval)

    def test_completion_text_handles_conversational_completion(self):
        completion = [{"role": "assistant", "content": "  Allin   punchaw.  "}]

        self.assertEqual(train_gspo.completion_text(completion), "Allin punchaw.")

    def test_token_f1_rewards_overlap(self):
        self.assertGreater(
            train_gspo.token_f1("Allin punchaw mamay.", "Allin punchaw taytay."),
            train_gspo.token_f1("Imaynalla.", "Allin punchaw taytay."),
        )

    def test_latest_eval_metrics_returns_last_eval_record(self):
        history = [
            {"loss": 0.4},
            {"eval_reward": 0.1, "eval_loss": 0.2},
            {"eval_reward": 0.3, "eval_loss": 0.1},
        ]

        self.assertEqual(train_gspo.latest_eval_metrics(history)["eval_reward"], 0.3)

    def test_length_ratio_score_penalizes_large_mismatch(self):
        self.assertGreater(
            train_gspo.length_ratio_score("huk iskay kimsa", "huk iskay kimsa"),
            train_gspo.length_ratio_score("huk", "huk iskay kimsa tawa pichqa"),
        )

    def test_spanish_leakage_penalty_detects_spanish_stopwords(self):
        self.assertGreater(
            train_gspo.spanish_leakage_penalty("el señor de la casa"),
            train_gspo.spanish_leakage_penalty("allin punchaw taytay"),
        )

    def test_source_copy_ratio_detects_copied_source_tokens(self):
        self.assertGreater(
            train_gspo.source_copy_ratio("Buenos dias autoridad.", "Buenos dias autoridad."),
            train_gspo.source_copy_ratio("Allin punchaw kamachiq.", "Buenos dias autoridad."),
        )

    def test_severity_penalty_flags_exact_source_copy(self):
        self.assertGreaterEqual(
            train_gspo.severity_penalty(
                "Buenos dias autoridad.",
                "Allin punchaw kamachiq.",
                "Buenos dias autoridad.",
                chrf=0.1,
                f1=0.0,
                length_score=0.8,
            ),
            0.55,
        )

    def test_reward_combines_metrics_and_penalty(self):
        with mock.patch.object(train_gspo, "sentence_chrfpp", return_value=0.8), mock.patch.object(
            train_gspo, "sentence_bleu", return_value=0.4
        ):
            good = train_gspo.chanka_reward(
                completions=["Allin punchaw taytay."],
                target=["Allin punchaw taytay."],
            )[0]
            leaked = train_gspo.chanka_reward(
                completions=["el de la Allin punchaw taytay."],
                target=["Allin punchaw taytay."],
            )[0]

        self.assertGreater(good, leaked)

    def test_vibethinker_profile_rewards_diverse_group(self):
        base = [0.2, 0.2, 0.2, 0.2]
        sources = ["a", "a", "a", "a"]
        repeated = train_gspo.add_vibethinker_diversity_bonus(base, ["x", "x", "x", "x"], sources)
        diverse = train_gspo.add_vibethinker_diversity_bonus(base, ["x", "y", "z", "w"], sources)

        self.assertGreater(sum(diverse), sum(repeated))


if __name__ == "__main__":
    unittest.main()
