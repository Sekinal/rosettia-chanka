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
        self.assertIsNone(args.resume_from_checkpoint)

    def test_resume_checkpoint_cli_is_optional(self):
        args = train_gspo.parse_args(
            ["--resume-from-checkpoint", "outputs/run/chanka_gspo/checkpoint-56"]
        )

        self.assertEqual(
            str(args.resume_from_checkpoint), "outputs/run/chanka_gspo/checkpoint-56"
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

    def test_chat_artifact_penalty_detects_role_tokens(self):
        self.assertGreater(
            train_gspo.chat_artifact_penalty("Allin punchaw user Allin punchaw assistant <think></think>"),
            train_gspo.chat_artifact_penalty("Allin punchaw taytay"),
        )

    def test_strip_chat_artifacts_keeps_translation_prefix(self):
        self.assertEqual(
            train_gspo.strip_chat_artifacts("Allin punchaw user Allin punchaw assistant <think></think>"),
            "Allin punchaw",
        )
        self.assertEqual(train_gspo.strip_chat_artifacts("Allin punchaw taytay"), "Allin punchaw taytay")

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
            artifact = train_gspo.chanka_reward(
                completions=["Allin punchaw taytay. user Allin punchaw taytay. assistant <think></think>"],
                target=["Allin punchaw taytay."],
            )[0]

        self.assertGreater(good, leaked)
        self.assertGreater(good, artifact)

    def test_vibethinker_profile_rewards_diverse_group(self):
        base = [0.2, 0.2, 0.2, 0.2]
        sources = ["a", "a", "a", "a"]
        repeated = train_gspo.add_vibethinker_diversity_bonus(base, ["x", "x", "x", "x"], sources)
        diverse = train_gspo.add_vibethinker_diversity_bonus(base, ["x", "y", "z", "w"], sources)

        self.assertGreater(sum(diverse), sum(repeated))

    def test_mixed_profiles_are_available_for_canary_sweeps(self):
        expected = {
            "mix_severity_verifier",
            "mix_verifier_vibe",
            "mix_all_strict",
            "rosettia_guard_v1",
            "rosettia_guard_v2",
            "learned_verifier_2511",
            "learned_verifier_vibe_2511",
        }

        self.assertTrue(expected.issubset(set(train_gspo.REWARD_PROFILES)))

    def test_rosettia_guard_penalizes_source_copy(self):
        with mock.patch.object(train_gspo, "sentence_chrfpp", return_value=0.65), mock.patch.object(
            train_gspo, "sentence_bleu", return_value=0.30
        ):
            translated = train_gspo.reward_score(
                "Allin punchaw kamachiq.",
                "Allin punchaw kamachiq.",
                "Buenos dias autoridad.",
                "rosettia_guard_v2",
            )
            copied = train_gspo.reward_score(
                "Buenos dias autoridad.",
                "Allin punchaw kamachiq.",
                "Buenos dias autoridad.",
                "rosettia_guard_v2",
            )

        self.assertGreater(translated, copied)

    def test_mix_verifier_vibe_uses_diversity_bonus(self):
        with mock.patch.object(train_gspo, "reward_score", return_value=0.2):
            repeated = train_gspo.chanka_reward(
                completions=["x", "x", "x", "x"],
                target=["a", "a", "a", "a"],
                source=["s", "s", "s", "s"],
                profile="mix_verifier_vibe",
            )
            diverse = train_gspo.chanka_reward(
                completions=["x", "y", "z", "w"],
                target=["a", "a", "a", "a"],
                source=["s", "s", "s", "s"],
                profile="mix_verifier_vibe",
            )

        self.assertGreater(sum(diverse), sum(repeated))

    def test_parse_verifier_score_accepts_json_and_clamps(self):
        self.assertEqual(train_gspo.parse_verifier_score('{"score":0.87,"severity":"none"}'), 0.87)
        self.assertEqual(train_gspo.parse_verifier_score('{"score":1.9}'), 1.0)
        self.assertEqual(train_gspo.parse_verifier_score("puntaje: 0.42"), 0.42)

    def test_learned_verifier_profile_requires_adapter_path(self):
        with self.assertRaises(ValueError):
            train_gspo.make_reward_fn("learned_verifier_2511")
        with self.assertRaises(ValueError):
            train_gspo.make_reward_fn("learned_verifier_vibe_2511")

    def test_learned_verifier_reward_blends_model_score_with_guards(self):
        class FixedScorer:
            def score_many(self, sources, references, hypotheses):
                return [0.9 for _ in hypotheses]

        with mock.patch.object(train_gspo, "sentence_chrfpp", return_value=0.7), mock.patch.object(
            train_gspo, "sentence_bleu", return_value=0.3
        ):
            translated = train_gspo.learned_verifier_rewards(
                FixedScorer(),
                ["Allin punchaw kamachiq."],
                ["Allin punchaw kamachiq."],
                ["Buenos dias autoridad."],
            )[0]
            copied = train_gspo.learned_verifier_rewards(
                FixedScorer(),
                ["Buenos dias autoridad."],
                ["Allin punchaw kamachiq."],
                ["Buenos dias autoridad."],
            )[0]

        self.assertGreater(translated, copied)

    def test_learned_verifier_vibe_adds_diversity_bonus(self):
        class FixedScorer:
            def score_many(self, sources, references, hypotheses):
                return [0.5 for _ in hypotheses]

        with mock.patch.object(train_gspo, "LearnedVerifierScorer", return_value=FixedScorer()), mock.patch.object(
            train_gspo, "sentence_chrfpp", return_value=0.5
        ), mock.patch.object(train_gspo, "sentence_bleu", return_value=0.2):
            reward_fn = train_gspo.make_reward_fn(
                "learned_verifier_vibe_2511",
                verifier_adapter_path=mock.Mock(),
            )
            repeated = reward_fn(
                completions=["x", "x", "x", "x"],
                target=["a", "a", "a", "a"],
                source=["s", "s", "s", "s"],
            )
            diverse = reward_fn(
                completions=["x", "y", "z", "w"],
                target=["a", "a", "a", "a"],
                source=["s", "s", "s", "s"],
            )

        self.assertGreater(sum(diverse), sum(repeated))

    def test_learned_verifier_scorer_tokenizes_prompts_as_text(self):
        class FakeTensor(dict):
            @property
            def shape(self):
                return (1, 3)

        class FakeInputs(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            eos_token = "<eos>"
            pad_token = "<eos>"
            eos_token_id = 0

            def __call__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return FakeInputs({"input_ids": FakeTensor()})

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "prompt"

            def decode(self, ids, skip_special_tokens=True):
                return '{"score":0.5}'

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                self.generate_kwargs = kwargs

                class FakeOutput:
                    def __getitem__(self, key):
                        return [4]

                return FakeOutput()

            def eval(self):
                return None

        scorer = train_gspo.LearnedVerifierScorer.__new__(train_gspo.LearnedVerifierScorer)
        scorer.torch = mock.Mock()
        scorer.torch.inference_mode.return_value.__enter__ = mock.Mock(return_value=None)
        scorer.torch.inference_mode.return_value.__exit__ = mock.Mock(return_value=None)
        scorer.max_seq_length = 512
        scorer.max_new_tokens = 32
        scorer.batch_size = 1
        scorer.model = FakeModel()
        scorer.tokenizer = FakeTokenizer()

        self.assertEqual(scorer.score_many(["src"], ["ref"], ["hyp"]), [0.5])
        self.assertEqual(scorer.tokenizer.args, ())
        self.assertEqual(scorer.tokenizer.kwargs["text"], ["prompt"])
        self.assertEqual(scorer.model.generate_kwargs["eos_token_id"], 0)


if __name__ == "__main__":
    unittest.main()
