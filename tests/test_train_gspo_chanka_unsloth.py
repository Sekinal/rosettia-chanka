from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from scripts import train_gspo_chanka_unsloth as train_gspo


class DummyParameter:
    def __init__(self, size: int, requires_grad: bool = False):
        self._size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._size

    def requires_grad_(self, value: bool):
        self.requires_grad = value


class DummyModel:
    def __init__(self, named_parameters):
        self._named_parameters = named_parameters

    def parameters(self):
        return iter(parameter for _, parameter in self._named_parameters)

    def named_parameters(self):
        return iter(self._named_parameters)


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

    def test_fast_canary_cli_can_skip_trainer_eval_and_cap_final_metrics(self):
        args = train_gspo.parse_args(
            [
                "--no-trainer-eval",
                "--final-metrics-max-samples",
                "16",
                "--final-generation-batch-size",
                "8",
                "--predictions-jsonl",
                "outputs/run/final_predictions.jsonl",
            ]
        )

        self.assertFalse(args.trainer_eval)
        self.assertEqual(args.final_metrics_max_samples, 16)
        self.assertEqual(args.final_generation_batch_size, 8)
        self.assertEqual(args.predictions_jsonl, Path("outputs/run/final_predictions.jsonl"))

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

    def test_prompt_can_include_optional_terminology(self):
        messages = train_gspo.prompt_messages(
            "La señora vive en Quinua.",
            [("señora", "mama"), ("vive", "tiyan")],
        )

        self.assertIn("Glosario sugerido", messages[1]["content"])
        self.assertIn("- señora = mama", messages[1]["content"])
        self.assertIn("no fuerces", messages[1]["content"])
        self.assertIn("La señora vive en Quinua.", messages[1]["content"])

    def test_prompt_can_include_few_shot_examples(self):
        messages = train_gspo.prompt_messages(
            "Yo vivo en Quinua.",
            few_shot_examples=[("Yo vivo en Ayacucho.", "Ayacuchopim tiyani.")],
        )

        self.assertIn("Ejemplos de referencia", messages[1]["content"])
        self.assertIn("Español: Yo vivo en Ayacucho.", messages[1]["content"])
        self.assertIn("Quechua chanka: Ayacuchopim tiyani.", messages[1]["content"])
        self.assertIn("Español: Yo vivo en Quinua.", messages[1]["content"])

    def test_prompt_can_request_self_verification_format(self):
        messages = train_gspo.prompt_messages("No es un buen esposo.", self_verification=True)

        self.assertIn("Formato obligatorio", messages[1]["content"])
        self.assertIn("Traduccion final:", messages[1]["content"])
        self.assertIn("Autoevaluacion:", messages[1]["content"])
        self.assertIn("Puntaje:", messages[1]["content"])

    def test_prompt_can_request_bounded_thinking_self_verification_format(self):
        messages = train_gspo.prompt_messages(
            "No es un buen esposo.",
            self_verification=True,
            self_verification_thinking=True,
        )

        self.assertIn("Analisis de traduccion:", messages[1]["content"])
        self.assertIn("Traduccion final:", messages[1]["content"])
        self.assertIn("1 a 2 chequeos", messages[1]["content"])
        self.assertIn("[SIGNIFICADO]", messages[1]["content"])
        self.assertIn("[ANTI_COPIA]", messages[1]["content"])
        self.assertIn("maximo 35 palabras", messages[1]["content"])
        self.assertIn("debe ser corto", messages[1]["content"])

    def test_prompt_can_request_compact_thinking_self_verification_format(self):
        messages = train_gspo.prompt_messages(
            "No es un buen esposo.",
            self_verification=True,
            self_verification_thinking=True,
            self_verification_compact=True,
        )

        self.assertIn("exactamente 3 lineas", messages[1]["content"])
        self.assertIn("Analisis:", messages[1]["content"])
        self.assertIn("Final:", messages[1]["content"])
        self.assertIn("Puntaje:", messages[1]["content"])
        self.assertIn("No escribas Autoevaluacion", messages[1]["content"])

    def test_chat_template_helper_disables_thinking_when_supported(self):
        class ThinkingAwareTokenizer:
            def apply_chat_template(self, messages, enable_thinking=True, **kwargs):
                self.enable_thinking = enable_thinking
                self.kwargs = kwargs
                return "prompt"

        tokenizer = ThinkingAwareTokenizer()

        rendered = train_gspo.apply_chat_template_no_thinking(
            tokenizer,
            train_gspo.prompt_messages("Buenos dias."),
            tokenize=False,
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "prompt")
        self.assertFalse(tokenizer.enable_thinking)
        self.assertEqual(tokenizer.kwargs["add_generation_prompt"], True)

    def test_force_tokenizer_no_thinking_template_closes_qwen_think_tag(self):
        class Tokenizer:
            chat_template = """prefix
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}
{%- endif %}
suffix"""

        tokenizer = Tokenizer()

        patched = train_gspo.force_tokenizer_no_thinking_template(tokenizer)

        self.assertTrue(patched)
        self.assertIn("</think>", tokenizer.chat_template)
        self.assertNotIn("enable_thinking is defined", tokenizer.chat_template)

    def test_select_terminology_prefers_longest_matches_and_dedupes_targets(self):
        selected = train_gspo.select_terminology(
            "La madre abandonada hablo con la señora.",
            [
                ("madre abandonada", "saqisqa mama"),
                ("madre", "mama"),
                ("señora", "mama"),
                ("el", "dummy"),
            ],
            top_k=3,
        )

        self.assertEqual(selected, [("madre abandonada", "saqisqa mama"), ("madre", "mama")])

    def test_build_dataset_can_use_terminology_prompts(self):
        dataset = train_gspo.build_dataset(
            [
                {
                    "source": "¿Es usted casado?",
                    "target": "¿Warmiyuqchu kanki?",
                    "source_name": "manual",
                    "variant": "quy/chanka",
                }
            ],
            terminology_entries=[("Casado", "Warmiyuq")],
            terminology_top_k=1,
        )

        user_message = dataset[0]["prompt"][1]["content"]
        self.assertIn("Glosario sugerido", user_message)
        self.assertIn("- Casado = Warmiyuq", user_message)

    def test_build_dataset_can_use_self_verification_prompts(self):
        dataset = train_gspo.build_dataset(
            [
                {
                    "source": "¿Es usted casado?",
                    "target": "¿Warmiyuqchu kanki?",
                    "source_name": "manual",
                    "variant": "quy/chanka",
                }
            ],
            self_verification=True,
        )

        self.assertIn("Traduccion final:", dataset[0]["prompt"][1]["content"])

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

    def test_enable_peft_adapter_training_only_unfreezes_adapter_weights(self):
        lora = DummyParameter(10)
        dense = DummyParameter(100)
        saved = DummyParameter(3)
        model = DummyModel(
            [
                ("base.layer.weight", dense),
                ("base.lora_A.default.weight", lora),
                ("base.modules_to_save.default.weight", saved),
            ]
        )

        enabled = train_gspo.enable_peft_adapter_training(model)

        self.assertEqual(enabled, 13)
        self.assertFalse(dense.requires_grad)
        self.assertTrue(lora.requires_grad)
        self.assertTrue(saved.requires_grad)
        self.assertEqual(train_gspo.trainable_parameter_count(model), 13)

    def test_completion_text_handles_conversational_completion(self):
        completion = [{"role": "assistant", "content": "  Allin   punchaw.  "}]

        self.assertEqual(train_gspo.completion_text(completion), "Allin punchaw.")

    def test_parse_self_verification_output_extracts_translation_and_score(self):
        parsed = train_gspo.parse_self_verification_output(
            "Traduccion final: Mana allin qusachu. Autoevaluacion: no veo errores. Puntaje: \\boxed{0.93}"
        )

        self.assertEqual(parsed["translation"], "Mana allin qusachu.")
        self.assertEqual(parsed["self_score"], 0.93)
        self.assertTrue(parsed["has_format"])

    def test_extract_translation_strips_self_evaluation_without_final_marker(self):
        translation = train_gspo.extract_translation_from_structured_output(
            "¿Ima kallpim tiyanki? Autoevaluacion: Manam error kanchu Puntaje: \\boxed{1.0}"
        )

        self.assertEqual(translation, "¿Ima kallpim tiyanki?")

    def test_parse_self_verification_output_extracts_bounded_thinking(self):
        parsed = train_gspo.parse_self_verification_output(
            "Analisis de traduccion: conserva el significado y evita copia del espanol. "
            "Traduccion final: Mana allin qusachu. "
            "Autoevaluacion: no veo errores. Puntaje: \\boxed{0.93}"
        )

        self.assertEqual(parsed["translation"], "Mana allin qusachu.")
        self.assertIn("conserva el significado", parsed["thinking"])
        self.assertIn("no veo errores", parsed["self_evaluation"])
        self.assertTrue(parsed["has_thinking_format"])

    def test_parse_self_verification_output_extracts_compact_thinking(self):
        parsed = train_gspo.parse_self_verification_output(
            "Analisis: [SIGNIFICADO] conserva negacion; [GRAMATICA] usa -chu. "
            "Final: Mana allin qusachu. "
            "Puntaje: \\boxed{0.84}"
        )

        self.assertEqual(parsed["translation"], "Mana allin qusachu.")
        self.assertEqual(parsed["thinking"], "[SIGNIFICADO] conserva negacion; [GRAMATICA] usa -chu.")
        self.assertEqual(parsed["self_score"], 0.84)
        self.assertTrue(parsed["has_format"])
        self.assertTrue(parsed["has_thinking_format"])

    def test_translation_thinking_score_prefers_primitive_tags(self):
        plain = train_gspo.translation_thinking_score(
            "conserva el significado y evita copia del espanol"
        )
        tagged = train_gspo.translation_thinking_score(
            "[SIGNIFICADO] conserva el sentido; [ANTI_COPIA] evita copia del espanol"
        )

        self.assertGreater(tagged, plain)
        self.assertEqual(
            train_gspo.thinking_primitive_count(
                "[SIGNIFICADO] conserva; [GRAMATICA] revisa verbo; [UNKNOWN] nada"
            ),
            2,
        )

    def test_meta_verifier_prompt_includes_candidate_and_analysis(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "prompt"

        tokenizer = Tokenizer()

        rendered = train_gspo.meta_verifier_prompt_text(
            tokenizer,
            "Hola",
            "Rimaykullayki",
            "Hola",
            "Puntaje: \\boxed{0.2}",
        )

        self.assertEqual(rendered, "prompt")
        self.assertIn("Candidata: Hola", tokenizer.messages[1]["content"])
        self.assertIn("Analisis: Puntaje", tokenizer.messages[1]["content"])
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

    def test_structured_translation_extractor_falls_back_to_plain_text(self):
        self.assertEqual(
            train_gspo.extract_translation_from_structured_output("Allin punchaw."),
            "Allin punchaw.",
        )

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
        self.assertEqual(
            train_gspo.strip_chat_artifacts("Thinking Process: analyze. Final Answer: Allin punchaw"),
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
            "learned_verifier_bleu_margin_vibe_2511",
            "deepseekmath_final_verifier_2511",
            "reference_rerank_vibe_v1",
            "self_verifiable_translation_2511",
            "self_verifiable_thinking_translation_2511",
        }

        self.assertTrue(expected.issubset(set(train_gspo.REWARD_PROFILES)))

    def test_deepseekmath_final_verifier_reward_keeps_policy_final_only(self):
        class Scorer:
            def score_many(self, sources, references, hypotheses):
                return [0.95 if "Allin" in hypothesis else 0.20 for hypothesis in hypotheses]

        with mock.patch.object(train_gspo, "sentence_chrfpp", side_effect=[0.8, 0.8, 0.2, 0.2]), mock.patch.object(
            train_gspo, "sentence_bleu", side_effect=[0.4, 0.4, 0.0, 0.0]
        ):
            rewards = train_gspo.deepseekmath_final_verifier_rewards(
                Scorer(),
                [
                    "Allin punchaw kamachiq.",
                    "Analisis: user assistant <think> Buenos dias autoridad.",
                ],
                ["Allin punchaw kamachiq.", "Allin punchaw kamachiq."],
                ["Buenos dias autoridad.", "Buenos dias autoridad."],
            )

        self.assertGreater(rewards[0], rewards[1])

    def test_deepseekmath_final_verifier_profile_requires_verifier(self):
        with self.assertRaisesRegex(ValueError, "verifier"):
            train_gspo.make_reward_fn("deepseekmath_final_verifier_2511")

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

    def test_self_verifiable_translation_reward_penalizes_false_self_score(self):
        good = (
            "Traduccion final: Allin punchaw kamachiq. "
            "Autoevaluacion: no veo errores importantes. "
            "Puntaje: \\boxed{0.95}"
        )
        copied = (
            "Traduccion final: Buenos dias autoridad. "
            "Autoevaluacion: no veo errores importantes. "
            "Puntaje: \\boxed{0.95}"
        )
        with mock.patch.object(train_gspo, "sentence_chrfpp", side_effect=[0.8, 0.1]), mock.patch.object(
            train_gspo, "sentence_bleu", side_effect=[0.4, 0.0]
        ):
            good_reward = train_gspo.self_verifiable_translation_reward(
                good,
                "Allin punchaw kamachiq.",
                "Buenos dias autoridad.",
            )
            copied_reward = train_gspo.self_verifiable_translation_reward(
                copied,
                "Allin punchaw kamachiq.",
                "Buenos dias autoridad.",
            )

        self.assertGreater(good_reward, copied_reward)

    def test_self_verifiable_translation_reward_accepts_learned_meta_score(self):
        parsed = train_gspo.parse_self_verification_output(
            "Traduccion final: Allin punchaw kamachiq. "
            "Autoevaluacion: no veo errores importantes. "
            "Puntaje: \\boxed{0.95}"
        )
        with mock.patch.object(train_gspo, "sentence_chrfpp", return_value=0.8), mock.patch.object(
            train_gspo, "sentence_bleu", return_value=0.4
        ):
            high_meta = train_gspo.self_verifiable_translation_reward_from_parsed(
                parsed,
                "Allin punchaw kamachiq.",
                "Buenos dias autoridad.",
                learned_meta_score=1.0,
            )
            low_meta = train_gspo.self_verifiable_translation_reward_from_parsed(
                parsed,
                "Allin punchaw kamachiq.",
                "Buenos dias autoridad.",
                learned_meta_score=0.0,
            )

        self.assertGreater(high_meta, low_meta)

    def test_self_verification_diagnostics_tracks_format_and_calibration(self):
        raw_predictions = [
            "Analisis de traduccion: conserva el significado y evita copia del espanol. "
            "Traduccion final: Allin punchaw kamachiq. "
            "Autoevaluacion: no veo errores importantes. "
            "Puntaje: \\boxed{0.90}",
            "Traduccion final: Buenos dias autoridad. Autoevaluacion: no veo errores.",
        ]

        with mock.patch.object(
            train_gspo,
            "bounded_translation_quality_score",
            side_effect=[0.70, 0.10],
        ):
            diagnostics = train_gspo.self_verification_diagnostics(
                raw_predictions,
                ["Allin punchaw kamachiq.", "Allin punchaw kamachiq."],
                ["Buenos dias autoridad.", "Buenos dias autoridad."],
                require_thinking=True,
            )

        self.assertEqual(diagnostics["self_verification_format_rate"], 50.0)
        self.assertEqual(diagnostics["self_verification_required_format_rate"], 50.0)
        self.assertEqual(diagnostics["self_verification_missing_score_rate"], 50.0)
        self.assertGreater(diagnostics["self_verification_avg_thinking_score"], 0.0)
        self.assertIn("self_verification_avg_thinking_primitives", diagnostics)

    def test_write_predictions_jsonl_preserves_structured_self_verification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.jsonl"
            train_gspo.write_predictions_jsonl(
                path,
                [
                    {
                        "source": "Buenos dias.",
                        "target": "Allin punchaw.",
                        "source_name": "manual",
                        "variant": "quy/chanka",
                    }
                ],
                ["Allin punchaw."],
                [
                    "Analisis de traduccion: [SIGNIFICADO] conserva. "
                    "Traduccion final: Allin punchaw. "
                    "Autoevaluacion: no veo errores. "
                    "Puntaje: \\boxed{0.95}"
                ],
                self_verification=True,
            )

            payload = json.loads(path.read_text().strip())

        self.assertEqual(payload["source"], "Buenos dias.")
        self.assertEqual(payload["reference"], "Allin punchaw.")
        self.assertEqual(payload["prediction"], "Allin punchaw.")
        self.assertEqual(payload["self_verification"]["translation"], "Allin punchaw.")
        self.assertEqual(payload["self_verification"]["self_score"], 0.95)
        self.assertTrue(payload["self_verification"]["has_thinking_format"])

    def test_reference_rerank_metric_profile_penalizes_source_copy(self):
        with mock.patch.object(train_gspo, "sentence_chrfpp", return_value=0.7), mock.patch.object(
            train_gspo, "sentence_bleu", return_value=0.3
        ):
            translated = train_gspo.reward_score(
                "Allin punchaw kamachiq.",
                "Allin punchaw kamachiq.",
                "Buenos dias autoridad.",
                "reference_rerank_vibe_v1",
            )
            copied = train_gspo.reward_score(
                "Buenos dias autoridad.",
                "Allin punchaw kamachiq.",
                "Buenos dias autoridad.",
                "reference_rerank_vibe_v1",
            )

        self.assertGreater(translated, copied)

    def test_token_precision_rewards_concise_overlap(self):
        self.assertGreater(
            train_gspo.token_precision("allin punchaw", "allin punchaw taytay"),
            train_gspo.token_precision("allin punchaw huk iskay", "allin punchaw taytay"),
        )

    def test_learned_verifier_bleu_margin_penalizes_source_copy(self):
        class FixedScorer:
            def score_many(self, sources, references, hypotheses):
                return [0.8 for _ in hypotheses]

        with mock.patch.object(train_gspo, "sentence_chrfpp", return_value=0.7), mock.patch.object(
            train_gspo, "sentence_bleu", return_value=0.3
        ):
            translated = train_gspo.learned_verifier_bleu_margin_rewards(
                FixedScorer(),
                ["Allin punchaw kamachiq."],
                ["Allin punchaw kamachiq."],
                ["Buenos dias autoridad."],
            )[0]
            copied = train_gspo.learned_verifier_bleu_margin_rewards(
                FixedScorer(),
                ["Buenos dias autoridad."],
                ["Allin punchaw kamachiq."],
                ["Buenos dias autoridad."],
            )[0]

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
            generation_config = mock.Mock()

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
        scorer.tokenizer.padding_side = "right"

        self.assertEqual(scorer.score_many(["src"], ["ref"], ["hyp"]), [0.5])
        self.assertEqual(scorer.tokenizer.args, ())
        self.assertEqual(scorer.tokenizer.kwargs["text"], ["prompt"])
        self.assertEqual(scorer.tokenizer.padding_side, "left")
        self.assertEqual(scorer.model.generate_kwargs["eos_token_id"], 0)

    def test_configure_generation_uses_left_padding(self):
        class GenerationConfig:
            eos_token_id = None
            pad_token_id = None

        class Model:
            generation_config = GenerationConfig()

        class Tokenizer:
            eos_token = "<eos>"
            pad_token = None
            eos_token_id = 7
            padding_side = "right"

        tokenizer = Tokenizer()
        model = Model()

        train_gspo.configure_left_padded_generation(model, tokenizer)

        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(model.generation_config.eos_token_id, 7)
        self.assertEqual(model.generation_config.pad_token_id, 7)

    def test_configure_generation_updates_processor_inner_tokenizer(self):
        class GenerationConfig:
            eos_token_id = None
            pad_token_id = None

        class Model:
            generation_config = GenerationConfig()

        class InnerTokenizer:
            eos_token = "<eos>"
            pad_token = None
            eos_token_id = 9
            padding_side = "right"

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return "encoded"

        class Processor:
            tokenizer = InnerTokenizer()
            eos_token = "<processor-eos>"
            pad_token = None
            eos_token_id = 3
            padding_side = "right"

        processor = Processor()
        model = Model()

        train_gspo.configure_left_padded_generation(model, processor)
        encoded = train_gspo.tokenize_generation_prompts(processor, ["prompt"], padding=True)

        self.assertEqual(processor.padding_side, "left")
        self.assertEqual(processor.tokenizer.padding_side, "left")
        self.assertEqual(processor.tokenizer.pad_token, "<eos>")
        self.assertEqual(model.generation_config.eos_token_id, 9)
        self.assertEqual(encoded, "encoded")
        self.assertEqual(processor.tokenizer.kwargs["text"], ["prompt"])

    def test_parse_args_can_disable_truncated_completion_masking(self):
        args = train_gspo.parse_args(["--no-mask-truncated-completions"])

        self.assertFalse(args.mask_truncated_completions)

    def test_score_box_stop_token_id_uses_single_closing_brace_token(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.text = text
                self.add_special_tokens = add_special_tokens
                return [123]

        tokenizer = Tokenizer()

        self.assertEqual(train_gspo.score_box_stop_token_id(tokenizer), 123)
        self.assertEqual(tokenizer.text, "}")
        self.assertFalse(tokenizer.add_special_tokens)

    def test_score_box_stop_token_id_rejects_multi_token_brace(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2]

        self.assertIsNone(train_gspo.score_box_stop_token_id(Tokenizer()))


if __name__ == "__main__":
    unittest.main()
