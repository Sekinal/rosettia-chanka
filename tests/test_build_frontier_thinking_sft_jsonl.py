from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_frontier_thinking_sft_jsonl as builder


class BuildFrontierThinkingSftJsonlTests(unittest.TestCase):
    def test_payload_uses_deepseek_v4_pro_and_thinking_by_default(self):
        args = builder.parse_args(["--output-jsonl", "out.jsonl"])

        payload = builder.request_payload(args, "Buenos dias.", "Allin punchaw.")

        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["thinking"]["type"], "enabled")
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_prompt_can_include_reviewed_few_shot_primitive_examples(self):
        rows = [
            {"source": "Buenos dias.", "target": "Allin punchaw."},
            {"source": "Hola.", "target": "Rimaykullayki."},
            {"source": "Gracias.", "target": "Añay."},
        ]

        few_shots = builder.select_few_shots(rows, rows[0], count=2)
        messages = builder.prompt_messages(rows[0]["source"], rows[0]["target"], few_shots)

        self.assertEqual(len(few_shots), 2)
        self.assertNotIn("Buenos dias.", {example["source"] for example in few_shots})
        self.assertIn("Good examples", messages[1]["content"])
        self.assertIn("[SIGNIFICADO]", messages[1]["content"])
        self.assertIn("[GRAMATICA]", messages[1]["content"])
        self.assertIn("JSON target", messages[1]["content"])

    def test_request_payload_with_few_shots_includes_examples(self):
        args = builder.parse_args(["--output-jsonl", "out.jsonl"])
        few_shots = [
            {
                "source": "Hola.",
                "reference": "Rimaykullayki.",
                "target": builder.few_shot_target("Rimaykullayki.", 0),
            }
        ]

        payload = builder.request_payload_with_few_shots(args, "Buenos dias.", "Allin punchaw.", few_shots)

        self.assertIn("Good examples", payload["messages"][1]["content"])
        self.assertIn("Rimaykullayki.", payload["messages"][1]["content"])

    def test_dry_run_audit_payload_uses_audit_model_when_set(self):
        args = builder.parse_args(
            [
                "--output-jsonl",
                "out.jsonl",
                "--audit",
                "--audit-model",
                "deepseek-v4-flash",
            ]
        )

        payload = builder.chat_payload(
            args,
            args.audit_model,
            builder.audit_messages(
                "Buenos dias.",
                "Allin punchaw.",
                {
                    "analysis": "[SIGNIFICADO] conserva; [ANTI_COPIA] evita copia.",
                    "translation": "Allin punchaw.",
                    "self_evaluation": "Riesgo bajo.",
                    "score": 0.95,
                },
            ),
            128,
        )

        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertIn("pass (boolean)", payload["messages"][1]["content"])

    def test_parse_frontier_json_and_build_target_keeps_reference_by_default(self):
        parsed = builder.parse_frontier_json(
            json.dumps(
                {
                    "analysis": "[SIGNIFICADO] conserva saludo; [ANTI_COPIA] no copia espanol.",
                    "translation": "Mala traduccion sintetica.",
                    "self_evaluation": "Riesgo bajo.",
                    "score": 0.91,
                }
            )
        )

        target = builder.build_target(parsed, "Allin punchaw.", allow_model_translation=False)

        self.assertIn("Analisis de traduccion:", target)
        self.assertIn("Traduccion final: Allin punchaw.", target)
        self.assertNotIn("Mala traduccion", target)
        self.assertIn("Puntaje: \\boxed{0.91}", target)

    def test_record_filter_requires_primitive_tags(self):
        good = {
            "analysis": "[SIGNIFICADO] conserva; [GRAMATICA] revisa sufijo.",
            "translation": "Allin punchaw.",
            "self_evaluation": "Riesgo bajo.",
            "score": 0.9,
        }
        bad = {**good, "analysis": "conserva significado sin etiquetas"}

        self.assertTrue(builder.record_passes(good, min_primitive_tags=2))
        self.assertFalse(builder.record_passes(bad, min_primitive_tags=2))

    def test_parse_and_gate_audit_json(self):
        audit = builder.parse_audit_json('{"pass": true, "score": 0.82, "reason": "concise"}')

        self.assertTrue(builder.audit_passes(audit, min_score=0.75))
        self.assertFalse(builder.audit_passes(audit, min_score=0.9))

    def test_load_source_jsonl_accepts_reference_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text('{"source":"Hola","reference":"Rimaykullayki"}\n')
            args = builder.parse_args(["--source-jsonl", str(path), "--output-jsonl", "out.jsonl"])

            rows = builder.load_rows(args)

        self.assertEqual(rows[0]["source"], "Hola")
        self.assertEqual(rows[0]["target"], "Rimaykullayki")

    def test_main_resumes_and_appends_incrementally(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_jsonl = tmp / "rows.jsonl"
            output_jsonl = tmp / "out.jsonl"
            failures_jsonl = tmp / "failures.jsonl"
            source_jsonl.write_text(
                '{"source":"Buenos dias.","reference":"Allin punchaw."}\n'
                '{"source":"Hola","reference":"Rimaykullayki"}\n'
            )
            output_jsonl.write_text(
                json.dumps(
                    {
                        "row_key": builder.row_key("Buenos dias.", "Allin punchaw."),
                        "source": "Buenos dias.",
                        "reference": "Allin punchaw.",
                    }
                )
                + "\n"
            )
            response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "analysis": "[SIGNIFICADO] conserva saludo; [ANTI_COPIA] evita copia.",
                                    "translation": "Rimaykullayki",
                                    "self_evaluation": "Riesgo bajo.",
                                    "score": 0.94,
                                }
                            )
                        }
                    }
                ]
            }

            with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}, clear=False), mock.patch.object(
                builder, "call_chat_completion", return_value=response
            ) as mocked_call:
                builder.main_from_args(
                    [
                        "--source-jsonl",
                        str(source_jsonl),
                        "--output-jsonl",
                        str(output_jsonl),
                        "--failures-jsonl",
                        str(failures_jsonl),
                        "--max-rows",
                        "2",
                    ]
                )

            rows = list(builder.iter_jsonl(output_jsonl))

        self.assertEqual(mocked_call.call_count, 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["source"], "Hola")
        self.assertFalse(failures_jsonl.exists())


if __name__ == "__main__":
    unittest.main()
