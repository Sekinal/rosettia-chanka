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

    def test_expected_primitives_are_row_specific_and_enforced(self):
        expected = builder.expected_primitives_for_row(
            "El juez Juan aprobo 3 documentos.",
            "Juan sutiyuq juezqa kimsa qillqata chaskirqan.",
        )
        parsed = {
            "analysis": "[SIGNIFICADO] conserva; [GRAMATICA] revisa sufijo; [ENTIDADES] mantiene Juan y 3.",
            "translation": "Juan sutiyuq juezqa kimsa qillqata chaskirqan.",
            "self_evaluation": "Riesgo bajo.",
            "score": 0.9,
        }
        missing = {**parsed, "analysis": "[SIGNIFICADO] conserva; [GRAMATICA] revisa sufijo."}

        self.assertIn("[ENTIDADES]", expected)
        self.assertTrue(builder.record_passes(parsed, min_primitive_tags=2, required_primitives=expected))
        self.assertFalse(builder.record_passes(missing, min_primitive_tags=2, required_primitives=expected))

    def test_select_rows_balances_expected_primitives(self):
        rows = [
            {"source": "Hola.", "target": "Rimaykullayki."},
            {"source": "Buenos dias.", "target": "Allin punchaw."},
            {"source": "Gracias.", "target": "Anay."},
            {"source": "Hay 3 documentos.", "target": "Kimsa qillqakuna kan."},
            {"source": "El tribunal revisa el proceso.", "target": "Tribunalqa procesota qawan."},
            {"source": "Hasta manana.", "target": "Paqarinkama."},
        ]

        selected = builder.select_rows(rows, offset=0, max_rows=3, seed=7, stratify_primitives=True)
        selected_tags = {
            tag
            for row in selected
            for tag in builder.expected_primitives_for_row(row["source"], row["target"])
        }

        self.assertEqual(len(selected), 3)
        self.assertIn("[ENTIDADES]", selected_tags)
        self.assertIn("[TERMINOLOGIA]", selected_tags)

    def test_select_rows_dedupes_before_spending_frontier_calls(self):
        rows = [
            {"source": "Hola.", "target": "Rimaykullayki."},
            {"source": "Hola.", "target": "Rimaykullayki."},
            {"source": "Hay 3 documentos.", "target": "Kimsa qillqakuna kan."},
        ]

        selected = builder.select_rows(rows, offset=0, max_rows=3, seed=7, stratify_primitives=True)

        self.assertEqual(len(selected), 2)
        self.assertEqual(len({builder.row_key(row["source"], row["target"]) for row in selected}), 2)

    def test_select_rows_can_keep_seeded_random_slice(self):
        rows = [
            {"source": "Hola.", "target": "Rimaykullayki."},
            {"source": "Buenos dias.", "target": "Allin punchaw."},
            {"source": "Gracias.", "target": "Anay."},
        ]

        selected = builder.select_rows(rows, offset=0, max_rows=2, seed=1, stratify_primitives=False)

        self.assertEqual(selected, builder.select_rows(rows, offset=0, max_rows=2, seed=1, stratify_primitives=False))

    def test_selection_report_summarizes_expected_primitive_curriculum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"source": "Hola.", "target": "Rimaykullayki."},
                {"source": "Hay 3 documentos.", "target": "Kimsa qillqakuna kan."},
            ]
            args = builder.parse_args(["--output-jsonl", str(Path(tmpdir) / "out.jsonl"), "--max-rows", "2"])

            payload = builder.selection_report(rows, args, input_rows=rows + [rows[0]])

        self.assertEqual(payload["selected_rows"], 2)
        self.assertEqual(payload["input_rows"], 3)
        self.assertEqual(payload["duplicate_input_rows"], 1)
        self.assertEqual(payload["pending_rows"], 2)
        self.assertEqual(payload["estimated_generation_requests"], 2)
        self.assertGreaterEqual(payload["expected_primitive_counts"]["[SIGNIFICADO]"], 2)
        self.assertGreaterEqual(payload["expected_primitive_counts"]["[ENTIDADES]"], 1)
        self.assertEqual(len(payload["samples"]), 2)

    def test_selection_report_counts_resume_pending_requests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output = tmp / "out.jsonl"
            failures = tmp / "failures.jsonl"
            rows = [
                {"source": "Hola.", "target": "Rimaykullayki."},
                {"source": "Hay 3 documentos.", "target": "Kimsa qillqakuna kan."},
                {"source": "Gracias.", "target": "Anay."},
            ]
            output.write_text(
                json.dumps(
                    {
                        "row_key": builder.row_key("Hola.", "Rimaykullayki."),
                        "source": "Hola.",
                        "reference": "Rimaykullayki.",
                    }
                )
                + "\n"
            )
            failures.write_text(
                json.dumps(
                    {
                        "row_key": builder.row_key("Gracias.", "Anay."),
                        "source": "Gracias.",
                        "reference": "Anay.",
                    }
                )
                + "\n"
            )
            args = builder.parse_args(
                [
                    "--output-jsonl",
                    str(output),
                    "--failures-jsonl",
                    str(failures),
                    "--audit",
                    "--max-retries",
                    "4",
                ]
            )

            payload = builder.selection_report(rows, args, input_rows=rows)

        self.assertEqual(payload["selected_existing_accepted_rows"], 1)
        self.assertEqual(payload["selected_existing_failed_rows"], 1)
        self.assertEqual(payload["pending_rows"], 1)
        self.assertEqual(payload["estimated_generation_requests"], 1)
        self.assertEqual(payload["estimated_max_audit_requests"], 1)
        self.assertEqual(payload["estimated_max_frontier_requests"], 2)
        self.assertEqual(payload["estimated_max_http_attempts"], 8)

    def test_selected_row_records_include_resume_statuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output = tmp / "out.jsonl"
            failures = tmp / "failures.jsonl"
            rows = [
                {"source": "Hola.", "target": "Rimaykullayki."},
                {"source": "Hay 3 documentos.", "target": "Kimsa qillqakuna kan."},
                {"source": "Gracias.", "target": "Anay."},
            ]
            output.write_text(
                json.dumps({"row_key": builder.row_key("Hola.", "Rimaykullayki.")}) + "\n"
            )
            failures.write_text(
                json.dumps({"row_key": builder.row_key("Gracias.", "Anay.")}) + "\n"
            )

            records = builder.selected_row_records(rows, output, failures, resume=True, retry_failures=False)

        statuses = {record["source"]: record["resume_status"] for record in records}
        self.assertEqual(statuses["Hola."], "existing_accepted")
        self.assertEqual(statuses["Gracias."], "existing_failed")
        self.assertEqual(statuses["Hay 3 documentos."], "pending")
        self.assertIn("[ENTIDADES]", records[1]["expected_primitives"])

    def test_prompt_preview_records_include_exact_generation_payload_without_secret(self):
        rows = [
            {"source": "Hola.", "target": "Rimaykullayki."},
            {"source": "Hay 3 documentos.", "target": "Kimsa qillqakuna kan."},
        ]
        args = builder.parse_args(["--output-jsonl", "out.jsonl", "--few-shot-count", "1"])
        selected_records = [
            {
                "row_key": builder.row_key(row["source"], row["target"]),
                "resume_status": "pending",
            }
            for row in rows
        ]

        previews = builder.prompt_preview_records(rows, selected_records, args)

        self.assertEqual(len(previews), 2)
        self.assertEqual(previews[0]["frontier_model"], "deepseek-v4-pro")
        self.assertEqual(previews[0]["generation_payload"]["model"], "deepseek-v4-pro")
        self.assertEqual(previews[0]["generation_payload"]["thinking"]["type"], "enabled")
        self.assertNotIn("Authorization", json.dumps(previews[0], ensure_ascii=False))
        self.assertNotIn("DEEPSEEK_API_KEY", json.dumps(previews[0], ensure_ascii=False))
        self.assertIn("[SIGNIFICADO]", previews[0]["generation_payload"]["messages"][1]["content"])
        self.assertIn("[GRAMATICA]", previews[0]["generation_payload"]["messages"][1]["content"])
        self.assertEqual(previews[0]["few_shot_sources"], ["Hay 3 documentos."])

    def test_selection_only_writes_reports_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_jsonl = tmp / "rows.jsonl"
            report_json = tmp / "selection.json"
            report_md = tmp / "selection.md"
            selection_jsonl = tmp / "selection_rows.jsonl"
            preview_jsonl = tmp / "prompt_preview.jsonl"
            source_jsonl.write_text(
                '{"source":"Hola.","reference":"Rimaykullayki."}\n'
                '{"source":"Hay 3 documentos.","reference":"Kimsa qillqakuna kan."}\n'
            )

            with mock.patch.dict("os.environ", {}, clear=True):
                builder.main_from_args(
                    [
                        "--source-jsonl",
                        str(source_jsonl),
                        "--output-jsonl",
                        str(tmp / "out.jsonl"),
                        "--selection-report-json",
                        str(report_json),
                        "--selection-report-md",
                        str(report_md),
                        "--selection-jsonl",
                        str(selection_jsonl),
                        "--prompt-preview-jsonl",
                        str(preview_jsonl),
                        "--selection-only",
                    ]
                )

            payload = json.loads(report_json.read_text())
            markdown = report_md.read_text()
            selected_rows = [json.loads(line) for line in selection_jsonl.read_text().splitlines()]
            previews = [json.loads(line) for line in preview_jsonl.read_text().splitlines()]

        self.assertEqual(payload["selected_rows"], 2)
        self.assertIn("[ENTIDADES]", payload["expected_primitive_counts"])
        self.assertEqual(len(selected_rows), 2)
        self.assertEqual({row["resume_status"] for row in selected_rows}, {"pending"})
        self.assertEqual(len(previews), 2)
        self.assertEqual(previews[0]["generation_payload"]["model"], "deepseek-v4-pro")
        self.assertNotIn("Authorization", json.dumps(previews, ensure_ascii=False))
        self.assertIn("duplicate input rows skipped", markdown)
        self.assertIn("Frontier Source Selection Report", markdown)

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
                                    "analysis": (
                                        "[SIGNIFICADO] conserva saludo; [GRAMATICA] forma natural; "
                                        "[ANTI_COPIA] evita copia."
                                    ),
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

    def test_main_stops_at_runtime_api_request_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_jsonl = tmp / "rows.jsonl"
            output_jsonl = tmp / "out.jsonl"
            summary_json = tmp / "summary.json"
            source_jsonl.write_text(
                '{"source":"Buenos dias.","reference":"Allin punchaw."}\n'
                '{"source":"Hola","reference":"Rimaykullayki"}\n'
            )
            response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "analysis": (
                                        "[SIGNIFICADO] conserva saludo; [GRAMATICA] forma natural; "
                                        "[ANTI_COPIA] evita copia."
                                    ),
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
                        "--summary-json",
                        str(summary_json),
                        "--max-rows",
                        "2",
                        "--max-api-requests",
                        "1",
                    ]
                )

            rows = list(builder.iter_jsonl(output_jsonl))
            summary = json.loads(summary_json.read_text())["summary"]

        self.assertEqual(mocked_call.call_count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["api_requests_used"], 1)
        self.assertEqual(summary["max_api_requests"], 1)
        self.assertTrue(summary["stopped_by_api_request_budget"])

    def test_audit_mode_reserves_two_runtime_api_requests_per_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_jsonl = tmp / "rows.jsonl"
            output_jsonl = tmp / "out.jsonl"
            summary_json = tmp / "summary.json"
            source_jsonl.write_text('{"source":"Hola","reference":"Rimaykullayki"}\n')

            with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}, clear=False), mock.patch.object(
                builder, "call_chat_completion"
            ) as mocked_call:
                builder.main_from_args(
                    [
                        "--source-jsonl",
                        str(source_jsonl),
                        "--output-jsonl",
                        str(output_jsonl),
                        "--summary-json",
                        str(summary_json),
                        "--audit",
                        "--max-api-requests",
                        "1",
                    ]
                )

            summary = json.loads(summary_json.read_text())["summary"]

        self.assertEqual(mocked_call.call_count, 0)
        self.assertFalse(output_jsonl.exists())
        self.assertEqual(summary["api_requests_used"], 0)
        self.assertTrue(summary["stopped_by_api_request_budget"])


if __name__ == "__main__":
    unittest.main()
