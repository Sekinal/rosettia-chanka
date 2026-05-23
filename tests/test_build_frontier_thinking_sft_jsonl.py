from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_frontier_thinking_sft_jsonl as builder


class BuildFrontierThinkingSftJsonlTests(unittest.TestCase):
    def test_payload_uses_deepseek_v4_pro_and_thinking_by_default(self):
        args = builder.parse_args(["--output-jsonl", "out.jsonl"])

        payload = builder.request_payload(args, "Buenos dias.", "Allin punchaw.")

        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["thinking"]["type"], "enabled")
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertEqual(payload["response_format"], {"type": "json_object"})

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

    def test_load_source_jsonl_accepts_reference_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text('{"source":"Hola","reference":"Rimaykullayki"}\n')
            args = builder.parse_args(["--source-jsonl", str(path), "--output-jsonl", "out.jsonl"])

            rows = builder.load_rows(args)

        self.assertEqual(rows[0]["source"], "Hola")
        self.assertEqual(rows[0]["target"], "Rimaykullayki")


if __name__ == "__main__":
    unittest.main()
