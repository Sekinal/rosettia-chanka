from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import check_frontier_prompt_preview as checker


def preview_record(row_key: str = "hola\trimaykullayki") -> dict:
    return {
        "row_key": row_key,
        "expected_primitives": ["[SIGNIFICADO]", "[GRAMATICA]", "[ANTI_COPIA]"],
        "few_shot_row_keys": ["gracias\tanay"],
        "generation_payload": {
            "model": "deepseek-v4-pro",
            "messages": [
                {"role": "system", "content": "Return only valid JSON."},
                {
                    "role": "user",
                    "content": (
                        "Required primitive tags for this row: [SIGNIFICADO], [GRAMATICA], [ANTI_COPIA].\n"
                        "Source terms to consider: Hola.\n"
                        "Reference terms to consider: Rimaykullayki.\n"
                        "Do not write generic checks like 'correcto'. "
                        "Each tag must mention a concrete source/reference token. "
                        "Use at least six non-tag words in the analysis."
                    ),
                },
            ],
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
        },
    }


class CheckFrontierPromptPreviewTests(unittest.TestCase):
    def test_gate_passes_for_valid_preview(self):
        metrics, passed, reasons = checker.gate_prompt_preview(
            [preview_record()],
            expected_model="deepseek-v4-pro",
            min_preview_rows=1,
            selected_rows=1,
            expected_reasoning_effort="max",
        )

        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        self.assertEqual(metrics["preview_rows"], 1)
        self.assertEqual(metrics["secret_marker_records"], 0)
        self.assertEqual(metrics["missing_anti_vacuous_instruction_records"], 0)

    def test_gate_fails_on_missing_expected_tags_and_secret_marker(self):
        bad = preview_record()
        bad["generation_payload"]["messages"][1]["content"] = "Required primitive tags for this row: [SIGNIFICADO]."
        bad["generation_payload"]["headers"] = {"Authorization": "Bearer should-not-exist"}

        _, passed, reasons = checker.gate_prompt_preview(
            [bad],
            expected_model="deepseek-v4-pro",
            min_preview_rows=1,
            selected_rows=1,
        )

        self.assertFalse(passed)
        self.assertIn("record 1 contains secret/auth marker", reasons)
        self.assertIn("record 1 required primitive line missing expected tags: [GRAMATICA], [ANTI_COPIA]", reasons)

    def test_gate_fails_when_prompt_lacks_anti_vacuous_instructions(self):
        bad = preview_record()
        bad["generation_payload"]["messages"][1]["content"] = (
            "Required primitive tags for this row: [SIGNIFICADO], [GRAMATICA], [ANTI_COPIA]."
        )

        metrics, passed, reasons = checker.gate_prompt_preview(
            [bad],
            expected_model="deepseek-v4-pro",
            min_preview_rows=1,
            selected_rows=1,
        )

        self.assertFalse(passed)
        self.assertEqual(metrics["missing_anti_vacuous_instruction_records"], 1)
        self.assertTrue(any("prompt missing anti-vacuous instructions" in reason for reason in reasons))

    def test_gate_fails_on_wrong_model_or_thinking_shape(self):
        bad = preview_record()
        bad["generation_payload"]["model"] = "deepseek-chat"
        bad["generation_payload"]["thinking"] = {"type": "disabled"}
        bad["generation_payload"]["response_format"] = {"type": "text"}

        _, passed, reasons = checker.gate_prompt_preview(
            [bad],
            expected_model="deepseek-v4-pro",
            min_preview_rows=1,
            selected_rows=1,
        )

        self.assertFalse(passed)
        self.assertIn("record 1 model 'deepseek-chat' != 'deepseek-v4-pro'", reasons)
        self.assertIn("record 1 does not enable thinking", reasons)
        self.assertIn("record 1 does not require JSON response_format", reasons)

    def test_gate_fails_on_current_row_few_shot_leak_and_preview_count(self):
        bad = preview_record()
        bad["few_shot_row_keys"] = [bad["row_key"]]

        _, passed, reasons = checker.gate_prompt_preview(
            [bad],
            expected_model="deepseek-v4-pro",
            min_preview_rows=2,
            selected_rows=3,
        )

        self.assertFalse(passed)
        self.assertIn("preview_rows 1 < 2", reasons)
        self.assertIn("preview_rows 1 != selected_rows 3", reasons)
        self.assertIn("record 1 uses current row as a few-shot example", reasons)

    def test_main_writes_gate_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            preview_path = root / "preview.jsonl"
            selection_path = root / "selection.jsonl"
            output_path = root / "gate.json"
            preview_path.write_text(json.dumps(preview_record(), ensure_ascii=False) + "\n")
            selection_path.write_text(json.dumps({"row_key": "hola\trimaykullayki"}) + "\n")

            status = checker.main(
                [
                    "--prompt-preview-jsonl",
                    str(preview_path),
                    "--selection-jsonl",
                    str(selection_path),
                    "--output-json",
                    str(output_path),
                    "--expected-reasoning-effort",
                    "max",
                ]
            )
            payload = json.loads(output_path.read_text())

        self.assertEqual(status, 0)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["preview_rows"], 1)


if __name__ == "__main__":
    unittest.main()
