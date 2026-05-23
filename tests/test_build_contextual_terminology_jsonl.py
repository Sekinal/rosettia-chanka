import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts import build_contextual_terminology_jsonl as builder


class BuildContextualTerminologyJsonlTests(unittest.TestCase):
    def test_valid_entry_rejects_stopwords_digits_and_long_terms(self):
        args = Namespace(
            min_source_chars=3,
            max_source_chars=16,
            max_target_chars=16,
            max_source_words=2,
            max_target_words=2,
        )

        self.assertTrue(builder.valid_entry("abogado", "amachaq", args))
        self.assertFalse(builder.valid_entry("de", "manta", args))
        self.assertFalse(builder.valid_entry("articulo 12", "chunka iskay", args))
        self.assertFalse(builder.valid_entry("padres (padre y madre)", "taytamama", args))
        self.assertFalse(builder.valid_entry("termino demasiado largo", "simi", args))

    @patch("scripts.build_contextual_terminology_jsonl.gspo.load_terminology_entries")
    def test_build_rows_wraps_terms_in_context_templates(self, mock_entries):
        mock_entries.return_value = [
            ("abogado", "amachaq"),
            ("juez", "taripaq"),
            ("abogado", "amachaq"),
            ("articulo 12", "chunka iskay"),
        ]
        args = builder.parse_args(
            [
                "--output-jsonl",
                "unused.jsonl",
                "--templates-per-term",
                "2",
                "--max-terms",
                "2",
            ]
        )

        rows, metrics = builder.build_rows(args)

        self.assertEqual(metrics["accepted_terms"], 2)
        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(rows[0]["source"], "Explique abogado.")
        self.assertEqual(rows[0]["target"], "amachaqta sut'ichay.")
        self.assertEqual(rows[1]["source"], "Explique el caso de abogado.")
        self.assertEqual(rows[1]["target"], "amachaq kasutam sut'ichay.")
        self.assertEqual(rows[0]["label_type"], "terminology_context_explain_term")

    def test_write_jsonl_creates_parent_directory(self):
        rows = [
            {
                "source": "Explique abogado.",
                "target": "amachaqta sut'ichay.",
                "label_type": "terminology_context_explain_term",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "context.jsonl"
            builder.write_jsonl(path, rows)
            written = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(written, rows)


if __name__ == "__main__":
    unittest.main()
