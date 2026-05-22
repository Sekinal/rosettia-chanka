from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_terminology_pair_jsonl as builder


class BuildTerminologyPairJsonlTests(unittest.TestCase):
    def test_build_rows_filters_and_dedupes_terms(self):
        args = argparse.Namespace(
            dataset_repo="dataset",
            terminology_file="terms.parquet",
            min_source_chars=3,
            max_source_chars=20,
            max_target_chars=20,
            max_rows=None,
        )

        with mock.patch.object(
            builder.gspo,
            "load_terminology_entries",
            return_value=[
                ("Casado", "Warmiyuq"),
                ("casado", "Warmiyuq"),
                ("el", "dummy"),
                ("Este termino es demasiado largo", "dummy"),
                ("Valido", "x" * 21),
            ],
        ):
            rows = builder.build_rows(args)

        self.assertEqual(
            rows,
            [
                {
                    "source": "Casado",
                    "target": "Warmiyuq",
                    "reference": "Warmiyuq",
                    "source_name": "terms.parquet",
                    "variant": "quy/chanka_terminology",
                    "label_type": "terminology_pair",
                }
            ],
        )

    def test_write_jsonl_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "terms.jsonl"
            builder.write_jsonl(
                path,
                [
                    {
                        "source": "Casado",
                        "target": "Warmiyuq",
                        "reference": "Warmiyuq",
                        "source_name": "terms.parquet",
                        "variant": "quy/chanka_terminology",
                        "label_type": "terminology_pair",
                    }
                ],
            )

            self.assertIn('"source": "Casado"', path.read_text())


if __name__ == "__main__":
    unittest.main()
