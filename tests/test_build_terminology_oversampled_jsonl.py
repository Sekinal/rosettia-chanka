from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_terminology_oversampled_jsonl as builder


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class BuildTerminologyOversampledJsonlTests(unittest.TestCase):
    def test_build_rows_repeats_only_terminology_matched_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.jsonl"
            write_jsonl(
                path,
                [
                    {"source": "¿Es usted casado?", "target": "¿Warmiyuqchu kanki?", "label_type": "clean"},
                    {"source": "Buenos dias.", "target": "Allin punchaw.", "label_type": "clean"},
                ],
            )
            args = argparse.Namespace(
                input_jsonl=[path],
                output_jsonl=Path(tmpdir) / "out.jsonl",
                source_field="source",
                target_field="target",
                reference_field="reference",
                dataset_repo="dataset",
                terminology_file="terms.parquet",
                terminology_top_k=1,
                terminology_min_source_chars=3,
                repeat_matched=3,
                max_extra_rows=None,
                seed=1,
                no_shuffle=True,
            )

            with mock.patch.object(
                builder.gspo,
                "load_terminology_entries",
                return_value=[("casado", "Warmiyuq")],
            ):
                rows, metrics = builder.build_rows(args)

        self.assertEqual(metrics["input_rows"], 2)
        self.assertEqual(metrics["matched_base_rows"], 1)
        self.assertEqual(metrics["extra_rows"], 2)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[2]["label_type"], "clean_terminology_oversample")
        self.assertEqual(rows[2]["terminology"], [{"source_term": "casado", "target_term": "Warmiyuq"}])

    def test_max_extra_rows_caps_repeats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.jsonl"
            write_jsonl(
                path,
                [
                    {"source": "casado", "target": "Warmiyuq"},
                    {"source": "casada", "target": "Qusayuq"},
                ],
            )
            args = argparse.Namespace(
                input_jsonl=[path],
                output_jsonl=Path(tmpdir) / "out.jsonl",
                source_field="source",
                target_field="target",
                reference_field="reference",
                dataset_repo="dataset",
                terminology_file="terms.parquet",
                terminology_top_k=1,
                terminology_min_source_chars=3,
                repeat_matched=4,
                max_extra_rows=1,
                seed=1,
                no_shuffle=True,
            )

            with mock.patch.object(
                builder.gspo,
                "load_terminology_entries",
                return_value=[("casado", "Warmiyuq"), ("casada", "Qusayuq")],
            ):
                rows, metrics = builder.build_rows(args)

        self.assertEqual(metrics["extra_rows"], 1)
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
