from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_mixed_sft_jsonl as build_mixed


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class BuildMixedSftJsonlTests(unittest.TestCase):
    def test_pseudo_rows_normalize_prediction_to_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mbr.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "source": " Buenos dias. ",
                        "prediction": " Allin punchaw. ",
                        "reference": "Allin punchaw.",
                        "variant": "quy/chanka",
                    }
                ],
            )
            args = build_mixed.parse_args(["--output-jsonl", str(Path(tmpdir) / "out.jsonl"), "--pseudo-jsonl", str(path), "--no-clean-anchors"])

            rows = build_mixed.build_rows(args)

        self.assertEqual(rows[0]["source"], "Buenos dias.")
        self.assertEqual(rows[0]["target"], "Allin punchaw.")
        self.assertEqual(rows[0]["reference"], "Allin punchaw.")
        self.assertEqual(rows[0]["label_type"], "pseudo_mbr")

    def test_clean_anchors_are_added_before_pseudo_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mbr.jsonl"
            write_jsonl(path, [{"source": "S1", "prediction": "T1", "reference": "T1"}])
            args = build_mixed.parse_args(["--output-jsonl", str(Path(tmpdir) / "out.jsonl"), "--pseudo-jsonl", str(path), "--no-shuffle"])

            clean_rows = [
                {"source": "S1", "target": "T1", "source_name": "clean", "variant": "quy/chanka"},
                {"source": "S2", "target": "T2", "source_name": "clean", "variant": "quy/chanka"},
            ]
            with mock.patch.object(build_mixed.gspo, "load_chanka_rows", return_value=clean_rows), mock.patch.object(
                build_mixed.gspo,
                "split_rows",
                return_value=(clean_rows, []),
            ):
                rows = build_mixed.build_rows(args)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["label_type"], "clean_anchor")
        self.assertEqual(rows[0]["source"], "S1")
        self.assertEqual(rows[1]["label_type"], "clean_anchor")
        self.assertEqual(rows[1]["source"], "S2")

    def test_write_jsonl_outputs_target_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.jsonl"
            build_mixed.write_jsonl(
                path,
                [{"source": "S", "target": "T", "reference": "R", "source_name": "x", "variant": "v", "label_type": "pseudo_mbr"}],
            )

            record = json.loads(path.read_text())

        self.assertEqual(record["target"], "T")
        self.assertEqual(record["label_type"], "pseudo_mbr")


if __name__ == "__main__":
    unittest.main()
