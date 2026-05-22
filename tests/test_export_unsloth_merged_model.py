from __future__ import annotations

import unittest
from pathlib import Path

from scripts import export_unsloth_merged_model as export_merged


class ExportUnslothMergedModelTests(unittest.TestCase):
    def test_parse_args_defaults_to_merged_16bit(self):
        args = export_merged.parse_args(
            [
                "--adapter-path",
                "outputs/run/checkpoint-8",
                "--output-dir",
                "outputs/merged/run",
            ]
        )

        self.assertEqual(args.adapter_path, Path("outputs/run/checkpoint-8"))
        self.assertEqual(args.output_dir, Path("outputs/merged/run"))
        self.assertEqual(args.max_seq_length, 128)
        self.assertEqual(args.save_method, "merged_16bit")


if __name__ == "__main__":
    unittest.main()
