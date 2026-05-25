from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class DeepSeekMathShellWorkflowTests(unittest.TestCase):
    def test_all_in_one_sft_manifest_records_frontier_lineage(self):
        script = (REPO_ROOT / "experiments/gspo/run_deepseek_v4_pro_thinking_sft_then_gspo.sh").read_text()
        function_body = script.split("write_sft_seed_manifest() {", 1)[1].split("\n}", 1)[0]

        for expected in [
            "--frontier-jsonl",
            "--frontier-report-json",
            "--frontier-summary-json",
            "--frontier-paid-gate-json",
            "SFT_MANIFEST_FRONTIER_ARGS",
        ]:
            self.assertIn(expected, function_body)

    def test_split_sft_manifest_records_frontier_lineage(self):
        script = (REPO_ROOT / "experiments/gspo/run_deepseek_v4_pro_sft_from_frontier_data.sh").read_text()

        for expected in [
            "--frontier-jsonl",
            "--frontier-report-json",
            "--frontier-summary-json",
            "--frontier-paid-gate-json",
            "SFT_MANIFEST_FRONTIER_ARGS",
        ]:
            self.assertIn(expected, script)


if __name__ == "__main__":
    unittest.main()
