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

    def test_frontier_sft_wrappers_use_row_scaled_schedule(self):
        scripts = [
            REPO_ROOT / "experiments/gspo/run_deepseek_v4_pro_sft_from_frontier_data.sh",
            REPO_ROOT / "experiments/gspo/run_deepseek_v4_pro_thinking_sft_then_gspo.sh",
        ]

        for path in scripts:
            with self.subTest(path=path.name):
                content = path.read_text()
                for expected in [
                    "AUTO_FRONTIER_SFT_SCHEDULE",
                    "FRONTIER_ACCEPTED_ROWS",
                    "SFT_MAX_STEPS=$(( (FRONTIER_ACCEPTED_ROWS * 3 + 7) / 8 ))",
                    "SFT_LEARNING_RATE=\"1e-6\"",
                    "Frontier SFT schedule:",
                ]:
                    self.assertIn(expected, content)

    def test_all_in_one_sft_eval_uses_long_completion_cap(self):
        script = (REPO_ROOT / "experiments/gspo/run_deepseek_v4_pro_thinking_sft_then_gspo.sh").read_text()

        self.assertIn('SFT_EVAL_MAX_COMPLETION_LENGTH:-256', script)
        self.assertNotIn('SFT_EVAL_MAX_COMPLETION_LENGTH:-112', script)


if __name__ == "__main__":
    unittest.main()
