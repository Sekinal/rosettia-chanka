from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_gspo_canaries as summarize


class SummarizeGspoCanariesTests(unittest.TestCase):
    def test_collect_metrics_sorts_by_reward_then_quality_and_copy(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for name, reward, chrf, copy in [
                ("weak", 0.1, 30.0, 1.0),
                ("best", 0.2, 25.0, 0.5),
                ("tie_lower_quality", 0.2, 20.0, 0.1),
            ]:
                metrics_path = root / name / "chanka_gspo" / "final_metrics.json"
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(
                    json.dumps(
                        {
                            "reward_profile": name,
                            "trainer_eval_reward": reward,
                            "chrf++": chrf,
                            "source_copy_ratio": copy,
                        }
                    )
                )

            records = summarize.collect_metrics(root)

        self.assertEqual([record["reward_profile"] for record in records], ["best", "tie_lower_quality", "weak"])

    def test_write_markdown_includes_profile_table(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "summary.md"
            summarize.write_markdown(
                [
                    {
                        "reward_profile": "rosettia_guard_v2",
                        "trainer_eval_reward": 0.4,
                        "chrf++": 50.0,
                        "bleu": 10.0,
                        "token_f1": 30.0,
                        "source_copy_ratio": 1.0,
                        "exact_source_copy_rate": 0.0,
                        "spanish_leakage_penalty": 0.0,
                    }
                ],
                path,
            )

            content = path.read_text()

        self.assertIn("rosettia_guard_v2", content)
        self.assertIn("Eval reward", content)


if __name__ == "__main__":
    unittest.main()
