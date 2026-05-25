from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_deepseekmath_next_action as runner


def status(stage: str, command: str) -> dict:
    return {"next_action": {"stage": stage, "command": command}}


def touch_script(root: Path, script: str) -> None:
    path = root / script
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nexit 0\n")


class RunDeepSeekMathNextActionTests(unittest.TestCase):
    def test_approves_frontier_generation_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = "experiments/gspo/run_deepseek_v4_pro_paid_generation_smoke.sh"
            touch_script(root, script)

            report = runner.validate_action(
                status("frontier_generation", f"DATA_DIR=outputs/frontier {script}"),
                root,
            )

        self.assertTrue(report["approved"])
        self.assertEqual(report["env"], {"DATA_DIR": "outputs/frontier"})
        self.assertEqual(report["script"], script)

    def test_approves_promoted_manifest_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = "experiments/gspo/run_hardcase_meta_then_followup_gspo_cycle.sh"
            touch_script(root, script)

            report = runner.validate_action(
                status("promoted_policy", f"BASE_CYCLE_MANIFEST=outputs/gspo/cycle_manifest.json {script}"),
                root,
            )

        self.assertTrue(report["approved"])
        self.assertEqual(report["env"]["BASE_CYCLE_MANIFEST"], "outputs/gspo/cycle_manifest.json")

    def test_approves_hardcase_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = "experiments/gspo/run_next_meta_verifier_from_hardcases.sh"
            touch_script(root, script)

            report = runner.validate_action(
                status("hardcase_iteration", f"GSPO_META_JSONL=outputs/hard.jsonl {script}"),
                root,
            )

        self.assertTrue(report["approved"])
        self.assertEqual(report["env"]["GSPO_META_JSONL"], "outputs/hard.jsonl")

    def test_rejects_unknown_script_extra_args_and_unknown_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = "experiments/gspo/run_deepseek_v4_pro_paid_generation_smoke.sh"
            touch_script(root, script)

            report = runner.validate_action(
                status("frontier_generation", f"BAD=1 DATA_DIR=out {script} --unsafe"),
                root,
            )

        self.assertFalse(report["approved"])
        self.assertIn("unexpected command arguments: --unsafe", report["reasons"])
        self.assertIn("env vars not allowlisted for frontier_generation: BAD", report["reasons"])

    def test_rejects_non_runnable_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = runner.validate_action(status("inspect_sft_seed", "cat manifest.json"), root)

        self.assertFalse(report["approved"])
        self.assertIn("stage is not runnable: inspect_sft_seed", report["reasons"])

    def test_main_writes_dry_run_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script = "experiments/gspo/run_deepseek_v4_pro_paid_generation_smoke.sh"
            touch_script(root, script)
            status_json = root / "status.json"
            output_json = root / "next.json"
            status_json.write_text(json.dumps(status("frontier_generation", f"DATA_DIR=out {script}")))

            code = runner.main(
                [
                    "--status-json",
                    str(status_json),
                    "--repo-root",
                    str(root),
                    "--output-json",
                    str(output_json),
                ]
            )
            payload = json.loads(output_json.read_text())

        self.assertEqual(code, 0)
        self.assertTrue(payload["approved"])
        self.assertFalse(payload["execute"])

    def test_main_writes_structured_failure_for_missing_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_json = root / "next.json"
            code = runner.main(
                [
                    "--status-json",
                    str(root / "missing.json"),
                    "--repo-root",
                    str(root),
                    "--output-json",
                    str(output_json),
                ]
            )
            payload = json.loads(output_json.read_text())

        self.assertEqual(code, 1)
        self.assertFalse(payload["approved"])
        self.assertIn("status JSON missing or invalid", payload["reasons"][0])


if __name__ == "__main__":
    unittest.main()
