from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import preflight_deepseekmath_language_loop as preflight


class PreflightDeepSeekMathLanguageLoopTests(unittest.TestCase):
    def test_fails_without_required_api_key_or_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            root = Path(tmpdir)
            args = argparse.Namespace(
                base_adapter=root / "missing",
                output_root=root,
                api_key_env="DEEPSEEK_API_KEY",
                require_api_key=True,
                min_free_gb=0.0,
                required_path=[],
                default_required_paths=False,
            )

            report = preflight.build_report(args, root=root)

        self.assertFalse(report["passed"])
        self.assertIn("missing API key environment variable: DEEPSEEK_API_KEY", report["failures"])
        self.assertIn("missing base adapter:", " ".join(report["failures"]))
        self.assertFalse(report["api_key_set"])
        self.assertNotIn("sk-", str(report))

    def test_passes_with_adapter_api_key_and_required_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "secret-value"}, clear=True
        ):
            root = Path(tmpdir)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            required = root / "script.py"
            required.write_text("print('ok')\n")
            args = argparse.Namespace(
                base_adapter=adapter,
                output_root=root / "outputs",
                api_key_env="DEEPSEEK_API_KEY",
                require_api_key=True,
                min_free_gb=0.0,
                required_path=[Path("script.py")],
                default_required_paths=False,
            )

            report = preflight.build_report(args, root=root)

        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])
        self.assertTrue(report["api_key_set"])
        self.assertNotIn("secret-value", str(report))

    def test_can_disable_api_key_requirement_for_dry_infra_checks(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            root = Path(tmpdir)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            args = argparse.Namespace(
                base_adapter=adapter,
                output_root=root,
                api_key_env="DEEPSEEK_API_KEY",
                require_api_key=False,
                min_free_gb=0.0,
                required_path=[],
                default_required_paths=False,
            )

            report = preflight.build_report(args, root=root)

        self.assertTrue(report["api_key_set"] is False)
        self.assertNotIn("missing API key", " ".join(report["failures"]))


if __name__ == "__main__":
    unittest.main()
