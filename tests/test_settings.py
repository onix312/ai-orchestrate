import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_orchestrate.core import OrchestratorError
from ai_orchestrate.settings import SettingsStore, default_settings, normalize_settings


class SettingsTests(unittest.TestCase):
    def test_settings_persist_across_store_instances_and_do_not_accept_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            first = SettingsStore(path)
            saved = first.save({"mode": "quick", "merge_target": "github", "default_checks": "pytest -q\n",
                                "usage_log_path": str(Path(tmp) / "telemetry" / "tokens.jsonl"),
                                "journal_path": str(Path(tmp) / "history" / "runs.jsonl")})
            self.assertEqual(saved["mode"], "quick")
            self.assertEqual(saved["usage_log_path"], str(Path(tmp) / "telemetry" / "tokens.jsonl"))
            self.assertEqual(saved["journal_path"], str(Path(tmp) / "history" / "runs.jsonl"))
            self.assertEqual(saved["default_checks"], "pytest -q")
            second = SettingsStore(path)
            self.assertEqual(second.load()["merge_target"], "github")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["mode"], "quick")
            with self.assertRaisesRegex(OrchestratorError, "Неизвестные настройки"):
                second.save({"jev_api_key": "must-not-be-stored"})
            self.assertNotIn("must-not-be-stored", path.read_text(encoding="utf-8"))

    def test_custom_settings_path_does_not_change_parent_directory_permissions(self):
        if os.name == "nt":
            self.skipTest("POSIX directory mode bits are not portable to Windows")
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared"
            shared.mkdir(mode=0o755)
            shared.chmod(0o755)
            SettingsStore(shared / "settings.json").save({"mode": "quick"})
            self.assertEqual(shared.stat().st_mode & 0o777, 0o755)
            self.assertEqual((shared / "settings.json").stat().st_mode & 0o777, 0o600)

    def test_settings_validate_merge_policy_and_model_names(self):
        with self.assertRaisesRegex(OrchestratorError, "Полный цикл"):
            normalize_settings({"mode": "quick", "merge_policy": "jev_auto"})
        with self.assertRaisesRegex(OrchestratorError, "Модель"):
            normalize_settings({"luna_model": "--dangerous"})
        with self.assertRaisesRegex(OrchestratorError, "Префикс"):
            normalize_settings({"branch_prefix": "../outside"})
        for key in ("mode", "router", "merge_policy", "merge_target", "merge_method", "lane"):
            with self.subTest(key=key), self.assertRaises(OrchestratorError):
                normalize_settings({key: []})

    def test_defaults_cover_reusable_run_and_merge_configuration(self):
        settings = default_settings()
        self.assertEqual(settings["merge_policy"], "confirm")
        self.assertEqual(settings["merge_target"], "local")
        self.assertTrue(settings["save_journal"])
        self.assertEqual(settings["mode"], "full")

    def test_chatgpt_executor_and_limit_fallback_are_validated(self):
        settings = normalize_settings({"executor": "chatgpt", "limit_fallback": "chatgpt"})
        self.assertEqual(settings["executor"], "chatgpt")
        self.assertEqual(settings["limit_fallback"], "chatgpt")
        self.assertGreaterEqual(settings["relay_timeout"], 120)
        self.assertEqual(normalize_settings({"limit_fallback": "api"})["limit_fallback"], "api")
        self.assertEqual(normalize_settings({"limit_fallback": "off"})["limit_fallback"], "off")
        for bad in ({"executor": "manual"}, {"limit_fallback": "auto"}, {"relay_timeout": 5},
                    {"relay_timeout": "3600"}):
            with self.subTest(bad=bad), self.assertRaises(OrchestratorError):
                normalize_settings(bad)


if __name__ == "__main__":
    unittest.main()
