import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrate import secrets
from ai_orchestrate.core import OrchestratorError
from ai_orchestrate.settings import SettingsStore, default_settings

KEY = "tsk_live_0123456789ABCDEF"


class SecretsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.key_file = Path(self._tmp.name) / "jev-key"
        patcher = patch.dict(os.environ, {
            "AI_ORCHESTRATE_JEV_KEY_FILE": str(self.key_file),
            "AI_ORCHESTRATE_HOME": self._tmp.name,
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("TYPESAFE_API_KEY", None)
        secrets.reset_activation_state()

    def test_key_is_stored_outside_settings_json_with_private_permissions(self):
        status = secrets.save_jev_key(f"  {KEY}\n")
        self.assertTrue(status["available"])
        self.assertEqual(status["source"], "file")
        self.assertNotIn(KEY, status["masked"])
        self.assertEqual(self.key_file.read_text(encoding="utf-8").strip(), KEY)
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(self.key_file.stat().st_mode), 0o600)
        settings_path = Path(self._tmp.name) / "settings.json"
        store = SettingsStore(settings_path)
        store.save(default_settings())
        self.assertNotIn(KEY, settings_path.read_text(encoding="utf-8"))
        self.assertNotIn("jev", " ".join(default_settings()).lower())

    def test_environment_key_wins_over_stored_file_and_is_reported_as_such(self):
        secrets.save_jev_key(KEY)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "tsk_env_99999999"}):
            status = secrets.jev_key_status()
            self.assertEqual(status["source"], "environment")
            self.assertTrue(status["environment_overrides_file"])
            self.assertEqual(secrets.active_jev_key(), "tsk_env_99999999")

    def test_activation_at_startup_and_clear(self):
        self.assertEqual(secrets.activate_stored_jev_key(), "")
        secrets.save_jev_key(KEY)
        os.environ.pop("TYPESAFE_API_KEY", None)
        self.assertEqual(secrets.activate_stored_jev_key(), "file")
        self.assertEqual(os.environ["TYPESAFE_API_KEY"], KEY)
        status = secrets.clear_jev_key()
        self.assertFalse(status["available"])
        self.assertFalse(self.key_file.exists())
        self.assertNotIn("TYPESAFE_API_KEY", os.environ)

    def test_clear_keeps_an_exported_environment_key_and_says_so(self):
        secrets.save_jev_key(KEY)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": KEY}):
            secrets.reset_activation_state()
            status = secrets.clear_jev_key()
            self.assertTrue(status["available"])
            self.assertIn("окружении", status["note"])

    def test_invalid_keys_are_rejected_before_anything_is_written(self):
        for bad in ("", "   ", "short", "has space", "line\nbreak", "ключ-в-unicode", "{" + "a" * 20):
            with self.assertRaises(OrchestratorError):
                secrets.save_jev_key(bad)
        self.assertFalse(self.key_file.exists())


if __name__ == "__main__":
    unittest.main()
