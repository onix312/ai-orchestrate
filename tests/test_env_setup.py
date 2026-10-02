import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrate import env_setup
from ai_orchestrate.core import OrchestratorError, doctor

FAKE_CODEX = """#!/bin/sh
case "$1" in
  --version) echo "codex-cli 9.9.9" ;;
  login) exit 0 ;;
esac
exit 0
"""


def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


class EnvSetupTests(unittest.TestCase):
    def setUp(self):
        env_setup.clear_auth_cache()
        env_setup._npm_prefix = None

    def tearDown(self):
        env_setup.clear_auth_cache()
        env_setup._npm_prefix = None

    def test_refresh_path_finds_tools_installed_outside_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex = _write_executable(Path(tmp) / ".local" / "bin" / "codex", FAKE_CODEX)
            with patch.dict(os.environ, {"HOME": tmp, "PATH": "/usr/bin"}), \
                 patch("ai_orchestrate.env_setup.npm_global_prefix", return_value=""):
                self.assertNotIn(str(codex.parent), os.environ["PATH"].split(os.pathsep))
                # find_tool also scans well-known install directories, so it sees the binary already.
                self.assertEqual(env_setup.find_tool("codex"), str(codex))
                added = env_setup.refresh_path()
                self.assertIn(str(codex.parent), added)
                self.assertIn(str(codex.parent), os.environ["PATH"].split(os.pathsep))
                self.assertEqual(shutil.which("codex"), str(codex))
                self.assertIn("9.9.9", env_setup.tool_version(str(codex)))

    def test_report_marks_missing_codex_as_blocker_and_logged_in_codex_as_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HOME": tmp, "PATH": "/usr/bin"}), \
                 patch("ai_orchestrate.env_setup.npm_global_prefix", return_value=""), \
                 patch("ai_orchestrate.env_setup.candidate_bin_dirs", return_value=[]), \
                 patch.dict(os.environ, {"AI_ORCHESTRATE_JEV_KEY_FILE": str(Path(tmp) / "no-key")}):
                blocked = env_setup.environment_report()
                self.assertFalse(blocked["ready"])
                self.assertIn("codex.missing", [item["id"] for item in blocked["problems"]])
                self.assertEqual(blocked["tools"]["codex"]["state"], "missing")
                # Ручной исполнитель «ChatGPT (обычный чат)» работает без Codex CLI.
                manual = env_setup.environment_report(executor="chatgpt")
                self.assertNotIn("codex.missing", [item["id"] for item in manual["problems"]
                                                   if item["severity"] == "blocker"])
                self.assertEqual(manual["executor"], "chatgpt")
                self.assertFalse(manual["tools"]["codex"]["required"])
                api = env_setup.environment_report(executor="api")
                self.assertEqual(api["executor"], "api")

                codex = _write_executable(Path(tmp) / ".local" / "bin" / "codex", FAKE_CODEX)
                env_setup.clear_auth_cache()
                with patch("ai_orchestrate.env_setup.candidate_bin_dirs", return_value=[codex.parent]):
                    ready = env_setup.environment_report()
                self.assertEqual(ready["tools"]["codex"]["state"], "ok")
                self.assertTrue(ready["tools"]["codex"]["login"]["authenticated"])
                self.assertNotIn("codex.missing", [item["id"] for item in ready["problems"]])

    def test_install_tool_streams_output_and_reports_failure_for_unknown_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex = _write_executable(Path(tmp) / "bin" / "codex", FAKE_CODEX)
            lines: list[str] = []
            with patch("ai_orchestrate.env_setup._install_candidates",
                       return_value=[{"command": ["sh", "-c", "echo installing-codex"], "available": True, "reason": ""}]), \
                 patch("ai_orchestrate.env_setup.find_tool",
                       side_effect=lambda name: {"codex": str(codex), "sh": "/bin/sh"}.get(name, "")), \
                 patch("ai_orchestrate.env_setup.npm_global_prefix", return_value=""):
                result = env_setup.install_tool("codex", on_output=lines.append)
            self.assertTrue(result["ok"])
            self.assertEqual(result["returncode"], 0)
            self.assertIn("installing-codex", "\n".join(result["output"]))
            self.assertTrue(any("installing-codex" in line for line in lines))
            with self.assertRaises(OrchestratorError):
                env_setup.install_tool("docker")

    def test_doctor_rows_carry_a_hint_for_every_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HOME": tmp, "PATH": "/usr/bin",
                                         "AI_ORCHESTRATE_JEV_KEY_FILE": str(Path(tmp) / "no-key")}), \
                 patch("ai_orchestrate.env_setup.npm_global_prefix", return_value=""), \
                 patch("ai_orchestrate.env_setup.candidate_bin_dirs", return_value=[]):
                rows = doctor()
        self.assertTrue(all(len(row) == 4 for row in rows))
        codex_row = next(row for row in rows if row[0] == "Codex CLI")
        self.assertFalse(codex_row[1])
        self.assertIn("codex", codex_row[3])


if __name__ == "__main__":
    unittest.main()
