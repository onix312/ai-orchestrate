import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from ai_orchestrate import autofill
from ai_orchestrate.settings import default_settings
from ai_orchestrate.web import RunManager, make_handler


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "-m", "initial"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "branch", "-m", "main"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    (path / "tests").mkdir(exist_ok=True)
    return path


class AutofillTests(unittest.TestCase):
    def test_autofill_detects_repo_checks_branch_and_downgrades_jev_without_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _git_repo(Path(tmp) / "project")
            settings = default_settings()
            settings.update({"router": "jev", "merge_policy": "jev_auto", "mode": "full", "max_model_calls": 2})
            os.environ.pop("TYPESAFE_API_KEY", None)
            env = {"AI_ORCHESTRATE_JEV_KEY_FILE": str(Path(tmp) / "no-key"), "AI_ORCHESTRATE_HOME": tmp}
            from unittest.mock import patch

            with patch.dict(os.environ, env):
                updates, report = autofill.autofill(repo, settings)
            self.assertEqual(updates["default_repo"], str(repo.resolve()))
            self.assertTrue(updates["default_checks"].strip())
            self.assertEqual(updates["base_branch"], "main")
            self.assertEqual(updates["router"], "local")
            self.assertEqual(updates["merge_policy"], "confirm")
            self.assertEqual(updates["max_model_calls"], 3)
            self.assertTrue(any(item["key"] == "router" for item in report["applied"]))

    def test_autofill_keeps_manual_checks_and_respects_a_higher_call_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _git_repo(Path(tmp) / "project")
            settings = default_settings()
            settings.update({"default_checks": "python -m compileall -q .", "max_model_calls": 7})
            updates, _ = autofill.autofill(repo, settings)
            self.assertNotIn("default_checks", updates)
            self.assertNotIn("max_model_calls", updates)


class SetupApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.root = root
        self.repo = _git_repo(root / "project")
        env = {
            "CODEX_HOME": str(root / "codex-home"),
            "AI_ORCHESTRATE_JEV_KEY_FILE": str(root / "jev-key"),
            "AI_ORCHESTRATE_HOME": str(root),
        }
        from unittest.mock import patch

        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("TYPESAFE_API_KEY", None)
        self.manager = RunManager(root, usage_path=root / "usage.jsonl", settings_path=root / "settings.json",
                                  journal_path=root / "journal.jsonl")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.manager))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def _get(self, path):
        with urlopen(self.base + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path, payload):
        request = Request(self.base + path, data=json.dumps(payload).encode("utf-8"), method="POST",
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_environment_projects_and_defaults_endpoints(self):
        environment = self._get("/api/environment")
        self.assertIn("codex", environment["tools"])
        self.assertIsInstance(environment["problems"], list)
        self.assertIn("jev", environment)
        defaults = self._get("/api/defaults")
        self.assertEqual(defaults["settings"]["mode"], "full")
        catalog = self._get("/api/projects")
        self.assertIn(str(self.repo.resolve()), [item["path"] for item in catalog["projects"]])

    def test_jev_key_round_trip_never_returns_the_key(self):
        key = "tsk_live_web_api_round_trip_123"
        before = self._get("/api/jev-key")
        self.assertFalse(before["available"])
        saved = self._post("/api/jev-key", {"key": key})
        self.assertTrue(saved["available"])
        self.assertEqual(saved["source"], "file")
        self.assertNotIn(key, json.dumps(saved))
        settings_file = self.root / "settings.json"
        self._post("/api/settings", {"settings": default_settings()})
        self.assertNotIn(key, settings_file.read_text(encoding="utf-8"))
        cleared = self._post("/api/jev-key", {"clear": True})
        self.assertFalse(cleared["available"])

    def test_autofill_endpoint_saves_generated_settings(self):
        result = self._post("/api/autofill", {"repo": str(self.repo)})
        self.assertEqual(result["settings"]["default_repo"], str(self.repo.resolve()))
        self.assertEqual(result["settings"]["base_branch"], "main")
        self.assertTrue(result["report"]["applied"])
        persisted = self._get("/api/settings")
        self.assertEqual(persisted["settings"]["base_branch"], "main")

    def test_setup_job_streams_installer_output(self):
        from unittest.mock import patch

        # The report is stubbed so this exercises the job machinery on any machine,
        # whether or not Codex CLI happens to be installed here.
        fake_report = {"tools": {"codex": {"found": False, "title": "Codex CLI", "path": "",
                                           "install": {"command": "npm install -g @openai/codex"}}}}

        def fake_install(tool, *, on_output=None, timeout=900):
            for line in ("$ npm install -g @openai/codex", "added 1 package"):
                if on_output:
                    on_output(line)
            return {"tool": tool, "command": "npm install -g @openai/codex", "ok": True,
                    "returncode": 0, "output": ["$ npm install -g @openai/codex", "added 1 package"]}

        with patch("ai_orchestrate.env_setup.environment_report", return_value=fake_report), \
             patch("ai_orchestrate.env_setup.install_tool", side_effect=fake_install):
            job = self._post("/api/setup", {"tool": "codex"})
            self.assertEqual(job["status"], "running")
            deadline = time.monotonic() + 10
            current = job
            while current["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.05)
                current = self._get(f"/api/setup/{job['id']}")
        self.assertEqual(current["status"], "complete")
        self.assertIn("added 1 package", "\n".join(current["output"]))


if __name__ == "__main__":
    unittest.main()
