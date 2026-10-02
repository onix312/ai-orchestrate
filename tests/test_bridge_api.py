"""HTTP-level tests for the provider key store and the ChatGPT bridge endpoints."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from ai_orchestrate.gitops import create_worktree
from ai_orchestrate.settings import default_settings
from ai_orchestrate.web import RunJob, RunManager, RunSubmission, make_handler

CHECK = "python -m unittest discover -s tests"


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    # A real machine has a configured identity; worktree commits need one too.
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "-m", "initial"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "branch", "-m", "main"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    return path


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.root = root
        self.repo = _git_repo(root / "project")
        env = {"AI_ORCHESTRATE_HOME": str(root),
               "AI_ORCHESTRATE_JEV_KEY_FILE": str(root / "jev-key"),
               "AI_ORCHESTRATE_OPENAI_KEY_FILE": str(root / "openai-key"),
               "AI_ORCHESTRATE_OPENROUTER_KEY_FILE": str(root / "openrouter-key")}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            os.environ.pop(name, None)
        from ai_orchestrate import secrets
        secrets.reset_activation_state()
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

    def test_provider_keys_are_stored_masked_and_never_in_settings(self):
        key = "sk-openai-web-round-trip-99"
        before = self._get("/api/keys")
        self.assertFalse(before["keys"]["openai"]["available"])

        saved = self._post("/api/keys/openai", {"key": key})
        self.assertTrue(saved["changed"]["available"])
        self.assertEqual(saved["changed"]["source"], "file")
        self.assertNotIn(key, json.dumps(saved))

        stored = self.root / "openai-key"
        self.assertEqual(stored.read_text(encoding="utf-8").strip(), key)
        if os.name != "nt":
            self.assertEqual(stored.stat().st_mode & 0o777, 0o600)

        self._post("/api/settings", {"settings": default_settings()})
        self.assertNotIn(key, (self.root / "settings.json").read_text(encoding="utf-8"))

        cleared = self._post("/api/keys/openai", {"clear": True})
        self.assertFalse(cleared["changed"]["available"])
        self.assertFalse(stored.exists())

    def test_bridge_budget_stop_is_a_json_error_not_a_disconnected_request(self):
        from ai_orchestrate.workflow import WorkflowStopped
        with patch.object(self.manager, "bridge_apply", side_effect=WorkflowStopped("budget exhausted")):
            with self.assertRaises(HTTPError) as caught:
                self._post("/api/bridge/apply", {"run": "id", "answer": "text"})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(json.loads(caught.exception.read())["error"], "budget exhausted")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(Exception):
            self._post("/api/keys/chatgpt-app", {"key": "whatever-1234"})

    def test_bridge_endpoints_refuse_a_run_without_a_worktree(self):
        with self.assertRaises(Exception):
            self._post("/api/bridge/prompt", {"run": "does-not-exist"})


class BridgeRoundTripTests(unittest.TestCase):
    """The human pastes a ChatGPT answer; the panel applies it and re-checks."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.root = root
        self.repo = _git_repo(root / "project")
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "test_app.py").write_text(
            "import unittest\nfrom app import total\n\n\n"
            "class TotalTests(unittest.TestCase):\n"
            "    def test_total_is_rounded(self):\n"
            '        self.assertEqual(total([1.005, 2.004]), 3.01)\n',
            encoding="utf-8")
        (self.repo / "app.py").write_text("def total(items):\n    return sum(items)\n", encoding="utf-8")
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "add", "-A"], cwd=self.repo, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "commit", "-q", "-m", "app"], cwd=self.repo, check=True, stdout=subprocess.DEVNULL)
        env = {"AI_ORCHESTRATE_HOME": str(root)}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager = RunManager(root, usage_path=root / "usage.jsonl", settings_path=root / "settings.json",
                                  journal_path=root / "journal.jsonl")

    def _job_with_worktree(self) -> RunJob:
        settings = default_settings()
        settings.update({"mode": "quick", "default_checks": CHECK, "merge_target": "local", "merge_policy": "confirm"})
        submission = RunSubmission(self.repo, "Округлять сумму до двух знаков", [CHECK], settings)
        job = RunJob(id="bridge0001", submission=submission)
        worktree = create_worktree(self.repo, self.root / "worktrees", job_id=job.id,
                                   branch_prefix=settings["branch_prefix"], base_ref="main",
                                   base_branch="main")
        job.worktree = worktree
        job.status = "incomplete"
        self.manager._jobs[job.id] = job
        self.manager._bridge_checks(job)
        return job

    def test_prompt_contains_the_failing_test_and_apply_makes_checks_pass(self):
        job = self._job_with_worktree()

        prompt = self.manager.bridge_prompt({"run": job.id})
        self.assertIn("Округлять сумму", prompt["prompt"])
        self.assertEqual(len(prompt["failed_checks"]), 1)
        self.assertIn(CHECK, prompt["failed_checks"][0]["command"])

        answer = ("### FILE: app.py\n```python\n"
                  "def total(items):\n    return round(sum(items), 2)\n```\n")
        summary = self.manager.bridge_apply({"run": job.id, "answer": answer})

        self.assertTrue(summary["checks_passed"], summary["checks"])
        self.assertEqual([item["path"] for item in summary["report"]["applied"]], ["app.py"])
        self.assertEqual(self.manager._jobs[job.id].status, "awaiting_confirmation")
        self.assertTrue(summary and self.manager.fetch(job.id)["can_confirm"])
        committed = subprocess.run(["git", "show", "--stat", "--oneline", "HEAD"], cwd=job.worktree.path,
                                   capture_output=True, text=True, check=True).stdout
        self.assertIn("app.py", committed)

    def test_wrong_answer_keeps_the_run_unmergeable(self):
        job = self._job_with_worktree()
        summary = self.manager.bridge_apply({
            "run": job.id,
            "answer": "### FILE: app.py\n```python\ndef total(items):\n    return sum(items)\n```\n",
        })
        self.assertFalse(summary["checks_passed"])
        self.assertEqual(self.manager._jobs[job.id].status, "incomplete")
        self.assertFalse(self.manager.fetch(job.id)["can_confirm"])

    def test_bridge_refuses_to_touch_files_outside_the_worktree(self):
        job = self._job_with_worktree()
        summary = self.manager.bridge_apply({
            "run": job.id, "answer": "### FILE: ../../escaped.txt\n```\npwned\n```\n"})
        self.assertEqual(summary["report"]["applied"], [])
        self.assertFalse((self.root.parent / "escaped.txt").exists())
        self.assertEqual(self.manager._jobs[job.id].status, "incomplete")


if __name__ == "__main__":
    unittest.main()
