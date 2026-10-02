"""HTTP-level tests for the provider key store and the ChatGPT bridge endpoints."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from ai_orchestrate import relay as relay_module
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

    def test_desktop_task_handoff_opens_a_fresh_codex_chat_with_prefilled_prompt(self):
        payload = self._post("/api/chatgpt/new-task", {
            "repo": str(self.repo), "task": "Добавь настройку темы", "checks": "python -m unittest discover -s tests -t .",
            "github_item": "https://github.com/example/project/issues/7",
        })
        link = urlsplit(payload["url"])
        query = parse_qs(link.query)
        self.assertEqual((link.scheme, link.netloc), ("codex", "new"))
        self.assertEqual(query["path"], [str(self.repo.resolve())])
        self.assertTrue(payload["prompt_in_url"])
        self.assertIn("Добавь настройку темы", query["prompt"][0])
        self.assertIn("python -m unittest discover -s tests -t .", query["prompt"][0])
        self.assertIn("https://github.com/example/project/issues/7", query["prompt"][0])
        self.assertEqual(set(query), {"path", "prompt"})

    def test_desktop_handoff_keeps_long_prompt_out_of_deep_link_without_truncation(self):
        task = "A" * 48_000
        payload = self._post("/api/chatgpt/new-task", {"repo": str(self.repo), "task": task, "checks": ""})
        query = parse_qs(urlsplit(payload["url"]).query)
        self.assertFalse(payload["prompt_in_url"])
        self.assertNotIn("prompt", query)
        self.assertEqual(query["path"], [str(self.repo.resolve())])
        self.assertIn(task, payload["prompt"])

    def test_desktop_handoff_rejects_paths_outside_allowed_workspace(self):
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaises(HTTPError) as caught:
                self._post("/api/chatgpt/new-task", {"repo": outside, "task": "Read files outside workspace"})
        self.assertEqual(caught.exception.code, 400)

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


class ManualRelayApiTests(unittest.TestCase):
    """Панель отдаёт промпт в обычный ChatGPT и принимает вставленный ответ по HTTP."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.root = root
        self.repo = _git_repo(root / "project")
        patcher = patch.dict(os.environ, {"AI_ORCHESTRATE_HOME": str(root)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager = RunManager(root, usage_path=root / "usage.jsonl", settings_path=root / "settings.json",
                                  journal_path=root / "journal.jsonl")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.manager))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def _post(self, path, payload, expect_error=False):
        request = Request(self.base + path, data=json.dumps(payload).encode("utf-8"), method="POST",
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if not expect_error:
                raise
            return {"status": error.code, "error": json.loads(error.read())["error"]}

    def _get(self, path):
        with urlopen(self.base + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _waiting_job(self):
        settings = default_settings()
        settings.update({"mode": "quick", "executor": "chatgpt", "default_checks": CHECK})
        submission = RunSubmission(self.repo, "Добавь функцию", [CHECK], settings)
        job = RunJob(id="relay00001", submission=submission, status="awaiting_answer")
        job.relay = relay_module.ManualRelay(
            on_event=lambda event: self.manager._workflow_event(job, event), timeout=30)
        self.manager._jobs[job.id] = job
        self.manager._active_job = job.id
        return job

    def test_answer_endpoint_resumes_a_waiting_manual_step(self):
        job = self._waiting_job()
        received = {}

        def worker():
            received["answer"] = job.relay.request(
                kind="code", stage="implementation", role="Разработчик",
                title="Правки кода для обычного ChatGPT", instructions="Верни файлы.",
                prompt="Ты — инженер. Верни файлы целиком.")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not job.relay.waiting() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(job.relay.waiting())

        payload = self._get(f"/api/runs/{job.id}")
        self.assertEqual(payload["status"], "awaiting_answer")
        self.assertTrue(payload["can_answer"])
        self.assertEqual(payload["manual"]["request"]["kind"], "code")
        self.assertIn("Верни файлы целиком", payload["manual"]["request"]["prompt"])

        answer = "### FILE: app.py\n```\nprint('ok')\n```\n"
        response = self._post(f"/api/runs/{job.id}/answer", {"answer": answer})
        self.assertTrue(response["accepted"])
        thread.join(timeout=5)
        self.assertEqual(received["answer"], answer)
        self.assertEqual(job.status, "running")
        self.assertFalse(self._get(f"/api/runs/{job.id}")["can_answer"])

    def test_answer_endpoint_rejects_when_nothing_is_expected(self):
        job = self._waiting_job()
        refused = self._post(f"/api/runs/{job.id}/answer", {"answer": "text"}, expect_error=True)
        self.assertEqual(refused["status"], 409)
        self.assertIn("не ожидается", refused["error"])
        empty = self._post(f"/api/runs/{job.id}/answer", {"answer": "   "}, expect_error=True)
        self.assertEqual(empty["status"], 409)
        missing = self._post("/api/runs/does-not-exist/answer", {"answer": "text"}, expect_error=True)
        self.assertEqual(missing["status"], 409)

    def test_cancel_releases_a_waiting_manual_step(self):
        job = self._waiting_job()
        result = {}

        def worker():
            try:
                job.relay.request(kind="plan", stage="planning", role="Аналитик", title="План",
                                  instructions="Верни план.", prompt="План, пожалуйста.")
            except Exception as exc:  # RelayStopped наследуется от OrchestratorError
                result["error"] = str(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not job.relay.waiting() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.manager.cancel(job.id))
        thread.join(timeout=5)
        self.assertIn("остановлено", result["error"])
        self.assertTrue(job.cancel.is_set())


if __name__ == "__main__":
    unittest.main()
