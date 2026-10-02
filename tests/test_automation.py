import tempfile
import time
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrate.core import CodexUsage, Decision, OrchestratorError
from ai_orchestrate.settings import default_settings
from ai_orchestrate.gitops import merge_local as perform_local_merge
from ai_orchestrate.usage import append_usage
from ai_orchestrate.web import RunManager


class AutomationTests(unittest.TestCase):
    def _repo(self, root: Path) -> Path:
        root.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Automation Test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "automation@example.invalid"], cwd=root, check=True)
        (root / "README.md").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True)
        return root

    def _settings(self, root: Path, repo: Path, **overrides):
        settings = default_settings()
        settings.update({
            "default_repo": str(repo),
            "default_checks": "python --version",
            "mode": "quick",
            "max_repairs": 0,
            "max_model_calls": 1,
            "worktree_root": str(root / "worktrees"),
            "merge_target": "local",
            "merge_policy": "confirm",
        })
        settings.update(overrides)
        return settings

    @staticmethod
    def _wait(manager: RunManager, job_id: str, statuses: set[str], timeout: float = 8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = manager.fetch(job_id)
            if result and result["status"] in statuses:
                return result
            time.sleep(0.02)
        raise AssertionError(f"Job did not reach {statuses}; latest={manager.fetch(job_id)}")

    def test_success_waits_for_confirm_before_fast_forward_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            manager = RunManager(root, usage_path=root / "usage.jsonl", settings_path=root / "settings.json",
                                 journal_path=root / "journal.jsonl")
            usage_file = root / "logs" / "tokens.jsonl"
            journal_file = root / "logs" / "runs.jsonl"
            settings = self._settings(root, repo, project_contexts={str(repo): "Use the existing project conventions."},
                                      usage_log_path=str(usage_file), journal_path=str(journal_file))

            def fake_workflow(request, *, emit, cancel_event, usage_path):
                self.assertEqual(usage_path, usage_file.resolve())
                self.assertNotEqual(request.repo.resolve(), repo.resolve())
                self.assertIn("Use the existing project conventions.", request.task)
                (request.repo / "feature.txt").write_text("implemented in isolated worktree\n", encoding="utf-8")
                return {"status": "complete", "checks": [{"command": "python --version", "returncode": 0}],
                        "review": "", "model_calls": 1, "run_tokens": 10, "plan": ""}

            with patch("ai_orchestrate.web.run_workflow", side_effect=fake_workflow):
                job = manager.start({"repo": str(repo), "task": "Add a feature", "checks": "python --version",
                                     "settings": settings})
                pending = self._wait(manager, job.id, {"awaiting_confirmation"})
                self.assertTrue(pending["can_confirm"])
                self.assertTrue(pending["can_discard"])
                self.assertFalse((repo / "feature.txt").exists(), "The base checkout must remain untouched before approval.")
                self.assertEqual(subprocess.run(["git", "-C", str(repo), "branch", "--show-current"],
                                                capture_output=True, text=True, check=True).stdout.strip(), "main")
                self.assertTrue(manager.confirm(job.id))
                complete = self._wait(manager, job.id, {"complete"})

            self.assertTrue((repo / "feature.txt").exists())
            self.assertEqual(complete["result"]["merge_status"], "merged")
            history = manager.history()
            self.assertTrue(history)
            self.assertEqual(history[0]["status"], "complete")
            self.assertTrue(journal_file.is_file())
            self.assertFalse((root / "journal.jsonl").exists())

    def test_final_jev_approval_triggers_automatic_merge_only_after_full_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            manager = RunManager(root, usage_path=root / "usage.jsonl", settings_path=root / "settings.json",
                                 journal_path=root / "journal.jsonl")
            settings = self._settings(root, repo, mode="full", max_repairs=0, max_model_calls=3,
                                      merge_policy="jev_auto")

            def fake_workflow(request, *, emit, cancel_event, usage_path):
                (request.repo / "safe.txt").write_text("checked\n", encoding="utf-8")
                return {"status": "complete", "checks": [{"command": "python --version", "returncode": 0}],
                        "review": "PASS\nNo blocking findings.", "model_calls": 3, "run_tokens": 30,
                        "plan": "requirements", "changed_files": ["safe.txt"]}

            def jev_response(state, question, choices, **kwargs):
                self.assertEqual(state["phase"], "final_merge_gate")
                self.assertIn("+checked", state["diff"])
                self.assertEqual(set(choices), {"APPROVE", "HOLD", "REJECT"})
                return Decision("APPROVE", 0.98, {"APPROVE": 0.98, "HOLD": 0.01, "REJECT": 0.01})

            with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key-for-automation"}), \
                 patch("ai_orchestrate.web.run_workflow", side_effect=fake_workflow), \
                 patch("ai_orchestrate.web.jev_choice", side_effect=jev_response):
                job = manager.start({"repo": str(repo), "task": "Implement and verify", "checks": "python --version",
                                     "settings": settings})
                complete = self._wait(manager, job.id, {"complete"})
            self.assertTrue((repo / "safe.txt").exists())
            self.assertEqual(complete["result"]["jev_decision"], "APPROVE")
            self.assertEqual(complete["result"]["merge"]["initiated_by"], "jev")

    def test_jev_approved_merge_failure_remains_confirmable_in_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            manager = RunManager(root, usage_path=root / "usage.jsonl", settings_path=root / "settings.json")
            settings = self._settings(root, repo, mode="full", max_repairs=0, max_model_calls=3,
                                      merge_policy="jev_auto")

            def fake_workflow(request, *, emit, cancel_event, usage_path):
                (request.repo / "safe.txt").write_text("checked\n", encoding="utf-8")
                return {"status": "complete", "checks": [{"command": "python --version", "returncode": 0}],
                        "review": "PASS\nNo blocking findings.", "model_calls": 3, "run_tokens": 30,
                        "plan": "requirements", "changed_files": ["safe.txt"]}

            merge_attempts = 0

            def flaky_merge(worktree):
                nonlocal merge_attempts
                merge_attempts += 1
                if merge_attempts == 1:
                    raise OrchestratorError("simulated temporary merge failure")
                return perform_local_merge(worktree)

            decision = Decision("APPROVE", 0.98, {"APPROVE": 0.98, "HOLD": 0.01, "REJECT": 0.01})
            with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key-for-automation"}), \
                 patch("ai_orchestrate.web.run_workflow", side_effect=fake_workflow), \
                 patch("ai_orchestrate.web.jev_choice", return_value=decision), \
                 patch("ai_orchestrate.web.merge_local", side_effect=flaky_merge):
                job = manager.start({"repo": str(repo), "task": "Implement and verify", "checks": "python --version",
                                     "settings": settings})
                waiting = self._wait(manager, job.id, {"awaiting_confirmation"})
                self.assertTrue(waiting["can_confirm"])
                self.assertEqual(waiting["result"]["jev_decision"], "APPROVE")
                self.assertIn("simulated temporary merge failure", waiting["error"])
                self.assertTrue(manager.confirm(job.id))
                complete = self._wait(manager, job.id, {"complete"})

            self.assertTrue((repo / "safe.txt").exists())
            self.assertEqual(complete["result"]["merge"]["initiated_by"], "user")
            self.assertEqual(merge_attempts, 2)

    def test_daily_token_budget_fails_before_creating_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            usage_file = root / "usage.jsonl"
            append_usage(usage_file, model="m", effort="low", role="coder", attempt=1, returncode=0,
                         usage=CodexUsage(60, 0, 40))
            manager = RunManager(root, usage_path=usage_file, settings_path=root / "settings.json")
            settings = self._settings(root, repo, daily_token_budget=100)
            with self.assertRaisesRegex(OrchestratorError, "Дневной лимит уже израсходован"):
                manager.start({"repo": str(repo), "task": "task", "settings": settings})
            self.assertIsNone(manager._active_job)
            self.assertFalse((root / "worktrees").exists())

    def test_missing_jev_router_key_fails_before_creating_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            manager = RunManager(root, settings_path=root / "settings.json")
            settings = self._settings(root, repo, router="jev")
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(OrchestratorError, "Триаж Jev требует TYPESAFE_API_KEY"):
                    manager.start({"repo": str(repo), "task": "task", "settings": settings})
            self.assertIsNone(manager._active_job)
            self.assertFalse((root / "worktrees").exists())

    def test_missing_jev_key_fails_closed_before_creating_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            manager = RunManager(root, usage_path=root / "usage.jsonl", settings_path=root / "settings.json")
            settings = self._settings(root, repo, mode="full", merge_policy="jev_auto")
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(OrchestratorError, "требует TYPESAFE_API_KEY"):
                    manager.start({"repo": str(repo), "task": "task", "settings": settings})

    def test_failed_checks_never_offer_merge_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            manager = RunManager(root, usage_path=root / "usage.jsonl", settings_path=root / "settings.json")
            settings = self._settings(root, repo)

            def fake_workflow(request, *, emit, cancel_event, usage_path):
                (request.repo / "unsafe.txt").write_text("not merged\n", encoding="utf-8")
                return {"status": "complete", "checks": [{"command": "python --version", "returncode": 1}],
                        "review": "", "model_calls": 1, "run_tokens": 10}

            with patch("ai_orchestrate.web.run_workflow", side_effect=fake_workflow):
                job = manager.start({"repo": str(repo), "task": "Do not merge this", "checks": "python --version",
                                     "settings": settings})
                result = self._wait(manager, job.id, {"incomplete"})
            self.assertFalse(result["can_confirm"])
            self.assertFalse((repo / "unsafe.txt").exists())
            self.assertEqual(result["result"]["merge_status"], "blocked")


if __name__ == "__main__":
    unittest.main()
