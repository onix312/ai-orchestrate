import subprocess
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from ai_orchestrate.core import LANES, CodexResult, CodexUsage, OrchestratorError, is_usage_limit_error
from ai_orchestrate.prompts import ROLE_PROMPTS, build_developer_prompt, build_planning_prompt
from ai_orchestrate.web import RunManager
from ai_orchestrate.workflow import WorkflowRequest, run_workflow, suggest_checks

LIMIT_MESSAGE = ("You've hit your usage limit. Upgrade to Pro "
                 "(https://chatgpt.com/explore/pro) or try again at Oct 3rd, 2026 12:09 AM.")


class FakeRelay:
    """Человек, который всегда отвечает заранее заготовленным текстом."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def request(self, *, kind, stage, role, title, instructions, prompt):
        self.requests.append({"kind": kind, "stage": stage, "role": role, "title": title,
                              "instructions": instructions, "prompt": prompt})
        if not self.answers:
            raise AssertionError("мост запросил больше ответов, чем подготовлено в тесте")
        return self.answers.pop(0)


class WorkflowTests(unittest.TestCase):
    def _git_repo(self, path: Path) -> Path:
        path.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=path, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "commit", "--allow-empty", "-m", "initial"], cwd=path, check=True,
                       stdout=subprocess.DEVNULL)
        return path

    @staticmethod
    def _which(command):
        return {"codex": "codex", "git": "git", "python": "python", "pytest": "pytest"}.get(command)

    def test_profession_prompt_library_has_planner_and_specific_coder_roles(self):
        self.assertIn("analyst", ROLE_PROMPTS)
        self.assertIn("architect", ROLE_PROMPTS)
        self.assertIn("reviewer", ROLE_PROMPTS)
        plan = build_planning_prompt("Add a settings page")
        coder = build_developer_prompt("frontend", "Add a settings page", plan="Use existing routes")
        self.assertIn("КРИТЕРИИ", plan)
        self.assertIn("frontend/UX-инженер", coder)
        self.assertIn("Use existing routes", coder)

    def test_check_suggestion_detects_existing_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "tests").mkdir()
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=self._which):
                self.assertEqual(suggest_checks(repo), ["pytest -q"])

    def test_unittest_suggestion_anchors_importable_test_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            tests_dir = repo / "tests"
            tests_dir.mkdir()
            which = lambda command: "python.exe" if command == "python" else None
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=which):
                self.assertEqual(suggest_checks(repo), ["python -m unittest discover -s tests"])
                (tests_dir / "__init__.py").write_text("", encoding="utf-8")
                self.assertEqual(suggest_checks(repo), ["python -m unittest discover -s tests -t ."])

    def test_full_cycle_emits_professions_calls_codex_and_runs_review_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._git_repo(root / "project")
            usage_log = root / "usage.jsonl"
            request = WorkflowRequest(
                repo=repo, task="Add a small settings page", checks=["pytest -q"],
                mode="full", profession="frontend", max_repairs=1, max_model_calls=5,
                prompt_token_budget=16000, max_run_tokens=10000, daily_token_budget=20000,
            )
            results = [
                CodexResult(0, CodexUsage(100, 50, 10), "Цель и критерии: настройки. План: добавить экран."),
                CodexResult(0, CodexUsage(120, 40, 15), "Страница добавлена."),
                CodexResult(0, CodexUsage(90, 30, 8), "PASS\nЗамечаний нет."),
            ]
            events = []
            checks = [{"command": "pytest -q", "returncode": 0, "output": "1 passed"}]
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=self._which), \
                 patch("ai_orchestrate.workflow.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.workflow.run_codex", side_effect=results) as codex, \
                 patch("ai_orchestrate.workflow.run_checks", return_value=checks), \
                 patch("ai_orchestrate.workflow.git_snapshot", return_value=("diff", " M app.py")):
                result = run_workflow(request, emit=events.append, cancel_event=Event(), usage_path=usage_log)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["model_calls"], 3)
            self.assertEqual(result["run_tokens"], 343)
            self.assertEqual(codex.call_count, 3)
            self.assertEqual(codex.call_args_list[0].kwargs["sandbox"], "read-only")
            self.assertEqual(codex.call_args_list[1].kwargs["sandbox"], "workspace-write")
            self.assertEqual(codex.call_args_list[2].kwargs["sandbox"], "read-only")
            self.assertIn("frontend/UX-инженер", codex.call_args_list[1].args[1])
            stages = {event["stage"] for event in events}
            self.assertTrue({"routing", "planning", "implementation", "testing", "review", "final"}.issubset(stages))
            roles = {event.get("role") for event in events}
            self.assertIn("Аналитик + архитектор", roles)
            self.assertIn("Ревьюер", roles)

    def test_quick_workflow_uses_one_model_call_and_local_check_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._git_repo(root / "project")
            request = WorkflowRequest(
                repo=repo, task="Fix a typo", checks=["python --version"], mode="quick",
                profession="bugfixer", max_repairs=0, max_model_calls=1,
            )
            result = CodexResult(0, CodexUsage(30, 0, 5), "Fixed.")
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=self._which), \
                 patch("ai_orchestrate.workflow.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.workflow.run_codex", return_value=result) as codex, \
                 patch("ai_orchestrate.workflow.run_checks", return_value=[{"command": "python --version", "returncode": 0, "output": "Python 3"}]):
                result_summary = run_workflow(request, emit=lambda _: None, cancel_event=Event(),
                                               usage_path=root / "usage.jsonl")
            self.assertEqual(result_summary["status"], "complete")
            self.assertEqual(result_summary["model_calls"], 1)
            codex.assert_called_once()

    def test_incomplete_result_explains_executor_and_check_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._git_repo(root / "project")
            request = WorkflowRequest(repo=repo, task="Fix typo", checks=["pytest -q"],
                                      mode="quick", max_repairs=0)
            events = []
            failed = CodexResult(1, CodexUsage(30, 0, 5), "", stderr="Model unavailable")
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=self._which), \
                 patch("ai_orchestrate.workflow.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.workflow.run_codex", return_value=failed), \
                 patch("ai_orchestrate.workflow.run_checks", return_value=[
                     {"command": "pytest -q", "returncode": 1, "output": "AssertionError"}]):
                result = run_workflow(request, emit=events.append, cancel_event=Event(),
                                      usage_path=root / "usage.jsonl")
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("Model unavailable", result["failure_reason"])
            self.assertIn("AssertionError", result["failure_reason"])
            self.assertIn(result["failure_reason"], events[-1]["message"])
            completed = next(e for e in events if e["event"] == "model.completed")
            self.assertEqual(completed["data"]["stderr"], "Model unavailable")

    def test_chatgpt_executor_runs_whole_cycle_through_the_manual_relay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._git_repo(root / "project")
            relay = FakeRelay([
                "ЦЕЛЬ И КРИТЕРИИ: добавить файл. ПЛАН: создать feature.py.",
                "### FILE: feature.py\n```\ndef ready():\n    return True\n```\n",
                "PASS\nЗамечаний нет.",
            ])
            request = WorkflowRequest(
                repo=repo, task="Создай feature.py", checks=["pytest -q"], mode="full",
                profession="developer", executor="chatgpt", relay=relay,
                max_repairs=1, max_model_calls=5, limit_fallback="chatgpt",
            )
            events = []
            checks = [{"command": "pytest -q", "returncode": 0, "output": "1 passed"}]
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=self._which), \
                 patch("ai_orchestrate.workflow.route_with_lane", return_value=("SMALL", ("chatgpt-manual", "low"))), \
                 patch("ai_orchestrate.workflow.run_codex") as codex, \
                 patch("ai_orchestrate.workflow.run_checks", return_value=checks), \
                 patch("ai_orchestrate.workflow.git_snapshot", return_value=("diff", " M feature.py")):
                result = run_workflow(request, emit=events.append, cancel_event=Event(),
                                      usage_path=root / "usage.jsonl")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["executor"], "chatgpt")
            self.assertEqual(result["model_calls"], 3)
            self.assertEqual(result["run_tokens"], 0)
            codex.assert_not_called()
            self.assertTrue((repo / "feature.py").is_file())
            self.assertIn("return True", (repo / "feature.py").read_text(encoding="utf-8"))
            self.assertEqual([item["kind"] for item in relay.requests], ["plan", "code", "review"])
            self.assertIn("ЦЕЛЬ И КРИТЕРИИ", relay.requests[0]["prompt"])
            self.assertIn("*** Begin Patch", relay.requests[1]["prompt"])
            self.assertIn("feature.py", relay.requests[1]["prompt"])
            kinds = [event["event"] for event in events]
            self.assertIn("manual.usage", kinds)
            self.assertNotIn("usage.unknown", kinds)  # ручные шаги не считаются измеряемыми

    def test_codex_usage_limit_switches_remaining_steps_to_chatgpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._git_repo(root / "project")
            relay = FakeRelay([
                "ЦЕЛЬ И КРИТЕРИИ: исправить. ПЛАН: создать fix.py.",
                "*** Begin Patch\n*** Add File: fix.py\n+ANSWER = 42\n*** End Patch\n",
                "PASS\nОк.",
            ])
            request = WorkflowRequest(
                repo=repo, task="Исправь лимит", checks=["pytest -q"], mode="full",
                executor="codex", relay=relay, limit_fallback="chatgpt",
                max_repairs=0, max_model_calls=5,
            )
            events = []
            limited = CodexResult(1, CodexUsage(), "", stderr=LIMIT_MESSAGE)
            checks = [{"command": "pytest -q", "returncode": 0, "output": "1 passed"}]
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=self._which), \
                 patch("ai_orchestrate.workflow.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.workflow.run_codex", return_value=limited) as codex, \
                 patch("ai_orchestrate.workflow.run_checks", return_value=checks), \
                 patch("ai_orchestrate.workflow.git_snapshot", return_value=("diff", " M fix.py")):
                result = run_workflow(request, emit=events.append, cancel_event=Event(),
                                      usage_path=root / "usage.jsonl")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["executor"], "chatgpt")
            codex.assert_called_once()  # только первый шаг успел уйти в Codex
            self.assertTrue((repo / "fix.py").is_file())
            self.assertEqual([item["kind"] for item in relay.requests], ["plan", "code", "review"])
            kinds = [event["event"] for event in events]
            self.assertIn("executor.limit", kinds)
            self.assertIn("executor.fallback", kinds)
            fallback = next(event for event in events if event["event"] == "executor.fallback")
            self.assertEqual(fallback["data"]["to"], "chatgpt")

    def test_usage_limit_without_fallback_keeps_the_plain_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._git_repo(root / "project")
            request = WorkflowRequest(
                repo=repo, task="Обычная задача", checks=["pytest -q"], mode="quick",
                executor="codex", limit_fallback="off", max_model_calls=2, max_repairs=0,
            )
            limited = CodexResult(1, CodexUsage(), "", stderr=LIMIT_MESSAGE)
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=self._which), \
                 patch("ai_orchestrate.workflow.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.workflow.run_codex", return_value=limited), \
                 patch("ai_orchestrate.workflow.run_checks", return_value=[
                     {"command": "pytest -q", "returncode": 1, "output": "AssertionError"}]):
                result = run_workflow(request, emit=lambda _: None, cancel_event=Event(),
                                      usage_path=root / "usage.jsonl")
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("usage limit", result["failure_reason"])

    def test_usage_limit_without_bypass_explains_what_to_enable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._git_repo(root / "project")
            events = []
            request = WorkflowRequest(
                repo=repo, task="Обычная задача", checks=["pytest -q"], mode="quick",
                executor="codex", limit_fallback="api", max_model_calls=2, max_repairs=0,
            )
            limited = CodexResult(1, CodexUsage(20, 0, 5), "", stderr=LIMIT_MESSAGE)
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=self._which), \
                 patch("ai_orchestrate.workflow.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.workflow.run_codex", return_value=limited), \
                 patch("ai_orchestrate.workflow.run_checks", return_value=[
                     {"command": "pytest -q", "returncode": 1, "output": "AssertionError"}]):
                run_workflow(request, emit=events.append, cancel_event=Event(),
                             usage_path=root / "usage.jsonl")
            limit_event = next(event for event in events if event["event"] == "executor.limit")
            self.assertEqual(limit_event["data"]["fallback"], "")
            self.assertIn("имя модели и ключ", limit_event["message"])

    def test_chatgpt_executor_requires_the_panel_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._git_repo(root / "project")
            request = WorkflowRequest(repo=repo, task="Задача", checks=["python --version"],
                                      executor="chatgpt", relay=None)
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=self._which):
                with self.assertRaisesRegex(OrchestratorError, "обычный чат"):
                    run_workflow(request, emit=lambda _: None, cancel_event=Event(),
                                 usage_path=root / "usage.jsonl")

    def test_usage_limit_detection_covers_cli_and_api_messages(self):
        self.assertTrue(is_usage_limit_error(LIMIT_MESSAGE))
        self.assertTrue(is_usage_limit_error('{"error": {"code": "insufficient_quota"}}'))
        self.assertTrue(is_usage_limit_error("HTTP 429 Too Many Requests"))
        self.assertFalse(is_usage_limit_error("AssertionError: expected 2"))
        self.assertFalse(is_usage_limit_error("", None))

    def test_ui_workspace_rejects_paths_outside_configured_root(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            manager = RunManager(Path(tmp))
            inside = Path(tmp) / "project"
            inside.mkdir()
            self.assertEqual(manager._safe_repo_path(str(inside)), inside.resolve())
            with self.assertRaisesRegex(OrchestratorError, "внутри"):
                manager._safe_repo_path(outside)


if __name__ == "__main__":
    unittest.main()
