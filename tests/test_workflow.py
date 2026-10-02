import subprocess
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from ai_orchestrate.core import LANES, CodexResult, CodexUsage, OrchestratorError
from ai_orchestrate.prompts import ROLE_PROMPTS, build_developer_prompt, build_planning_prompt
from ai_orchestrate.web import RunManager
from ai_orchestrate.workflow import WorkflowRequest, run_workflow, suggest_checks


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
