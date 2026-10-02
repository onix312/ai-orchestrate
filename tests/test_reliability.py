"""Regression tests for the reliability/security audit. No external model calls."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch
from types import SimpleNamespace

from ai_orchestrate.core import CodexResult, CodexUsage, OrchestratorError, git_snapshot, run_codex, run_checks
from ai_orchestrate.endpoints import endpoint_provider
from ai_orchestrate.gitops import (create_worktree, commit_worktree, git, merge_local,
                                  verify_worktree, _changed_paths, worktree_digest)
from ai_orchestrate.llm_api import ApiConfig, run_llm_api, safe_join, _NoRedirect
from ai_orchestrate.chatgpt_bridge import parse_answer, apply_operations
from ai_orchestrate.settings import SettingsStore, default_settings
from ai_orchestrate.web import RunManager, RunSubmission, RunJob
from ai_orchestrate.workflow import _api_config, WorkflowRequest
from ai_orchestrate.autofill import detect_base_branch


def tool(name, args, content=None):
    return {"choices": [{"message": {"content": content, "tool_calls": [
        {"id": "t", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


class ApiReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_api(self, response, **kwargs):
        with patch("ai_orchestrate.llm_api._post", return_value=response) as post:
            result = run_llm_api(self.root, "task", "model",
                                 config=ApiConfig(max_rounds=1), **kwargs)
        return result, post

    def test_budget_exhaustion_is_not_success_and_does_not_write(self):
        result, _ = self.run_api(tool("write_file", {"path": "x", "content": "bad"}), token_budget=12)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertFalse((self.root / "x").exists())

    def test_zero_budget_makes_no_request(self):
        result, post = self.run_api({}, token_budget=0)
        self.assertNotEqual(result.returncode, 0)
        post.assert_not_called()

    def test_intermediate_text_does_not_turn_round_exhaustion_into_success(self):
        result, _ = self.run_api(tool("list_dir", {}, content="I will continue"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("лимит итераций", result.stderr)

    def test_malformed_tool_arguments_fail_safely(self):
        for args in ([], "string", 12, None):
            with self.subTest(args=args):
                result, _ = self.run_api(tool("finish", args))
                self.assertNotEqual(result.returncode, 0)

    def test_empty_and_truncated_responses_are_not_success(self):
        for response in ([], {"choices": []}, {"choices": [{"message": {}}]},
                         {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}):
            result, _ = self.run_api(response)
            self.assertNotEqual(result.returncode, 0)

    def test_unknown_usage_cannot_bypass_budget(self):
        response = tool("write_file", {"path": "x", "content": "bad"})
        response.pop("usage")
        result, _ = self.run_api(response, token_budget=500)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "x").exists())

    def test_no_unsandboxed_commands_are_offered_or_executed(self):
        result, post = self.run_api(tool("run_command", {"command": 'python -c "raise Exception()"'}))
        tools = post.call_args.args[1]["tools"]
        self.assertNotIn("run_command", [t["function"]["name"] for t in tools])
        self.assertNotEqual(result.returncode, 0)

    def test_finish_stops_later_mutations_in_same_response(self):
        response = tool("finish", {"summary": "done"})
        response["choices"][0]["message"]["tool_calls"].extend(
            tool("write_file", {"path": "x", "content": "bad"})["choices"][0]["message"]["tool_calls"])
        result, _ = self.run_api(response)
        self.assertEqual(result.returncode, 0)
        self.assertFalse((self.root / "x").exists())

    def test_cancel_during_request_accounts_usage_but_does_not_apply_tools(self):
        cancel = Event()
        def respond(*args):
            cancel.set()
            return tool("write_file", {"path": "x", "content": "bad"})
        with patch("ai_orchestrate.llm_api._post", side_effect=respond):
            result = run_llm_api(self.root, "task", "model", config=ApiConfig(), cancel_event=cancel)
        self.assertTrue(result.cancelled)
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertFalse((self.root / "x").exists())

    def test_prior_usage_survives_later_http_failure(self):
        with patch("ai_orchestrate.llm_api._post", side_effect=[tool("list_dir", {}), OrchestratorError("offline")]):
            result = run_llm_api(self.root, "task", "model", config=ApiConfig())
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertIn("offline", result.stderr)

    def test_git_metadata_and_symlink_alias_are_blocked(self):
        for name in (".git", ".git/config", "sub/../.git", "sub/.git/hooks/a", "x:stream"):
            with self.assertRaises(OrchestratorError):
                safe_join(self.root, name)
        (self.root / ".git").mkdir()
        (self.root / "alias").symlink_to(self.root / ".git", target_is_directory=True)
        with self.assertRaises(OrchestratorError):
            safe_join(self.root, "alias/config")


class EndpointTests(unittest.TestCase):
    def test_exact_host_and_https_required_for_saved_keys(self):
        self.assertEqual(endpoint_provider("https://api.openai.com/v1"), "openai")
        self.assertEqual(endpoint_provider("https://openrouter.ai/api/v1"), "openrouter")
        self.assertIsNone(endpoint_provider("http://[::1]:1234/v1"))
        for url in ("https://openrouter.ai.evil.test/v1", "https://evil.test/localhost", "http://api.openai.com/v1",
                    "https://localhost@evil.test/v1", "https://api.openai.com:444/v1", "http://0.0.0.0:1234/v1"):
            with self.subTest(url=url), self.assertRaises(OrchestratorError):
                endpoint_provider(url)

    def test_local_models_do_not_receive_saved_openai_key(self):
        with patch("ai_orchestrate.workflow.active_key", return_value="secret") as key:
            config = _api_config(WorkflowRequest(Path("."), "task", [], api_base_url="http://localhost:11434/v1"))
        self.assertEqual(config.api_key, "")
        key.assert_not_called()

    def test_redirects_never_forward_authorization(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 307, "", {}, "https://evil.test"))


class PatchReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "a").write_text("one\nmiddle\nlast\n")

    def test_bad_second_file_does_not_change_first(self):
        report = apply_operations(self.root, parse_answer(
            "### FILE: a\n```\nchanged\n```\n### FILE: ../escape\n```\nbad\n```"))
        self.assertFalse(report.ok)
        self.assertEqual(report.applied, [])
        self.assertEqual((self.root / "a").read_text(), "one\nmiddle\nlast\n")

    def test_truncated_answers_are_rejected_before_writing(self):
        for answer in ("### FILE: a\n```\npartial", "*** Begin Patch\n*** Delete File: a",
                       "### FILE: a\nunfenced"):
            with self.assertRaises(OrchestratorError):
                parse_answer(answer)

    def test_multiple_noncontiguous_hunks(self):
        answer = "*** Begin Patch\n*** Update File: a\n@@\n-one\n+first\n@@\n-last\n+end\n*** End Patch"
        report = apply_operations(self.root, parse_answer(answer))
        self.assertTrue(report.ok, report.rejected)
        self.assertEqual((self.root / "a").read_text(), "first\nmiddle\nend\n")

    def test_ambiguous_hunk_is_rejected(self):
        (self.root / "a").write_text("one\none\n")
        report = apply_operations(self.root, parse_answer("*** Begin Patch\n*** Update File: a\n@@\n-one\n+new\n*** End Patch"))
        self.assertFalse(report.ok)
        self.assertEqual((self.root / "a").read_text(), "one\none\n")

    def test_patch_text_inside_file_is_not_executed(self):
        report = apply_operations(self.root, parse_answer(
            "### FILE: instructions.txt\n```text\n*** Begin Patch\n*** Delete File: a\n*** End Patch\n```"))
        self.assertTrue(report.ok)
        self.assertTrue((self.root / "a").exists())

    def test_duplicate_paths_and_add_over_existing_are_rejected(self):
        for answer in ("### FILE: a\n```\none\n```\n### FILE: ./a\n```\ntwo\n```",
                       "*** Begin Patch\n*** Add File: a\n+overwrite\n*** End Patch"):
            report = apply_operations(self.root, parse_answer(answer))
            self.assertFalse(report.ok)
            self.assertEqual(report.applied, [])

    def test_io_failure_rolls_back_previous_writes(self):
        from ai_orchestrate.chatgpt_bridge import _atomic_write
        count = 0
        def flaky(path, content, mode):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("disk full")
            return _atomic_write(path, content, mode)
        with patch("ai_orchestrate.chatgpt_bridge._atomic_write", side_effect=flaky):
            report = apply_operations(self.root, parse_answer(
                "### FILE: a\n```\nchanged\n```\n### FILE: sub/new\n```\nx\n```"))
        self.assertFalse(report.ok)
        self.assertEqual((self.root / "a").read_text(), "one\nmiddle\nlast\n")
        self.assertFalse((self.root / "sub").exists())


class GitAndBridgeReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.repo)], check=True)
        git(self.repo, ["config", "user.name", "Test"])
        git(self.repo, ["config", "user.email", "test@example.invalid"])
        (self.repo / "a").write_text("base\n")
        git(self.repo, ["add", "a"])
        git(self.repo, ["commit", "-qm", "base"])
        self.worktree = create_worktree(self.repo, self.root / "trees", job_id="test1", branch_prefix="agent",
                                       base_ref="main", base_branch="main")
        self.manager = RunManager(self.root, usage_path=self.root / "usage.jsonl",
                                  settings_path=self.root / "settings.json", journal_path=self.root / "journal.jsonl")
        settings = default_settings()
        settings.update(mode="quick", save_journal=False)
        self.job = RunJob("test1", RunSubmission(self.repo, "Fix", ["python --version"], settings),
                          status="incomplete", worktree=self.worktree)
        self.manager._jobs[self.job.id] = self.job

    def answer(self, text="fixed"):
        return {"run": self.job.id, "answer": f"### FILE: a\n```\n{text}\n```"}

    def test_digest_binds_commit_to_checked_files(self):
        (self.worktree.path / "a").write_text("checked")
        digest = worktree_digest(self.worktree.path)
        (self.worktree.path / "a").write_text("not checked")
        with self.assertRaisesRegex(OrchestratorError, "после проверок"):
            commit_worktree(self.worktree, "blocked", verified_digest=digest)

    def test_digest_is_stable_across_commit_and_includes_new_files(self):
        (self.worktree.path / "new").write_text("new file")
        digest = worktree_digest(self.worktree.path)
        result = commit_worktree(self.worktree, "checked", verified_digest=digest)
        self.assertTrue(result["committed"])
        self.assertEqual(worktree_digest(self.worktree.path), digest)

    def test_bridge_review_cannot_reset_model_call_budget(self):
        self.job.submission.settings["mode"] = "full"
        self.job.model_calls_spent = self.job.submission.settings["max_model_calls"]
        with patch("ai_orchestrate.web.run_workflow") as workflow:
            from ai_orchestrate.workflow import WorkflowStopped
            with self.assertRaisesRegex(WorkflowStopped, "обходить бюджет"):
                self.manager.bridge_apply(self.answer())
        workflow.assert_not_called()
        self.assertFalse(self.manager.fetch(self.job.id)["can_confirm"])

    def test_cleanup_does_not_force_remove_new_unchecked_files(self):
        (self.worktree.path / "new").write_text("user work")
        self.job.status = "running"
        self.manager._cleanup(self.job, delete_branch=True, force=True)
        self.assertTrue((self.worktree.path / "new").exists())
        self.assertTrue(any(e["event"] == "warning" for e in self.job.events))

    def test_dirty_worktree_blocks_merge(self):
        (self.worktree.path / "a").write_text("checked\n")
        result = commit_worktree(self.worktree, "checked")
        (self.worktree.path / "a").write_text("unchecked\n")
        with self.assertRaises(OrchestratorError):
            merge_local(self.worktree, expected_sha=result["sha"])
        self.assertEqual((self.repo / "a").read_text(), "base\n")

    def test_changed_head_blocks_merge(self):
        (self.worktree.path / "a").write_text("checked\n")
        result = commit_worktree(self.worktree, "checked")
        (self.worktree.path / "a").write_text("unchecked\n")
        git(self.worktree.path, ["commit", "-qam", "unchecked"])
        with self.assertRaises(OrchestratorError):
            merge_local(self.worktree, expected_sha=result["sha"])

    def test_model_created_commit_is_not_silently_treated_as_no_changes(self):
        git(self.worktree.path, ["commit", "--allow-empty", "-qm", "model commit"])
        with self.assertRaises(OrchestratorError):
            commit_worktree(self.worktree, "automatic")

    def test_unusual_sensitive_filename_is_detected(self):
        folder = self.worktree.path / "кириллица\nfolder"
        folder.mkdir()
        (folder / ".env").write_text("secret")
        with self.assertRaisesRegex(OrchestratorError, "секретные"):
            commit_worktree(self.worktree, "unsafe")

    def test_rename_parser_checks_both_names(self):
        self.assertEqual(_changed_paths("R  public.txt\0.env\0?? a -> b\0"), ["public.txt", ".env", "a -> b"])

    def test_snapshot_includes_uncommitted_diff_against_base(self):
        (self.worktree.path / "a").write_text("uncommitted\n")
        diff, _ = git_snapshot(self.worktree.path, base=self.worktree.base_sha)
        self.assertIn("+uncommitted", diff)

    def test_snapshot_does_not_read_external_symlink(self):
        outside = self.root / "outside"
        outside.write_text("VERY_PRIVATE_CONTENT")
        (self.worktree.path / "link").symlink_to(outside)
        diff, _ = git_snapshot(self.worktree.path)
        self.assertNotIn("VERY_PRIVATE_CONTENT", diff)

    def test_local_autofill_uses_checkout_not_remote_head(self):
        git(self.repo, ["update-ref", "refs/remotes/origin/other", "HEAD"])
        git(self.repo, ["symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/other"])
        self.assertEqual(detect_base_branch(self.repo, local=True), "main")
        self.assertEqual(detect_base_branch(self.repo), "other")

    def test_failed_discard_is_not_reported_as_success(self):
        with patch("ai_orchestrate.web.remove_worktree", side_effect=OrchestratorError("locked")):
            self.manager._discard_worker(self.job)
        self.assertEqual(self.job.status, "incomplete")
        self.assertEqual(self.job.error, "locked")
        self.assertFalse(any(e["event"] == "worktree.discard.completed" for e in self.job.events))
        self.assertIsNotNone(self.job.worktree)

    def test_successful_discard_disables_repeat_action(self):
        self.manager._discard_worker(self.job)
        self.assertFalse(self.manager.fetch(self.job.id)["can_discard"])
        self.assertFalse(self.worktree.path.exists())

    def test_bridge_rejects_concurrent_operation(self):
        with self.manager._bridge_operation({"run": self.job.id}):
            with self.assertRaises(OrchestratorError):
                self.manager.bridge_apply(self.answer())
            self.assertFalse(self.manager.confirm(self.job.id))
            self.assertFalse(self.manager.discard(self.job.id))

    def test_rejected_bridge_answer_invalidates_previous_approval(self):
        self.manager.bridge_apply(self.answer())
        self.assertTrue(self.manager.fetch(self.job.id)["can_confirm"])
        result = self.manager.bridge_apply({"run": self.job.id,
                                           "answer": "### FILE: ../bad\n```\nwrong\n```"})
        self.assertFalse(result["checks_passed"])
        self.assertFalse(self.manager.fetch(self.job.id)["can_confirm"])

    def test_check_exception_invalidates_previous_approval(self):
        self.manager.bridge_apply(self.answer())
        with patch("ai_orchestrate.web.run_checks", side_effect=OrchestratorError("check failed to start")):
            with self.assertRaises(OrchestratorError):
                self.manager.bridge_apply(self.answer("changed again"))
        self.assertEqual(self.job.status, "incomplete")
        self.assertFalse(self.manager.fetch(self.job.id)["can_confirm"])

    def test_bridge_prompt_uses_evidence_without_running_tests(self):
        with patch("ai_orchestrate.web.run_checks") as checks:
            self.manager.bridge_prompt({"run": self.job.id})
        checks.assert_not_called()

    def test_full_bridge_requires_new_read_only_review(self):
        self.job.submission.settings["mode"] = "full"
        import shutil
        which = shutil.which
        for review, expected in (("ISSUES\nNot ready", False), ("PASS\nOK", True)):
            with patch("ai_orchestrate.workflow.shutil.which", side_effect=lambda name: "codex" if name == "codex" else which(name)), \
                 patch("ai_orchestrate.workflow.run_codex", return_value=CodexResult(0, CodexUsage(10, 0, 2), review)) as model:
                result = self.manager.bridge_apply(self.answer())
            model.assert_called_once()
            self.assertEqual(model.call_args.kwargs["sandbox"], "read-only")
            self.assertEqual(result["review_passed"], expected)
            self.assertEqual(self.manager.fetch(self.job.id)["can_confirm"], expected)

    def test_workflow_terminal_event_does_not_end_job_before_commit(self):
        self.job.status = "running"
        self.manager._workflow_event(self.job, {"event": "run.completed", "data": {}})
        self.assertEqual(self.job.events, [])

    def test_successful_second_bridge_can_merge_existing_commit(self):
        self.manager.bridge_apply(self.answer())
        sha = self.job.result["commit_sha"]
        self.manager.bridge_apply(self.answer())
        self.assertEqual(self.job.result["commit_sha"], sha)
        self.assertTrue(self.manager.fetch(self.job.id)["can_confirm"])


class CodexDefaultsTests(unittest.TestCase):
    def test_empty_model_uses_cli_config_and_persists_session(self):
        with patch("ai_orchestrate.core.shutil.which", return_value="codex"):
            with patch("builtins.print"):
                calls = []
                def runner(command, **kwargs):
                    calls.append(command)
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                run_codex(Path("."), "task", "", "low", runner=runner)
        self.assertNotIn("-m", calls[0])
        self.assertNotIn("--ephemeral", calls[0])

    def test_json_failure_is_preserved_even_with_zero_process_exit(self):
        with patch("ai_orchestrate.core.shutil.which", return_value="codex"):
            result = run_codex(Path("."), "task", "", "low", runner=lambda *a, **k: SimpleNamespace(
                returncode=0, stdout=json.dumps({"type": "turn.failed", "error": {"message": "Model not found"}}), stderr=""))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Model not found", result.stderr)

    def test_old_placeholder_defaults_migrate_to_cli_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({"luna_model": "gpt-6-luna", "sol_model": "gpt-6-sol"}))
            settings = SettingsStore(path).load()
        self.assertEqual(settings["luna_model"], "")
        self.assertEqual(settings["sol_model"], "")


class CheckOutputTests(unittest.TestCase):
    def test_large_output_is_drained_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "loud.py"
            script.write_text("import sys\nsys.stdout.write('x' * 8_000_000 + 'TAIL_MARKER')\n")
            result = run_checks(Path(tmp), [f'"{sys.executable}" "{script}"'], timeout=10)[0]
        self.assertEqual(result["returncode"], 0)
        self.assertLessEqual(len(result["output"]), 4000)
        self.assertTrue(result["output"].endswith("TAIL_MARKER"))

    def test_checks_do_not_receive_provider_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "env.py"
            script.write_text("import os\nprint(os.environ.get('OPENAI_API_KEY', 'ABSENT'))\n")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "private-provider-key"}):
                result = run_checks(Path(tmp), [f'"{sys.executable}" "{script}"'])[0]
        self.assertEqual(result["output"].strip(), "ABSENT")
