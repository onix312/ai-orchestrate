import json
import os
import subprocess
import tempfile
import time
import unittest
from threading import Event, Timer
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_orchestrate.core import (
    LANES,
    CodexResult,
    CodexUsage,
    Decision,
    OrchestratorError,
    _codex_command,
    _codex_env,
    ensure_clean_git,
    estimate_prompt_tokens,
    local_lane,
    next_lane,
    next_lane_name,
    parse_codex_events,
    review_passed,
    route,
    route_with_lane,
    run_checks,
    run_codex,
)
from ai_orchestrate.cli import _run
from ai_orchestrate.usage import append_usage, read_usage_entries, tokens_for_date


class CodexCommandResolutionTests(unittest.TestCase):
    def test_resolves_codex_to_absolute_path(self):
        with patch("ai_orchestrate.core.shutil.which", return_value="/usr/local/bin/codex") as which:
            self.assertEqual(_codex_command(), ["/usr/local/bin/codex"])
        which.assert_called_once_with("codex")

    def test_missing_codex_cli_is_reported_before_launch(self):
        with patch("ai_orchestrate.core.shutil.which", return_value=None):
            with self.assertRaisesRegex(OrchestratorError, "Codex CLI was not found on PATH"):
                _codex_command()

    def test_windows_cmd_shim_is_launched_through_comspec(self):
        with patch("ai_orchestrate.core.os.name", "nt"), \
             patch.dict(os.environ, {"COMSPEC": r"C:\Windows\System32\cmd.exe"}), \
             patch("ai_orchestrate.core.shutil.which", return_value=r"C:\Users\test\npm\codex.cmd"):
            self.assertEqual(
                _codex_command(),
                [r"C:\Windows\System32\cmd.exe", "/d", "/s", "/c", r"C:\Users\test\npm\codex.cmd"],
            )

    def test_windows_executable_is_launched_directly(self):
        executable = r"C:\Program Files\Codex\codex.exe"
        with patch("ai_orchestrate.core.os.name", "nt"), \
             patch("ai_orchestrate.core.shutil.which", return_value=executable):
            self.assertEqual(_codex_command(), [executable])

    def test_resolved_path_is_used_by_popen(self):
        class FakeInput:
            closed = False

            def write(self, value):
                pass
            def flush(self):
                pass
            def close(self):
                self.closed = True

        class FakeProcess:
            pid = 12345
            returncode = 0
            stdin = FakeInput()
            stdout = StringIO("")
            stderr = StringIO("")

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

        process = FakeProcess()
        with patch("ai_orchestrate.core.shutil.which", return_value="/opt/codex/bin/codex"), \
             patch("ai_orchestrate.core.subprocess.Popen", return_value=process) as popen:
            run_codex(Path("."), "task", "model", "low")
        self.assertEqual(popen.call_args.args[0][0], "/opt/codex/bin/codex")

    def test_launch_error_names_the_resolved_executable(self):
        executable = "/opt/codex/bin/codex"
        with patch("ai_orchestrate.core.shutil.which", return_value=executable), \
             patch("ai_orchestrate.core.subprocess.Popen", side_effect=FileNotFoundError):
            with self.assertRaises(OrchestratorError) as raised:
                run_codex(Path("."), "task", "model", "low")
        self.assertIn(executable, str(raised.exception))
        self.assertIn("FileNotFoundError", str(raised.exception))


class CoreTests(unittest.TestCase):
    def test_local_router_is_free_and_uses_lowest_tier_for_small_work(self):
        with patch("ai_orchestrate.core.jev_choice", side_effect=AssertionError):
            self.assertEqual(route("Fix a typo", dry_run=True), LANES["SMALL"])
            lane, model = route_with_lane("Add Russian authentication and security checks")
        self.assertEqual(lane, "HIGH")
        self.assertEqual(model, LANES["HIGH"])

    def test_route_accepts_explicit_lane_and_dry_run_never_calls_jev(self):
        with patch("ai_orchestrate.core.jev_choice", side_effect=AssertionError):
            self.assertEqual(route("task", lane="MEDIUM", dry_run=True), LANES["MEDIUM"])
        with self.assertRaises(OrchestratorError):
            route("task", lane="unknown", dry_run=True)
        with self.assertRaisesRegex(OrchestratorError, "cannot call Jev"):
            route("task", router="jev", dry_run=True)

    def test_local_lane_scales_for_scope(self):
        self.assertEqual(local_lane("Fix typo"), "SMALL")
        self.assertEqual(local_lane("Please refactor several modules and add tests"), "MEDIUM")
        self.assertEqual(local_lane("Review production security and data loss risks"), "HIGH")

    def test_jev_parses_choice_and_requires_schema(self):
        payload = {"answers": {"decision": {"choice": "SMALL", "confidence": .9,
                                             "probabilities": {"SMALL": .9, "MEDIUM": .04,
                                                                "HIGH": .03, "ESCALATE": .03}}}}

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps(payload).encode()

        events = []
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret"}), patch(
            "ai_orchestrate.core.urllib.request.urlopen", return_value=Response()
        ) as open_url:
            lane_name, decision = route_with_lane("task", router="jev", on_event=lambda name, data: events.append((name, data)))
        self.assertEqual(lane_name, "SMALL")
        self.assertEqual(decision, LANES["SMALL"])
        self.assertEqual([event[0] for event in events], ["jev.started", "jev.completed"])
        self.assertEqual(events[1][1]["probabilities"]["SMALL"], .9)
        self.assertNotIn("secret", repr(open_url.call_args))

    def test_jev_missing_key_is_clear(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(OrchestratorError, "TYPESAFE_API_KEY"):
                route("task", router="jev")

    def test_jev_rejects_bad_probabilities_and_hides_exception_text(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return b'{"answers":{"decision":{"choice":"SMALL","confidence":1,"probabilities":{"SMALL":0.4}}}}'

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret"}), patch(
            "ai_orchestrate.core.urllib.request.urlopen", return_value=Response()
        ):
            with self.assertRaisesRegex(OrchestratorError, "probabilities"):
                route("task", router="jev")
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret"}), patch(
            "ai_orchestrate.core.urllib.request.urlopen", side_effect=RuntimeError("secret response")
        ):
            with self.assertRaises(OrchestratorError) as raised:
                route("task", router="jev")
        self.assertNotIn("secret response", str(raised.exception))

    def test_lane_progression_is_bounded(self):
        self.assertEqual(next_lane(*LANES["SMALL"]), LANES["MEDIUM"])
        self.assertEqual(next_lane(*LANES["HIGH"]), LANES["ESCALATE"])
        self.assertEqual(next_lane(*LANES["ESCALATE"]), LANES["ESCALATE"])
        self.assertEqual(next_lane_name("SMALL"), "MEDIUM")
        self.assertIsNone(next_lane_name("ESCALATE"))

    def test_codex_jsonl_parser_aggregates_usage_and_final_message(self):
        events = "\n".join([
            '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":70,"output_tokens":10}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"first"}}',
            '{"type":"turn.completed","usage":{"input_tokens":40,"cached_input_tokens":30,"output_tokens":8}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
        ])
        usage, message = parse_codex_events(events)
        self.assertEqual(usage, CodexUsage(140, 100, 18))
        self.assertEqual(usage.total_tokens, 158)
        self.assertEqual(message, "done")

    def test_prompt_budget_is_approximate_utf8_text_count(self):
        self.assertEqual(estimate_prompt_tokens(""), 0)
        self.assertGreater(estimate_prompt_tokens("задача"), 0)

    def test_codex_child_does_not_receive_jev_credentials_and_collects_usage(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret", "OPENROUTER_API_KEY": "also-secret"}):
            env = _codex_env()
        self.assertNotIn("TYPESAFE_API_KEY", env)
        self.assertNotIn("OPENROUTER_API_KEY", env)

        captured = {}
        stream = '{"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":5,"output_tokens":2}}\n'
        stream += '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n'

        def runner(command, **kwargs):
            captured.update(kwargs)
            captured["command"] = command
            return SimpleNamespace(returncode=0, stdout=stream, stderr="")

        with patch("ai_orchestrate.core.shutil.which", return_value="/usr/bin/codex"):
            result = run_codex(Path("."), "do work", "gpt-6-luna", "low", runner=runner)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertEqual(result.final_message, "done")
        self.assertEqual(captured["input"], "do work")
        self.assertEqual(captured["command"][-1], "-")
        self.assertIn("--json", captured["command"])
        self.assertNotIn("--ephemeral", captured["command"])
        self.assertIn("workspace-write", captured["command"])
        self.assertNotIn("TYPESAFE_API_KEY", captured["env"])

    def test_default_codex_runner_uses_process_group_and_captures_json_telemetry(self):
        stream = '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}\n'

        class FakeInput:
            def __init__(self):
                self.value = ""
                self.closed = False
            def write(self, value):
                self.value += value
            def flush(self):
                pass
            def close(self):
                self.closed = True

        class FakeProcess:
            pid = 12345
            returncode = 0

            def __init__(self):
                self.stdin = FakeInput()
                self.stdout = StringIO(stream)
                self.stderr = StringIO("")

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

        process = FakeProcess()
        events = []
        with patch("ai_orchestrate.core.shutil.which", return_value="/usr/bin/codex"), \
             patch("ai_orchestrate.core.subprocess.Popen", return_value=process) as popen:
            result = run_codex(Path("."), "task", "model", "low", timeout=15, on_event=events.append)
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertEqual(process.stdin.value, "task")
        self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.PIPE)
        self.assertEqual([event["type"] for event in events], ["turn.completed"])
        if os.name != "nt":
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    @unittest.skipIf(os.name == "nt", "process-group timeout integration test is POSIX-only")
    def test_live_codex_timeout_kills_process_group_and_cancel_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            codex = bin_dir / "codex"
            codex.write_text(
                """#!/bin/sh
printf '%s\\n' '{"type":"turn.started"}'
sleep 30
""",
                encoding="utf-8",
            )
            codex.chmod(0o755)
            repo = Path(tmp) / "repo"
            repo.mkdir()
            fake_path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            with patch.dict(os.environ, {"PATH": fake_path}):
                started = time.monotonic()
                timed_out = run_codex(repo, "task", "model", "low", timeout=1)
                self.assertEqual(timed_out.returncode, 124)
                self.assertLess(time.monotonic() - started, 5)
                cancel = Event()
                timer = Timer(.2, cancel.set)
                timer.start()
                try:
                    cancelled = run_codex(repo, "task", "model", "low", timeout=10, cancel_event=cancel)
                finally:
                    timer.cancel()
                self.assertEqual(cancelled.returncode, 130)
                self.assertTrue(cancelled.cancelled)

    @unittest.skipIf(os.name == "nt", "process-group timeout integration test is POSIX-only")
    def test_check_timeout_and_user_cancel_kill_process_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            command = bin_dir / "slow-check"
            command.write_text(
                """#!/bin/sh
sleep 30
""",
                encoding="utf-8",
            )
            command.chmod(0o755)
            repo = Path(tmp) / "repo"
            repo.mkdir()
            fake_path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            with patch.dict(os.environ, {"PATH": fake_path}):
                started = time.monotonic()
                timed_out = run_checks(repo, ["slow-check"], timeout=1)
                self.assertEqual(timed_out[0]["returncode"], 124)
                self.assertLess(time.monotonic() - started, 5)
                cancel = Event()
                timer = Timer(.2, cancel.set)
                timer.start()
                events = []
                try:
                    cancelled = run_checks(repo, ["slow-check"], timeout=10, cancel_event=cancel,
                                           on_event=lambda name, data: events.append(name))
                finally:
                    timer.cancel()
                self.assertEqual(cancelled[0]["returncode"], 130)
                self.assertEqual(events, ["check.started", "check.completed"])

    def test_codex_can_run_read_only_and_timeout_is_bounded(self):
        commands = []

        def runner(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("ai_orchestrate.core.shutil.which", return_value="/usr/bin/codex"):
            run_codex(Path("."), "review", "review-model", "low", sandbox="read-only", runner=runner)
            self.assertIn("read-only", commands[0])
            with self.assertRaises(OrchestratorError):
                run_codex(Path("."), "review", "review-model", "low", sandbox="danger-full-access", runner=runner)

            def timed_out(*args, **kwargs):
                raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

            result = run_codex(Path("."), "task", "model", "low", timeout=3, runner=timed_out)
        self.assertEqual(result.returncode, 124)

    def test_clean_git_gate_includes_untracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "--allow-empty", "-m", "initial"], cwd=repo, check=True,
                           stdout=subprocess.DEVNULL)
            ensure_clean_git(repo)
            (repo / "new.txt").write_text("untracked")
            with self.assertRaisesRegex(OrchestratorError, "including untracked"):
                ensure_clean_git(repo)

    def test_review_must_explicitly_start_with_pass(self):
        self.assertTrue(review_passed("PASS\nNo actionable findings."))
        self.assertFalse(review_passed("ISSUES\nFix X"))
        self.assertFalse(review_passed("I think this passes."))


class UsageTests(unittest.TestCase):
    def test_usage_ledger_stores_counts_not_prompts_and_sums_daily_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            usage = CodexUsage(input_tokens=90, cached_input_tokens=50, output_tokens=10)
            self.assertTrue(append_usage(path, model="small", effort="low", role="coder", attempt=1,
                                         returncode=0, usage=usage))
            today = datetime.now().astimezone().date().isoformat()
            self.assertEqual(tokens_for_date(path, today), 100)
            entries = read_usage_entries(path)
            self.assertEqual(len(entries), 1)
            self.assertNotIn("prompt", entries[0])
            self.assertEqual(entries[0]["cached_input_tokens"], 50)

    def test_custom_usage_path_does_not_change_parent_permissions_and_keeps_file_private(self):
        if os.name == "nt":
            self.skipTest("POSIX permission bits are not portable to Windows")
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp) / "shared"
            shared.mkdir(mode=0o755)
            shared.chmod(0o755)
            path = shared / "usage.jsonl"
            self.assertTrue(append_usage(path, model="m", effort="low", role="coder", attempt=1,
                                         returncode=0, usage=CodexUsage(10, 0, 2)))
            self.assertEqual(shared.stat().st_mode & 0o777, 0o755)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_unknown_usage_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            self.assertFalse(append_usage(path, model="m", effort="low", role="coder", attempt=1,
                                          returncode=1, usage=CodexUsage()))
            self.assertFalse(path.exists())


class CliTests(unittest.TestCase):
    def _args(self, repo: str, **overrides):
        values = dict(
            repo=repo,
            check=["python"],
            task="do task",
            dry_run=False,
            lane=None,
            router="local",
            max_attempts=1,
            max_model_calls=None,
            max_run_tokens=None,
            daily_token_budget=None,
            prompt_token_budget=12000,
            usage_log=str(Path(repo) / "usage.jsonl"),
            review=False,
            review_model=None,
            codex_timeout=10,
            check_timeout=10,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def _available_commands(self, command):
        return {"codex": "codex", "python": "python", "git": "git"}.get(command)

    def test_failed_required_check_cannot_complete_or_call_jev(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)
            with patch("ai_orchestrate.cli.ensure_clean_git"), \
                 patch("ai_orchestrate.cli.shutil.which", side_effect=self._available_commands), \
                 patch("ai_orchestrate.cli.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.cli.run_codex", return_value=CodexResult(0)), \
                 patch("ai_orchestrate.cli.run_checks", return_value=[{"command": "python", "returncode": 1, "output": "failed"}]), \
                 redirect_stdout(StringIO()):
                self.assertEqual(_run(args), 2)

    def test_success_uses_one_coder_call_and_no_jev_or_reviewer_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp)
            result = CodexResult(0, CodexUsage(30, 20, 5), "Implemented.")
            with patch("ai_orchestrate.cli.ensure_clean_git"), \
                 patch("ai_orchestrate.cli.shutil.which", side_effect=self._available_commands), \
                 patch("ai_orchestrate.cli.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.cli.run_codex", return_value=result) as codex, \
                 patch("ai_orchestrate.cli.run_checks", return_value=[{"command": "python", "returncode": 0, "output": "ok"}]), \
                 redirect_stdout(StringIO()) as output:
                self.assertEqual(_run(args), 0)
            codex.assert_called_once()
            self.assertIn("total=35 tokens", output.getvalue())
            self.assertIn("COMPLETE", output.getvalue())
            entries = read_usage_entries(Path(args.usage_log))
            self.assertEqual(entries[0]["total_tokens"], 35)

    def test_failed_check_retries_once_at_next_lane_with_bounded_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp, max_attempts=2, prompt_token_budget=1000)
            results = [CodexResult(0, CodexUsage(20, 0, 5)), CodexResult(0, CodexUsage(25, 0, 4))]
            checks = [
                [{"command": "python", "returncode": 1, "output": "assertion failed"}],
                [{"command": "python", "returncode": 0, "output": "ok"}],
            ]
            with patch("ai_orchestrate.cli.ensure_clean_git"), \
                 patch("ai_orchestrate.cli.shutil.which", side_effect=self._available_commands), \
                 patch("ai_orchestrate.cli.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.cli.run_codex", side_effect=results) as codex, \
                 patch("ai_orchestrate.cli.run_checks", side_effect=checks), \
                 redirect_stdout(StringIO()) as output:
                self.assertEqual(_run(args), 0)
            self.assertEqual(codex.call_count, 2)
            self.assertIn("assertion failed", codex.call_args_list[1].args[1])
            self.assertEqual(codex.call_args_list[1].args[2], LANES["MEDIUM"][0])
            self.assertIn("escalating to MEDIUM", output.getvalue())

    def test_unknown_usage_stops_budgeted_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp, max_attempts=2, max_run_tokens=100)
            with patch("ai_orchestrate.cli.ensure_clean_git"), \
                 patch("ai_orchestrate.cli.shutil.which", side_effect=self._available_commands), \
                 patch("ai_orchestrate.cli.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.cli.run_codex", return_value=CodexResult(1)) as codex, \
                 patch("ai_orchestrate.cli.run_checks", return_value=[{"command": "python", "returncode": 1, "output": "failed"}]), \
                 redirect_stdout(StringIO()) as output:
                self.assertEqual(_run(args), 2)
            codex.assert_called_once()
            self.assertIn("did not report usage", output.getvalue())

    def test_reviewer_runs_read_only_as_separate_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp, review=True)
            results = [
                CodexResult(0, CodexUsage(20, 0, 5), "Coded."),
                CodexResult(0, CodexUsage(25, 0, 4), "PASS\nNo actionable issues."),
            ]
            with patch("ai_orchestrate.cli.ensure_clean_git"), \
                 patch("ai_orchestrate.cli.shutil.which", side_effect=self._available_commands), \
                 patch("ai_orchestrate.cli.route_with_lane", return_value=("SMALL", LANES["SMALL"])), \
                 patch("ai_orchestrate.cli.run_codex", side_effect=results) as codex, \
                 patch("ai_orchestrate.cli.run_checks", return_value=[{"command": "python", "returncode": 0, "output": "ok"}]), \
                 patch("ai_orchestrate.cli.git_snapshot", return_value=("diff", " M file.py")), \
                 redirect_stdout(StringIO()) as output:
                self.assertEqual(_run(args), 0)
            self.assertEqual(codex.call_count, 2)
            self.assertEqual(codex.call_args_list[1].kwargs["sandbox"], "read-only")
            self.assertIn("independent review found no actionable issue", output.getvalue())

    def test_daily_budget_stops_before_router_or_codex(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "usage.jsonl"
            append_usage(log, model="m", effort="low", role="coder", attempt=1, returncode=0,
                         usage=CodexUsage(60, 0, 40))
            args = self._args(tmp, usage_log=str(log), daily_token_budget=100)
            with patch("ai_orchestrate.cli.ensure_clean_git"), \
                 patch("ai_orchestrate.cli.shutil.which", side_effect=self._available_commands), \
                 patch("ai_orchestrate.cli.route_with_lane", side_effect=AssertionError), \
                 patch("ai_orchestrate.cli.run_codex", side_effect=AssertionError), \
                 redirect_stdout(StringIO()), self.assertRaisesRegex(OrchestratorError, "already exhausted"):
                _run(args)

    def test_prompt_budget_rejects_oversized_task_before_model_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(tmp, task="ж" * 1000, prompt_token_budget=10)
            with patch("ai_orchestrate.cli.ensure_clean_git"), \
                 patch("ai_orchestrate.cli.shutil.which", side_effect=self._available_commands), \
                 patch("ai_orchestrate.cli.run_codex", side_effect=AssertionError), \
                 redirect_stdout(StringIO()), self.assertRaisesRegex(OrchestratorError, "above --prompt-token-budget"):
                _run(args)


if __name__ == "__main__":
    unittest.main()
