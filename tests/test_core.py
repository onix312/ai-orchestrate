import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_orchestrate.core import (LANES, Decision, OrchestratorError, _codex_env, ensure_clean_git,
                                 jev_choice, next_lane, route, run_codex)
from ai_orchestrate.cli import _run


class CoreTests(unittest.TestCase):
    def test_dry_run_uses_explicit_lane_without_jev(self):
        with patch("ai_orchestrate.core.jev_choice", side_effect=AssertionError):
            self.assertEqual(route("task", lane="MEDIUM", dry_run=True), LANES["MEDIUM"])

    def test_dry_run_requires_explicit_valid_lane(self):
        with self.assertRaises(OrchestratorError):
            route("task", dry_run=True)
        with self.assertRaises(OrchestratorError):
            route("task", lane="unknown", dry_run=True)

    def test_jev_parses_choice_and_requires_schema(self):
        payload = {"answers": {"decision": {"choice": "SMALL", "confidence": .9,
                                             "probabilities": {"SMALL": .9, "MEDIUM": .1}}}}

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps(payload).encode()

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret"}), patch(
            "ai_orchestrate.core.urllib.request.urlopen", return_value=Response()
        ) as open_url:
            decision = jev_choice({}, "question", {"SMALL": "s", "MEDIUM": "m"})
        self.assertEqual(decision.choice, "SMALL")
        self.assertEqual(decision.confidence, .9)
        self.assertNotIn("secret", repr(open_url.call_args))

    def test_jev_missing_key_is_clear(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(OrchestratorError, "TYPESAFE_API_KEY"):
                jev_choice({}, "question", {"SMALL": "s"})

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
                jev_choice({}, "question", {"SMALL": "s"})
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret"}), patch(
            "ai_orchestrate.core.urllib.request.urlopen", side_effect=RuntimeError("secret response")
        ):
            with self.assertRaises(OrchestratorError) as raised:
                jev_choice({}, "question", {"SMALL": "s"})
        self.assertNotIn("secret response", str(raised.exception))

    def test_lanes_progress_then_escalate(self):
        self.assertEqual(next_lane(*LANES["SMALL"]), LANES["MEDIUM"])
        self.assertEqual(next_lane(*LANES["HIGH"]), LANES["ESCALATE"])
        self.assertEqual(next_lane(*LANES["ESCALATE"]), LANES["ESCALATE"])

    def test_codex_child_does_not_receive_jev_credentials(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "secret", "OPENROUTER_API_KEY": "also-secret"}):
            env = _codex_env()
        self.assertNotIn("TYPESAFE_API_KEY", env)
        self.assertNotIn("OPENROUTER_API_KEY", env)

        captured = {}
        class Process:
            returncode = 0
        def runner(command, **kwargs):
            captured.update(kwargs)
            captured["command"] = command
            return Process()
        self.assertEqual(run_codex(Path("."), "do work", "gpt-6-luna", "low", runner=runner), 0)
        self.assertEqual(captured["input"], "do work")
        self.assertEqual(captured["command"][-1], "-")
        self.assertNotIn("TYPESAFE_API_KEY", captured["env"])

    def test_clean_git_gate_includes_untracked(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            import subprocess
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "--allow-empty", "-m", "initial"], cwd=repo, check=True,
                           stdout=subprocess.DEVNULL)
            ensure_clean_git(repo)
            (repo / "new.txt").write_text("untracked")
            with self.assertRaisesRegex(OrchestratorError, "including untracked"):
                ensure_clean_git(repo)

    def test_failed_required_check_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(repo=tmp, check=["python"], task="do task", dry_run=False,
                                   lane=None, max_attempts=1, codex_timeout=10, check_timeout=10)
            with patch("ai_orchestrate.cli.ensure_clean_git"), \
                 patch("ai_orchestrate.cli.route", return_value=LANES["SMALL"]), \
                 patch("ai_orchestrate.cli.run_codex", return_value=0), \
                 patch("ai_orchestrate.cli.git_snapshot", return_value=("diff", " M file")), \
                 patch("ai_orchestrate.cli.run_checks", return_value=[{"command": "python", "returncode": 1, "output": "failed"}]), \
                 patch("ai_orchestrate.cli.jev_choice") as jev, redirect_stdout(StringIO()):
                self.assertEqual(_run(args), 2)
            jev.assert_not_called()

    def test_low_confidence_jev_complete_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(repo=tmp, check=["python"], task="do task", dry_run=False,
                                   lane=None, max_attempts=1, codex_timeout=10, check_timeout=10)
            with patch("ai_orchestrate.cli.ensure_clean_git"), \
                 patch("ai_orchestrate.cli.route", return_value=LANES["SMALL"]), \
                 patch("ai_orchestrate.cli.run_codex", return_value=0), \
                 patch("ai_orchestrate.cli.git_snapshot", return_value=("diff", " M file")), \
                 patch("ai_orchestrate.cli.run_checks", return_value=[{"command": "python", "returncode": 0, "output": "ok"}]), \
                 patch("ai_orchestrate.cli.jev_choice", return_value=Decision("COMPLETE", .5)), \
                 redirect_stdout(StringIO()):
                self.assertEqual(_run(args), 2)


if __name__ == "__main__":
    unittest.main()
