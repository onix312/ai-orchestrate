"""Tests for the OpenAI-compatible executor and the ChatGPT bridge."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from ai_orchestrate import secrets
from ai_orchestrate.chatgpt_bridge import apply_operations, build_bridge_prompt, parse_answer
from ai_orchestrate.core import OrchestratorError
from ai_orchestrate.llm_api import ApiConfig, READ_ONLY_TOOLS, run_llm_api, safe_join
from ai_orchestrate.workflow import WorkflowRequest, _prepare_request, _validate_request


class FakeOpenAIServer:
    """Minimal /v1/chat/completions endpoint that drives the executor with tool calls."""

    def __init__(self, plan):
        self.plan = plan
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(204)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                outer.requests.append(body)
                status, payload = outer.plan(len(outer.requests), body)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def tool_message(calls):
    return {"choices": [{"message": {"content": None, "tool_calls": calls}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30,
                      "prompt_tokens_details": {"cached_tokens": 10}}}


def call(name, arguments):
    return {"id": f"call-{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def finish(summary="Готово."):
    return tool_message([call("finish", {"summary": summary})])


class LlmApiExecutorTests(unittest.TestCase):
    def test_agent_loop_writes_file_runs_command_and_reports_usage(self):
        def plan(index, body):
            if index == 1:
                return 200, tool_message([call("write_file", {"path": "hello.txt", "content": "42\n"})])
            if index == 2:
                return 200, tool_message([call("run_command", {"command": "cat hello.txt"})])
            return 200, finish("Файл создан.")

        server = FakeOpenAIServer(plan)
        self.addCleanup(server.stop)
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            result = run_llm_api(
                repo, "Создай hello.txt", "fake-model",
                config=ApiConfig(base_url=server.base_url, api_key="sk-test", timeout=60, max_rounds=6),
                on_event=events.append,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((repo / "hello.txt").read_text(encoding="utf-8"), "42\n")
            self.assertEqual(result.final_message, "Файл создан.")
            # Three rounds of 100/30 with 10 cached tokens each.
            self.assertEqual((result.usage.input_tokens, result.usage.cached_input_tokens,
                              result.usage.output_tokens, result.usage.total_tokens), (300, 30, 90, 390))
        kinds = [event.get("type") for event in events]
        self.assertIn("item.completed", kinds)
        self.assertIn("turn.completed", kinds)
        # The payload must stay OpenAI-shaped and name the routed model.
        self.assertEqual(server.requests[0]["model"], "fake-model")
        self.assertIn("tools", server.requests[0])

    def test_read_only_sandbox_never_exposes_write_tools(self):
        server = FakeOpenAIServer(lambda index, body: (200, finish()))
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            run_llm_api(Path(tmp), "Посмотри проект", "fake-model",
                        config=ApiConfig(base_url=server.base_url, api_key="k", timeout=30),
                        sandbox="read-only")
        offered = {tool["function"]["name"] for tool in server.requests[0]["tools"]}
        self.assertTrue(offered.issubset(READ_ONLY_TOOLS | {"finish"}), offered)
        self.assertNotIn("write_file", offered)
        self.assertNotIn("str_replace", offered)

    def test_write_tool_refuses_paths_outside_the_worktree(self):
        def plan(index, body):
            if index == 1:
                return 200, tool_message([call("write_file", {"path": "../escaped.txt", "content": "x"})])
            return 200, finish()

        server = FakeOpenAIServer(plan)
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            result = run_llm_api(repo, "Задача", "fake-model",
                                 config=ApiConfig(base_url=server.base_url, api_key="k", timeout=30))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((repo.parent / "escaped.txt").exists())

    def test_dangerous_command_is_refused_without_running(self):
        def plan(index, body):
            if index == 1:
                return 200, tool_message([call("run_command", {"command": "rm -rf / --no-preserve-root"})])
            return 200, finish()

        server = FakeOpenAIServer(plan)
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            run_llm_api(Path(tmp), "Задача", "fake-model",
                        config=ApiConfig(base_url=server.base_url, api_key="k", timeout=30))
        # The refusal is reported back to the model instead of being executed.
        self.assertEqual(len(server.requests), 2)

    def test_http_errors_become_readable_orchestrator_errors(self):
        server = FakeOpenAIServer(lambda index, body: (
            401, {"error": {"message": "Incorrect API key provided"}}))
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OrchestratorError) as caught:
                run_llm_api(Path(tmp), "Задача", "fake-model",
                            config=ApiConfig(base_url=server.base_url, api_key="bad", timeout=30))
        message = str(caught.exception)
        self.assertIn("401", message)
        self.assertIn("ключ", message.lower())

    def test_round_limit_stops_the_loop(self):
        server = FakeOpenAIServer(lambda index, body: (
            200, tool_message([call("list_dir", {"path": "."})])))
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_llm_api(Path(tmp), "Задача", "fake-model",
                                 config=ApiConfig(base_url=server.base_url, api_key="k",
                                                  timeout=60, max_rounds=2))
        self.assertEqual(len(server.requests), 2)
        self.assertNotEqual(result.returncode, 0)


class SafeJoinTests(unittest.TestCase):
    def test_absolute_paths_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for raw in ("/etc/passwd", "../outside.txt", "sub/../../outside.txt", ""):
                with self.assertRaises(OrchestratorError, msg=raw):
                    safe_join(root, raw)
            self.assertEqual(safe_join(root, "a/b.txt"), root / "a" / "b.txt")


class ChatGptBridgeTests(unittest.TestCase):
    ANSWER = """Вот исправления.

### FILE: src/app.py
```python
print("fixed")
```

```
*** Begin Patch
*** Update File: src/util.py
@@
 def total(items):
-    return sum(i.price for i in items)
+    return round(sum(i.price for i in items), 2)
*** Add File: src/notes.txt
+заметка
*** Delete File: src/dead.py
*** End Patch
```
"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "util.py").write_text(
            "def total(items):\n    return sum(i.price for i in items)\n", encoding="utf-8")
        (self.root / "src" / "dead.py").write_text("x = 1\n", encoding="utf-8")

    def test_full_file_patch_add_and_delete_are_applied(self):
        report = apply_operations(self.root, parse_answer(self.ANSWER))
        self.assertEqual(report.rejected, [])
        self.assertEqual([item["path"] for item in report.applied],
                         ["src/app.py", "src/util.py", "src/notes.txt", "src/dead.py"])
        self.assertIn('print("fixed")', (self.root / "src" / "app.py").read_text(encoding="utf-8"))
        self.assertEqual((self.root / "src" / "util.py").read_text(encoding="utf-8"),
                         "def total(items):\n    return round(sum(i.price for i in items), 2)\n")
        self.assertEqual((self.root / "src" / "notes.txt").read_text(encoding="utf-8"), "заметка\n")
        self.assertFalse((self.root / "src" / "dead.py").exists())

    def test_answer_cannot_write_outside_the_worktree(self):
        report = apply_operations(self.root, parse_answer("### FILE: ../escaped.txt\n```\npwned\n```\n"))
        self.assertEqual(report.applied, [])
        self.assertIn("за пределы", report.rejected[0]["error"])
        self.assertFalse((self.root.parent / "escaped.txt").exists())

    def test_patch_with_wrong_context_is_rejected_instead_of_applied(self):
        report = apply_operations(
            self.root, parse_answer("*** Begin Patch\n*** Update File: src/util.py\n@@\n-nothing\n+x\n*** End Patch\n"))
        self.assertEqual(report.applied, [])
        self.assertIn("Контекст патча", report.rejected[0]["error"])

    def test_answer_without_file_blocks_is_rejected(self):
        with self.assertRaises(OrchestratorError):
            parse_answer("Просто перепиши функцию total, там не хватает round.")

    def test_prompt_contains_task_diff_and_failing_checks(self):
        prompt = build_bridge_prompt(
            "Починить формат цены", "diff --git a/src/util.py b/src/util.py", " M src/util.py",
            [{"command": "pytest -q", "returncode": 1, "output": "FAILED test_price"},
             {"command": "ruff check", "returncode": 0, "output": "All checks passed"}],
            plan="Добавить round(..., 2)")
        self.assertIn("Починить формат цены", prompt)
        self.assertIn("FAILED test_price", prompt)
        self.assertIn("Добавить round(..., 2)", prompt)
        self.assertIn("*** Begin Patch", prompt)
        self.assertIn("### FILE:", prompt)
        self.assertNotIn("All checks passed", prompt)  # passing checks stay out of the prompt


class ExecutorValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        self._home = tempfile.TemporaryDirectory()
        self.addCleanup(self._home.cleanup)
        patcher = patch.dict("os.environ", {"AI_ORCHESTRATE_HOME": self._home.name}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("OPENAI_API_KEY", None)
        secrets.reset_activation_state()
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "init"],
                       cwd=self.repo, check=True, env={**os.environ, "GIT_AUTHOR_NAME": "t",
                                                       "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                                                       "GIT_COMMITTER_EMAIL": "t@t"})

    def _request(self, **kwargs):
        base = {"repo": self.repo, "task": "Задача", "checks": ["echo ok"], "executor": "api",
                "api_model": "llama3.1"}
        base.update(kwargs)
        return WorkflowRequest(**base)

    def test_api_executor_forces_every_lane_onto_the_same_model(self):
        prepared = _prepare_request(self._request(api_model="qwen2.5-coder", review_model=""))
        self.assertEqual((prepared.luna_model, prepared.sol_model, prepared.review_model),
                         ("qwen2.5-coder",) * 3)

    def test_api_executor_needs_a_model_and_a_key(self):
        with self.assertRaises(OrchestratorError) as missing_model:
            _validate_request(self._request(api_model=""))
        self.assertIn("имя модели", str(missing_model.exception))
        with self.assertRaises(OrchestratorError) as missing_key:
            _validate_request(self._request(api_base_url="https://api.openai.com/v1"))
        self.assertIn("ключ", str(missing_key.exception).lower())

    def test_local_endpoint_is_allowed_without_a_key(self):
        _validate_request(self._request(api_base_url="http://127.0.0.1:11434/v1"))

    def test_stored_key_unlocks_a_remote_endpoint(self):
        secrets.save_key("openai", "sk-abcdefghij123456")
        _validate_request(self._request(api_base_url="https://api.openai.com/v1"))

    def test_codex_executor_still_requires_the_cli(self):
        with patch("ai_orchestrate.workflow.shutil.which", return_value=None):
            with self.assertRaises(OrchestratorError) as caught:
                _validate_request(self._request(executor="codex"))
        self.assertIn("Codex CLI", str(caught.exception))

    def test_unknown_executor_is_rejected(self):
        with self.assertRaises(OrchestratorError):
            _validate_request(self._request(executor="chatgpt-app"))



if __name__ == "__main__":
    unittest.main()
