"""Тесты ручного моста: ожидание ответа и сборка контекста для обычного ChatGPT."""

import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ai_orchestrate.core import OrchestratorError
from ai_orchestrate.relay import (
    CHATGPT_URL,
    ManualRelay,
    RelayStopped,
    build_manual_prompt,
    chatgpt_link,
    list_worktree_files,
    pack_repository_context,
)


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True,
                   stdout=subprocess.DEVNULL)
    (path / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (path / "README.md").write_text("# Project\n", encoding="utf-8")
    (path / "app_test.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "initial"], cwd=path, check=True, stdout=subprocess.DEVNULL)
    return path


class ManualRelayTests(unittest.TestCase):
    def test_answer_round_trip_publishes_prompt_and_history(self):
        events = []
        relay = ManualRelay(on_event=events.append, timeout=30)
        seen = {}

        def worker():
            seen["answer"] = relay.request(kind="code", stage="implementation", role="Разработчик",
                                           title="Правки кода", instructions="Верни файлы.",
                                           prompt="Ты — инженер. Верни файлы.")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not relay.waiting() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(relay.waiting())
        state = relay.public()
        self.assertTrue(state["waiting"])
        self.assertEqual(state["request"]["kind"], "code")
        self.assertEqual(state["request"]["chars"], len("Ты — инженер. Верни файлы."))
        self.assertTrue(state["request"]["url"].startswith(CHATGPT_URL))
        self.assertTrue(state["request"]["prompt_in_url"])
        self.assertTrue(relay.deliver("### FILE: app.py\n```\nprint('hello')\n```\n"))
        thread.join(timeout=5)
        self.assertEqual(seen["answer"], "### FILE: app.py\n```\nprint('hello')\n```\n")
        self.assertFalse(relay.waiting())
        self.assertFalse(relay.deliver("ещё ответ"))
        self.assertEqual(relay.public()["answered"], 1)
        self.assertEqual([event["event"] for event in events], ["manual.requested", "manual.answered"])

    def test_cancel_and_timeout_release_the_worker_thread(self):
        relay = ManualRelay(timeout=30)
        result = {}

        def worker():
            try:
                relay.request(kind="plan", stage="planning", role="Аналитик", title="План",
                              instructions="Верни план.", prompt="План, пожалуйста.")
            except RelayStopped as exc:
                result["error"] = str(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not relay.waiting() and time.monotonic() < deadline:
            time.sleep(0.01)
        relay.cancel()
        thread.join(timeout=5)
        self.assertIn("остановлено", result["error"])

        expired = ManualRelay(timeout=30)
        expired.timeout = 0.3
        with self.assertRaises(RelayStopped):
            expired.request(kind="review", stage="review", role="Ревьюер", title="Ревью",
                            instructions="Проверь.", prompt="Проверь diff.")

    def test_empty_prompt_and_empty_answer_are_rejected(self):
        relay = ManualRelay()
        with self.assertRaisesRegex(OrchestratorError, "пуст"):
            relay.request(kind="plan", stage="planning", role="Аналитик", title="План",
                          instructions="", prompt="   ")
        with self.assertRaisesRegex(OrchestratorError, "пустой"):
            relay.deliver("   ")

    def test_long_prompt_stays_out_of_the_link(self):
        short = chatgpt_link("План")
        self.assertTrue(short["prompt_in_url"])
        self.assertIn("q=", short["url"])
        long_prompt = chatgpt_link("x" * 20_000)
        self.assertFalse(long_prompt["prompt_in_url"])
        self.assertEqual(long_prompt["url"], CHATGPT_URL)


class ContextPackingTests(unittest.TestCase):
    def test_context_includes_tree_and_source_files_but_never_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _git_repo(Path(tmp) / "project")
            (repo / ".env").write_text("OPENAI_API_KEY=sk-live-secret\n", encoding="utf-8")
            (repo / "deploy.key").write_text("private\n", encoding="utf-8")
            (repo / "logo.png").write_bytes(b"\x89PNG\x00binary")
            (repo / "src").mkdir()
            (repo / "src" / "settings.py").write_text("SETTINGS = {'dark': True}\n", encoding="utf-8")
            packed = pack_repository_context(repo, task="Добавь тёмную тему в settings.py")
            self.assertIn("app.py", packed["text"])
            self.assertIn("src/settings.py", packed["text"])
            self.assertIn("SETTINGS", packed["text"])
            self.assertNotIn("sk-live-secret", packed["text"])
            self.assertNotIn("deploy.key", packed["text"])
            self.assertNotIn("logo.png", packed["text"])
            # Имена секретных файлов тоже не уходят во внешний чат.
            self.assertNotIn(".env", packed["text"])
            self.assertGreaterEqual(packed["total_files"], 4)
            self.assertEqual(packed["files"][0], "src/settings.py")

    def test_pack_ignores_dependency_folders_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _git_repo(Path(tmp) / "project")
            modules = repo / "node_modules" / "left-pad"
            modules.mkdir(parents=True)
            (modules / "index.js").write_text("module.exports = null\n", encoding="utf-8")
            first = pack_repository_context(repo, task="Исправь app.py и тесты")
            second = pack_repository_context(repo, task="Исправь app.py и тесты")
            self.assertEqual(first["text"], second["text"])
            self.assertNotIn("left-pad", first["text"])
            self.assertEqual(first["files"][0], "app.py")

    def test_tight_budget_limits_the_number_of_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _git_repo(Path(tmp) / "project")
            packed = pack_repository_context(repo, task="app.py", max_files=1, budget_chars=200)
            self.assertEqual(len(packed["files"]), 1)
            self.assertLessEqual(len(packed["text"]), 400)

    def test_worktree_listing_prefers_git_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = _git_repo(Path(tmp) / "project")
            (repo / "untracked.txt").write_text("new\n", encoding="utf-8")
            (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
            (repo / "ignored.txt").write_text("ignored\n", encoding="utf-8")
            names = list_worktree_files(repo)
            self.assertIn("untracked.txt", names)
            self.assertNotIn("ignored.txt", names)
            self.assertNotIn(".gitignore", [name for name in names if name.startswith(".git/")])

    def test_manual_prompt_contains_context_role_and_answer_format(self):
        prompt = build_manual_prompt(
            "РОЛЬ — разработчик. Реализуй задачу.",
            kind="code", context_text="### Файл: app.py", context_files=["app.py"],
            check_commands=["pytest -q"], extra_note="Верни файлы целиком.",
        )
        self.assertIn("РОЛЬ — разработчик", prompt)
        self.assertIn("app.py", prompt)
        self.assertIn("pytest -q", prompt)
        self.assertIn("*** Begin Patch", prompt)
        review = build_manual_prompt("РОЛЬ — ревьюер.", kind="review")
        self.assertIn("PASS", review)
        plan = build_manual_prompt("РОЛЬ — аналитик.", kind="plan")
        self.assertIn("ЦЕЛЬ И КРИТЕРИИ", plan)


if __name__ == "__main__":
    unittest.main()
