import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ai_orchestrate.settings import default_settings
from ai_orchestrate.web import RunJob, RunManager, RunSubmission, make_handler


class WebTests(unittest.TestCase):
    def test_custom_journal_path_does_not_change_parent_permissions_and_keeps_file_private(self):
        if os.name == "nt":
            self.skipTest("POSIX permission bits are not portable to Windows")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "project"
            repo.mkdir()
            shared = root / "shared"
            shared.mkdir(mode=0o755)
            shared.chmod(0o755)
            journal_file = shared / "runs.jsonl"
            settings = default_settings()
            settings["journal_path"] = str(journal_file)
            manager = RunManager(root, settings_path=root / "settings.json")
            submission = RunSubmission(repo, "task", [], settings)
            manager._record_journal(RunJob("test-job", submission, status="complete"))
            self.assertEqual(shared.stat().st_mode & 0o777, 0o755)
            self.assertEqual(journal_file.stat().st_mode & 0o777, 0o600)

    def test_project_context_is_persisted_separately_per_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            manager = RunManager(root, settings_path=root / "settings.json")
            manager.save_settings({"default_repo": str(first), "project_context": "First repo rules."})
            manager.save_settings({"default_repo": str(second), "project_context": "Second repo rules."})
            contexts = manager.settings_store.load()["project_contexts"]
            self.assertEqual(contexts[str(first.resolve())], "First repo rules.")
            self.assertEqual(contexts[str(second.resolve())], "Second repo rules.")

    def test_ui_and_api_serve_same_origin_and_reject_cross_origin_mutations(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            manager = RunManager(root, usage_path=root / "usage.jsonl", settings_path=root / "settings.json",
                                 journal_path=root / "journal.jsonl")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(manager))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(thread.join, 2)
            self.addCleanup(server.shutdown)
            base = f"http://127.0.0.1:{server.server_port}"

            with urlopen(base + "/", timeout=3) as response:
                page = response.read().decode("utf-8")
                self.assertEqual(response.status, 200)
                self.assertIn("Расход до запуска", page)
                self.assertIn("Jev дирижирует, Codex играет", page)
                self.assertIn('data-scene="idle"', page)
                self.assertIn("router.jev.completed", page)
                self.assertIn("confirmMergeButton", page)
                self.assertIn("usageLogPath", page)
                self.assertIn("journalPath", page)
                self.assertIn("Сохранить настройки", page)
                self.assertIn("githubItem", page)
                self.assertIn("Автоматически после финального APPROVE от Jev", page)
            with urlopen(base + "/api/status", timeout=3) as response:
                status = response.read().decode("utf-8")
                self.assertIn(str(root.resolve()), status)
                self.assertIn('"professions"', status)
                self.assertIn('"role_prompts"', status)
                self.assertIn("senior code reviewer", status)
            with urlopen(base + "/api/settings", timeout=3) as response:
                self.assertIn('"merge_policy": "confirm"', response.read().decode("utf-8"))
            usage_file = root / "logs" / "tokens.jsonl"
            journal_file = root / "logs" / "runs.jsonl"
            save_body = json.dumps({"settings": {"mode": "quick", "default_repo": str(project),
                                                    "usage_log_path": str(usage_file), "journal_path": str(journal_file),
                                                    "project_context": "Use the current project conventions."}}).encode()
            request = Request(base + "/api/settings", data=save_body, method="POST",
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=3) as response:
                self.assertEqual(response.status, 200)
                self.assertIn('"saved": true', response.read().decode("utf-8"))
            settings_json = (root / "settings.json").read_text(encoding="utf-8")
            self.assertIn('"mode": "quick"', settings_json)
            self.assertIn(str(usage_file.resolve()), settings_json)
            self.assertIn(str(journal_file.resolve()), settings_json)
            with urlopen(base + "/api/status", timeout=3) as response:
                status = json.loads(response.read().decode("utf-8"))
                self.assertEqual(status["usage_file"], str(usage_file.resolve()))
                self.assertEqual(status["journal_file"], str(journal_file.resolve()))
            with urlopen(base + "/api/project-context?repo=" + quote(str(project)), timeout=3) as response:
                self.assertIn("Use the current project conventions", response.read().decode("utf-8"))

            with urlopen(base + "/api/checks?repo=" + quote(str(project)), timeout=3) as response:
                self.assertIn('"checks": []', response.read().decode("utf-8"))

            request = Request(
                base + "/api/runs", data=b"{}", method="POST",
                headers={"Content-Type": "application/json", "Origin": "https://attacker.invalid"},
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=3)
            self.assertEqual(raised.exception.code, 403)

            request = Request(base + "/api/checks?repo=" + quote(outside), method="GET")
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=3)
            self.assertEqual(raised.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
