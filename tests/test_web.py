import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from ai_orchestrate.web import RunManager, make_handler


class WebTests(unittest.TestCase):
    def test_ui_and_api_serve_same_origin_and_reject_cross_origin_mutations(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            manager = RunManager(root, usage_path=root / "usage.jsonl")
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
            with urlopen(base + "/api/status", timeout=3) as response:
                status = response.read().decode("utf-8")
                self.assertIn(str(root.resolve()), status)
                self.assertIn('"professions"', status)
                self.assertIn('"role_prompts"', status)
                self.assertIn("senior code reviewer", status)
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
