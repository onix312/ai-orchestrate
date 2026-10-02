import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrate import projects


def _rollout(sessions_dir: Path, day: str, name: str, cwd: str, timestamp: str, branch: str = "main") -> Path:
    folder = sessions_dir / "2026" / "10" / day
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": timestamp,
        "type": "session_meta",
        "payload": {"id": name, "timestamp": timestamp, "cwd": cwd,
                    "originator": "codex_cli_rs", "git": {"branch": branch, "repository_url": ""}},
    }
    path = folder / name
    path.write_text(json.dumps(meta) + "\n" + json.dumps({"type": "turn.started"}) + "\n", encoding="utf-8")
    return path


class ProjectDiscoveryTests(unittest.TestCase):
    def test_codex_session_store_lists_projects_with_session_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex_home = root / "codex-home"
            inside = root / "workspace" / "app"
            inside.mkdir(parents=True)
            (inside / ".git").mkdir()
            outside = root / "elsewhere" / "tool"
            outside.mkdir(parents=True)
            sessions = codex_home / "sessions"
            _rollout(sessions, "01", "rollout-2026-10-01T09-00-00-a.jsonl", str(inside), "2026-10-01T09:00:00+03:00")
            _rollout(sessions, "02", "rollout-2026-10-02T09-00-00-b.jsonl", str(inside), "2026-10-02T09:00:00+03:00",
                     branch="feature/x")
            _rollout(sessions, "02", "rollout-2026-10-02T10-00-00-c.jsonl", str(outside), "2026-10-02T10:00:00+03:00")
            (sessions / "2026" / "10" / "02" / "broken.jsonl").write_text("not json\n", encoding="utf-8")

            with patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}):
                catalog = projects.discover_projects(root / "workspace")

            self.assertTrue(catalog["codex_sessions_found"])
            by_name = {item["name"]: item for item in catalog["projects"]}
            self.assertEqual(by_name["app"]["sessions"], 2)
            self.assertEqual(by_name["app"]["last_used"], "2026-10-02T09:00:00+03:00")
            self.assertEqual(by_name["app"]["branch"], "feature/x")
            self.assertTrue(by_name["app"]["selectable"])
            self.assertIn("codex", by_name["app"]["sources"])
            self.assertIn("workspace", by_name["app"]["sources"])
            # Known to Codex but outside the allowed workspace: visible, clearly not selectable.
            self.assertFalse(by_name["tool"]["selectable"])
            self.assertFalse(by_name["tool"]["inside_workspace"])

    def test_journal_entries_for_deleted_folders_are_not_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alive = root / "alive"
            alive.mkdir()
            journal = root / "journal.jsonl"
            journal.write_text(
                json.dumps({"repo": str(alive)}) + "\n" + json.dumps({"repo": str(root / "gone")}) + "\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": str(root / "codex-home")}):
                catalog = projects.discover_projects(root, journal_path=journal)
            names = [item["path"] for item in catalog["projects"]]
            self.assertIn(str(alive), names)
            self.assertNotIn(str(root / "gone"), names)

    def test_empty_session_store_is_reported_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"CODEX_HOME": str(root / "codex-home")}):
                catalog = projects.discover_projects(root)
            self.assertEqual(catalog["projects"], [])
            self.assertFalse(catalog["codex_sessions_found"])
            self.assertEqual(catalog["codex_sessions_dir"], str(root / "codex-home" / "sessions"))


if __name__ == "__main__":
    unittest.main()
