import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_orchestrate.core import OrchestratorError
from ai_orchestrate.gitops import (
    branch_sha,
    commit_worktree,
    create_worktree,
    current_branch,
    git,
    merge_local,
    remove_worktree,
)


class GitOpsTests(unittest.TestCase):
    def _repo(self, root: Path) -> Path:
        root.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Agent Test"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "agent@example.invalid"], cwd=root, check=True)
        (root / "README.md").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=root, check=True)
        return root

    def test_worktree_is_isolated_and_local_merge_is_fast_forward_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            base_sha = branch_sha(repo, "main")
            worktree = create_worktree(
                repo, root / "worktrees", job_id="abc123", branch_prefix="agent",
                base_ref="refs/heads/main", base_branch="main",
            )
            self.assertEqual(current_branch(repo), "main")
            self.assertEqual(git(repo, ["rev-parse", "HEAD"]).stdout.strip(), base_sha)
            self.assertNotEqual(worktree.path, repo)
            (worktree.path / "feature.txt").write_text("isolated feature\n", encoding="utf-8")
            commit = commit_worktree(worktree, "ai: add feature")
            self.assertTrue(commit["committed"])
            self.assertIn("feature.txt", commit["files"])
            merged_sha = merge_local(worktree)
            self.assertEqual(git(repo, ["rev-parse", "HEAD"]).stdout.strip(), merged_sha)
            self.assertTrue((repo / "feature.txt").exists())
            remove_worktree(worktree, delete_branch=True)
            self.assertFalse(worktree.path.exists())
            self.assertNotEqual(git(repo, ["show-ref", "--verify", f"refs/heads/{worktree.branch}"], check=False).returncode, 0)

    def test_sensitive_file_is_never_auto_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            worktree = create_worktree(
                repo, root / "worktrees", job_id="secret1", branch_prefix="agent",
                base_ref="refs/heads/main", base_branch="main",
            )
            (worktree.path / ".env.local").write_text("TOKEN=do-not-commit\n", encoding="utf-8")
            with self.assertRaisesRegex(OrchestratorError, "секретные файлы"):
                commit_worktree(worktree, "unsafe")
            self.assertEqual(git(worktree.path, ["log", "-1", "--format=%s"]).stdout.strip(), "initial")
            remove_worktree(worktree, delete_branch=True, force=True)

    def test_merge_refuses_if_user_changes_base_branch_while_job_waits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root / "project")
            worktree = create_worktree(
                repo, root / "worktrees", job_id="stale1", branch_prefix="agent",
                base_ref="refs/heads/main", base_branch="main",
            )
            (worktree.path / "feature.txt").write_text("feature\n", encoding="utf-8")
            commit_worktree(worktree, "feature")
            (repo / "later.txt").write_text("new base commit\n", encoding="utf-8")
            subprocess.run(["git", "add", "later.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "user update"], cwd=repo, check=True)
            with self.assertRaisesRegex(OrchestratorError, "изменилась после запуска"):
                merge_local(worktree)
            remove_worktree(worktree, delete_branch=True, force=True)


if __name__ == "__main__":
    unittest.main()
