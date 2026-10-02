import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_orchestrate.core import OrchestratorError
from ai_orchestrate.github import (
    GitHubRepository,
    publish_and_merge,
    resolve_item,
    task_with_github_context,
)


def completed(args, stdout="", stderr="", code=0):
    return subprocess.CompletedProcess(args, code, stdout, stderr)


class GitHubTests(unittest.TestCase):
    def test_issue_url_loads_context_and_marks_content_untrusted(self):
        repo_info = {"nameWithOwner": "example/project", "url": "https://github.com/example/project",
                     "defaultBranchRef": {"name": "main"}}
        issue = {"number": 27, "title": "Add export", "body": "Please expose CSV export.",
                 "state": "OPEN", "url": "https://github.com/example/project/issues/27"}
        with tempfile.TemporaryDirectory() as tmp, patch("ai_orchestrate.github.shutil.which", return_value="gh"), \
             patch("ai_orchestrate.github.subprocess.run", side_effect=[
                 completed([], json.dumps(repo_info)), completed([], json.dumps(issue))
             ]) as run:
            item = resolve_item(tmp, "https://github.com/example/project/issues/27")
        self.assertEqual(item.kind, "issue")
        self.assertEqual(item.number, 27)
        prompt = task_with_github_context("Implement the issue", item)
        self.assertIn("<github_context>", prompt)
        self.assertIn("Please expose CSV export", prompt)
        self.assertIn("недоверенные данные", prompt)
        self.assertEqual(run.call_count, 2)

    def test_fork_pull_request_is_rejected_before_automatic_write(self):
        repo_info = {"nameWithOwner": "example/project", "url": "https://github.com/example/project",
                     "defaultBranchRef": {"name": "main"}}
        pr = {"number": 7, "title": "External patch", "body": "body", "state": "OPEN",
              "url": "https://github.com/example/project/pull/7", "baseRefName": "main",
              "headRefName": "patch", "headRefOid": "deadbeef",
              "headRepository": {"name": "project", "owner": {"login": "contributor"}},
              "headRepositoryOwner": {"login": "contributor"}}
        with tempfile.TemporaryDirectory() as tmp, patch("ai_orchestrate.github.shutil.which", return_value="gh"), \
             patch("ai_orchestrate.github.subprocess.run", side_effect=[
                 completed([], json.dumps(repo_info)), completed([], json.dumps(pr))
             ]):
            with self.assertRaisesRegex(OrchestratorError, "из fork"):
                resolve_item(tmp, "https://github.com/example/project/pull/7")

    def test_repeat_after_remote_merge_is_idempotent_and_only_then_deletes_remote_branch(self):
        repo = GitHubRepository("example/project", "https://github.com/example/project", "main")
        merged_pr = {"number": 45, "url": "https://github.com/example/project/pull/45", "state": "MERGED",
                     "mergedAt": "2026-10-02T10:00:00Z", "autoMergeRequest": None,
                     "baseRefName": "main", "headRefName": "agent/job-1", "headRefOid": "abc"}
        with patch("ai_orchestrate.github._gh", return_value=completed([], json.dumps(merged_pr))) as gh_call, \
             patch("ai_orchestrate.github.git", return_value=completed([], "")) as git_call:
            result = publish_and_merge(
                "/tmp/repo", repository=repo, branch="agent/job-1", base_branch="main",
                task="Implement export", checks=["pytest -q"], item=None,
                merge_method="squash", wait_for_checks=True, delete_branch=True,
            )
        self.assertEqual(result["status"], "merged")
        self.assertTrue(result["remote_branch_deleted"])
        self.assertEqual(gh_call.call_count, 1)
        self.assertEqual(gh_call.call_args.args[1][:3], ["pr", "view", "agent/job-1"])
        git_call.assert_called_once()
        self.assertEqual(git_call.call_args.args[1], ["push", "origin", "--delete", "agent/job-1"])

    def test_confirmed_publish_uses_native_auto_merge_and_does_not_bypass_checks(self):
        repo = GitHubRepository("example/project", "https://github.com/example/project", "main")
        calls = []

        def gh_mock(repo_path, args, **kwargs):
            calls.append(args)
            if args[:3] == ["pr", "view", "agent/job-1"]:
                return completed(args, "", "not found", 1)
            if args[:2] == ["pr", "create"]:
                return completed(args, "https://github.com/example/project/pull/45\n")
            if args[:3] == ["pr", "view", "https://github.com/example/project/pull/45"]:
                return completed(args, json.dumps({"number": 45, "url": "https://github.com/example/project/pull/45",
                                                   "state": "OPEN", "baseRefName": "main",
                                                   "headRefName": "agent/job-1", "headRefOid": "abc"}))
            if args[:2] == ["pr", "merge"]:
                return completed(args, "auto merge enabled")
            if args[:3] == ["pr", "view", "45"]:
                return completed(args, json.dumps({"number": 45, "url": "https://github.com/example/project/pull/45",
                                                   "state": "OPEN", "mergedAt": None,
                                                   "autoMergeRequest": {"enabledAt": "now"}}))
            raise AssertionError(args)

        with patch("ai_orchestrate.github._gh", side_effect=gh_mock), \
             patch("ai_orchestrate.github.git", return_value=completed([], "")) as git_call:
            result = publish_and_merge(
                "/tmp/repo", repository=repo, branch="agent/job-1", base_branch="main",
                task="Implement export", checks=["pytest -q"], item=None,
                merge_method="squash", wait_for_checks=True, delete_branch=True,
            )
        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["number"], 45)
        merge_call = next(args for args in calls if args[:2] == ["pr", "merge"])
        self.assertIn("--auto", merge_call)
        self.assertIn("--squash", merge_call)
        self.assertNotIn("--delete-branch", merge_call)
        self.assertNotIn("--admin", merge_call)
        self.assertTrue(result["branch_cleanup_pending"])
        git_call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
