"""Discover the projects a user already works with.

Sources, most useful first:

1. the Codex CLI session store — ``<CODEX_HOME or ~/.codex>/sessions/YYYY/MM/DD/rollout-*.jsonl``.
   Each rollout opens with a ``session_meta`` record that carries the working directory, so the
   folders where Codex has actually been used can be listed without running Codex;
2. the panel's own journal and settings (repositories used from this panel);
3. a shallow scan of the allowed workspace root for Git repositories.

The Codex transcript format is not a published contract, so every field is optional and any row
that does not parse is skipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_ROLLOUT_FILES = 600
MAX_ROLLOUT_PREFIX_BYTES = 65_536
MAX_WALK_DEPTH = 4


def codex_home() -> Path:
    override = os.environ.get("CODEX_HOME")
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return (Path.home() / ".codex").resolve(strict=False)


def codex_sessions_dir() -> Path:
    return codex_home() / "sessions"


def _walk_rollout_files(root: Path, *, max_files: int = MAX_ROLLOUT_FILES) -> list[Path]:
    """Depth-limited, newest-first walk of the date-partitioned rollout store."""
    found: list[tuple[float, Path]] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if depth + 1 <= MAX_WALK_DEPTH:
                                stack.append((Path(entry.path), depth + 1))
                        elif entry.name.startswith("rollout-") and entry.name.endswith(".jsonl"):
                            found.append((entry.stat().st_mtime, Path(entry.path)))
                    except OSError:
                        continue
        except OSError:
            continue
    found.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in found[:max_files]]


def _session_meta(path: Path) -> dict[str, Any]:
    """Read only a bounded prefix of a rollout file and return its session_meta payload."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            head = source.read(MAX_ROLLOUT_PREFIX_BYTES)
    except OSError:
        return {}
    for line in head.splitlines()[:20]:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = record
        kind = record.get("type") or payload.get("type")
        if "cwd" not in payload and kind != "session_meta":
            continue
        return payload
    return {}


def _as_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).isoformat(timespec="seconds")
    except ValueError:
        return value.strip()[:32]


@dataclass
class ProjectCandidate:
    path: str
    name: str = ""
    sources: list[str] = field(default_factory=list)
    sessions: int = 0
    last_used: str = ""
    branch: str = ""
    repository_url: str = ""
    exists: bool = False
    is_git: bool = False
    inside_workspace: bool = False

    @property
    def selectable(self) -> bool:
        return self.inside_workspace and self.exists

    def public(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "sources": sorted(set(self.sources)),
            "sessions": self.sessions,
            "last_used": self.last_used,
            "branch": self.branch,
            "repository_url": self.repository_url,
            "exists": self.exists,
            "is_git": self.is_git,
            "inside_workspace": self.inside_workspace,
            "selectable": self.selectable,
        }


def codex_projects(*, max_files: int = MAX_ROLLOUT_FILES) -> dict[str, dict[str, Any]]:
    """Aggregate Codex rollouts per working directory. Returns ``{path: info}``."""
    sessions_dir = codex_sessions_dir()
    aggregated: dict[str, dict[str, Any]] = {}
    if not sessions_dir.is_dir():
        return aggregated
    for rollout in _walk_rollout_files(sessions_dir, max_files=max_files):
        payload = _session_meta(rollout)
        cwd = payload.get("cwd")
        if not isinstance(cwd, str) or not cwd.strip():
            continue
        try:
            resolved = str(Path(cwd.strip()).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            continue
        timestamp = _as_timestamp(payload.get("timestamp") or payload.get("started_at"))
        git_info = payload.get("git") if isinstance(payload.get("git"), dict) else {}
        entry = aggregated.setdefault(resolved, {"sessions": 0, "last_used": "", "branch": "", "repository_url": ""})
        entry["sessions"] += 1
        if timestamp > entry["last_used"]:
            entry["last_used"] = timestamp
            branch = git_info.get("branch")
            if isinstance(branch, str):
                entry["branch"] = branch
            repository_url = git_info.get("repository_url") or git_info.get("url")
            if isinstance(repository_url, str):
                entry["repository_url"] = repository_url
    return aggregated


def _workspace_repositories(root: Path, *, max_depth: int = 2, limit: int = 200) -> list[Path]:
    if not root.is_dir():
        return []
    if (root / ".git").exists():
        return [root]
    found: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(found) < limit:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False) or entry.name.startswith("."):
                        continue
                    path = Path(entry.path)
                    if (path / ".git").exists():
                        found.append(path)
                    elif depth + 1 < max_depth:
                        stack.append((path, depth + 1))
        except OSError:
            continue
    return sorted(found)


def _journal_repositories(journal_path: Path, limit: int = 60) -> list[str]:
    if not journal_path.exists():
        return []
    paths: list[str] = []
    try:
        with journal_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                repo = record.get("repo") if isinstance(record, dict) else None
                if isinstance(repo, str) and repo.strip() and repo not in paths:
                    paths.append(repo)
    except OSError:
        return []
    return paths[-limit:]


def discover_projects(
    workspace_root: Path,
    *,
    settings: dict[str, Any] | None = None,
    journal_path: Path | None = None,
    max_files: int = MAX_ROLLOUT_FILES,
) -> dict[str, Any]:
    """Merge every known project source into one ordered, UI-ready list."""
    workspace_root = workspace_root.expanduser().resolve(strict=False)
    settings = settings or {}
    candidates: dict[str, ProjectCandidate] = {}

    def candidate_for(raw_path: str) -> ProjectCandidate | None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        try:
            resolved = Path(raw_path.strip()).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return None
        key = str(resolved)
        existing = candidates.get(key)
        if existing is None:
            existing = ProjectCandidate(path=key, name=resolved.name)
            candidates[key] = existing
        return existing

    for path, info in codex_projects(max_files=max_files).items():
        candidate = candidate_for(path)
        if candidate is None:
            continue
        candidate.sources.append("codex")
        candidate.sessions = info["sessions"]
        candidate.last_used = info["last_used"]
        candidate.branch = info["branch"]
        candidate.repository_url = info["repository_url"]

    for key in list(settings.get("project_contexts", {}) or {}):
        candidate = candidate_for(str(key))
        if candidate is not None:
            candidate.sources.append("settings")
    default_repo = settings.get("default_repo")
    if default_repo:
        candidate = candidate_for(str(default_repo))
        if candidate is not None:
            candidate.sources.append("settings")

    if journal_path is not None:
        for raw in _journal_repositories(journal_path):
            candidate = candidate_for(raw)
            if candidate is not None and "journal" not in candidate.sources:
                candidate.sources.append("journal")

    for repository in _workspace_repositories(workspace_root):
        candidate = candidate_for(str(repository))
        if candidate is not None and "workspace" not in candidate.sources:
            candidate.sources.append("workspace")

    for candidate in candidates.values():
        path = Path(candidate.path)
        candidate.exists = path.is_dir()
        candidate.is_git = (path / ".git").exists()
        try:
            path.relative_to(workspace_root)
            candidate.inside_workspace = True
        except ValueError:
            candidate.inside_workspace = False
        if not candidate.sources:
            candidate.sources.append("manual")

    # Journal/settings entries for folders that no longer exist are noise; Codex history stays
    # visible because it explains what the CLI has touched even when a disk is unmounted.
    candidates = {
        key: candidate for key, candidate in candidates.items()
        if Path(key).is_dir() or "codex" in candidate.sources
    }
    # Stable multi-pass sort: usable projects first, newest Codex activity next, name as tie-break.
    ordered = sorted(candidates.values(), key=lambda item: item.name.lower())
    ordered.sort(key=lambda item: item.last_used, reverse=True)
    ordered.sort(key=lambda item: not item.selectable)
    return {
        "workspace_root": str(workspace_root),
        "codex_home": str(codex_home()),
        "codex_sessions_dir": str(codex_sessions_dir()),
        "codex_sessions_found": any("codex" in item.sources for item in candidates.values()),
        "projects": [item.public() for item in ordered],
    }


def summarize(catalog: dict[str, Any]) -> str:
    total = len(catalog["projects"])
    from_codex = sum(1 for item in catalog["projects"] if "codex" in item["sources"])
    return f"{total} проектов ({from_codex} из истории Codex), каталог сессий: {catalog['codex_sessions_dir']}"
