from __future__ import annotations

import os
import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .core import OrchestratorError


@dataclass(frozen=True)
class Worktree:
    repo: Path
    path: Path
    branch: str
    base_branch: str
    base_sha: str
    base_ref: str


def _clean_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value or ""
    for name, secret in os.environ.items():
        if len(secret) >= 8 and any(part in name.upper() for part in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            text = text.replace(secret, "[REDACTED]")
    return text.strip()


def git(repo: Path, args: Sequence[str], *, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    if not shutil.which("git"):
        raise OrchestratorError("Git не найден в PATH.")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise OrchestratorError(f"Команда Git превысила таймаут ({timeout} с).") from exc
    except OSError as exc:
        raise OrchestratorError(f"Не удалось запустить Git ({type(exc).__name__}).") from exc
    if check and result.returncode != 0:
        detail = _clean_output(result.stderr or result.stdout)
        if len(detail) > 1200:
            detail = detail[-1200:]
        action = args[0] if args else "command"
        suffix = f": {detail}" if detail else ""
        raise OrchestratorError(f"Git {action} завершился с кодом {result.returncode}{suffix}")
    return result


def current_branch(repo: Path) -> str:
    branch = git(repo, ["branch", "--show-current"]).stdout.strip()
    if not branch:
        raise OrchestratorError("Исходный репозиторий находится в detached HEAD. Выбери базовую ветку в настройках.")
    return branch


def branch_sha(repo: Path, branch: str) -> str:
    # Verify the requested branch is a local branch and not an arbitrary revision expression.
    if not branch or git(repo, ["check-ref-format", "--branch", branch], check=False).returncode != 0:
        raise OrchestratorError("Некорректное имя базовой ветки.")
    result = git(repo, ["show-ref", "--verify", "--hash", f"refs/heads/{branch}"], check=False)
    if result.returncode != 0:
        raise OrchestratorError(f"Локальная базовая ветка не найдена: {branch}")
    return result.stdout.strip()


def remote_branch_sha(repo: Path, branch: str, *, remote: str = "origin") -> str:
    if not branch or git(repo, ["check-ref-format", "--branch", branch], check=False).returncode != 0:
        raise OrchestratorError("Некорректное имя базовой ветки.")
    git(repo, ["fetch", "--no-tags", remote, f"refs/heads/{branch}:refs/remotes/{remote}/{branch}"], timeout=180)
    result = git(repo, ["rev-parse", "--verify", f"refs/remotes/{remote}/{branch}"])
    return result.stdout.strip()


def pull_request_head_sha(repo: Path, number: int) -> str:
    if not 1 <= number <= 2**31 - 1:
        raise OrchestratorError("Некорректный номер pull request.")
    destination = f"refs/ai-orchestrate/pull/{number}"
    git(repo, ["fetch", "--no-tags", "origin", f"refs/pull/{number}/head:{destination}"], timeout=180)
    return git(repo, ["rev-parse", "--verify", destination]).stdout.strip()


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    return (result or "project")[:70]


def create_worktree(
    repo: Path,
    worktree_root: Path,
    *,
    job_id: str,
    branch_prefix: str,
    base_ref: str,
    base_branch: str,
    source_kind: str = "task",
    source_number: int | None = None,
) -> Worktree:
    repo = repo.expanduser().resolve(strict=True)
    try:
        root = worktree_root.expanduser().resolve(strict=False)
        root.relative_to(repo)
    except ValueError:
        pass
    except (OSError, RuntimeError) as exc:
        raise OrchestratorError("Путь к каталогу worktree некорректен.") from exc
    else:
        raise OrchestratorError("Каталог worktree должен располагаться вне выбранного Git-репозитория.")

    prefix = branch_prefix.strip("/")
    suffix = f"{source_kind}-{source_number}" if source_number is not None else source_kind
    branch = f"{prefix}/{_slug(suffix)}-{_slug(job_id)}"
    checked = git(repo, ["check-ref-format", "--branch", branch], check=False)
    if checked.returncode != 0:
        raise OrchestratorError("Сформированное имя рабочей ветки не является допустимым Git ref.")

    path = root / _slug(repo.name) / _slug(job_id)
    if path.exists():
        raise OrchestratorError(f"Каталог рабочей копии уже существует: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not allow a configured symlink to redirect creation inside the original checkout.
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(repo)
    except ValueError:
        pass
    else:
        raise OrchestratorError("Рабочая копия не может создаваться внутри исходного репозитория.")

    base_sha = git(repo, ["rev-parse", "--verify", base_ref]).stdout.strip()
    git(repo, ["worktree", "add", "-b", branch, str(path), base_sha], timeout=180)
    return Worktree(repo=repo, path=path.resolve(), branch=branch, base_branch=base_branch,
                    base_sha=base_sha, base_ref=base_ref)


_SENSITIVE_NAME = re.compile(
    r"(^|/)(?:\.env(?:\..*)?|\.npmrc|\.pypirc|credentials?(?:\.[^/]*)?|"
    r"secrets?(?:\.[^/]*)?|id_(?:rsa|ed25519)|[^/]+\.(?:pem|key|p12|pfx))$",
    re.IGNORECASE,
)


def _changed_paths(status: str) -> list[str]:
    """Parse porcelain v1 -z, including both sides of renames; never unquote names."""
    entries = iter(status.split("\0"))
    paths: list[str] = []
    for entry in entries:
        if len(entry) < 4:
            continue
        paths.append(entry[3:])
        if "R" in entry[:2] or "C" in entry[:2]:
            source = next(entries, "")
            if source:
                paths.append(source)
    return paths


def verify_worktree(worktree: Worktree, expected_sha: str, *, clean: bool = True) -> None:
    if current_branch(worktree.path) != worktree.branch:
        raise OrchestratorError("Рабочая копия переключена на другую ветку; операция остановлена.")
    actual = git(worktree.path, ["rev-parse", "--verify", "HEAD"]).stdout.strip()
    if actual != expected_sha:
        raise OrchestratorError("Рабочая ветка изменилась после проверки; требуется новый цикл проверок и ревью.")
    if clean and git(worktree.path, ["status", "--porcelain", "--untracked-files=all"]).stdout.strip():
        raise OrchestratorError("После проверки в worktree появились изменения; слияние запрещено.")


def worktree_digest(repo: Path) -> str:
    """Bind verification to file contents, including untracked files and executable bits.

    Git-ignored build/dependency files are excluded. Symlinks are hashed, never followed.
    The digest stays stable when these exact files are staged and committed.
    """
    names = git(repo, ["ls-files", "--cached", "--others", "--exclude-standard", "-z"]).stdout.split("\0")
    digest = hashlib.sha256()
    for name in sorted(set(names) - {""}):
        path = repo / name
        if not path.exists() and not path.is_symlink():
            continue
        digest.update(name.encode("utf-8") + b"\0")
        if path.is_symlink():
            digest.update(b"symlink:" + os.fsencode(os.readlink(path)) + b"\0")
        elif path.is_file():
            try:
                path.resolve().relative_to(repo.resolve())
            except ValueError as exc:
                raise OrchestratorError("Файл перенаправлен за пределы worktree: " + name) from exc
            digest.update(b"executable:" + str(bool(path.stat().st_mode & 0o111)).encode() + b"\0")
            content = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    content.update(chunk)
            digest.update(content.digest())
        else:
            raise OrchestratorError("Автоматическая проверка вложенных Git/submodule-каталогов не поддерживается: " + name)
    return digest.hexdigest()


def commit_worktree(worktree: Worktree, message: str, *, expected_head: str | None = None, verified_digest: str | None = None) -> dict[str, str | bool | list[str]]:
    verify_worktree(worktree, expected_head or worktree.base_sha, clean=False)
    if verified_digest and worktree_digest(worktree.path) != verified_digest:
        raise OrchestratorError("Файлы изменились после проверок/ревью; commit запрещён.")
    status = git(worktree.path, ["status", "--porcelain", "-z", "--untracked-files=all"]).stdout
    if not status.strip():
        return {"committed": False, "sha": "", "files": []}
    sensitive = [name for name in _changed_paths(status) if _SENSITIVE_NAME.search(name)]
    if sensitive:
        raise OrchestratorError(
            "В изменениях найдены потенциально секретные файлы, их нельзя включать в автоматический commit: "
            + ", ".join(sensitive[:8])
        )
    git(worktree.path, ["add", "--all"])
    staged = git(worktree.path, ["diff", "--cached", "--name-only", "-z"]).stdout.rstrip("\0").split("\0")
    if not any(staged):
        return {"committed": False, "sha": "", "files": []}
    checked = git(worktree.path, ["diff", "--cached", "--check"], check=False)
    if checked.returncode != 0:
        raise OrchestratorError("Git diff --check обнаружил whitespace-ошибки; commit не создан.")
    safe_message = message.strip().replace("\r", " ").replace("\n", " ")[:180]
    if not safe_message:
        safe_message = "AI-assisted change"
    git(worktree.path, ["commit", "-m", safe_message], timeout=180)
    if verified_digest and worktree_digest(worktree.path) != verified_digest:
        raise OrchestratorError("Git hook или внешний процесс изменил проверенные файлы; слияние запрещено.")
    sha = git(worktree.path, ["rev-parse", "--verify", "HEAD"]).stdout.strip()
    return {"committed": True, "sha": sha, "files": staged}


def verify_merge_target(worktree: Worktree) -> None:
    branch = current_branch(worktree.repo)
    if branch != worktree.base_branch:
        raise OrchestratorError(
            f"Слияние остановлено: в исходном checkout выбрана ветка {branch}, ожидалась {worktree.base_branch}."
        )
    current_sha = git(worktree.repo, ["rev-parse", "--verify", "HEAD"]).stdout.strip()
    if current_sha != worktree.base_sha:
        raise OrchestratorError(
            "Слияние остановлено: базовая ветка изменилась после запуска задачи. "
            "Проверь изменения и создай новую задачу от актуальной ветки."
        )
    status = git(worktree.repo, ["status", "--porcelain", "--untracked-files=all"]).stdout
    if status.strip():
        raise OrchestratorError(
            "Локальное fast-forward-слияние остановлено: в исходном checkout появились незакоммиченные изменения."
        )
    ancestor = git(worktree.repo, ["merge-base", "--is-ancestor", worktree.base_sha, worktree.branch], check=False)
    if ancestor.returncode != 0:
        raise OrchestratorError("Рабочая ветка больше не основана на сохранённом состоянии целевой ветки.")


def merge_local(worktree: Worktree, *, expected_sha: str | None = None) -> str:
    sha = expected_sha or git(worktree.path, ["rev-parse", "--verify", "HEAD"]).stdout.strip()
    verify_worktree(worktree, sha)
    verify_merge_target(worktree)
    git(worktree.repo, ["merge", "--ff-only", "--no-edit", sha], timeout=180)
    return git(worktree.repo, ["rev-parse", "--verify", "HEAD"]).stdout.strip()


def remove_worktree(worktree: Worktree, *, delete_branch: bool = False, force: bool = False, force_branch: bool = False) -> None:
    if worktree.path.exists():
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(worktree.path))
        git(worktree.repo, args, timeout=120)
    if delete_branch:
        result = git(worktree.repo, ["branch", "-D" if force or force_branch else "-d", worktree.branch], check=False)
        if result.returncode != 0:
            # A successful merge can make the branch removable; report real failures only when it remains.
            if git(worktree.repo, ["show-ref", "--verify", f"refs/heads/{worktree.branch}"], check=False).returncode == 0:
                detail = _clean_output(result.stderr or result.stdout)
                raise OrchestratorError(f"Не удалось удалить ветку {worktree.branch}: {detail[:500]}")
