"""Generate as much of the configuration as possible instead of asking for it.

``autofill`` only touches values it can justify from the local machine: the selected
repository, its checks, its base branch, and limits that contradict each other. Anything
ambiguous is returned as a suggestion and left for the user to decide.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import env_setup, secrets
from .core import OrchestratorError
from .gitops import git
from .workflow import suggest_checks

_GITHUB_REMOTE_RE = re.compile(r"github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


def detect_base_branch(repo: Path) -> str:
    """Prefer the remote HEAD, then the checked-out branch."""
    for args in (["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"],
                 ["rev-parse", "--abbrev-ref", "HEAD"]):
        try:
            result = git(repo, args, check=False, timeout=10)
        except OrchestratorError:
            continue
        value = (result.stdout or "").strip().splitlines()
        if result.returncode == 0 and value:
            branch = value[0].strip().removeprefix("origin/")
            if branch and branch != "HEAD":
                return branch
    return ""


def detect_github_remote(repo: Path) -> str:
    try:
        result = git(repo, ["remote", "get-url", "origin"], check=False, timeout=10)
    except OrchestratorError:
        return ""
    if result.returncode != 0:
        return ""
    match = _GITHUB_REMOTE_RE.search((result.stdout or "").strip())
    return match.group(1).removesuffix(".git") if match else ""


def autofill(
    repo: Path,
    settings: dict[str, Any],
    *,
    overwrite_checks: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(updates, report)`` where updates can be handed to ``SettingsStore.save``."""
    updates: dict[str, Any] = {}
    applied: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []

    def apply(key: str, label: str, value: Any, reason: str) -> None:
        updates[key] = value
        applied.append({"key": key, "label": label, "value": value, "reason": reason})

    def keep(key: str, label: str, value: Any, reason: str) -> None:
        applied.append({"key": key, "label": label, "value": value, "reason": reason, "kept": True})

    repo = repo.expanduser().resolve(strict=False)
    apply("default_repo", "Папка проекта", str(repo), "выбранный репозиторий")

    checks = [line.strip() for line in str(settings.get("default_checks", "")).splitlines() if line.strip()]
    detected = suggest_checks(repo)
    if not checks or overwrite_checks:
        if detected:
            apply("default_checks", "Автопроверки", "\n".join(detected),
                  "найдены по файлам проекта (package.json / pytest / tests)")
        elif not checks:
            suggestions.append({
                "key": "default_checks", "label": "Автопроверки",
                "reason": "В проекте не нашлось готовой тестовой команды — добавь её вручную.",
            })
    else:
        keep("default_checks", "Автопроверки", "\n".join(checks), "уже заданы вручную")

    base_branch = detect_base_branch(repo)
    if base_branch:
        apply("base_branch", "Базовая ветка", base_branch, "определена из origin/HEAD или текущей ветки")
    else:
        suggestions.append({
            "key": "base_branch", "label": "Базовая ветка",
            "reason": "Не удалось определить ветку; будет использована текущая.",
        })

    jev_ready = bool(secrets.active_jev_key())
    if settings.get("router") == "jev" and not jev_ready:
        apply("router", "Триаж задачи", "local", "Jev без ключа — включён бесплатный локальный триаж")
    else:
        keep("router", "Триаж задачи", settings.get("router", "local"), "оставлен выбранный роутер")
    if settings.get("merge_policy") == "jev_auto" and not jev_ready:
        apply("merge_policy", "Когда сливать", "confirm",
              "автослияние после Jev недоступно без ключа — нужна кнопка подтверждения")
    else:
        keep("merge_policy", "Когда сливать", settings.get("merge_policy", "confirm"), "политика слияния не менялась")

    mode = settings.get("mode", "full")
    needed = 3 if mode == "full" else 1
    max_calls = int(settings.get("max_model_calls", 5))
    if max_calls < needed:
        apply("max_model_calls", "Максимум вызовов", needed,
              f"в режиме «{mode}» нужно минимум {needed} вызовов, иначе цикл остановится до ревью")
    else:
        keep("max_model_calls", "Максимум вызовов", max_calls, "лимита хватает на выбранный режим")

    run_budget = int(settings.get("max_run_tokens", 60000))
    daily_budget = int(settings.get("daily_token_budget", 120000))
    if daily_budget < run_budget:
        apply("daily_token_budget", "Токенов в день", run_budget,
              "дневной лимит был меньше лимита одной задачи")
    else:
        keep("daily_token_budget", "Токенов в день", daily_budget, "дневной лимит не менялся")

    remote = detect_github_remote(repo)
    if remote:
        suggestions.append({
            "key": "merge_target", "label": "Куда сливать",
            "reason": f"Найден GitHub origin {remote}: для pull request выбери цель «GitHub».",
        })
    if not env_setup.find_tool("gh"):
        suggestions.append({
            "key": "merge_target", "label": "Куда сливать",
            "reason": "gh не установлен: GitHub-цель недоступна, локальное слияние работает.",
        })
    return updates, {
        "repo": str(repo),
        "applied": applied,
        "suggestions": suggestions,
        "base_branch": base_branch,
        "checks": detected,
        "github_remote": remote,
    }
