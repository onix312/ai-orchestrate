from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

from .core import OrchestratorError


def default_settings() -> dict[str, Any]:
    return {
        "default_repo": "",
        "default_checks": "",
        "project_contexts": {},
        "mode": "full",
        "profession": "developer",
        "router": "local",
        "lane": "",
        "luna_model": "gpt-6-luna",
        "sol_model": "gpt-6-sol",
        "review_model": "",
        "max_repairs": 1,
        "max_model_calls": 5,
        "prompt_token_budget": 16000,
        "max_run_tokens": 60000,
        "daily_token_budget": 120000,
        "codex_timeout": 1800,
        "check_timeout": 300,
        "merge_policy": "confirm",
        "merge_target": "local",
        "merge_method": "squash",
        "wait_for_github_checks": True,
        "delete_branch": True,
        "base_branch": "",
        "branch_prefix": "ai-orchestrate",
        "worktree_root": str(Path.home() / ".ai-orchestrate" / "worktrees"),
        "usage_log_path": "",
        "journal_path": "",
        "save_journal": True,
    }


_INT_RANGES = {
    "max_repairs": (0, 3),
    "max_model_calls": (1, 10),
    "prompt_token_budget": (100, 100000),
    "max_run_tokens": (100, 1000000),
    "daily_token_budget": (100, 10000000),
    "codex_timeout": (10, 7200),
    "check_timeout": (1, 3600),
}
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$")
_BRANCH_PREFIX_RE = re.compile(r"^[A-Za-z0-9._/-]{1,100}$")
_ALLOWED_KEYS = frozenset(default_settings())


def _resolved_path(value: str, key: str) -> str:
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError) as exc:
        raise OrchestratorError(f"Некорректный путь настройки {key}.") from exc


def normalize_settings(value: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OrchestratorError("Настройки должны быть JSON-объектом.")
    unknown = set(value) - _ALLOWED_KEYS
    if unknown:
        raise OrchestratorError("Неизвестные настройки: " + ", ".join(sorted(unknown)))

    result = default_settings()
    if base:
        result.update(base)
    result.update(value)

    for key, maximum in (("default_repo", 2048), ("worktree_root", 2048),
                         ("usage_log_path", 2048), ("journal_path", 2048),
                         ("default_checks", 6000), ("base_branch", 200)):
        item = result.get(key)
        if not isinstance(item, str) or len(item) > maximum or "\x00" in item:
            raise OrchestratorError(f"Некорректное значение настройки {key}.")
        result[key] = item.strip()
    try:
        result["default_repo"] = str(Path(result["default_repo"]).expanduser()) if result["default_repo"] else ""
    except (OSError, RuntimeError, ValueError) as exc:
        raise OrchestratorError("Некорректный путь настройки default_repo.") from exc
    for key in ("usage_log_path", "journal_path"):
        result[key] = _resolved_path(result[key], key) if result[key] else ""
    if result["worktree_root"]:
        result["worktree_root"] = _resolved_path(result["worktree_root"], "worktree_root")
    else:
        result["worktree_root"] = _resolved_path(str(Path.home() / ".ai-orchestrate" / "worktrees"), "worktree_root")

    checks = [line.strip() for line in result["default_checks"].splitlines() if line.strip()]
    if len(checks) > 12:
        raise OrchestratorError("В настройках можно сохранить не более 12 команд проверки.")
    result["default_checks"] = "\n".join(checks)

    contexts = result.get("project_contexts", {})
    if not isinstance(contexts, dict) or len(contexts) > 100:
        raise OrchestratorError("Контексты проектов должны быть объектом не более чем для 100 папок.")
    normalized_contexts: dict[str, str] = {}
    for raw_path, context in contexts.items():
        if (not isinstance(raw_path, str) or len(raw_path) > 2048 or "\x00" in raw_path
                or not isinstance(context, str) or len(context) > 12000):
            raise OrchestratorError("Некорректный путь или размер контекста проекта (максимум 12 000 символов).")
        key = _resolved_path(raw_path, "project_contexts")
        normalized_contexts[key] = context.strip()
    result["project_contexts"] = normalized_contexts

    if not isinstance(result["mode"], str) or result["mode"] not in {"quick", "full"}:
        raise OrchestratorError("Режим должен быть quick или full.")
    if not isinstance(result["router"], str) or result["router"] not in {"local", "jev"}:
        raise OrchestratorError("Роутер должен быть local или jev.")
    if not isinstance(result["merge_policy"], str) or result["merge_policy"] not in {"confirm", "jev_auto"}:
        raise OrchestratorError("Политика слияния должна быть confirm или jev_auto.")
    if not isinstance(result["merge_target"], str) or result["merge_target"] not in {"local", "github"}:
        raise OrchestratorError("Цель слияния должна быть local или github.")
    if not isinstance(result["merge_method"], str) or result["merge_method"] not in {"squash", "merge", "rebase"}:
        raise OrchestratorError("Способ GitHub-слияния должен быть squash, merge или rebase.")
    if not isinstance(result["lane"], str) or result["lane"] not in {"", "SMALL", "MEDIUM", "HIGH", "ESCALATE"}:
        raise OrchestratorError("Неизвестная стартовая полоса модели.")

    for key in ("profession",):
        item = result.get(key)
        if not isinstance(item, str) or not item or len(item) > 64:
            raise OrchestratorError(f"Некорректное значение настройки {key}.")
    for key in ("luna_model", "sol_model"):
        item = result.get(key)
        if not isinstance(item, str) or not _MODEL_RE.fullmatch(item):
            raise OrchestratorError(f"Модель {key} должна содержать только буквы, цифры и . _ : / -.")
    review_model = result.get("review_model")
    if not isinstance(review_model, str) or (review_model and not _MODEL_RE.fullmatch(review_model)):
        raise OrchestratorError("Некорректное имя модели ревьюера.")
    for key, (minimum, maximum) in _INT_RANGES.items():
        item = result.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum:
            raise OrchestratorError(f"Настройка {key} должна быть целым числом от {minimum} до {maximum}.")
    for key in ("wait_for_github_checks", "delete_branch", "save_journal"):
        if not isinstance(result.get(key), bool):
            raise OrchestratorError(f"Настройка {key} должна быть логическим значением.")

    prefix = result.get("branch_prefix")
    if not isinstance(prefix, str) or not _BRANCH_PREFIX_RE.fullmatch(prefix):
        raise OrchestratorError("Префикс веток может содержать только буквы, цифры, точку, _, / и -.")
    if (prefix.startswith("/") or prefix.endswith("/") or ".." in prefix or "//" in prefix
            or "@{" in prefix or prefix.endswith(".lock")):
        raise OrchestratorError("Префикс веток не является безопасным Git ref.")

    if result["merge_policy"] == "jev_auto" and result["mode"] != "full":
        raise OrchestratorError("Автослияние после Jev доступно только в режиме «Полный цикл» с независимым ревью.")
    if result["merge_target"] == "github" and not result["wait_for_github_checks"]:
        # Direct merges can bypass a repository's pending required checks. Keep this deliberate and explicit.
        pass
    return result


class SettingsStore:
    """Small atomic, user-local settings store. Secrets are intentionally not stored here."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or (Path.home() / ".ai-orchestrate" / "settings.json")).expanduser().resolve(strict=False)
        self._lock = RLock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return default_settings()
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise OrchestratorError(f"Не удалось прочитать файл настроек ({type(exc).__name__}): {self.path}") from exc
            if not isinstance(raw, dict):
                raise OrchestratorError(f"Файл настроек должен содержать JSON-объект: {self.path}")
            # Fill newly introduced fields from defaults while validating existing values.
            return normalize_settings(raw)

    def save(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = self.load()
            normalized = normalize_settings(updates, current)
            temporary: Path | None = None
            try:
                self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd, temporary_name = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=self.path.parent)
                temporary = Path(temporary_name)
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
                    json.dump(normalized, output, ensure_ascii=False, indent=2)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    os.chmod(temporary, 0o600)
                except OSError:
                    pass
                os.replace(temporary, self.path)
            except OSError as exc:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise OrchestratorError(f"Не удалось сохранить настройки ({type(exc).__name__}): {self.path}") from exc
            return normalized
