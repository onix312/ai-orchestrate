"""Обычный ChatGPT как исполнитель: человек переносит промпт и ответ.

Панель не автоматизирует chatgpt.com, не читает cookies и не использует
неофициальные API. Она собирает самодостаточный промпт (задача, структура
репозитория, содержимое ключевых файлов и правила формата ответа), ждёт текст,
который пользователь вставит из обычного чата ChatGPT, и продолжает цикл:
применяет файлы, запускает проверки, обновляет ревью.

Модуль используется в двух случаях:

* исполнитель ``chatgpt`` — весь цикл идёт через обычный чат;
* аварийный обход ``limit_fallback=chatgpt`` — Codex/API упёрся в лимит
  использования, и оставшиеся этапы продолжает человек.

Ожидание ответа — обычный rendezvous на ``threading.Condition``: рабочий поток
оркестратора блокируется до ответа, отмены задачи или таймаута, а панель
показывает ожидающий запрос через :meth:`ManualRelay.public`.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Condition
from typing import Any, Callable
from urllib.parse import quote

from .chatgpt_bridge import answer_format_instructions
from .core import OrchestratorError, _redact_secrets

CHATGPT_URL = "https://chatgpt.com/"
MAX_PROMPT_CHARS = 200_000
MAX_URL_PROMPT_CHARS = 6_000
MAX_CONTEXT_CHARS = 60_000
MAX_CONTEXT_FILE_CHARS = 14_000
MAX_CONTEXT_FILES = 24
MAX_CONTEXT_TREE_LINES = 160
MAX_FILE_BYTES = 400_000
DEFAULT_TIMEOUT = 3_600.0

IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", "dist", "build",
    "target", "coverage", ".next", ".nuxt", ".svelte-kit", ".idea", ".vscode",
    ".gradle", ".terraform", "vendor",
})
IGNORED_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svgz", ".pdf", ".zip",
    ".gz", ".tar", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib",
    ".bin", ".pyc", ".pyo", ".class", ".jar", ".o", ".a", ".obj", ".lib", ".woff",
    ".woff2", ".ttf", ".otf", ".mp3", ".mp4", ".mov", ".avi", ".wav", ".sqlite",
    ".sqlite3", ".db", ".lock", ".map", ".min.js", ".min.css",
)
# Файлы с секретами не попадают в контекст для внешнего чата даже частично.
SECRET_NAMES = frozenset({".env", ".netrc", ".npmrc", ".pypirc", "credentials", "credentials.json", "secrets.json"})
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".asc")
BASE_FILES = frozenset({"readme.md", "readme.ru.md", "agents.md", "package.json", "pyproject.toml", "setup.py"})

_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_./-]{4,}")


class RelayStopped(OrchestratorError):
    """Ожидание ответа человека прервано: задача отменена или истёк таймаут."""


@dataclass(frozen=True)
class RelayRequest:
    """Один вопрос к обычному чату ChatGPT и подсказки для панели."""

    kind: str  # "plan" | "code" | "review" | "repair"
    stage: str
    role: str
    title: str
    instructions: str
    prompt: str
    url: str = CHATGPT_URL
    prompt_in_url: bool = False
    created_at: str = ""


@dataclass
class RelayState:
    request: RelayRequest | None = None
    answers: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def chatgpt_link(prompt: str) -> dict[str, Any]:
    """Ссылка на обычный чат. Длинный промпт в URL не влезает — остаётся копирование."""
    if not prompt:
        return {"url": CHATGPT_URL, "prompt_in_url": False}
    if len(prompt) <= MAX_URL_PROMPT_CHARS:
        return {"url": f"{CHATGPT_URL}?q={quote(prompt, safe='')}", "prompt_in_url": True}
    return {"url": CHATGPT_URL, "prompt_in_url": False}


def _is_secret_path(relative: str) -> bool:
    name = os.path.basename(relative).casefold()
    if name in SECRET_NAMES or name.startswith(".env"):
        return True
    if name.startswith(("id_rsa", "id_ed25519", "id_ecdsa")):
        return True
    return name.endswith(SECRET_SUFFIXES)


def _is_binary_free(text: str) -> bool:
    return "\x00" not in text


def list_worktree_files(root: Path) -> list[str]:
    """Файлы worktree: индекс Git с учётом .gitignore, иначе обход каталогов."""
    root = Path(root)
    names: list[str] = []
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=False, timeout=30,
        )
        if completed.returncode == 0:
            names = [item for item in completed.stdout.decode("utf-8", "replace").split("\0") if item]
    except (OSError, subprocess.SubprocessError):
        names = []
    if not names:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in IGNORED_DIRS and not name.startswith(".git") and name != ".ai-orchestrate"
            )
            for name in sorted(filenames):
                names.append(str((Path(dirpath) / name).relative_to(root)).replace(os.sep, "/"))
    cleaned = []
    for name in names:
        normalized = name.replace("\\", "/").strip("/")
        if not normalized or normalized.startswith(".git/") or _is_secret_path(normalized):
            continue
        if any(part in IGNORED_DIRS for part in normalized.split("/")[:-1]):
            continue
        if normalized.casefold().endswith(IGNORED_SUFFIXES):
            continue
        cleaned.append(normalized)
    return sorted(dict.fromkeys(cleaned))


def _relevance(name: str, keywords: list[str]) -> int:
    """Чем больше слов задачи в пути, тем вероятнее файл нужен модели."""
    lowered = name.casefold()
    score = sum(6 for keyword in keywords if keyword in lowered)
    if os.path.basename(lowered) in BASE_FILES:
        score += 3
    depth = lowered.count("/")
    score -= depth
    if depth == 0:
        score += 1
    return score


def _keywords(task: str) -> list[str]:
    words = {match.group(0).casefold() for match in _WORD_RE.finditer(task or "")}
    return sorted(word for word in words if len(word) <= 40)


def pack_repository_context(
    root: Path,
    *,
    task: str = "",
    budget_chars: int = MAX_CONTEXT_CHARS,
    max_files: int = MAX_CONTEXT_FILES,
) -> dict[str, Any]:
    """Собрать структуру проекта и содержимое ключевых файлов для внешнего чата.

    Возвращает ``{"text", "files", "omitted", "total_files"}``. Ничего не пишет
    на диск и не читает файлы вне worktree. Секреты и бинарные файлы исключены.
    """
    root = Path(root).expanduser().resolve(strict=True)
    names = list_worktree_files(root)
    keywords = _keywords(task)
    ranked = sorted(names, key=lambda name: (-_relevance(name, keywords), name))
    ordered = list(ranked)
    tree_lines = names[:MAX_CONTEXT_TREE_LINES]
    parts = [
        f"## Структура проекта ({root.name}; файлов в рабочей копии: {len(names)})",
        "```text",
        *[f"- {name}" for name in tree_lines],
        *(["…"] if len(names) > len(tree_lines) else []),
        "```",
        "",
        "## Содержимое ключевых файлов",
    ]
    included: list[str] = []
    used = sum(len(part) for part in parts)
    for name in ordered:
        if len(included) >= max_files or used >= budget_chars:
            break
        path = root / name
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        if not _is_binary_free(text):
            continue
        clipped = text[:MAX_CONTEXT_FILE_CHARS]
        if len(text) > len(clipped):
            clipped += "\n… (файл обрезан)"
        block = f"\n### Файл: {name}\n```\n{clipped}\n```\n"
        if used + len(block) > budget_chars and included:
            continue
        parts.append(block)
        included.append(name)
        used += len(block)
    text = _redact_secrets("\n".join(parts).strip() + "\n")
    return {
        "text": text,
        "files": included,
        "omitted": max(0, len(names) - len(included)),
        "total_files": len(names),
    }


def build_manual_prompt(
    base_prompt: str,
    *,
    kind: str,
    context_text: str = "",
    context_files: list[str] | None = None,
    check_commands: list[str] | None = None,
    extra_note: str = "",
) -> str:
    """Собрать финальный текст для обычного чата: роль, контекст, формат ответа."""
    parts: list[str] = [
        "Ты — инженер, который работает с существующим Git-репозиторием по переписке.",
        "Ниже роль, задача и контекст проекта. Не расширяй scope и не выдумывай файлы,",
        "которых нет в контексте. Секреты не запрашивай и не печатай.",
        "",
        base_prompt.strip(),
    ]
    if context_text.strip():
        parts += ["", context_text.strip()]
    if context_files:
        parts += ["", "## Какие файлы попали в контекст", ", ".join(context_files)]
    if check_commands:
        parts += [
            "",
            "## Локальные проверки проекта",
            *[f"- {command}" for command in check_commands],
            "Автор проверок — оркестратор: он выполнит их сам после твоих правок.",
        ]
    if extra_note.strip():
        parts += ["", extra_note.strip()]
    if kind in {"code", "repair"}:
        parts += ["", answer_format_instructions()]
    elif kind == "review":
        parts += [
            "",
            "## Формат ответа",
            "Первая строка строго PASS, если блокирующих замечаний нет, иначе ISSUES.",
            "Дальше — короткие пункты: файл/строка, последствие, минимальное исправление.",
            "Не пересказывай diff и не раскрывай цепочку рассуждений.",
        ]
    else:
        parts += [
            "",
            "## Формат ответа",
            "Верни ровно разделы: ЦЕЛЬ И КРИТЕРИИ, ДОПУЩЕНИЯ, ПЛАН, РИСКИ/ПРОВЕРКИ.",
            "Коротко, без скрытых рассуждений и без блоков с файлами.",
        ]
    prompt = "\n".join(parts).strip() + "\n"
    if len(prompt) > MAX_PROMPT_CHARS:
        raise OrchestratorError(f"Промпт для ChatGPT длиннее {MAX_PROMPT_CHARS} символов.")
    return prompt


class ManualRelay:
    """Точка встречи рабочего потока оркестратора и человека в панели."""

    def __init__(
        self,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: Any = None,
        timeout: float = DEFAULT_TIMEOUT,
        poll_interval: float = 0.5,
    ) -> None:
        self._condition = Condition()
        self._state = RelayState()
        self._answer: str | None = None
        self._cancelled = False
        self._closed = False
        self._on_event = on_event
        self._cancel_event = cancel_event
        self.timeout = max(60.0, float(timeout))
        self.poll_interval = max(0.1, min(float(poll_interval), 5.0))

    # -- наблюдаемое состояние -------------------------------------------------
    def waiting(self) -> bool:
        with self._condition:
            return self._state.request is not None and self._answer is None and not self._cancelled

    def public(self) -> dict[str, Any] | None:
        """Состояние для панели: ожидающий запрос и краткая история ответов."""
        with self._condition:
            request = self._state.request
            return {
                "waiting": request is not None and self._answer is None and not self._cancelled,
                "answered": self._state.answers,
                "timeout_seconds": int(self.timeout),
                "request": None if request is None else {
                    "kind": request.kind,
                    "stage": request.stage,
                    "role": request.role,
                    "title": request.title,
                    "instructions": request.instructions,
                    "prompt": request.prompt,
                    "chars": len(request.prompt),
                    "url": request.url,
                    "prompt_in_url": request.prompt_in_url,
                    "created_at": request.created_at,
                },
                "history": list(self._state.history[-5:]),
            }

    # -- жизненный цикл --------------------------------------------------------
    def cancel(self) -> None:
        """Остановить ожидание: задача отменена пользователем."""
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def reset(self) -> None:
        """Снова разрешить ожидание: панель перезапустила мост после отмены."""
        with self._condition:
            if self._closed:
                return
            self._cancelled = False
            self._answer = None
            self._condition.notify_all()

    def deliver(self, answer: str) -> bool:
        """Принять вставленный ответ человека. ``False`` — сейчас ответа никто не ждёт."""
        if not isinstance(answer, str) or not answer.strip():
            raise OrchestratorError("Ответ ChatGPT пустой.")
        with self._condition:
            if self._state.request is None or self._answer is not None or self._cancelled:
                return False
            self._answer = answer
            self._state.answers += 1
            self._state.history.append({
                "kind": self._state.request.kind,
                "stage": self._state.request.stage,
                "chars": len(answer),
                "answered_at": _now(),
            })
            self._condition.notify_all()
            return True

    # -- рабочий поток ---------------------------------------------------------
    def request(
        self,
        *,
        kind: str,
        stage: str,
        role: str,
        title: str,
        instructions: str,
        prompt: str,
    ) -> str:
        """Опубликовать запрос панели и дождаться вставленного ответа."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise OrchestratorError("Промпт для ChatGPT пуст.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise OrchestratorError(f"Промпт для ChatGPT длиннее {MAX_PROMPT_CHARS} символов.")
        link = chatgpt_link(prompt)
        request = RelayRequest(
            kind=kind, stage=stage, role=role, title=title, instructions=instructions,
            prompt=prompt, url=link["url"], prompt_in_url=link["prompt_in_url"], created_at=_now(),
        )
        with self._condition:
            if self._cancelled or self._closed:
                raise RelayStopped("Ожидание ответа ChatGPT остановлено до запроса.")
            self._state.request = request
            self._answer = None
            self._condition.notify_all()
        self._emit({
            "event": "manual.requested",
            "stage": stage,
            "role": role,
            "message": f"{title}: скопируй промпт в обычный ChatGPT и вставь ответ в панель.",
            "data": {
                "kind": kind,
                "chars": len(prompt),
                "prompt_in_url": link["prompt_in_url"],
                "url": link["url"],
                "timeout_seconds": int(self.timeout),
            },
        })
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while True:
                if self._cancelled or self._closed or self._cancel_requested():
                    self._clear_request()
                    raise RelayStopped("Ожидание ответа ChatGPT остановлено: задача отменена или мост закрыт.")
                if self._answer is not None:
                    answer = self._answer
                    self._answer = None
                    self._state.request = None
                    break
                if time.monotonic() >= deadline:
                    self._clear_request()
                    raise RelayStopped(
                        f"Ответ ChatGPT не получен за {int(self.timeout)} с. Увеличь таймаут ручного моста "
                        "или запусти задачу заново; рабочая ветка сохранена."
                    )
                self._condition.wait(timeout=self.poll_interval)
        self._emit({
            "event": "manual.answered",
            "stage": stage,
            "role": role,
            "message": f"{title}: ответ ChatGPT получен ({len(answer):,} символов).",
            "data": {"kind": kind, "chars": len(answer)},
        })
        return answer

    # -- внутреннее ------------------------------------------------------------
    def _cancel_requested(self) -> bool:
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _clear_request(self) -> None:
        self._state.request = None
        self._answer = None

    def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            # Наблюдаемость не должна ломать ожидание ответа.
            pass
