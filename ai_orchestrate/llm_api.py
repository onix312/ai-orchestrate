"""OpenAI-compatible executor: the same agent loop as Codex CLI, over a chat-completions API.

One code path serves the OpenAI API, OpenRouter and local OpenAI-compatible servers
(Ollama, LM Studio) — only ``base_url`` and the key change, and a local server needs no key
at all. There is no third-party SDK: plain :mod:`urllib`, like the Jev client.

Safety rules mirror the Codex path: the model only ever touches the isolated worktree, every
path is jailed inside it, child processes get the key-stripped environment from
:func:`ai_orchestrate.core._codex_env`, and all tool output is redacted and truncated.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .core import (
    CodexResult,
    CodexUsage,
    OrchestratorError,
    _redact_secrets,
    run_checks,
    truncate_text,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
MAX_TOOL_ROUNDS = 24
MAX_FILE_CHARS = 200_000
MAX_READ_CHARS = 24_000
MAX_OUTPUT_CHARS = 8_000
COMMAND_TIMEOUT = 120

# Cheap guardrails: the model works inside a disposable worktree, but there is no reason to
# let it start something destructive at the machine level.
BLOCKED_COMMAND_FRAGMENTS = (
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", "shutdown", "reboot",
    ":(){ :|:& };:", "> /dev/sd", "chmod -R 777 /",
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the project worktree. Returns at most ~24k characters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the worktree root."},
                    "start_line": {"type": "integer", "description": "Optional 1-based first line."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory entries (files and folders) inside the worktree.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory relative to the worktree root; empty means root."}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a whole file. Prefer str_replace for small edits; use this for new files or full rewrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the worktree root."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace the first exact occurrence of a snippet inside a file. The snippet must match byte for byte.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string", "description": "Exact existing snippet; must be unique enough to match once."},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command inside the worktree (tests, linters, git status). Output is truncated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command line, e.g. 'python -m unittest discover -s tests'."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Call when the task is done. Provide a short summary of what changed.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
]

READ_ONLY_TOOLS = {"read_file", "list_dir", "finish"}

SYSTEM_PROMPT = (
    "You are a careful software engineer working inside an isolated Git worktree. "
    "Read the code before changing it, make the smallest change that satisfies the task, and verify it "
    "with the project's own commands when they exist. Only touch files inside the worktree. "
    "Never print secrets. When you are done, call the finish tool with a short summary."
)


@dataclass(frozen=True)
class ApiConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = ""
    timeout: int = 300
    max_rounds: int = MAX_TOOL_ROUNDS
    extra_headers: dict[str, str] | None = None

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


def safe_join(root: Path, raw: str) -> Path:
    """Resolve a model-supplied path and refuse anything outside the worktree."""
    if not isinstance(raw, str) or not raw.strip():
        raise OrchestratorError("Путь к файлу пустой.")
    candidate = raw.strip().replace("\\", "/")
    if candidate.startswith("/") or (len(candidate) > 1 and candidate[1] == ":"):
        raise OrchestratorError(f"Нужен относительный путь внутри worktree, получен: {raw}")
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise OrchestratorError(f"Путь выходит за пределы worktree: {raw}") from exc
    return resolved


def _read(root: Path, args: dict[str, Any]) -> str:
    path = safe_join(root, str(args.get("path", "")))
    if not path.is_file():
        return f"ERROR: файл не найден: {args.get('path')}"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR: не удалось прочитать файл ({type(exc).__name__})"
    start = args.get("start_line")
    lines = text.splitlines()
    if isinstance(start, int) and start > 1:
        lines = lines[start - 1:]
    body = "\n".join(lines)
    if len(body) > MAX_READ_CHARS:
        body = body[:MAX_READ_CHARS] + "\n... [обрезано] ..."
    return body or "[пустой файл]"


def _list_dir(root: Path, args: dict[str, Any]) -> str:
    path = safe_join(root, str(args.get("path", "") or "."))
    if not path.is_dir():
        return f"ERROR: каталог не найден: {args.get('path')}"
    try:
        entries = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
    except OSError as exc:
        return f"ERROR: не удалось прочитать каталог ({type(exc).__name__})"
    lines = [f"{'d' if item.is_dir() else 'f'} {item.name}" for item in entries[:400]]
    return "\n".join(lines) or "[пусто]"


def _write(root: Path, args: dict[str, Any]) -> str:
    path = safe_join(root, str(args.get("path", "")))
    content = args.get("content")
    if not isinstance(content, str):
        return "ERROR: content должен быть строкой."
    if len(content) > MAX_FILE_CHARS:
        return f"ERROR: файл больше {MAX_FILE_CHARS} символов."
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    except OSError as exc:
        return f"ERROR: не удалось записать файл ({type(exc).__name__})"
    return f"OK: записано {len(content)} символов в {args.get('path')}"


def _str_replace(root: Path, args: dict[str, Any]) -> str:
    path = safe_join(root, str(args.get("path", "")))
    old = args.get("old_text")
    new = args.get("new_text")
    if not isinstance(old, str) or not isinstance(new, str):
        return "ERROR: old_text и new_text должны быть строками."
    if not path.is_file():
        return f"ERROR: файл не найден: {args.get('path')}"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"ERROR: не удалось прочитать файл ({type(exc).__name__})"
    if old not in text:
        return "ERROR: old_text не найден в файле — пришли точный фрагмент."
    if text.count(old) > 1:
        return f"ERROR: old_text встречается {text.count(old)} раз — добавь контекста, чтобы совпадение было единственным."
    try:
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
    except OSError as exc:
        return f"ERROR: не удалось записать файл ({type(exc).__name__})"
    return f"OK: заменено в {args.get('path')}"


def _run_command(root: Path, args: dict[str, Any], *, timeout: int) -> tuple[str, dict[str, Any] | None]:
    command = str(args.get("command", "")).strip()
    if not command:
        return "ERROR: команда пустая.", None
    lowered = command.lower()
    if any(fragment in lowered for fragment in BLOCKED_COMMAND_FRAGMENTS):
        return "ERROR: команда заблокирована как потенциально разрушительная.", None
    try:
        results = run_checks(root, [command], timeout=timeout)
    except OrchestratorError as exc:
        return f"ERROR: {exc}", None
    if not results:
        return "ERROR: команда не выполнена.", None
    item = results[0]
    output = truncate_text(_redact_secrets(str(item.get("output", ""))), MAX_OUTPUT_CHARS)
    event = {"command": command, "exit_code": item.get("returncode"),
             "aggregated_output": output}
    return f"exit {item.get('returncode')}\n{output}", event


def _post(config: ApiConfig, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    headers.update(config.extra_headers or {})
    request = urllib.request.Request(
        config.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
        except OSError:
            pass
        hints = {
            401: "ключ API не принят — проверь ключ в панели",
            402: "недостаточно средств/кредитов у провайдера",
            403: "доступ запрещён: проверь ключ и права",
            404: "модель или endpoint не найдены — проверь имя модели и base_url",
            429: "превышен лимит запросов провайдера",
        }
        hint = hints.get(exc.code, "")
        raise OrchestratorError(
            f"API вернул {exc.code}{': ' + hint if hint else ''}." + (f" {detail}" if detail else "")
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OrchestratorError(
            f"Не удалось достучаться до API ({type(exc).__name__}). Проверь base_url: {config.base_url}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise OrchestratorError("API вернул не-JSON ответ.") from exc


def _usage_from(payload: dict[str, Any]) -> CodexUsage:
    """Map OpenAI-style usage onto the same numbers the Codex ledger stores."""
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}

    def number(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    details = usage.get("prompt_tokens_details")
    cached = number(details.get("cached_tokens")) if isinstance(details, dict) else None
    return CodexUsage(
        input_tokens=number(usage.get("prompt_tokens")),
        cached_input_tokens=cached,
        output_tokens=number(usage.get("completion_tokens")),
    )


def run_llm_api(
    repo: Path,
    task: str,
    model: str,
    *,
    config: ApiConfig,
    sandbox: str = "workspace-write",
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: Any = None,
    token_budget: int | None = None,
    command_timeout: int = COMMAND_TIMEOUT,
) -> CodexResult:
    """Run one bounded agent turn against an OpenAI-compatible chat-completions endpoint."""
    if sandbox not in {"workspace-write", "read-only"}:
        raise OrchestratorError(f"Неизвестный режим sandbox: {sandbox!r}")
    if not model:
        raise OrchestratorError("Не указана модель API-исполнителя.")
    root = repo.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise OrchestratorError(f"Папка проекта не найдена: {root}")

    allowed = [tool for tool in TOOL_SCHEMAS
               if sandbox == "workspace-write" or tool["function"]["name"] in READ_ONLY_TOOLS]
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                      {"role": "user", "content": task}]
    totals = {"input": 0, "output": 0, "cached": 0}
    usage_seen = False
    final_message = ""
    finished = False

    def notify(event: dict[str, Any]) -> None:
        if on_event is not None:
            try:
                on_event(event)
            except Exception:
                # A disconnected UI must not stop the model loop.
                pass

    deadline = time.monotonic() + max(1, config.timeout)
    for round_index in range(1, config.max_rounds + 1):
        if cancel_event is not None and cancel_event.is_set():
            return CodexResult(130, _usage(totals, usage_seen), final_message, cancelled=True)
        if time.monotonic() > deadline:
            return CodexResult(124, _usage(totals, usage_seen), final_message,
                               stderr=f"Превышен таймаут {config.timeout} с")
        payload: dict[str, Any] = {"model": model, "messages": messages, "tools": allowed,
                                   "tool_choice": "auto", "temperature": 0.2}
        notify({"type": "turn.started"})
        response = _post(config, payload)
        usage = _usage_from(response)
        if usage.total_tokens is not None:
            usage_seen = True
            totals["input"] += usage.input_tokens or 0
            totals["output"] += usage.output_tokens or 0
            totals["cached"] += usage.cached_input_tokens or 0
            notify({"type": "turn.completed", "usage": {
                "input_tokens": usage.input_tokens, "cached_input_tokens": usage.cached_input_tokens,
                "output_tokens": usage.output_tokens,
            }})
        if token_budget is not None and totals["input"] + totals["output"] >= token_budget:
            return CodexResult(0, _usage(totals, usage_seen), final_message,
                               stderr="Достигнут токеновый лимит задачи внутри одного этапа.")

        choices = response.get("choices") or []
        if not choices:
            return CodexResult(1, _usage(totals, usage_seen), final_message, stderr="API вернул пустой choices.")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if isinstance(text, str) and text.strip():
            final_message = text.strip()
            notify({"type": "item.completed", "item": {"type": "agent_message", "text": text.strip()}})
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            finished = True
            break

        messages.append({"role": "assistant", "content": text if isinstance(text, str) else None,
                         "tool_calls": tool_calls})
        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            try:
                result_text, command_event = _dispatch(root, name, args, sandbox=sandbox,
                                                       command_timeout=command_timeout, notify=notify)
            except (OrchestratorError, OSError, UnicodeDecodeError, ValueError) as exc:
                # A failed tool call is a result for the model, not a reason to kill the run.
                result_text = f"ERROR: {type(exc).__name__}: {exc}"
                command_event = None
            if name == "finish":
                final_message = str(args.get("summary") or final_message or "Задача выполнена.")
                finished = True
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": truncate_text(result_text, MAX_OUTPUT_CHARS)})
            if command_event is not None:
                notify({"type": "item.completed", "item": {"type": "command_execution", **command_event}})
        if finished:
            break

    if not finished and not final_message:
        final_message = "Модель не вызвала finish и не вернула итоговое сообщение."
        return CodexResult(1, _usage(totals, usage_seen), final_message,
                           stderr=f"Исчерпан лимит итераций инструмента: {config.max_rounds}")
    return CodexResult(0, _usage(totals, usage_seen), final_message)


def _usage(totals: dict[str, int], seen: bool) -> CodexUsage:
    if not seen:
        return CodexUsage()
    return CodexUsage(input_tokens=totals["input"], cached_input_tokens=totals["cached"] or None,
                      output_tokens=totals["output"])


def _dispatch(root: Path, name: str, args: dict[str, Any], *, sandbox: str,
              command_timeout: int, notify: Callable[[dict[str, Any]], None]) -> tuple[str, dict[str, Any] | None]:
    if sandbox == "read-only" and name not in READ_ONLY_TOOLS:
        return "ERROR: этап read-only, изменение файлов и команды запрещены.", None
    if name == "read_file":
        return _read(root, args), None
    if name == "list_dir":
        return _list_dir(root, args), None
    if name == "write_file":
        result = _write(root, args)
        if result.startswith("OK"):
            notify({"type": "item.completed",
                    "item": {"type": "file_change", "path": str(args.get("path", "")), "status": "written"}})
        return result, None
    if name == "str_replace":
        result = _str_replace(root, args)
        if result.startswith("OK"):
            notify({"type": "item.completed",
                    "item": {"type": "file_change", "path": str(args.get("path", "")), "status": "edited"}})
        return result, None
    if name == "run_command":
        command = str(args.get("command", ""))
        notify({"type": "item.started", "item": {"type": "command_execution", "command": command}})
        return _run_command(root, args, timeout=command_timeout)
    if name == "finish":
        return "OK: этап завершён моделью.", None
    return f"ERROR: неизвестный инструмент {name!r}.", None
