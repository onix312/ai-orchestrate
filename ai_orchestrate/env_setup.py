"""Local environment discovery, PATH self-repair and one-step installation.

Everything here is a local, inspectable command. No model call is made and no
credential is requested: Codex and GitHub keep using their own logins.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import secrets
from .core import OrchestratorError

MAX_INSTALL_OUTPUT_LINES = 400
_AUTH_CACHE_TTL = 25.0

_auth_cache: dict[str, tuple[float, bool]] = {}
_auth_lock = threading.Lock()
_npm_prefix: str | None = None


def platform_name() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def candidate_bin_dirs() -> list[Path]:
    """Directories that commonly hold codex/gh/node but are often missing from PATH."""
    home = Path.home()
    directories: list[Path] = [
        home / ".local" / "bin",
        home / "bin",
        home / ".bun" / "bin",
        home / ".cargo" / "bin",
        home / ".volta" / "bin",
        home / ".nvm" / "current" / "bin",
        home / ".npm-global" / "bin",
        home / ".deno" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/opt/homebrew/sbin"),
        Path("/usr/local/bin"),
        Path("/usr/local/sbin"),
        Path("/snap/bin"),
    ]
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA") or (home / "AppData" / "Local"))
        roaming = Path(os.environ.get("APPDATA") or (home / "AppData" / "Roaming"))
        program_files = Path(os.environ.get("PROGRAMFILES") or "C:/Program Files")
        directories += [
            roaming / "npm",
            local / "Programs" / "codex",
            local / "codex" / "bin",
            local / "GitHubCLI",
            local / "Programs" / "GitHub CLI",
            local / "Microsoft" / "WinGet" / "Links",
            local / "Programs" / "nodejs",
            program_files / "GitHub CLI",
            program_files / "nodejs",
            home / "scoop" / "shims",
        ]
    nvm_versions = home / ".nvm" / "versions" / "node"
    try:
        if nvm_versions.is_dir():
            releases = sorted((item for item in nvm_versions.iterdir() if item.is_dir()), reverse=True)
            directories += [release / "bin" for release in releases[:3]]
    except OSError:
        pass
    prefix = npm_global_prefix()
    if prefix:
        directories.append(Path(prefix) if os.name == "nt" else Path(prefix) / "bin")
    unique: list[Path] = []
    for directory in directories:
        try:
            if directory.is_dir() and directory not in unique:
                unique.append(directory)
        except OSError:
            continue
    return unique


def npm_global_prefix() -> str:
    global _npm_prefix
    if _npm_prefix is not None:
        return _npm_prefix
    npm = shutil.which("npm")
    if not npm:
        _npm_prefix = ""
        return _npm_prefix
    try:
        result = subprocess.run(
            [npm, "prefix", "-g"], capture_output=True, text=True, timeout=12, check=False,
        )
        _npm_prefix = result.stdout.strip().splitlines()[0].strip() if result.returncode == 0 and result.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError, IndexError):
        _npm_prefix = ""
    return _npm_prefix


def refresh_path() -> list[str]:
    """Append known tool directories that are missing from PATH. Returns what was added."""
    current = os.environ.get("PATH", "")
    separator = os.pathsep
    existing = {Path(item) for item in current.split(separator) if item}
    added: list[str] = []
    for directory in candidate_bin_dirs():
        if directory in existing:
            continue
        existing.add(directory)
        current = f"{current}{separator}{directory}" if current else str(directory)
        added.append(str(directory))
    if added:
        os.environ["PATH"] = current
    return added


def find_tool(name: str) -> str:
    """Locate a tool on PATH or in a well-known install directory."""
    found = shutil.which(name)
    if found:
        return found
    suffixes = [".exe", ".cmd", ".bat"] if os.name == "nt" else [""]
    for directory in candidate_bin_dirs():
        for suffix in suffixes:
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file():
                return str(candidate)
    return ""


def tool_version(executable: str, args: tuple[str, ...] = ("--version",)) -> str:
    try:
        result = subprocess.run(
            [executable, *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=12, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    output = (result.stdout or result.stderr or "").strip().splitlines()
    return output[0].strip()[:200] if output else ""


def _cached_auth(cache_key: str, executable: str, args: list[str]) -> bool:
    now = time.monotonic()
    with _auth_lock:
        cached = _auth_cache.get(cache_key)
        if cached and now - cached[0] < _AUTH_CACHE_TTL:
            return cached[1]
    try:
        result = subprocess.run(
            [executable, *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15, check=False,
        )
        ok = result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    with _auth_lock:
        _auth_cache[cache_key] = (time.monotonic(), ok)
    return ok


def codex_authenticated() -> bool:
    executable = find_tool("codex")
    if not executable:
        return False
    return _cached_auth(f"codex:{executable}", executable, ["login", "status"])


def github_authenticated() -> bool:
    executable = find_tool("gh")
    if not executable:
        return False
    return _cached_auth(f"gh:{executable}", executable, ["auth", "status"])


def clear_auth_cache() -> None:
    with _auth_lock:
        _auth_cache.clear()


def _install_candidates(tool: str) -> list[dict[str, Any]]:
    platform = platform_name()
    plans: dict[str, dict[str, list[list[str]]]] = {
        "codex": {
            "windows": [["npm", "install", "-g", "@openai/codex"]],
            "macos": [["npm", "install", "-g", "@openai/codex"], ["brew", "install", "codex"]],
            "linux": [["npm", "install", "-g", "@openai/codex"]],
        },
        "gh": {
            "windows": [
                ["winget", "install", "--id", "GitHub.cli", "-e", "--accept-source-agreements",
                 "--accept-package-agreements"],
                ["scoop", "install", "gh"],
            ],
            "macos": [["brew", "install", "gh"], ["port", "install", "gh"]],
            "linux": [["brew", "install", "gh"], ["apt-get", "install", "-y", "gh"],
                      ["dnf", "install", "-y", "gh"], ["pacman", "-S", "--noconfirm", "gh"]],
        },
    }
    candidates = plans.get(tool, {}).get(platform, [])
    result: list[dict[str, Any]] = []
    for command in candidates:
        # apt-get/dnf/pacman need root; suggest them instead of hanging on a sudo prompt.
        if command[0] in {"apt-get", "dnf", "pacman"} and not _is_root():
            result.append({"command": command, "available": False,
                           "reason": "нужны права root — запусти команду вручную"})
            continue
        if command[0] == "npm" and not find_tool("npm"):
            result.append({"command": command, "available": False, "reason": "npm не найден"})
            continue
        executable = find_tool(command[0])
        result.append({
            "command": command,
            "available": bool(executable),
            "reason": "" if executable else f"{command[0]} не найден",
        })
    return result


def install_tool(
    tool: str,
    *,
    on_output: Callable[[str], None] | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    """Run the first available installer for a tool and capture its output."""
    if tool not in {"codex", "gh"}:
        raise OrchestratorError(f"Автоустановка недоступна для {tool!r}.")
    candidates = _install_candidates(tool)
    usable = next((item for item in candidates if item["available"]), None)
    if usable is None:
        listing = "; ".join(
            " ".join(item["command"]) + (f" ({item['reason']})" if item["reason"] else "")
            for item in candidates
        ) or "нет известного установщика для этой системы"
        raise OrchestratorError(f"Не удалось установить {tool} автоматически: {listing}.")
    command = [find_tool(usable["command"][0]) or usable["command"][0], *usable["command"][1:]]
    printable = " ".join(usable["command"])
    lines: list[str] = [f"$ {printable}"]
    if on_output:
        on_output(lines[0])
    clear_auth_cache()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        lines.append(f"Установка прервана: превышен таймаут {timeout} с.")
        return {"tool": tool, "command": printable, "ok": False, "returncode": 124,
                "output": lines[-MAX_INSTALL_OUTPUT_LINES:]}
    except OSError as exc:
        lines.append(f"Не удалось запустить установщик ({type(exc).__name__}).")
        return {"tool": tool, "command": printable, "ok": False, "returncode": 1,
                "output": lines[-MAX_INSTALL_OUTPUT_LINES:]}
    for stream in (result.stdout, result.stderr):
        for line in (stream or "").splitlines():
            lines.append(line)
            if on_output:
                on_output(line)
    ok = result.returncode == 0 and bool(find_tool(tool))
    if result.returncode == 0 and not find_tool(tool):
        lines.append("Установщик завершился без ошибки, но команда всё ещё не находится. "
                     "Перезапусти терминал/панель или добавь каталог установки в PATH.")
        ok = False
    lines.append(f"Код возврата: {result.returncode}")
    refresh_path()
    if on_output:
        on_output(lines[-1])
    return {"tool": tool, "command": printable, "ok": ok, "returncode": result.returncode,
            "output": lines[-MAX_INSTALL_OUTPUT_LINES:]}


def manual_instructions(tool: str) -> str:
    if tool == "codex":
        return ("npm install -g @openai/codex  →  codex login   "
                "(документация: https://github.com/openai/codex)")
    if tool == "gh":
        return ("Windows: winget install --id GitHub.cli · macOS: brew install gh · "
                "Linux: https://github.com/cli/cli/blob/trunk/docs/install_linux.md  →  gh auth login")
    return ""


def _tool_report(key: str, *, title: str, purpose: str, required: bool,
                 login_command: str = "", authenticated: bool | None = None,
                 version_args: tuple[str, ...] = ("--version",)) -> dict[str, Any]:
    path = find_tool(key)
    candidates = _install_candidates(key) if key in {"codex", "gh"} else []
    usable = next((item for item in candidates if item["available"]), None)
    detected_via = ""
    if path:
        detected_via = "PATH" if shutil.which(key) else "сканирование каталогов установки"
    report: dict[str, Any] = {
        "key": key,
        "title": title,
        "purpose": purpose,
        "required": required,
        "found": bool(path),
        "path": path,
        "detected_via": detected_via,
        "version": tool_version(path, version_args) if path else "",
        "install": {
            "supported": bool(candidates),
            "command": " ".join(usable["command"]) if usable else "",
            "candidates": [
                {"command": " ".join(item["command"]), "available": item["available"], "reason": item["reason"]}
                for item in candidates
            ],
            "manual": manual_instructions(key),
        },
        "login": {"required": bool(login_command), "command": login_command,
                  "authenticated": bool(authenticated) if path and login_command else None},
    }
    if path and not login_command:
        report["state"] = "ok"
    elif path and authenticated:
        report["state"] = "ok"
    elif path:
        report["state"] = "needs_login"
    else:
        report["state"] = "missing"
    return report


def environment_report(*, include_auth: bool = True, path_added: list[str] | None = None,
                       executor: str = "codex", api_base_url: str = "") -> dict[str, Any]:
    """Describe everything the panel needs to run, with a fix for each gap.

    ``executor`` decides which executor is a blocker: ``api`` and ``chatgpt``
    работают без Codex CLI — первый через ключ или локальный сервер, второй
    через обычный чат ChatGPT, куда промпт и ответ переносит человек.
    """
    api_mode = executor == "api"
    manual_mode = executor == "chatgpt"
    tools = {
        "git": _tool_report(
            "git", title="Git", required=True,
            purpose="Создаёт изолированную ветку и worktree, выполняет слияние.",
        ),
        "codex": _tool_report(
            "codex", title="Codex CLI", required=executor == "codex",
            purpose=("Исполнитель: пишет код и запускает команды в изолированном worktree." if executor == "codex" else
                     "Не обязателен для выбранного исполнителя: правки приносит API или обычный чат ChatGPT."),
            login_command="codex login",
            authenticated=codex_authenticated() if include_auth else None,
        ),
        "gh": _tool_report(
            "gh", title="GitHub CLI (gh)", required=False,
            purpose="Нужен только для GitHub-цели: загрузка issue/PR, push, создание PR и merge.",
            login_command="gh auth login",
            authenticated=github_authenticated() if include_auth else None,
        ),
        "node": _tool_report("node", title="Node.js", required=False,
                             purpose="Нужен для установки Codex CLI через npm и для npm-проверок."),
        "npm": _tool_report("npm", title="npm", required=False,
                            purpose="Пакетный менеджер, которым ставится Codex CLI."),
    }
    jev = secrets.jev_key_status()

    api_keys = secrets.all_key_status()
    from .endpoints import endpoint_provider
    problems: list[dict[str, Any]] = []
    try:
        api_key_provider = endpoint_provider(api_base_url)
    except OrchestratorError as exc:
        api_key_provider = None
        if api_mode:
            problems.append({"id": "api.endpoint", "tool": "api", "severity": "blocker",
                             "title": "Некорректный API-адрес", "detail": str(exc),
                             "fix": "Исправь API-адрес в настройках.", "can_install": False})
    remote_endpoint = api_key_provider is not None

    if not tools["codex"]["found"]:
        problems.append({
            "id": "codex.missing", "tool": "codex",
            "severity": "optional" if (api_mode or manual_mode) else "blocker",
            "title": "Codex CLI не найден",
            "detail": ("Выбран API-исполнитель, поэтому Codex CLI не обязателен. Он понадобится, "
                       "если вернёшь исполнителя «Codex CLI»."
                       if api_mode else
                       "Выбран ручной исполнитель «ChatGPT (обычный чат)»: промпт и ответ переносятся "
                       "через chatgpt.com, Codex CLI не нужен."
                       if manual_mode else "Без Codex CLI панель не может выполнить ни одну задачу."),
            "fix": tools["codex"]["install"]["command"] or "npm install -g @openai/codex",
            "can_install": bool(tools["codex"]["install"]["command"]),
        })
    elif not tools["codex"]["login"]["authenticated"] and include_auth:
        problems.append({
            "id": "codex.login", "tool": "codex", "severity": "optional" if (api_mode or manual_mode) else "blocker",
            "title": "Codex CLI без входа",
            "detail": ("Выбран API-исполнитель: вход в Codex CLI для запуска не нужен."
                       if api_mode else
                       "Выбран ручной исполнитель «ChatGPT (обычный чат)»: вход в Codex CLI не нужен."
                       if manual_mode else
                       "CLI установлен, но не авторизован. Вход интерактивный — выполни команду в терминале."),
            "fix": "codex login", "can_install": False,
        })
    if api_mode and remote_endpoint and not api_keys[api_key_provider]["available"]:
        problems.append({
            "id": "api.key", "tool": api_key_provider, "severity": "blocker",
            "title": f"API-исполнитель без ключа ({api_keys[api_key_provider]['label']})",
            "detail": "Для внешнего API нужен ключ. Локальные Ollama и LM Studio работают без ключа.",
            "fix": "Вставь ключ в раздел «Ключи API» или укажи локальный сервер, например http://127.0.0.1:11434/v1.",
            "can_install": False,
        })
    if not tools["git"]["found"]:
        problems.append({
            "id": "git.missing", "tool": "git", "severity": "blocker",
            "title": "Git не найден", "detail": "Git обязателен: без него нет веток и worktree.",
            "fix": "Установи Git: https://git-scm.com/downloads",
            "can_install": False,
        })
    if not tools["gh"]["found"]:
        problems.append({
            "id": "gh.missing", "tool": "gh", "severity": "optional",
            "title": "GitHub CLI (gh) не найден",
            "detail": "Локальный цикл работает и без него. Он нужен только для GitHub issue/PR, push и merge.",
            "fix": tools["gh"]["install"]["command"] or manual_instructions("gh"),
            "can_install": bool(tools["gh"]["install"]["command"]),
        })
    elif not tools["gh"]["login"]["authenticated"] and include_auth:
        problems.append({
            "id": "gh.login", "tool": "gh", "severity": "optional",
            "title": "gh установлен, но нет входа",
            "detail": "Для GitHub-цели выполни вход в терминале.",
            "fix": "gh auth login", "can_install": False,
        })
    if not jev["available"]:
        problems.append({
            "id": "jev.key", "tool": "jev", "severity": "optional",
            "title": "Jev без ключа",
            "detail": "Без ключа работают бесплатный локальный триаж и кнопка подтверждения слияния. "
                      "Ключ нужен только для Jev-триажа и автослияния после APPROVE.",
            "fix": "Вставь ключ в поле «Ключ Jev» слева или экспортируй TYPESAFE_API_KEY.",
            "can_install": False,
        })

    return {
        "platform": platform_name(),
        "path_added": path_added or [],
        "tools": tools,
        "jev": jev,
        "executor": executor,
        "api_keys": api_keys,
        "problems": problems,
        "blockers": [item for item in problems if item["severity"] == "blocker"],
        "ready": not any(item["severity"] == "blocker" for item in problems),
    }


def doctor_report() -> list[tuple[str, bool, str, str]]:
    """``(name, ok, detail, hint)`` rows for the CLI doctor command."""
    report = environment_report()
    rows: list[tuple[str, bool, str, str]] = [
        ("Python", True, f"{sys.version.split()[0]} · {platform_name()}", ""),
    ]
    for key in ("codex", "git", "gh", "node", "npm"):
        tool = report["tools"][key]
        detail = tool["version"] or ("найден: " + tool["path"] if tool["found"] else "не найден")
        if tool["found"] and tool["login"]["required"] and not tool["login"]["authenticated"]:
            ok = False
            detail = f"{detail} · нет входа"
            hint = tool["login"]["command"]
        elif tool["found"]:
            ok = True
            hint = ""
        else:
            ok = bool(tool["required"]) is False
            hint = tool["install"]["command"] or tool["install"]["manual"]
        if not tool["required"] and not tool["found"]:
            detail += " · опционально"
        rows.append((tool["title"], ok, detail, hint))
    jev = report["jev"]
    rows.append((
        "Jev router",
        True,
        ("ключ найден (" + ("окружение" if jev["source"] == "environment" else jev["path"]) + ")"
         if jev["available"] else "ключ не задан · опционально"),
        "" if jev["available"] else "python -m ai_orchestrate jev-key set",
    ))
    for provider_id in secrets.LLM_PROVIDERS:
        item = report["api_keys"][provider_id]
        rows.append((
            f"{item['label']} API",
            True,
            ("ключ найден (" + ("окружение" if item["source"] == "environment" else item["path"]) + ")"
             if item["available"] else "ключ не задан · нужен только для внешнего API"),
            "" if item["available"] else f"python -m ai_orchestrate keys set {provider_id}",
        ))
    if report["path_added"]:
        rows.append(("PATH", True, "автодобавлены каталоги: " + ", ".join(report["path_added"]), ""))
    return rows


def summarize_environment(report: dict[str, Any]) -> str:
    """One-line human summary used by the CLI and logs."""
    parts = [
        f"{tool['title']}: {'OK' if tool['state'] == 'ok' else tool['state']}"
        for tool in report["tools"].values()
    ]
    parts.append("Jev: " + ("ключ есть" if report["jev"]["available"] else "без ключа"))
    return " · ".join(parts)


def _selftest() -> int:  # pragma: no cover - manual helper
    print(json.dumps(environment_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_selftest())
