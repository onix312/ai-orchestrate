"""Local storage for optional API keys.

The Jev key is deliberately NOT part of ``settings.json``: it lives in its own
file with 0600 permissions, it is never echoed back to the UI (only a masked
form is), and :func:`ai_orchestrate.core._codex_env` keeps stripping it from
every Codex child process.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from .core import OrchestratorError, state_dir as _state_dir

JEV_ENV_VAR = "TYPESAFE_API_KEY"
JEV_KEY_FILENAME = "jev-key"

MIN_KEY_CHARS = 8
MAX_KEY_CHARS = 512
# Keys are opaque tokens; allow the characters real providers use and nothing else.
_KEY_RE = re.compile(r"^[A-Za-z0-9._~+/:=@-]+$")

# Set when the process itself loaded the key from the store, so the UI can tell
# "saved locally" apart from "exported in the shell before the panel started".
_activated_from_store = False


def state_dir() -> Path:
    """Same root as every other local file; see :func:`ai_orchestrate.core.state_dir`."""
    return _state_dir()


def jev_key_path() -> Path:
    override = os.environ.get("AI_ORCHESTRATE_JEV_KEY_FILE")
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return state_dir() / JEV_KEY_FILENAME


def read_jev_key_file() -> str:
    path = jev_key_path()
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError) as exc:
        raise OrchestratorError(
            f"Файл ключа Jev не читается ({type(exc).__name__}): {path}"
        ) from exc


def active_jev_key() -> str:
    """Environment wins over the stored file, so an exported key always applies."""
    return os.environ.get(JEV_ENV_VAR, "").strip() or read_jev_key_file()


def jev_key_source() -> str:
    env_value = os.environ.get(JEV_ENV_VAR, "").strip()
    stored = read_jev_key_file()
    if stored and env_value == stored and _activated_from_store:
        return "file"
    if env_value:
        return "environment"
    if stored:
        return "file"
    return ""


def mask_jev_key(key: str) -> str:
    if not key:
        return ""
    if len(key) < 12:
        return "•" * min(len(key), 8)
    return f"{key[:4]}…{key[-4:]} · {len(key)} символов"


def normalize_jev_key(raw: str) -> str:
    """Trim paste artefacts and reject anything that is not a plausible token."""
    if not isinstance(raw, str):
        raise OrchestratorError("Ключ Jev должен быть текстом.")
    key = raw.strip().strip("\"'").strip()
    if not key:
        raise OrchestratorError("Ключ Jev пустой.")
    if len(key) < MIN_KEY_CHARS:
        raise OrchestratorError(f"Ключ Jev слишком короткий: минимум {MIN_KEY_CHARS} символов.")
    if len(key) > MAX_KEY_CHARS:
        raise OrchestratorError(f"Ключ Jev слишком длинный: максимум {MAX_KEY_CHARS} символов.")
    if any(char.isspace() for char in key):
        raise OrchestratorError("Ключ Jev не должен содержать пробелы и переносы строк.")
    if not _KEY_RE.fullmatch(key):
        raise OrchestratorError(
            "Ключ Jev содержит недопустимые символы. Ожидаются буквы, цифры и . _ ~ + / : = @ -"
        )
    return key


def jev_key_status() -> dict:
    source = jev_key_source()
    key = active_jev_key()
    stored = read_jev_key_file()
    env_value = os.environ.get(JEV_ENV_VAR, "").strip()
    return {
        "available": bool(key),
        "source": source,
        "masked": mask_jev_key(key),
        "stored_masked": mask_jev_key(stored),
        "path": str(jev_key_path()),
        "env_var": JEV_ENV_VAR,
        "environment_overrides_file": bool(env_value and stored and env_value != stored),
        "note": (
            "Ключ взят из переменной окружения процесса."
            if source == "environment"
            else "Ключ сохранён локально в файле с правами 0600."
            if source == "file"
            else "Ключ не задан: Jev-триаж и автослияние после Jev недоступны."
        ),
    }


def save_jev_key(raw: str) -> dict:
    key = normalize_jev_key(raw)
    path = jev_key_path()
    temporary: Path | None = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=".jev-key-", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            output.write(key + "\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise OrchestratorError(f"Не удалось сохранить ключ Jev ({type(exc).__name__}): {path}") from exc
    # Activate immediately so the running panel does not need a restart.
    global _activated_from_store
    os.environ[JEV_ENV_VAR] = key
    _activated_from_store = True
    return jev_key_status()


def clear_jev_key() -> dict:
    path = jev_key_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise OrchestratorError(f"Не удалось удалить ключ Jev ({type(exc).__name__}): {path}") from exc
    # Only drop the process copy when the environment did not provide the key.
    global _activated_from_store
    if _activated_from_store:
        os.environ.pop(JEV_ENV_VAR, None)
        _activated_from_store = False
    status = jev_key_status()
    if status["available"]:
        status["note"] = (
            f"Локальный файл удалён, но {JEV_ENV_VAR} задана в окружении процесса; "
            "убери переменную и перезапусти панель, чтобы полностью отключить Jev."
        )
    else:
        status["note"] = "Ключ Jev удалён."
    return status


def activate_stored_jev_key() -> str:
    """Load a stored key into the process environment at startup.

    Returns the resulting source (``environment``, ``file`` or ``""``).
    """
    global _activated_from_store
    if os.environ.get(JEV_ENV_VAR, "").strip():
        return "environment"
    try:
        stored = read_jev_key_file()
    except OrchestratorError:
        return ""
    if stored:
        os.environ[JEV_ENV_VAR] = stored
        _activated_from_store = True
        return "file"
    return ""
