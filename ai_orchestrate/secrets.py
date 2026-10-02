"""Local storage for optional API keys.

No provider key is part of ``settings.json``: every key lives in its own file
with 0600 permissions, is never echoed back to the UI (only a masked form is),
and :func:`ai_orchestrate.core._codex_env` keeps the Jev key out of every Codex
child process.

Supported providers live in :data:`PROVIDERS`; ``jev`` keeps its original
dedicated helpers so existing callers and tests are unaffected.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .core import OrchestratorError, state_dir as _state_dir

JEV_ENV_VAR = "TYPESAFE_API_KEY"
JEV_KEY_FILENAME = "jev-key"

MIN_KEY_CHARS = 8
MAX_KEY_CHARS = 512
# Keys are opaque tokens; allow the characters real providers use and nothing else.
_KEY_RE = re.compile(r"^[A-Za-z0-9._~+/:=@-]+$")


@dataclass(frozen=True)
class ProviderKey:
    id: str
    label: str
    env_var: str
    filename: str
    optional: bool = True  # local Ollama/LM Studio work without any key


PROVIDERS: dict[str, ProviderKey] = {
    provider.id: provider
    for provider in (
        ProviderKey("jev", "Jev", JEV_ENV_VAR, JEV_KEY_FILENAME, optional=False),
        ProviderKey("openai", "OpenAI", "OPENAI_API_KEY", "openai-key"),
        ProviderKey("openrouter", "OpenRouter", "OPENROUTER_API_KEY", "openrouter-key"),
    )
}
JEV_PROVIDER = "jev"
LLM_PROVIDERS = ("openai", "openrouter")

# Set per provider when this process loaded the key from the store, so the UI can
# tell "saved locally" apart from "exported in the shell before the panel started".
_activated_from_store: dict[str, bool] = {}


def state_dir() -> Path:
    """Same root as every other local file; see :func:`ai_orchestrate.core.state_dir`."""
    return _state_dir()


def provider(provider_id: str) -> ProviderKey:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        raise OrchestratorError(f"Неизвестный провайдер ключа: {provider_id}") from None


def key_path(provider_id: str) -> Path:
    meta = provider(provider_id)
    if meta.id == JEV_PROVIDER:
        override = os.environ.get("AI_ORCHESTRATE_JEV_KEY_FILE")
    else:
        override = os.environ.get(f"AI_ORCHESTRATE_{meta.id.upper()}_KEY_FILE")
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return state_dir() / meta.filename


def read_key_file(provider_id: str) -> str:
    meta = provider(provider_id)
    path = key_path(meta.id)
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeDecodeError) as exc:
        raise OrchestratorError(
            f"Файл ключа {meta.label} не читается ({type(exc).__name__}): {path}"
        ) from exc


def active_key(provider_id: str) -> str:
    """Environment wins over the stored file, so an exported key always applies."""
    meta = provider(provider_id)
    return os.environ.get(meta.env_var, "").strip() or read_key_file(meta.id)


def key_source(provider_id: str) -> str:
    meta = provider(provider_id)
    env_value = os.environ.get(meta.env_var, "").strip()
    stored = read_key_file(meta.id)
    if stored and env_value == stored and _activated_from_store.get(meta.id):
        return "file"
    if env_value:
        return "environment"
    if stored:
        return "file"
    return ""


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) < 12:
        return "•" * min(len(key), 8)
    return f"{key[:4]}…{key[-4:]} · {len(key)} символов"


def normalize_key(raw: str, label: str) -> str:
    """Trim paste artefacts and reject anything that is not a plausible token."""
    if not isinstance(raw, str):
        raise OrchestratorError(f"Ключ {label} должен быть текстом.")
    key = raw.strip().strip("\"'").strip()
    if not key:
        raise OrchestratorError(f"Ключ {label} пустой.")
    if len(key) < MIN_KEY_CHARS:
        raise OrchestratorError(f"Ключ {label} слишком короткий: минимум {MIN_KEY_CHARS} символов.")
    if len(key) > MAX_KEY_CHARS:
        raise OrchestratorError(f"Ключ {label} слишком длинный: максимум {MAX_KEY_CHARS} символов.")
    if any(char.isspace() for char in key):
        raise OrchestratorError(f"Ключ {label} не должен содержать пробелы и переносы строк.")
    if not _KEY_RE.fullmatch(key):
        raise OrchestratorError(
            f"Ключ {label} содержит недопустимые символы. Ожидаются буквы, цифры и . _ ~ + / : = @ -"
        )
    return key


def key_status(provider_id: str) -> dict:
    meta = provider(provider_id)
    source = key_source(meta.id)
    key = active_key(meta.id)
    stored = read_key_file(meta.id)
    env_value = os.environ.get(meta.env_var, "").strip()
    empty_note = (
        "Ключ не задан: Jev-триаж и автослияние после Jev недоступны."
        if meta.id == JEV_PROVIDER
        else f"Ключ {meta.label} не задан: доступен только локальный сервер без ключа (Ollama, LM Studio)."
    )
    return {
        "provider": meta.id,
        "label": meta.label,
        "optional": meta.optional,
        "available": bool(key),
        "source": source,
        "masked": mask_key(key),
        "stored_masked": mask_key(stored),
        "path": str(key_path(meta.id)),
        "env_var": meta.env_var,
        "environment_overrides_file": bool(env_value and stored and env_value != stored),
        "note": (
            "Ключ взят из переменной окружения процесса."
            if source == "environment"
            else "Ключ сохранён локально в файле с правами 0600."
            if source == "file"
            else empty_note
        ),
    }


def save_key(provider_id: str, raw: str) -> dict:
    meta = provider(provider_id)
    key = normalize_key(raw, meta.label)
    path = key_path(meta.id)
    temporary: Path | None = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{meta.filename}-", suffix=".tmp", dir=path.parent)
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
        raise OrchestratorError(
            f"Не удалось сохранить ключ {meta.label} ({type(exc).__name__}): {path}"
        ) from exc
    # Activate immediately so the running panel does not need a restart.
    os.environ[meta.env_var] = key
    _activated_from_store[meta.id] = True
    return key_status(meta.id)


def clear_key(provider_id: str) -> dict:
    meta = provider(provider_id)
    path = key_path(meta.id)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise OrchestratorError(
            f"Не удалось удалить ключ {meta.label} ({type(exc).__name__}): {path}"
        ) from exc
    # Only drop the process copy when the environment did not provide the key.
    if _activated_from_store.get(meta.id):
        os.environ.pop(meta.env_var, None)
        _activated_from_store[meta.id] = False
    status = key_status(meta.id)
    if status["available"]:
        status["note"] = (
            f"Локальный файл удалён, но {meta.env_var} задана в окружении процесса; "
            f"убери переменную и перезапусти панель, чтобы полностью отключить {meta.label}."
        )
    else:
        status["note"] = f"Ключ {meta.label} удалён."
    return status


def activate_stored_key(provider_id: str) -> str:
    """Load a stored key into the process environment at startup.

    Returns the resulting source (``environment``, ``file`` or ``""``).
    """
    meta = provider(provider_id)
    if os.environ.get(meta.env_var, "").strip():
        return "environment"
    try:
        stored = read_key_file(meta.id)
    except OrchestratorError:
        return ""
    if stored:
        os.environ[meta.env_var] = stored
        _activated_from_store[meta.id] = True
        return "file"
    return ""


def all_key_status() -> dict[str, dict]:
    return {name: key_status(name) for name in PROVIDERS}


def reset_activation_state() -> None:
    """Forget which keys this process loaded from the store (used by tests)."""
    _activated_from_store.clear()


# --- Jev-specific helpers (original public API, kept stable) -------------------


def jev_key_path() -> Path:
    return key_path(JEV_PROVIDER)


def read_jev_key_file() -> str:
    return read_key_file(JEV_PROVIDER)


def active_jev_key() -> str:
    return active_key(JEV_PROVIDER)


def jev_key_source() -> str:
    return key_source(JEV_PROVIDER)


def mask_jev_key(key: str) -> str:
    return mask_key(key)


def normalize_jev_key(raw: str) -> str:
    return normalize_key(raw, "Jev")


def jev_key_status() -> dict:
    return key_status(JEV_PROVIDER)


def save_jev_key(raw: str) -> dict:
    return save_key(JEV_PROVIDER, raw)


def clear_jev_key() -> dict:
    return clear_key(JEV_PROVIDER)


def activate_stored_jev_key() -> str:
    return activate_stored_key(JEV_PROVIDER)
