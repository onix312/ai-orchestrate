"""Strict endpoint classification; never route saved credentials by URL substring."""
from urllib.parse import urlsplit

from .core import OrchestratorError

DEFAULT_API_URL = "https://api.openai.com/v1"


def endpoint_provider(value: str) -> str | None:
    """None denotes a local, keyless endpoint. Unknown remote hosts fail closed."""
    url = value.strip() or DEFAULT_API_URL
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise OrchestratorError("Некорректный API-адрес.") from exc
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(c.isspace() for c in url)):
        raise OrchestratorError("API-адрес должен быть HTTP(S) URL без логина, query и fragment.")
    host = parsed.hostname.lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return None
    providers = {"api.openai.com": "openai", "openrouter.ai": "openrouter"}
    if host not in providers or parsed.scheme != "https" or port not in {None, 443}:
        raise OrchestratorError(
            "Сохранённые API-ключи разрешены только для https://api.openai.com и "
            "https://openrouter.ai. Для локальной модели используй localhost, 127.0.0.1 или [::1]."
        )
    return providers[host]
