from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import CodexUsage, OrchestratorError


def default_usage_path(explicit: str | None = None) -> Path:
    value = explicit or os.environ.get("AI_ORCHESTRATE_USAGE_LOG")
    if value:
        return Path(value).expanduser().resolve()
    return Path.home() / ".ai-orchestrate" / "usage.jsonl"


def read_usage_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OrchestratorError(f"Usage log is invalid at line {line_number}: {path}") from exc
                if not isinstance(entry, dict):
                    raise OrchestratorError(f"Usage log entry {line_number} is not an object: {path}")
                total = entry.get("total_tokens")
                date = entry.get("date")
                if (not isinstance(total, int) or isinstance(total, bool) or total < 0
                        or not isinstance(date, str)):
                    raise OrchestratorError(f"Usage log entry {line_number} has invalid fields: {path}")
                entries.append(entry)
    except OSError as exc:
        raise OrchestratorError(f"Could not read usage log ({type(exc).__name__}): {path}") from exc
    return entries


def tokens_for_date(path: Path, date: str | None = None) -> int:
    target_date = date or datetime.now().astimezone().date().isoformat()
    return sum(entry["total_tokens"] for entry in read_usage_entries(path) if entry["date"] == target_date)


def append_usage(
    path: Path,
    *,
    model: str,
    effort: str,
    role: str,
    attempt: int,
    returncode: int,
    usage: CodexUsage,
) -> bool:
    """Append one telemetry record, never storing prompts, diffs, or credentials."""
    total = usage.total_tokens
    if total is None:
        return False
    now = datetime.now().astimezone()
    entry = {
        "date": now.date().isoformat(),
        "timestamp": now.isoformat(timespec="seconds"),
        "role": role,
        "attempt": attempt,
        "model": model,
        "effort": effort,
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": total,
        "returncode": returncode,
    }
    encoded = (json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # O_APPEND keeps concurrent short JSONL records from overwriting each other.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("short write to usage ledger")
                remaining = remaining[written:]
        finally:
            os.close(fd)
    except OSError as exc:
        raise OrchestratorError(f"Could not write usage log ({type(exc).__name__}): {path}") from exc
    return True


def usage_summary(path: Path) -> tuple[int, int, list[dict[str, Any]]]:
    entries = read_usage_entries(path)
    today = datetime.now().astimezone().date().isoformat()
    today_total = sum(entry["total_tokens"] for entry in entries if entry["date"] == today)
    all_time_total = sum(entry["total_tokens"] for entry in entries)
    return today_total, all_time_total, entries[-5:]
