"""ChatGPT bridge: a human-in-the-loop round trip through the ChatGPT app.

The panel cannot and must not automate the ChatGPT client — extracting output programmatically
is prohibited by OpenAI's terms of use, and this repository documents the same rule. What it can
do is prepare everything for you: build one copy-paste prompt (task + bounded diff + the exact
failing checks), and then take the answer you paste back, apply it inside the isolated worktree
and re-run the project's own checks.

Two answer formats are accepted, because chat models are unreliable at diff syntax:

1. Full file content::

       ### FILE: src/app.py
       ```python
       <полное содержимое файла>
       ```

2. The OpenAI ``apply_patch`` format, wrapped in a fence or not::

       *** Begin Patch
       *** Update File: src/app.py
       @@
        context line
       -old line
       +new line
       *** End Patch
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import OrchestratorError, format_checks, truncate_text
from .llm_api import MAX_FILE_CHARS, safe_join

MAX_ANSWER_CHARS = 200_000
MAX_FILES_PER_ANSWER = 40
_FILE_HEADER_RE = re.compile(r"^###\s*FILE:\s*(?P<path>\S.*?)\s*$", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[A-Za-z0-9_+-]*\s*$")


@dataclass
class FileOperation:
    op: str  # "write" | "patch" | "delete"
    path: str
    content: str = ""
    hunks: list[list[str]] = field(default_factory=list)


@dataclass
class ApplyReport:
    applied: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.applied) and not self.rejected

    def public(self) -> dict[str, Any]:
        return {"applied": self.applied, "rejected": self.rejected,
                "changed_files": [item["path"] for item in self.applied]}


def build_bridge_prompt(
    task: str,
    diff: str,
    status: str,
    checks: list[dict[str, Any]],
    *,
    plan: str = "",
    review: str = "",
    failed_only: bool = True,
) -> str:
    """Assemble the single message the user pastes into the ChatGPT app."""
    relevant = [item for item in checks if item.get("returncode") != 0] if failed_only else list(checks)
    parts = [
        "Ты — инженер, который исправляет конкретную проблему в существующем Git-репозитории.",
        "Отвечай ТОЛЬКО блоками файлов в описанном ниже формате: без пояснений, без markdown-заголовков,",
        "без нумерации шагов. Пути — относительные от корня репозитория. Не присылай файлы, которых нет",
        "в задаче или в diff, и не переписывай то, что уже работает.",
        "",
        "## Задача",
        truncate_text(task.strip() or "(не указана)", 4000),
    ]
    if plan.strip():
        parts += ["", "## План", truncate_text(plan.strip(), 2000)]
    if diff.strip():
        parts += ["", "## Текущий diff (может быть обрезан)", "```diff", truncate_text(diff, 12000), "```"]
    if status.strip():
        parts += ["", "## git status", "```", truncate_text(status, 1500), "```"]
    if relevant:
        parts += ["", "## Проверки, которые не прошли", "```",
                  truncate_text(format_checks(relevant, limit=8000), 8000), "```"]
    if review.strip():
        parts += ["", "## Замечания независимого ревью", truncate_text(review.strip(), 2500)]
    parts += [
        "",
        "## Формат ответа",
        "Маленький файл (до ~400 строк) пришли ПОЛНЫМ содержимым:",
        "### FILE: путь/к/файлу",
        "```",
        "<полное содержимое файла>",
        "```",
        "",
        "Для больших файлов пришли патч в формате apply_patch:",
        "*** Begin Patch",
        "*** Update File: путь/к/файлу",
        "@@",
        " контекстная строка без изменений",
        "-удаляемая строка",
        "+добавляемая строка",
        "*** End Patch",
        "",
        "Файл можно удалить блоком `*** Delete File: путь`. Новый файл — `*** Add File: путь` со строками `+`.",
    ]
    return "\n".join(parts)


def parse_answer(answer: str) -> list[FileOperation]:
    """Extract file operations from a pasted ChatGPT answer."""
    if not isinstance(answer, str) or not answer.strip():
        raise OrchestratorError("Ответ ChatGPT пустой.")
    if len(answer) > MAX_ANSWER_CHARS:
        raise OrchestratorError(f"Ответ длиннее {MAX_ANSWER_CHARS} символов — пришли его частями.")
    operations: list[FileOperation] = []
    operations.extend(_parse_file_blocks(answer))
    operations.extend(_parse_patch_blocks(answer))
    if not operations:
        raise OrchestratorError(
            "В ответе не нашлось ни одного блока файла. Нужны блоки `### FILE: путь` "
            "или `*** Begin Patch … *** End Patch`."
        )
    if len(operations) > MAX_FILES_PER_ANSWER:
        raise OrchestratorError(f"Слишком много файлов в одном ответе (>{MAX_FILES_PER_ANSWER}).")
    return operations


def _strip_fence(lines: list[str]) -> list[str]:
    if len(lines) >= 2 and _FENCE_RE.match(lines[0].strip()) and lines[-1].strip() == "```":
        return lines[1:-1]
    return lines


def _parse_file_blocks(answer: str) -> list[FileOperation]:
    operations: list[FileOperation] = []
    lines = answer.splitlines()
    index = 0
    while index < len(lines):
        header = _FILE_HEADER_RE.match(lines[index].strip())
        if not header:
            index += 1
            continue
        path = header.group("path").strip().strip("`").strip()
        index += 1
        if index < len(lines) and _FENCE_RE.match(lines[index].strip()):
            index += 1
            body: list[str] = []
            while index < len(lines) and lines[index].strip() != "```":
                body.append(lines[index])
                index += 1
            index += 1  # closing fence
        else:
            body = []
            while index < len(lines) and not _FILE_HEADER_RE.match(lines[index].strip()):
                body.append(lines[index])
                index += 1
        operations.append(FileOperation("write", path, "\n".join(body).rstrip("\n") + "\n"))
    return operations


def _parse_patch_blocks(answer: str) -> list[FileOperation]:
    operations: list[FileOperation] = []
    lines = _strip_fence(answer.splitlines())
    index = 0
    while index < len(lines):
        if lines[index].strip() != "*** Begin Patch":
            index += 1
            continue
        index += 1
        current: FileOperation | None = None
        while index < len(lines) and lines[index].strip() != "*** End Patch":
            line = lines[index]
            stripped = line.strip()
            if stripped.startswith("*** Update File:") or stripped.startswith("*** Add File:"):
                op = "patch" if stripped.startswith("*** Update File:") else "write"
                path = stripped.split(":", 1)[1].strip()
                current = FileOperation(op, path, hunks=[])
                operations.append(current)
            elif stripped.startswith("*** Delete File:"):
                operations.append(FileOperation("delete", stripped.split(":", 1)[1].strip()))
                current = None
            elif current is not None and stripped.startswith("@@"):
                pass  # hunk header: context marker only
            elif current is not None:
                current.hunks.append(line)
            index += 1
        index += 1
    # `*** Add File` carries its content as `+` lines inside the same patch block.
    for operation in operations:
        if operation.op == "write" and not operation.content and operation.hunks:
            added = [line[1:] if line.startswith("+") else "" for line in operation.hunks
                     if line.startswith("+") or not line.strip()]
            operation.content = "\n".join(added).rstrip("\n") + "\n"
    return operations


def apply_operations(worktree: Path, operations: list[FileOperation]) -> ApplyReport:
    """Apply parsed operations inside the worktree. Never touches anything outside it."""
    root = worktree.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise OrchestratorError(f"Рабочая копия не найдена: {root}")
    report = ApplyReport()
    for operation in operations:
        try:
            if operation.op == "delete":
                path = safe_join(root, operation.path)
                path.unlink()
                report.applied.append({"path": operation.path, "op": "delete"})
                continue
            path = safe_join(root, operation.path)
            if operation.op == "write":
                if len(operation.content) > MAX_FILE_CHARS:
                    raise OrchestratorError("Файл слишком большой.")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(operation.content, encoding="utf-8", newline="\n")
                report.applied.append({"path": operation.path, "op": "write",
                                       "chars": len(operation.content)})
                continue
            changed = _apply_hunks(path, operation.hunks)
            report.applied.append({"path": operation.path, "op": "patch", "changed_lines": changed})
        except (OrchestratorError, OSError, UnicodeDecodeError) as exc:
            report.rejected.append({"path": operation.path, "op": operation.op,
                                    "error": str(exc) or type(exc).__name__})
    return report


def _apply_hunks(path: Path, hunk_lines: list[str]) -> int:
    if not path.is_file():
        raise OrchestratorError("Файл для патча не найден — для нового файла используй *** Add File.")
    try:
        original = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise OrchestratorError(f"Не удалось прочитать файл ({type(exc).__name__}).") from exc

    search: list[str] = []
    replace: list[str] = []
    for line in hunk_lines:
        if not line:
            search.append("")
            replace.append("")
        elif line[0] == " ":
            search.append(line[1:])
            replace.append(line[1:])
        elif line[0] == "-":
            search.append(line[1:])
        elif line[0] == "+":
            replace.append(line[1:])
        else:
            # A bare line inside a hunk is treated as context, like a lost leading space.
            search.append(line)
            replace.append(line)
    if not search:
        raise OrchestratorError("В патче нет ни одной строки.")

    start = _find_sequence(original, search)
    if start < 0:
        raise OrchestratorError("Контекст патча не совпал с файлом — пришли файл целиком блоком ### FILE.")
    result = original[:start] + replace + original[start + len(search):]
    path.write_text("\n".join(result) + "\n", encoding="utf-8", newline="\n")
    return len(replace)


def _find_sequence(lines: list[str], search: list[str]) -> int:
    if not search:
        return -1
    for index in range(0, len(lines) - len(search) + 1):
        if lines[index:index + len(search)] == search:
            return index
    return -1
