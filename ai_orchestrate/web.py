from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .core import OrchestratorError, configured_lanes
from .prompts import PROFESSIONS, ROLE_PROMPTS
from .usage import default_usage_path, tokens_for_date
from .workflow import WorkflowRequest, WorkflowStopped, run_workflow, suggest_checks


MAX_REQUEST_BYTES = 128_000
MAX_EVENTS_PER_JOB = 1600
MAX_RETAINED_JOBS = 20


@dataclass
class RunJob:
    id: str
    status: str = "queued"
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str = ""
    cancel: Event = field(default_factory=Event)
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    next_event_id: int = 1


class RunManager:
    def __init__(self, workspace_root: Path, usage_path: Path | None = None) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        if not self.workspace_root.is_dir():
            raise OrchestratorError(f"Workspace root does not exist: {self.workspace_root}")
        self.usage_path = usage_path or default_usage_path()
        self._lock = RLock()
        self._jobs: dict[str, RunJob] = {}
        self._active_job: str | None = None

    def _safe_repo_path(self, value: str | None) -> Path:
        raw = (value or str(self.workspace_root)).strip()
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise OrchestratorError("Указанная папка проекта не найдена.") from exc
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise OrchestratorError(
                f"Папка должна находиться внутри разрешённой области: {self.workspace_root}"
            ) from exc
        if not resolved.is_dir():
            raise OrchestratorError("Выбранный путь не является папкой.")
        return resolved

    def status(self) -> dict[str, Any]:
        default_repo = self.workspace_root
        if not (default_repo / ".git").exists():
            candidates = sorted(path for path in self.workspace_root.iterdir()
                               if path.is_dir() and (path / ".git").exists())
            if candidates:
                default_repo = candidates[0]
        with self._lock:
            active = self._active_job
        try:
            usage_today = tokens_for_date(self.usage_path)
            usage_error = ""
        except OrchestratorError:
            usage_today = None
            usage_error = "Usage ledger недоступен или повреждён; лимит нельзя проверить."
        return {
            "workspace_root": str(self.workspace_root),
            "default_repo": str(default_repo),
            "codex_available": shutil.which("codex") is not None,
            "jev_available": bool(os.environ.get("TYPESAFE_API_KEY")),
            "usage_today_tokens": usage_today,
            "usage_error": usage_error,
            "models": configured_lanes(),
            "professions": [
                {"key": item.key, "title": item.title, "description": item.description}
                for item in PROFESSIONS
            ],
            "role_prompts": ROLE_PROMPTS,
            "active_job": active,
            "check_suggestions": suggest_checks(default_repo) if default_repo.is_dir() else [],
        }

    def start(self, payload: dict[str, Any]) -> RunJob:
        repo = self._safe_repo_path(str(payload.get("repo", "")))
        task = payload.get("task")
        if not isinstance(task, str) or len(task) > 48_000:
            raise OrchestratorError("Задача должна быть текстом до 48 000 символов.")
        raw_checks = payload.get("checks", "")
        if isinstance(raw_checks, str):
            checks = [line.strip() for line in raw_checks.splitlines() if line.strip()]
        elif isinstance(raw_checks, list) and all(isinstance(line, str) for line in raw_checks):
            checks = [line.strip() for line in raw_checks if line.strip()]
        else:
            raise OrchestratorError("Список проверок должен быть текстом или массивом команд.")
        if len(checks) > 12:
            raise OrchestratorError("Можно указать не более 12 команд проверки.")

        mode = str(payload.get("mode", "full"))
        defaults = {
            "max_repairs": 1 if mode == "full" else 0,
            "max_model_calls": 5 if mode == "full" else 1,
        }
        request = WorkflowRequest(
            repo=repo,
            task=task,
            checks=checks,
            mode=mode,
            profession=str(payload.get("profession", "developer")),
            router=str(payload.get("router", "local")),
            lane=str(payload["lane"]) if payload.get("lane") else None,
            max_repairs=_bounded_int(payload.get("max_repairs", defaults["max_repairs"]), 0, 3, "max_repairs"),
            max_model_calls=_bounded_int(payload.get("max_model_calls", defaults["max_model_calls"]), 1, 10, "max_model_calls"),
            prompt_token_budget=_bounded_int(payload.get("prompt_token_budget", 16000), 100, 100000, "prompt_token_budget"),
            max_run_tokens=_bounded_int(payload.get("max_run_tokens", 60000), 100, 1000000, "max_run_tokens"),
            daily_token_budget=_bounded_int(payload.get("daily_token_budget", 120000), 100, 10000000, "daily_token_budget"),
            codex_timeout=_bounded_int(payload.get("codex_timeout", 1800), 10, 7200, "codex_timeout"),
            check_timeout=_bounded_int(payload.get("check_timeout", 300), 1, 3600, "check_timeout"),
            allow_dirty=payload.get("allow_dirty") is True,
        )
        if request.router == "jev" and not os.environ.get("TYPESAFE_API_KEY"):
            raise OrchestratorError("Для Jev не задан TYPESAFE_API_KEY. Выбери локальный роутер или настрой ключ.")

        with self._lock:
            if self._active_job is not None:
                active = self._jobs.get(self._active_job)
                if active and active.status in {"queued", "running"}:
                    raise OrchestratorError("Уже выполняется одна задача. Дождись её завершения или отмени.")
            job = RunJob(id=uuid.uuid4().hex[:12])
            self._jobs[job.id] = job
            self._active_job = job.id
            self._trim_jobs()
            thread = Thread(target=self._worker, args=(job, request), daemon=True, name=f"orchestrator-{job.id}")
            thread.start()
            return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in {"queued", "running"}:
                return False
            job.cancel.set()
            self._append_event(job, {
                "event": "cancel.requested", "stage": "final", "role": "Оркестратор",
                "message": "Запрошена отмена; текущая команда будет остановлена или завершится первой.", "data": {},
            })
            return True

    def fetch(self, job_id: str, after: int = 0) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            events = [item for item in job.events if item["id"] > after]
            return {
                "id": job.id,
                "status": job.status,
                "created_at": job.created_at,
                "events": events,
                "result": job.result,
                "error": job.error,
                "last_event_id": job.next_event_id - 1,
            }

    def _append_event(self, job: RunJob, event: dict[str, Any]) -> None:
        item = {
            "id": job.next_event_id,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            **event,
        }
        job.next_event_id += 1
        job.events.append(item)
        if len(job.events) > MAX_EVENTS_PER_JOB:
            del job.events[: len(job.events) - MAX_EVENTS_PER_JOB]

    def _worker(self, job: RunJob, request: WorkflowRequest) -> None:
        with self._lock:
            job.status = "running"
            self._append_event(job, {
                "event": "run.started", "stage": "preflight", "role": "Оркестратор",
                "message": "Задача принята; запускаю пошаговый цикл.", "data": {},
            })
        try:
            result = run_workflow(
                request,
                emit=lambda event: self._job_event(job, event),
                cancel_event=job.cancel,
                usage_path=self.usage_path,
            )
            with self._lock:
                job.result = result
                job.status = "complete" if result.get("status") == "complete" else "incomplete"
        except Exception as exc:
            cancelled = job.cancel.is_set()
            incomplete = isinstance(exc, WorkflowStopped) and not cancelled
            with self._lock:
                job.status = "cancelled" if cancelled else "incomplete" if incomplete else "failed"
                job.error = str(exc)
                event_name = "run.cancelled" if cancelled else "run.incomplete" if incomplete else "run.failed"
                self._append_event(job, {
                    "event": event_name,
                    "stage": "final",
                    "role": "Оркестратор",
                    "message": str(exc),
                    "data": {},
                })
        finally:
            with self._lock:
                if self._active_job == job.id:
                    self._active_job = None

    def _job_event(self, job: RunJob, event: dict[str, Any]) -> None:
        with self._lock:
            if job.status in {"queued", "running"}:
                self._append_event(job, event)

    def _trim_jobs(self) -> None:
        if len(self._jobs) <= MAX_RETAINED_JOBS:
            return
        removable = [key for key, job in self._jobs.items()
                     if key != self._active_job and job.status not in {"queued", "running"}]
        for key in removable[: max(0, len(self._jobs) - MAX_RETAINED_JOBS)]:
            self._jobs.pop(key, None)


def _bounded_int(value: Any, minimum: int, maximum: int, field_name: str) -> int:
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise OrchestratorError(f"Некорректное значение лимита {field_name}.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OrchestratorError(f"Некорректное значение лимита {field_name}.") from exc
    if not minimum <= number <= maximum:
        raise OrchestratorError(f"Значение {field_name} должно быть от {minimum} до {maximum}.")
    return number


def make_handler(manager: RunManager) -> type[BaseHTTPRequestHandler]:
    static_file = Path(__file__).parent / "static" / "index.html"

    class Handler(BaseHTTPRequestHandler):
        server_version = "ai-orchestrate/0.3"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} {format % args}")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: Any) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _same_origin(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True
            origin_host = urlsplit(origin).netloc
            return origin_host == self.headers.get("Host", "")

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                try:
                    body = static_file.read_bytes()
                except OSError:
                    self._json(500, {"error": "UI static file is missing."})
                    return
                self._send(200, body, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                self._json(200, manager.status())
                return
            if parsed.path == "/api/checks":
                query = parse_qs(parsed.query)
                try:
                    repo = manager._safe_repo_path(query.get("repo", [""])[0])
                    self._json(200, {"repo": str(repo), "checks": suggest_checks(repo)})
                except OrchestratorError as exc:
                    self._json(400, {"error": str(exc)})
                return
            if parsed.path.startswith("/api/runs/"):
                parts = parsed.path.strip("/").split("/")
                if len(parts) != 3:
                    self._json(404, {"error": "Run not found."})
                    return
                query = parse_qs(parsed.query)
                try:
                    after = max(0, int(query.get("after", ["0"])[0]))
                except ValueError:
                    after = 0
                result = manager.fetch(parts[2], after=after)
                if result is None:
                    self._json(404, {"error": "Run not found."})
                else:
                    self._json(200, result)
                return
            self._json(404, {"error": "Not found."})

        def do_POST(self) -> None:
            if not self._same_origin():
                self._json(403, {"error": "Cross-origin request blocked."})
                return
            parsed = urlsplit(self.path)
            if parsed.path == "/api/runs":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    self._json(413, {"error": f"Request must be between 1 and {MAX_REQUEST_BYTES} bytes."})
                    return
                try:
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise OrchestratorError("Request body must be a JSON object.")
                    job = manager.start(payload)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self._json(400, {"error": "Invalid JSON."})
                    return
                except OrchestratorError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                self._json(202, {"id": job.id, "status": job.status})
                return
            if parsed.path.startswith("/api/runs/") and parsed.path.endswith("/cancel"):
                parts = parsed.path.strip("/").split("/")
                if len(parts) != 4:
                    self._json(404, {"error": "Run not found."})
                    return
                if manager.cancel(parts[2]):
                    self._json(202, {"cancel_requested": True})
                else:
                    self._json(404, {"error": "Run is missing or already finished."})
                return
            self._json(404, {"error": "Not found."})

    return Handler


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace_root: Path | None = None,
    usage_path: Path | None = None,
) -> int:
    root = workspace_root or Path.cwd()
    manager = RunManager(root, usage_path=usage_path)
    server = ThreadingHTTPServer((host, port), make_handler(manager))
    server.daemon_threads = True
    print(f"ai-orchestrate UI: http://{host}:{server.server_port}  (workspace: {manager.workspace_root})", flush=True)
    if host == "0.0.0.0":
        print("Warning: UI is reachable through the network; keep the preview/private workspace trusted.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping ai-orchestrate UI...")
    finally:
        server.server_close()
    return 0
