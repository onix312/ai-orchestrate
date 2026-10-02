from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable

from .core import (
    CodexResult,
    OrchestratorError,
    configured_lanes,
    ensure_clean_git,
    estimate_prompt_tokens,
    format_checks,
    git_snapshot,
    next_lane_name,
    review_passed,
    route_with_lane,
    run_checks,
    run_codex,
    split_command,
    truncate_text,
)
from .prompts import PROFESSIONS, build_developer_prompt, build_planning_prompt, build_reviewer_prompt
from .usage import append_usage, tokens_for_date


@dataclass(frozen=True)
class WorkflowRequest:
    repo: Path
    task: str
    checks: list[str]
    mode: str = "full"
    profession: str = "developer"
    router: str = "local"
    lane: str | None = None
    luna_model: str = ""
    sol_model: str = ""
    review_model: str = ""
    max_repairs: int = 1
    max_model_calls: int = 5
    prompt_token_budget: int = 16000
    max_run_tokens: int = 60000
    daily_token_budget: int = 120000
    codex_timeout: int = 1800
    check_timeout: int = 300
    allow_dirty: bool = False


class WorkflowStopped(Exception):
    pass


def _emit(callback: Callable[..., None], event: str, stage: str, message: str,
          *, role: str | None = None, data: dict[str, Any] | None = None) -> None:
    callback({
        "event": event,
        "stage": stage,
        "role": role,
        "message": message,
        "data": data or {},
    })


def _validate_request(request: WorkflowRequest) -> tuple[Path, list[str]]:
    repo = request.repo.expanduser().resolve()
    if not repo.is_dir():
        raise OrchestratorError(f"Папка проекта не найдена: {repo}")
    if not request.task.strip():
        raise OrchestratorError("Опиши задачу перед запуском.")
    if request.mode not in {"quick", "full"}:
        raise OrchestratorError("Неизвестный режим: выбери quick или full.")
    if request.router not in {"local", "jev"}:
        raise OrchestratorError("Роутер должен быть local или jev.")
    if request.profession not in {item.key for item in PROFESSIONS}:
        raise OrchestratorError("Выбрана неизвестная профессия.")
    model_lanes = configured_lanes(
        luna_model=request.luna_model or None,
        sol_model=request.sol_model or None,
    )
    if request.lane and request.lane not in model_lanes:
        raise OrchestratorError("Выбрана неизвестная полоса модели.")
    if request.lane and request.router == "jev":
        raise OrchestratorError("Выбери либо Jev-роутер, либо ручную полосу модели.")
    if not 0 <= request.max_repairs <= 3:
        raise OrchestratorError("Число автоматических исправлений должно быть от 0 до 3.")
    if not 1 <= request.max_model_calls <= 10:
        raise OrchestratorError("Лимит вызовов моделей должен быть от 1 до 10.")
    if min(request.prompt_token_budget, request.max_run_tokens, request.daily_token_budget,
           request.codex_timeout, request.check_timeout) < 1:
        raise OrchestratorError("Лимиты и таймауты должны быть положительными.")
    if not shutil.which("codex"):
        raise OrchestratorError("Codex CLI не найден в PATH. Установи его и выполни codex login.")

    try:
        if request.allow_dirty:
            subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=repo,
                           capture_output=True, text=True, check=True)
        else:
            ensure_clean_git(repo)
    except (subprocess.CalledProcessError, OSError) as exc:
        if request.allow_dirty:
            raise OrchestratorError("Папка должна быть Git-репозиторием с хотя бы одним коммитом.") from exc
        raise

    checks = [item.strip() for item in request.checks if item.strip()]
    if not checks:
        checks = suggest_checks(repo)
    if not checks:
        raise OrchestratorError("Автопроверки не найдены. Добавь хотя бы одну команду проверки в поле «Проверки».")
    for command in checks:
        try:
            args = split_command(command)
        except ValueError as exc:
            raise OrchestratorError(f"Не удалось разобрать команду проверки {command!r}: {exc}") from exc
        if not args or not shutil.which(args[0]):
            raise OrchestratorError(f"Команда проверки не найдена: {command}")
    return repo, checks


def suggest_checks(repo: Path) -> list[str]:
    """Suggest only checks that look configured in the selected repository."""
    package_json = repo / "package.json"
    if package_json.is_file():
        try:
            package = json.loads(package_json.read_text(encoding="utf-8"))
            if "test" in package.get("scripts", {}) and shutil.which("npm"):
                return ["npm test"]
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    if (repo / "pytest.ini").exists() or (repo / "tests").is_dir() and shutil.which("pytest"):
        if shutil.which("pytest"):
            return ["pytest -q"]
    if (repo / "tests").is_dir() and shutil.which("python"):
        return ["python -m unittest discover -s tests"]
    if (repo / "pyproject.toml").is_file() and shutil.which("python"):
        return ["python -m compileall -q ."]
    return []


def _format_codex_event(event: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    kind = event.get("type")
    if kind == "thread.started":
        return "codex.session", "Создана сессия Codex", {"thread_id": event.get("thread_id")}
    if kind == "turn.started":
        return "codex.turn", "Модель начала этап", {}
    if kind == "turn.completed":
        usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
        return "codex.turn", "Этап модели завершён", {
            key: usage.get(key) for key in ("input_tokens", "cached_input_tokens", "output_tokens")
            if usage.get(key) is not None
        }
    if kind == "turn.failed":
        error = event.get("error") if isinstance(event.get("error"), dict) else {}
        return "codex.error", "Codex завершил этап с ошибкой", {"message": str(error.get("message", ""))[:600]}
    if not isinstance(kind, str) or not kind.startswith("item."):
        return None
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    item_type = item.get("type") or item.get("item_type") or "item"
    if item_type == "reasoning":
        return "codex.activity", "Модель анализирует задачу", {"status": item.get("status", "")}
    if item_type == "command_execution":
        command = str(item.get("command", ""))[:240]
        if kind == "item.started":
            return "tool.started", f"Запущена команда: {command or 'команда'}", {"command": command}
        return "tool.completed", f"Команда завершена (exit {item.get('exit_code', '?')})", {
            "command": command,
            "exit_code": item.get("exit_code"),
            "output_chars": len(str(item.get("aggregated_output", ""))),
        }
    if item_type == "file_change":
        return "file.change", f"Изменение файла: {item.get('path', 'путь не указан')}", {
            "path": str(item.get("path", ""))[:300],
            "status": item.get("status", ""),
        }
    if item_type == "agent_message" and kind == "item.completed":
        message = str(item.get("text", ""))
        return "codex.message", "Codex подготовил сообщение по этапу", {"excerpt": message[:700]}
    if item_type == "mcp_tool_call":
        return "tool.mcp", f"Вызван MCP-инструмент: {item.get('server', item.get('name', 'tool'))}", {}
    if item_type == "web_search":
        return "tool.search", "Модель выполнила веб-поиск", {}
    return "codex.activity", f"Codex: {item_type}", {"status": item.get("status", "")}


def _review_model(writer_model: str, *, requested: str = "", luna_model: str = "", sol_model: str = "") -> str:
    import os

    configured = requested or os.environ.get("AI_ORCHESTRATE_REVIEW_MODEL")
    if configured:
        return configured
    lanes = configured_lanes(luna_model=luna_model or None, sol_model=sol_model or None)
    sol, luna = lanes["ESCALATE"][0], lanes["SMALL"][0]
    return sol if writer_model != sol else luna


def run_workflow(
    request: WorkflowRequest,
    *,
    emit: Callable[[dict[str, Any]], None],
    cancel_event: Event,
    usage_path: Path,
) -> dict[str, Any]:
    """Run the full bounded planner → coder → checks → reviewer → repair cycle."""
    started = time.monotonic()
    _emit(emit, "stage.started", "preflight", "Проверяю проект, Git и команды тестирования.")
    repo, checks = _validate_request(request)
    if cancel_event.is_set():
        raise WorkflowStopped("Задача отменена до старта.")
    _emit(emit, "stage.completed", "preflight", "Проект готов к запуску.", data={"repo": str(repo), "checks": checks})

    daily_start = tokens_for_date(usage_path)
    if daily_start >= request.daily_token_budget:
        raise OrchestratorError(
            f"Дневной лимит уже израсходован: {daily_start:,}/{request.daily_token_budget:,} токенов."
        )
    if not shutil.which("git"):
        raise OrchestratorError("Git не найден в PATH.")

    _emit(emit, "stage.started", "routing",
          "Отправляю задачу на триаж Jev." if request.router == "jev" and not request.lane
          else "Локально выбираю самый лёгкий подходящий уровень.",
          role="Jev" if request.router == "jev" and not request.lane else "Локальный роутер")
    lane_name, (model, effort) = route_with_lane(
        request.task,
        lane=request.lane,
        router=request.router,
        models={"luna_model": request.luna_model, "sol_model": request.sol_model},
        on_event=lambda name, data: _emit(
            emit,
            f"router.{name}",
            "routing",
            str(data.get("message") or name),
            role="Jev" if request.router == "jev" and not request.lane else "Локальный роутер",
            data=data,
        ),
    )
    _emit(emit, "stage.completed", "routing", f"Выбран маршрут {lane_name}: {model} / {effort}.",
          role="Jev" if request.router == "jev" and not request.lane else "Локальный роутер",
          data={"lane": lane_name, "model": model, "effort": effort})

    model_calls = 0
    prompt_tokens = 0
    run_tokens = 0
    usage_known = True
    ledger_ok = True
    plan = ""
    all_checks: list[dict[str, Any]] = []
    final_review = ""
    last_model_result = CodexResult(0)

    def ensure_can_call(prompt: str) -> int:
        nonlocal prompt_tokens
        if cancel_event.is_set():
            raise WorkflowStopped("Пользователь отменил задачу.")
        if model_calls >= request.max_model_calls:
            raise WorkflowStopped(f"Достигнут лимит вызовов моделей: {model_calls}/{request.max_model_calls}.")
        if not usage_known:
            raise WorkflowStopped("Codex не сообщил usage; останавливаю цикл, чтобы не обходить токеновый бюджет.")
        if not ledger_ok:
            raise WorkflowStopped("Не удалось обновить usage-лог; новые вызовы остановлены.")
        if run_tokens >= request.max_run_tokens:
            raise WorkflowStopped(f"Достигнут лимит задачи: {run_tokens:,}/{request.max_run_tokens:,} токенов.")
        if daily_start + run_tokens >= request.daily_token_budget:
            raise WorkflowStopped(
                f"Достигнут дневной лимит: {daily_start + run_tokens:,}/{request.daily_token_budget:,} токенов."
            )
        estimate = estimate_prompt_tokens(prompt)
        if prompt_tokens + estimate > request.prompt_token_budget:
            raise WorkflowStopped(
                f"Следующий промпт превышает лимит текста: ~{prompt_tokens + estimate:,}/"
                f"{request.prompt_token_budget:,} токенов."
            )
        prompt_tokens += estimate
        return estimate

    def model_call(role: str, stage: str, prompt: str, *, call_model: str, call_effort: str,
                   sandbox: str) -> CodexResult:
        nonlocal model_calls, run_tokens, usage_known, ledger_ok
        estimate = ensure_can_call(prompt)
        model_calls += 1
        _emit(emit, "model.started", stage,
              f"{role}: запуск {call_model} / {call_effort}.", role=role,
              data={"model": call_model, "effort": call_effort, "sandbox": sandbox,
                    "prompt_estimate": estimate, "call": model_calls, "call_limit": request.max_model_calls})
        call_started = time.monotonic()

        def on_codex_event(raw_event: dict[str, Any]) -> None:
            formatted = _format_codex_event(raw_event)
            if formatted:
                event_name, message, data = formatted
                _emit(emit, event_name, stage, message, role=role, data=data)

        result = run_codex(
            repo, prompt, call_model, call_effort, timeout=request.codex_timeout,
            sandbox=sandbox, on_event=on_codex_event, cancel_event=cancel_event,
        )
        elapsed = round(time.monotonic() - call_started, 2)
        total = result.usage.total_tokens
        usage_known &= total is not None
        if total is not None:
            run_tokens += total
            _emit(emit, "usage", stage, f"{role}: {total:,} токенов за вызов.", role=role,
                  data={"input_tokens": result.usage.input_tokens,
                        "cached_input_tokens": result.usage.cached_input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": total, "run_total": run_tokens,
                        "daily_total": daily_start + run_tokens})
            try:
                recorded = append_usage(
                    usage_path, model=call_model, effort=call_effort, role=role,
                    attempt=model_calls, returncode=result.returncode, usage=result.usage,
                )
                ledger_ok &= recorded
            except OrchestratorError as exc:
                ledger_ok = False
                _emit(emit, "warning", stage, f"Не удалось записать usage: {exc}", role=role)
        else:
            _emit(emit, "usage.unknown", stage,
                  "Codex не вернул token usage; дальнейшие вызовы будут остановлены при включённом лимите.",
                  role=role)
        if result.final_message:
            _emit(emit, "model.message", stage, f"{role} завершил этап.", role=role,
                  data={"excerpt": truncate_text(result.final_message, 1800)})
        _emit(emit, "model.completed", stage,
              f"{role}: завершено за {elapsed:.1f} с, exit {result.returncode}.", role=role,
              data={"model": call_model, "effort": call_effort,
                    "returncode": result.returncode, "elapsed_seconds": elapsed,
                    "cancelled": result.cancelled})
        return result

    if request.mode == "full":
        plan_prompt = build_planning_prompt(request.task)
        _emit(emit, "stage.started", "planning", "Аналитик формирует критерии, архитектор — короткий план.",
              role="Аналитик + архитектор")
        plan_result = model_call("Аналитик + архитектор", "planning", plan_prompt,
                                 call_model=model, call_effort="low", sandbox="read-only")
        if plan_result.cancelled or cancel_event.is_set():
            raise WorkflowStopped("Планирование отменено.")
        if plan_result.returncode != 0:
            raise WorkflowStopped("Не удалось завершить этап аналитика/архитектора.")
        plan = truncate_text(plan_result.final_message.strip(), 3500)
        _emit(emit, "stage.completed", "planning",
              "План готов." if plan else "План не получен; разработчик продолжит по исходной задаче.",
              role="Аналитик + архитектор", data={"plan": truncate_text(plan, 3500)})

    repair_feedback = ""
    repair_count = 0
    max_iterations = request.max_repairs + 1
    for iteration in range(1, max_iterations + 1):
        profession_name = next((item.title for item in PROFESSIONS if item.key == request.profession), "Разработчик")
        _emit(emit, "stage.started", "implementation",
              f"{profession_name} приступает к реализации.", role=profession_name,
              data={"iteration": iteration, "lane": lane_name, "model": model, "effort": effort})
        developer_prompt = build_developer_prompt(
            request.profession,
            request.task,
            plan=plan,
            repair_feedback=repair_feedback,
            checks=checks,
        )
        last_model_result = model_call(profession_name, "implementation", developer_prompt,
                                       call_model=model, call_effort=effort, sandbox="workspace-write")
        if last_model_result.cancelled or cancel_event.is_set():
            raise WorkflowStopped("Реализация отменена.")
        _emit(emit, "stage.completed", "implementation",
              "Модель завершила правки." if last_model_result.returncode == 0 else "Модель завершилась с ошибкой.",
              role=profession_name, data={"returncode": last_model_result.returncode,
                                          "passed": last_model_result.returncode == 0})

        _emit(emit, "stage.started", "testing", "Запускаю детерминированные проверки проекта.", role="QA / тестирование")
        all_checks = run_checks(repo, checks, timeout=request.check_timeout, on_event=lambda event, data: _emit(
            emit,
            event,
            "testing",
            (f"Запускаю {data.get('command', '')}" if event == "check.started" else
             f"Проверка завершена: exit {data.get('returncode', '?')}"),
            role="QA / тестирование",
            data=data,
        ), cancel_event=cancel_event)
        if cancel_event.is_set() or any(item.get("cancelled") is True for item in all_checks):
            raise WorkflowStopped("Проверки отменены.")
        if any(item["returncode"] == 124 for item in all_checks):
            raise WorkflowStopped("Проверка превысила таймаут; не запускаю повторный агент поверх неё.")
        for check in all_checks:
            _emit(emit, "check.result", "testing",
                  f"{'PASS' if check['returncode'] == 0 else 'FAIL'} — {check['command']}",
                  role="QA / тестирование",
                  data={"command": check["command"], "returncode": check["returncode"],
                        "output": truncate_text(check.get("output", ""), 1000)})
        checks_passed = bool(all_checks) and all(item["returncode"] == 0 for item in all_checks)
        _emit(emit, "stage.completed", "testing",
              "Все обязательные проверки прошли." if checks_passed else "Есть неуспешные проверки.",
              role="QA / тестирование", data={"passed": checks_passed, "count": len(all_checks)})
        implementation_passed = last_model_result.returncode == 0 and checks_passed

        if implementation_passed and request.mode == "full":
            _emit(emit, "stage.started", "review", "Ревьюер независимо проверяет diff и результаты тестов.",
                  role="Ревьюер")
            diff, status = git_snapshot(repo, max_chars=9000)
            reviewer_model = _review_model(
                model,
                requested=request.review_model,
                luna_model=request.luna_model,
                sol_model=request.sol_model,
            )
            review_text = build_reviewer_prompt(request.task, plan, diff, status, all_checks)
            review_result = model_call("Ревьюер", "review", review_text,
                                       call_model=reviewer_model, call_effort="low", sandbox="read-only")
            if review_result.cancelled or cancel_event.is_set():
                raise WorkflowStopped("Ревью отменено.")
            if review_result.returncode != 0:
                raise WorkflowStopped("Независимый ревьюер завершился с ошибкой.")
            final_review = review_result.final_message.strip()
            review_ok = review_passed(final_review)
            _emit(emit, "stage.completed", "review",
                  "Ревью пройдено." if review_ok else "Ревью нашло замечания или не вернуло PASS.",
                  role="Ревьюер", data={"passed": review_ok, "findings": truncate_text(final_review, 2500)})
            if review_ok:
                result = _finish("complete", request, lane_name, model_calls, run_tokens, prompt_tokens,
                                 started, all_checks, plan, final_review)
                _emit(emit, "run.completed", "final", "Полный цикл завершён успешно.", role="Итог", data=result)
                return result
            repair_feedback = "Ревьюер сообщил:\n" + truncate_text(final_review, 2500)

        elif implementation_passed:
            result = _finish("complete", request, lane_name, model_calls, run_tokens, prompt_tokens,
                             started, all_checks, plan, "")
            _emit(emit, "run.completed", "final", "Задача завершена: проверки прошли.", role="Итог", data=result)
            return result

        if not implementation_passed:
            repair_feedback = (
                f"Кодер завершился с exit {last_model_result.returncode}.\n"
                f"Проверки:\n{format_checks(all_checks, limit=3000)}"
            )
        if iteration >= max_iterations:
            break
        next_name = next_lane_name(lane_name)
        if next_name is not None:
            lane_name = next_name
            model, effort = configured_lanes(
                luna_model=request.luna_model or None,
                sol_model=request.sol_model or None,
            )[lane_name]
        repair_count += 1
        _emit(emit, "stage.retry", "implementation",
              f"Начинаю исправление {repair_count}/{request.max_repairs} с контекстом только по найденным проблемам.",
              role=profession_name, data={"lane": lane_name, "model": model, "effort": effort})

    final_status = "incomplete"
    result = _finish(final_status, request, lane_name, model_calls, run_tokens, prompt_tokens,
                     started, all_checks, plan, final_review)
    _emit(emit, "run.incomplete", "final", "Цикл остановлен: остались ошибки или замечания.",
          role="Итог", data=result)
    return result


def _finish(status: str, request: WorkflowRequest, lane: str, calls: int, tokens: int,
            prompt_tokens: int, started: float, checks: list[dict], plan: str, review: str) -> dict[str, Any]:
    return {
        "status": status,
        "repo": str(request.repo),
        "mode": request.mode,
        "profession": request.profession,
        "lane": lane,
        "model_calls": calls,
        "run_tokens": tokens,
        "prompt_estimate": prompt_tokens,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "checks": [{"command": item["command"], "returncode": item["returncode"]} for item in checks],
        "plan": truncate_text(plan, 3500),
        "review": truncate_text(review, 2500),
    }
