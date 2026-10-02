from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Any, Callable

from . import chatgpt_bridge
from .core import (
    CodexResult,
    CodexUsage,
    OrchestratorError,
    configured_lanes,
    ensure_clean_git,
    estimate_prompt_tokens,
    format_checks,
    git_snapshot,
    is_usage_limit_error,
    next_lane_name,
    review_passed,
    route_with_lane,
    run_checks,
    run_codex,
    split_command,
    truncate_text,
)
from .gitops import worktree_digest
from .endpoints import endpoint_provider
from .llm_api import ApiConfig, run_llm_api
from .prompts import PROFESSIONS, build_developer_prompt, build_planning_prompt, build_reviewer_prompt
from .relay import ManualRelay, RelayStopped, build_manual_prompt, pack_repository_context
from .secrets import active_key
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
    executor: str = "codex"
    api_model: str = ""
    api_base_url: str = ""
    api_max_rounds: int = 12
    limit_fallback: str = "chatgpt"
    relay: ManualRelay | None = None


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


def _api_provider_id(base_url: str) -> str | None:
    """Pick the stored key that matches the endpoint the request will hit."""
    return endpoint_provider(base_url)


def _api_config(request: WorkflowRequest) -> ApiConfig:
    base_url = request.api_base_url.strip() or "https://api.openai.com/v1"
    return ApiConfig(
        base_url=base_url,
        api_key=active_key(provider) if (provider := _api_provider_id(base_url)) else "",
        model=request.api_model.strip(),
        timeout=request.codex_timeout,
        max_rounds=request.api_max_rounds,
    )


def _api_key_or_local(base_url: str) -> bool:
    """A keyless endpoint is allowed for localhost (Ollama, LM Studio) and nothing else."""
    provider = endpoint_provider(base_url)
    return provider is None or bool(active_key(provider))


def _usage_limit_hint(request: WorkflowRequest) -> str:
    """Что сделать, если лимит исчерпан, а обход недоступен."""
    if request.limit_fallback == "off":
        return ("Настройка «Если лимит исчерпан» выключена. Выбери обход через обычный ChatGPT "
                "или API и запусти задачу заново — рабочая ветка сохранится.")
    if request.limit_fallback == "api":
        return ("Для обхода через API нужны имя модели и ключ провайдера (или локальный сервер "
                "Ollama/LM Studio): заполни их в настройках.")
    return "Обход через обычный ChatGPT доступен при запуске из панели ai-orchestrate."


def _usage_limit_fallback(request: WorkflowRequest, current: str) -> str:
    """Куда переключиться после ошибки лимита: ``""`` — обхода нет.

    ``limit_fallback=api`` сначала пробует OpenAI-совместимый API (ключ или
    локальный сервер), потому что он не требует человека; ``chatgpt`` всегда
    уходит в ручной мост через обычный чат.
    """
    if request.limit_fallback == "off":
        return ""
    candidates = ("api", "chatgpt") if request.limit_fallback == "api" else ("chatgpt",)
    for candidate in candidates:
        if candidate == current:
            continue
        if candidate == "api" and request.api_model.strip() and _api_key_or_local(request.api_base_url):
            return "api"
        if candidate == "chatgpt" and request.relay is not None:
            return "chatgpt"
    return ""


def _prepare_request(request: WorkflowRequest) -> WorkflowRequest:
    """У API-исполнителя ровно одна модель, а у ручного моста модель — сам ChatGPT.

    Поэтому полосы не должны выдумывать имена: подставляем понятную подпись,
    чтобы журнал и панель показывали, кто на самом деле выполняет шаг.
    """
    if request.executor == "chatgpt":
        label = request.api_model.strip() or "chatgpt-manual"
        return replace(request, luna_model=label, sol_model=label,
                       review_model=request.review_model.strip() or label)
    if request.executor != "api" or not request.api_model.strip():
        return request
    model = request.api_model.strip()
    return replace(request, luna_model=model, sol_model=model,
                   review_model=request.review_model.strip() or model)


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
    if request.executor not in {"codex", "api", "chatgpt"}:
        raise OrchestratorError("Исполнитель должен быть codex, api или chatgpt.")
    if request.limit_fallback not in {"chatgpt", "api", "off"}:
        raise OrchestratorError("Обход лимитов должен быть chatgpt, api или off.")
    if request.executor == "chatgpt":
        if request.relay is None:
            raise OrchestratorError(
                "Исполнитель «ChatGPT (обычный чат)» работает только через панель: "
                "в ней появляется промпт и поле для вставки ответа."
            )
    elif request.executor == "api":
        if not request.api_model.strip():
            raise OrchestratorError("Для API-исполнителя укажи имя модели (например gpt-5.1 или llama3.1).")
        if not 1 <= request.api_max_rounds <= 40:
            raise OrchestratorError("Число раундов API-исполнителя должно быть от 1 до 40.")
        if not _api_key_or_local(request.api_base_url):
            raise OrchestratorError(
                "Нет ключа API. Сохрани ключ OpenAI/OpenRouter в разделе «Ключи», "
                "или укажи локальный сервер без ключа (Ollama: http://127.0.0.1:11434/v1)."
            )
    elif request.executor == "codex" and not shutil.which("codex"):
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
        # When tests is a package, anchor imports at the repository root (works reliably on Windows too).
        top_level = " -t ." if (repo / "tests" / "__init__.py").is_file() else ""
        return [f"python -m unittest discover -s tests{top_level}"]
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
    verification_only: bool = False,
    review_base: str | None = None,
) -> dict[str, Any]:
    """Run the full bounded planner → coder → checks → reviewer → repair cycle."""
    started = time.monotonic()
    request = _prepare_request(request)
    _emit(emit, "stage.started", "preflight", "Проверяю проект, Git и команды тестирования.")
    repo, checks = _validate_request(request)
    if cancel_event.is_set():
        raise WorkflowStopped("Задача отменена до старта.")
    _emit(emit, "stage.completed", "preflight", "Проект готов к запуску.", data={"repo": str(repo), "checks": checks})

    daily_start = tokens_for_date(usage_path)
    # Ручной мост через обычный чат не тратит измеряемые токены, поэтому
    # исчерпанный дневной лимит не должен блокировать такой запуск.
    if daily_start >= request.daily_token_budget and request.executor != "chatgpt":
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
    active_executor = request.executor
    plan = ""
    all_checks: list[dict[str, Any]] = []
    final_review = ""
    last_model_result = CodexResult(0)

    def ensure_can_call(prompt: str, *, manual: bool) -> int:
        nonlocal prompt_tokens
        if cancel_event.is_set():
            raise WorkflowStopped("Пользователь отменил задачу.")
        if model_calls >= request.max_model_calls:
            raise WorkflowStopped(f"Достигнут лимит вызовов моделей: {model_calls}/{request.max_model_calls}.")
        if not manual:
            # Измеряемые лимиты относятся только к Codex/API: у ручного моста
            # расход считает подписка ChatGPT, а не локальный ledger.
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
        if not manual and prompt_tokens + estimate > request.prompt_token_budget:
            raise WorkflowStopped(
                f"Следующий промпт превышает лимит текста: ~{prompt_tokens + estimate:,}/"
                f"{request.prompt_token_budget:,} токенов."
            )
        prompt_tokens += estimate
        return estimate

    def run_measured_call(executor: str, prompt: str, call_model: str, call_effort: str,
                          sandbox: str, on_codex_event: Callable[[dict[str, Any]], None]) -> CodexResult:
        """Один вызов Codex CLI или OpenAI-совместимого API."""
        if executor == "api":
            config = _api_config(request)
            return run_llm_api(
                repo, prompt, call_model, config=config, sandbox=sandbox,
                on_event=on_codex_event, cancel_event=cancel_event,
                token_budget=max(min(request.max_run_tokens - run_tokens,
                                     request.daily_token_budget - daily_start - run_tokens), 0),
                command_timeout=request.check_timeout,
            )
        return run_codex(
            repo, prompt, call_model, call_effort, timeout=request.codex_timeout,
            sandbox=sandbox, on_event=on_codex_event, cancel_event=cancel_event,
        )

    def relay_call(role: str, stage: str, prompt: str, *, manual_kind: str, title: str,
                   instructions: str) -> CodexResult:
        """Показать промпт в панели, дождаться ответа человека и применить его."""
        assert request.relay is not None
        try:
            answer = request.relay.request(
                kind=manual_kind, stage=stage, role=role, title=title,
                instructions=instructions, prompt=prompt,
            )
        except RelayStopped as exc:
            raise WorkflowStopped(str(exc)) from exc
        if manual_kind not in {"code", "repair"}:
            return CodexResult(0, CodexUsage(), answer)
        try:
            operations = chatgpt_bridge.parse_answer(answer)
        except OrchestratorError as exc:
            return CodexResult(1, CodexUsage(), "", stderr=f"Ответ ChatGPT не разобран: {exc}")
        report = chatgpt_bridge.apply_operations(repo, operations)
        for item in report.applied:
            _emit(emit, "file.change", stage, f"Изменение файла: {item['path']}",
                  role=role, data={"path": item["path"], "status": "applied", "source": "chatgpt"})
        if not report.ok:
            errors = "; ".join(f"{item['path']}: {item['error']}" for item in report.rejected[:5])
            return CodexResult(1, CodexUsage(), "", stderr=f"Ответ ChatGPT отклонён: {errors or 'нет применённых файлов'}")
        return CodexResult(0, CodexUsage(), f"Применено файлов из ответа ChatGPT: {len(report.applied)}")

    def manual_prompt(base_prompt: str, *, kind: str, extra_note: str = "") -> str:
        """Дополнить промпт роли контекстом worktree и правилами ответа."""
        context = pack_repository_context(repo, task=request.task)
        return build_manual_prompt(
            base_prompt, kind=kind, context_text=context["text"], context_files=context["files"],
            check_commands=checks, extra_note=extra_note,
        )

    def model_call(role: str, stage: str, prompt: str, *, call_model: str, call_effort: str,
                   sandbox: str, manual_kind: str = "plan", manual_note: str = "") -> CodexResult:
        nonlocal model_calls, run_tokens, usage_known, ledger_ok, active_executor
        executor_for_call = active_executor
        manual = executor_for_call == "chatgpt"
        estimate = ensure_can_call(prompt, manual=manual)
        model_calls += 1
        _emit(emit, "model.started", stage,
              (f"{role}: готовлю промпт для обычного ChatGPT." if manual
               else f"{role}: запуск {call_model} / {call_effort}."), role=role,
              data={"model": "chatgpt-manual" if manual else call_model,
                    "effort": "manual" if manual else call_effort, "sandbox": sandbox,
                    "prompt_estimate": estimate, "call": model_calls,
                    "call_limit": request.max_model_calls, "executor": executor_for_call})
        call_started = time.monotonic()

        def on_codex_event(raw_event: dict[str, Any]) -> None:
            formatted = _format_codex_event(raw_event)
            if formatted:
                event_name, message, data = formatted
                _emit(emit, event_name, stage, message, role=role, data=data)

        manual_title = {
            "plan": "План от аналитика и архитектора",
            "code": "Правки кода для обычного ChatGPT",
            "repair": "Исправление после неуспешных проверок",
            "review": "Независимое ревью",
        }.get(manual_kind, role)

        if manual:
            prepared = manual_prompt(prompt, kind=manual_kind, extra_note=manual_note)
            result = relay_call(role, stage, prepared, manual_kind=manual_kind,
                                title=manual_title, instructions=manual_note)
        else:
            result = run_measured_call(executor_for_call, prompt, call_model, call_effort,
                                       sandbox, on_codex_event)
            if result.returncode != 0 and not result.cancelled and is_usage_limit_error(
                    result.stderr, result.final_message):
                fallback = _usage_limit_fallback(request, executor_for_call)
                if fallback:
                    detail = ("лимит Codex ChatGPT" if executor_for_call == "codex"
                              else "лимит API-провайдера")
                    _emit(emit, "executor.limit", stage,
                          f"Упёрлись в {detail}: {truncate_text(result.stderr or result.final_message, 400)}",
                          role=role, data={"executor": executor_for_call})
                    _emit(emit, "executor.fallback", stage,
                          ("Переключаю оставшиеся шаги на OpenAI-совместимый API."
                           if fallback == "api" else
                           "Переключаю оставшиеся шаги на обычный ChatGPT: промпт появится в панели, "
                           "ответ нужно вставить вручную."),
                          role=role, data={"from": executor_for_call, "to": fallback,
                                           "limit_fallback": request.limit_fallback})
                    active_executor = fallback
                    if fallback == "chatgpt":
                        prepared = manual_prompt(prompt, kind=manual_kind, extra_note=manual_note)
                        result = relay_call(role, stage, prepared, manual_kind=manual_kind,
                                            title=manual_title, instructions=manual_note)
                    else:
                        call_model = request.api_model.strip()
                        result = run_measured_call("api", prompt, call_model, call_effort,
                                                   sandbox, on_codex_event)
                else:
                    _emit(emit, "executor.limit", stage,
                          f"Лимит исчерпан ({executor_for_call}): "
                          + truncate_text(result.stderr or result.final_message, 300)
                          + " " + _usage_limit_hint(request),
                          role=role, data={"executor": executor_for_call, "fallback": "",
                                           "hint": _usage_limit_hint(request)})
        completed_manually = active_executor == "chatgpt"
        elapsed = round(time.monotonic() - call_started, 2)
        total = result.usage.total_tokens
        if completed_manually:
            _emit(emit, "manual.usage", stage,
                  "Шаг выполнен в обычном ChatGPT: подписочные токены не входят в локальный ledger.",
                  role=role, data={"metered": False})
        else:
            usage_known &= total is not None
        if not completed_manually and total is not None:
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
        elif not completed_manually:
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
                    "cancelled": result.cancelled, "stderr": truncate_text(result.stderr, 1800)})
        return result

    if request.mode == "full" and not verification_only:
        plan_prompt = build_planning_prompt(request.task)
        _emit(emit, "stage.started", "planning", "Аналитик формирует критерии, архитектор — короткий план.",
              role="Аналитик + архитектор")
        plan_result = model_call("Аналитик + архитектор", "planning", plan_prompt,
                                 call_model=model, call_effort="low", sandbox="read-only",
                                 manual_kind="plan",
                                 manual_note="Файлы не меняй: верни только план текстом.")
        if plan_result.cancelled or cancel_event.is_set():
            raise WorkflowStopped("Планирование отменено.")
        if plan_result.returncode != 0:
            raise WorkflowStopped(f"Не удалось завершить этап аналитика/архитектора (exit {plan_result.returncode}). "
                                  + truncate_text(plan_result.stderr, 1800))
        plan = truncate_text(plan_result.final_message.strip(), 3500)
        _emit(emit, "stage.completed", "planning",
              "План готов." if plan else "План не получен; разработчик продолжит по исходной задаче.",
              role="Аналитик + архитектор", data={"plan": truncate_text(plan, 3500)})

    repair_feedback = ""
    repair_count = 0
    max_iterations = 1 if verification_only else request.max_repairs + 1
    for iteration in range(1, max_iterations + 1):
        profession_name = next((item.title for item in PROFESSIONS if item.key == request.profession), "Разработчик")
        if not verification_only:
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
            last_model_result = model_call(
                profession_name, "implementation", developer_prompt,
                call_model=model, call_effort=effort, sandbox="workspace-write",
                manual_kind="repair" if repair_feedback else "code",
                manual_note=("Верни только изменённые файлы целым содержимым или патчем: оркестратор сам "
                             "применит их к worktree и запустит проверки." if not repair_feedback else
                             "Исправь только перечисленные проблемы и верни файлы целиком или патчем."),
            )
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
        verified_digest = worktree_digest(repo) if implementation_passed else ""

        if implementation_passed and request.mode == "full":
            _emit(emit, "stage.started", "review", "Ревьюер независимо проверяет diff и результаты тестов.",
                  role="Ревьюер")
            diff, status = git_snapshot(repo, max_chars=9000, **({"base": review_base} if review_base else {}))
            reviewer_model = _review_model(
                model,
                requested=request.review_model,
                luna_model=request.luna_model,
                sol_model=request.sol_model,
            )
            review_text = build_reviewer_prompt(request.task, plan, diff, status, all_checks)
            review_result = model_call(
                "Ревьюер", "review", review_text,
                call_model=reviewer_model, call_effort="low", sandbox="read-only",
                manual_kind="review",
                manual_note="Diff и проверки уже в промпте: ничего не редактируй, верни только вердикт и замечания.",
            )
            if review_result.cancelled or cancel_event.is_set():
                raise WorkflowStopped("Ревью отменено.")
            if review_result.returncode != 0:
                raise WorkflowStopped(f"Независимый ревьюер завершился с ошибкой (exit {review_result.returncode}). "
                                      + truncate_text(review_result.stderr, 1800))
            final_review = review_result.final_message.strip()
            review_ok = review_passed(final_review)
            _emit(emit, "stage.completed", "review",
                  "Ревью пройдено." if review_ok else "Ревью нашло замечания или не вернуло PASS.",
                  role="Ревьюер", data={"passed": review_ok, "findings": truncate_text(final_review, 2500)})
            if review_ok:
                result = _finish("complete", request, lane_name, model_calls, run_tokens, prompt_tokens,
                                 started, all_checks, plan, final_review)
                result["executor"] = active_executor
                result["verified_digest"] = verified_digest
                _emit(emit, "run.completed", "final", "Полный цикл завершён успешно.", role="Итог", data=result)
                return result
            repair_feedback = "Ревьюер сообщил:\n" + truncate_text(final_review, 2500)

        elif implementation_passed:
            result = _finish("complete", request, lane_name, model_calls, run_tokens, prompt_tokens,
                             started, all_checks, plan, "")
            result["executor"] = active_executor
            result["verified_digest"] = verified_digest
            _emit(emit, "run.completed", "final", "Задача завершена: проверки прошли.", role="Итог", data=result)
            return result

        if not implementation_passed:
            repair_feedback = (
                f"Кодер завершился с exit {last_model_result.returncode}.\n"
                f"{truncate_text(last_model_result.stderr, 1800)}\n"
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
    result["executor"] = active_executor
    result["failure_reason"] = repair_feedback
    _emit(emit, "run.incomplete", "final", "Цикл остановлен: " + repair_feedback,
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
