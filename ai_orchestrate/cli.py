from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .core import (
    CodexResult,
    OrchestratorError,
    configured_lanes,
    doctor,
    ensure_clean_git,
    estimate_prompt_tokens,
    format_checks,
    git_snapshot,
    next_lane_name,
    review_passed,
    review_prompt,
    route_with_lane,
    run_checks,
    run_codex,
    split_command,
    truncate_text,
)
from .usage import append_usage, default_usage_path, tokens_for_date, usage_summary


def _validate_checks(commands: list[str]) -> None:
    if not commands:
        raise OrchestratorError("At least one --check command is required; completion requires deterministic checks.")
    for command in commands:
        try:
            args = split_command(command)
        except ValueError as exc:
            raise OrchestratorError(f"Invalid check command {command!r}: {exc}") from exc
        if not args or not shutil.which(args[0]):
            raise OrchestratorError(f"Check command is unavailable: {command}")


def _report_usage(role: str, result: CodexResult) -> int:
    usage = result.usage
    total = usage.total_tokens
    if total is None:
        print(f"[usage] {role}: Codex did not report token usage")
        return 0
    cached = "unknown" if usage.cached_input_tokens is None else f"{usage.cached_input_tokens:,}"
    print(
        f"[usage] {role}: input={usage.input_tokens:,} (cached={cached}), "
        f"output={usage.output_tokens:,}, total={total:,} tokens"
    )
    return total


def _log_usage(path: Path, *, model: str, effort: str, role: str, attempt: int,
               result: CodexResult) -> bool:
    try:
        return append_usage(
            path,
            model=model,
            effort=effort,
            role=role,
            attempt=attempt,
            returncode=result.returncode,
            usage=result.usage,
        )
    except OrchestratorError as exc:
        print(f"[usage] warning: {exc}", file=sys.stderr)
        return False


def _can_call_model(
    *, calls: int, max_calls: int, run_tokens: int, max_run_tokens: int | None,
    daily_tokens: int, daily_budget: int | None, ledger_ok: bool, usage_known: bool,
) -> tuple[bool, str]:
    if calls >= max_calls:
        return False, f"model-call limit reached ({calls}/{max_calls})"
    if max_run_tokens is not None:
        if not usage_known:
            return False, "Codex did not report usage; stopping because the run token budget cannot be enforced"
        if run_tokens >= max_run_tokens:
            return False, f"run token budget reached ({run_tokens:,}/{max_run_tokens:,})"
    if daily_budget is not None:
        if not ledger_ok:
            return False, "usage ledger could not be updated; stopping to avoid exceeding the daily budget"
        if not usage_known:
            return False, "Codex did not report usage; stopping because the daily token budget cannot be enforced"
        if daily_tokens + run_tokens >= daily_budget:
            return False, f"daily token budget reached ({daily_tokens + run_tokens:,}/{daily_budget:,})"
    return True, ""


def _retry_prompt(task: str, result: CodexResult, checks: list[dict], reviewer_feedback: str = "") -> str:
    prompt = (
        "The previous implementation attempt did not pass. Continue from the current working tree; "
        "do not restart or expand scope. Fix only the concrete failure below, then rerun the required checks.\n\n"
        f"Original task:\n{task}\n\nCodex exit code: {result.returncode}\n"
        f"Required check results (bounded):\n{format_checks(checks, limit=3000)}"
    )
    if reviewer_feedback:
        prompt += "\n\nIndependent review findings (bounded):\n" + truncate_text(reviewer_feedback, 2000)
    return prompt


def _review_model(writer_model: str, requested: str | None) -> str:
    if requested:
        return requested
    configured = os.environ.get("AI_ORCHESTRATE_REVIEW_MODEL")
    if configured:
        return configured
    lanes = configured_lanes()
    sol, luna = lanes["ESCALATE"][0], lanes["SMALL"][0]
    return sol if writer_model != sol else luna


def _run(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        raise OrchestratorError(f"Repository directory does not exist: {repo}")
    _validate_checks(args.check)
    ensure_clean_git(repo)

    dry_run = bool(args.dry_run)
    router = getattr(args, "router", "local")
    explicit_lane = getattr(args, "lane", None)
    if explicit_lane and router == "jev":
        raise OrchestratorError("Use either --lane or --router jev, not both.")
    if not dry_run and shutil.which("codex") is None:
        raise OrchestratorError("Codex CLI is not available on PATH. Install it and sign in before a live run.")

    prompt_budget = getattr(args, "prompt_token_budget", 12000)
    initial_estimate = estimate_prompt_tokens(args.task)
    if initial_estimate > prompt_budget:
        raise OrchestratorError(
            f"Task prompt is about {initial_estimate:,} tokens, above --prompt-token-budget {prompt_budget:,}. "
            "Split the task or raise the budget explicitly."
        )

    usage_path = default_usage_path(getattr(args, "usage_log", None))
    daily_budget = getattr(args, "daily_token_budget", None)
    daily_tokens = tokens_for_date(usage_path) if daily_budget is not None and not dry_run else 0
    if daily_budget is not None and not dry_run:
        print(f"[budget] tracked today: {daily_tokens:,}/{daily_budget:,} tokens")
        if daily_tokens >= daily_budget:
            raise OrchestratorError("Today's recorded token budget is already exhausted; no model call was started.")

    lane_name, (model, effort) = route_with_lane(
        args.task,
        lane=explicit_lane,
        router=router,
        dry_run=dry_run,
    )
    if dry_run:
        estimate = estimate_prompt_tokens(args.task)
        print(
            f"DRY RUN: {lane_name} -> {model} / {effort}; prompt text ~{estimate:,} tokens; "
            "no Codex or Jev request and no files changed"
        )
        return 0

    max_attempts = getattr(args, "max_attempts", 1)
    review_enabled = bool(getattr(args, "review", False) or getattr(args, "review_model", None))
    configured_max_calls = getattr(args, "max_model_calls", None)
    max_calls = configured_max_calls or max_attempts * (2 if review_enabled else 1)
    prompt_tokens = 0
    run_tokens = 0
    calls = 0
    ledger_ok = True
    usage_known = True
    last_checks: list[dict] = []
    reviewer_feedback = ""
    prompt = args.task

    print(f"[route] {router}: {lane_name} -> {model} / {effort}")
    print(
        f"[budget] prompt text estimate ~{initial_estimate:,}/{prompt_budget:,} tokens "
        "(does not include Codex context, tools, or generated output)"
    )
    if review_enabled:
        print(f"[orchestra] independent read-only review enabled; max model calls: {max_calls}")
    else:
        print(f"[orchestra] one coder by default; max attempts: {max_attempts}; max model calls: {max_calls}")

    for attempt in range(1, max_attempts + 1):
        allowed, reason = _can_call_model(
            calls=calls,
            max_calls=max_calls,
            run_tokens=run_tokens,
            max_run_tokens=getattr(args, "max_run_tokens", None),
            daily_tokens=daily_tokens,
            daily_budget=daily_budget,
            ledger_ok=ledger_ok,
            usage_known=usage_known,
        )
        if not allowed:
            print(f"[result] INCOMPLETE: {reason}; refusing another model request")
            return 2
        estimate = estimate_prompt_tokens(prompt)
        if prompt_tokens + estimate > prompt_budget:
            print(
                f"[result] INCOMPLETE: next prompt is ~{estimate:,} tokens; total would exceed "
                f"--prompt-token-budget {prompt_budget:,}"
            )
            return 2
        prompt_tokens += estimate

        print(f"[agent] coding attempt {attempt}/{max_attempts}: {model} / {effort}")
        result = run_codex(repo, prompt, model, effort, timeout=args.codex_timeout)
        calls += 1
        run_tokens += _report_usage("coder", result)
        usage_known &= result.usage.total_tokens is not None
        ledger_ok &= _log_usage(
            usage_path, model=model, effort=effort, role="coder", attempt=attempt, result=result,
        )
        if result.final_message:
            print("[codex] " + truncate_text(result.final_message, 5000))
        if result.returncode != 0:
            print(f"[agent] Codex exited with code {result.returncode}")
            if result.stderr:
                print("[agent] " + truncate_text(result.stderr, 2000), file=sys.stderr)
        if result.returncode == 124:
            print("[result] INCOMPLETE: Codex timed out; stopped to avoid overlapping edit attempts")
            return 2

        last_checks = run_checks(repo, args.check, timeout=args.check_timeout)
        checks_passed = all(item["returncode"] == 0 for item in last_checks)
        for item in last_checks:
            print(f"[check] {'PASS' if item['returncode'] == 0 else 'FAIL'} {item['command']}")
        if any(item["returncode"] == 124 for item in last_checks):
            print("[result] INCOMPLETE: a check timed out; stopped to avoid overlapping processes")
            return 2

        implementation_passed = result.returncode == 0 and checks_passed
        if implementation_passed and review_enabled:
            allowed, reason = _can_call_model(
                calls=calls,
                max_calls=max_calls,
                run_tokens=run_tokens,
                max_run_tokens=getattr(args, "max_run_tokens", None),
                daily_tokens=daily_tokens,
                daily_budget=daily_budget,
                ledger_ok=ledger_ok,
                usage_known=usage_known,
            )
            if not allowed:
                print(f"[result] INCOMPLETE: {reason}; requested review was not run")
                return 2
            diff, status = git_snapshot(repo)
            review_task = review_prompt(args.task, diff, status, last_checks)
            review_estimate = estimate_prompt_tokens(review_task)
            if prompt_tokens + review_estimate > prompt_budget:
                print(
                    "[result] INCOMPLETE: independent-review prompt would exceed "
                    f"--prompt-token-budget {prompt_budget:,}"
                )
                return 2
            prompt_tokens += review_estimate
            reviewer_model = _review_model(model, getattr(args, "review_model", None))
            print(f"[review] read-only: {reviewer_model} / low")
            review_result = run_codex(
                repo, review_task, reviewer_model, "low", timeout=args.codex_timeout, sandbox="read-only",
            )
            calls += 1
            run_tokens += _report_usage("reviewer", review_result)
            usage_known &= review_result.usage.total_tokens is not None
            ledger_ok &= _log_usage(
                usage_path, model=reviewer_model, effort="low", role="reviewer",
                attempt=attempt, result=review_result,
            )
            reviewer_feedback = review_result.final_message
            if reviewer_feedback:
                print("[review] " + truncate_text(reviewer_feedback, 4000))
            if review_result.returncode == 124:
                print("[result] INCOMPLETE: reviewer timed out")
                return 2
            if review_result.returncode != 0:
                print(f"[result] INCOMPLETE: reviewer exited with code {review_result.returncode}")
                if review_result.stderr:
                    print("[review] " + truncate_text(review_result.stderr, 2000), file=sys.stderr)
                return 2
            if review_passed(reviewer_feedback):
                print("[result] COMPLETE (required checks passed; independent review found no actionable issue)")
                return 0
            print("[review] did not return PASS; treating review findings as a failed gate")
        elif implementation_passed:
            print("[result] COMPLETE (Codex succeeded and all required deterministic checks passed)")
            return 0

        if attempt == max_attempts:
            print("[result] INCOMPLETE: attempt limit reached with a failed implementation, check, or review")
            return 2
        next_name = next_lane_name(lane_name)
        if next_name is None:
            print("[result] INCOMPLETE: already at the highest lane; no further escalation is available")
            return 2
        lane_name = next_name
        model, effort = configured_lanes()[lane_name]
        prompt = _retry_prompt(args.task, result, last_checks, reviewer_feedback)
        print(f"[orchestrator] escalating to {lane_name} -> {model} / {effort}")

    return 2


def _print_usage(path: Path) -> int:
    today, all_time, latest = usage_summary(path)
    print(f"Usage ledger: {path}")
    print(f"Today: {today:,} tokens; all time: {all_time:,} tokens")
    if latest:
        print("Recent model calls (input includes cached input; total = input + output):")
        for item in latest:
            print(
                f"  {item.get('timestamp', '?')} {item.get('role', '?')} "
                f"{item.get('model', '?')}/{item.get('effort', '?')}: "
                f"in={item.get('input_tokens', '?')}, out={item.get('output_tokens', '?')}, "
                f"total={item.get('total_tokens', '?')}"
            )
    else:
        print("No token telemetry recorded yet.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-orchestrate",
        description="Cost-aware model orchestration for Codex CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check local prerequisites")
    usage = sub.add_parser("usage", help="Show token usage tracked from Codex CLI telemetry")
    usage.add_argument("--usage-log", help="Usage JSONL path (default: ~/.ai-orchestrate/usage.jsonl)")

    ui = sub.add_parser("ui", help="Open the local visual orchestration dashboard")
    ui.add_argument("--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 only for a trusted preview)")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--workspace-root", help="Restrict selectable projects to this directory (default: current directory)")
    ui.add_argument("--usage-log", help="Usage JSONL path (default: ~/.ai-orchestrate/usage.jsonl)")
    ui.add_argument("--settings-file", help="Persistent UI settings JSON path (default: ~/.ai-orchestrate/settings.json)")

    run = sub.add_parser("run", help="Route and run one task")
    run.add_argument("task", help="Task prompt sent to Codex")
    run.add_argument("--repo", required=True, help="Existing Git repository to edit")
    run.add_argument("--check", action="append", default=[], help="Required deterministic check; repeat for multiple")
    run.add_argument("--router", choices=("local", "jev"), default="local",
                     help="Free local heuristic by default; Jev routing is an explicit network request")
    run.add_argument("--lane", choices=tuple(configured_lanes()),
                     help="Explicitly select the starting model lane (avoids router calls)")
    run.add_argument("--max-attempts", type=int, default=1,
                     help="Maximum coding attempts; each retry can consume another model turn (default: 1)")
    run.add_argument("--max-model-calls", type=int,
                     help="Hard cap on coder plus reviewer calls; default derives from attempts and review mode")
    run.add_argument("--max-run-tokens", type=int,
                     help="Stop before another model call once reported usage reaches this run total")
    run.add_argument("--daily-token-budget", type=int,
                     help="Do not start more calls after the tracked daily total reaches this number")
    run.add_argument("--prompt-token-budget", type=int, default=12000,
                     help="Approximate total prompt-text budget across Codex calls (default: 12000)")
    run.add_argument("--usage-log", help="Usage JSONL path (default: ~/.ai-orchestrate/usage.jsonl)")
    run.add_argument("--review", action="store_true",
                     help="Add an independent read-only Codex review; uses another model call")
    run.add_argument("--review-model", help="Reviewer model; implies --review")
    run.add_argument("--codex-timeout", type=int, default=1800)
    run.add_argument("--check-timeout", type=int, default=300)
    run.add_argument("--dry-run", action="store_true", help="Print the selected local/explicit lane without model calls")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        failed = False
        for name, ok, detail in doctor():
            label = "OK" if ok else "MISSING"
            print(f"{label} {name}: {detail}")
            failed |= not ok
        return 1 if failed else 0
    if args.command == "usage":
        try:
            return _print_usage(default_usage_path(args.usage_log))
        except OrchestratorError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.command == "ui":
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        try:
            from .web import serve
            root = Path(args.workspace_root).expanduser() if args.workspace_root else Path.cwd()
            settings_path = Path(args.settings_file).expanduser() if args.settings_file else None
            return serve(args.host, args.port, root, default_usage_path(args.usage_log), settings_path)
        except (OrchestratorError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.review_model:
        args.review = True
    if not 1 <= args.max_attempts <= 5:
        parser.error("--max-attempts must be between 1 and 5")
    if args.max_model_calls is not None and not 1 <= args.max_model_calls <= 10:
        parser.error("--max-model-calls must be between 1 and 10")
    for name in ("codex_timeout", "check_timeout", "prompt_token_budget"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_run_tokens", "daily_token_budget"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        return _run(args)
    except (OrchestratorError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
