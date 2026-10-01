from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .core import (CONFIDENCE_THRESHOLD, OrchestratorError, configured_lanes, doctor,
                   ensure_clean_git, git_snapshot, jev_choice, next_lane, route, split_command,
                   run_checks, run_codex)


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


def _run(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        raise OrchestratorError(f"Repository directory does not exist: {repo}")
    _validate_checks(args.check)
    ensure_clean_git(repo)
    if not args.dry_run and shutil.which("codex") is None:
        raise OrchestratorError("Codex CLI is not available on PATH. Install it and sign in before a live run.")
    model, effort = route(args.task, lane=args.lane, dry_run=args.dry_run)
    if args.dry_run:
        lane_name = next(name for name, lane in configured_lanes().items() if lane == (model, effort))
        print(f"DRY RUN: {lane_name} -> {model} / {effort}; no network request or files changed")
        return 0

    last_checks: list[dict] = []
    prompt = args.task
    for attempt in range(1, args.max_attempts + 1):
        print(f"[agent] attempt {attempt}/{args.max_attempts}: {model} / {effort}")
        code = run_codex(repo, prompt, model, effort, timeout=args.codex_timeout)
        if code == 124:
            print("[result] INCOMPLETE: Codex timed out; stopped to avoid overlapping edit attempts")
            return 2
        if code != 0:
            print(f"[agent] Codex exited with code {code}")
        last_checks = run_checks(repo, args.check, timeout=args.check_timeout)
        passed = code == 0 and all(item["returncode"] == 0 for item in last_checks)
        for item in last_checks:
            print(f"[check] {'PASS' if item['returncode'] == 0 else 'FAIL'} {item['command']}")
        if any(item["returncode"] == 124 for item in last_checks):
            print("[result] INCOMPLETE: a check timed out; stopped to avoid overlapping processes")
            return 2
        diff, status = git_snapshot(repo)

        state = {
            "task": args.task,
            "attempt": attempt,
            "current_lane": f"{model}/{effort}",
            "codex_exit_code": code,
            "checks": [{"command": c["command"], "returncode": c["returncode"], "output": c["output"][-1000:]}
                       for c in last_checks],
            "diff": diff[-6000:],
            "git_status": status[-3000:],
        }
        if passed:
            decision = jev_choice(
                state,
                "Does the requested behavior appear implemented in the inspected changes? Choose COMPLETE only if it does and confidence is at least 0.8; otherwise choose VERIFY. Deterministic checks passed, but they do not prove scope.",
                {"COMPLETE": "The diff and Codex result indicate the requested behavior is implemented.",
                 "VERIFY": "The change appears incomplete, ambiguous, or needs human review."},
            )
            if decision.choice == "COMPLETE" and decision.confidence >= CONFIDENCE_THRESHOLD:
                print("[result] COMPLETE (checks passed; Jev judged the requested behavior implemented)")
                return 0
            print(f"[result] INCOMPLETE: Jev requested human verification (confidence {decision.confidence:.2f})")
            return 2

        if attempt == args.max_attempts:
            print("[result] INCOMPLETE: attempt limit reached with a failed Codex run or check")
            return 2
        if attempt == 1:
            model, effort = next_lane(model, effort)
            prompt = _retry_prompt(args.task, code, last_checks)
            continue
        decision = jev_choice(
            state,
            "Choose the next bounded action after this failed implementation/check cycle.",
            {"RETRY": "A concrete fix is likely within the current lane.",
             "ESCALATE": "Repeated failure or uncertainty warrants the Sol high lane.",
             "VERIFY": "More deterministic evidence is needed before changing anything.",
             "STOP": "Further automated changes are unlikely to help."},
        )
        if decision.choice == "RETRY":
            model, effort = next_lane(model, effort)
            prompt = _retry_prompt(args.task, code, last_checks)
        elif decision.choice == "ESCALATE":
            model, effort = configured_lanes()["ESCALATE"]
            prompt = _retry_prompt(args.task, code, last_checks)
        else:
            print(f"[result] INCOMPLETE: Jev chose {decision.choice}")
            return 2
    return 2


def _retry_prompt(task: str, code: int, checks: list[dict]) -> str:
    evidence = "\n".join(f"- {item['command']}: exit {item['returncode']}\n{item['output'][-2000:]}"
                          for item in checks)
    return (f"{task}\n\nPrevious implementation attempt did not pass. "
            f"Codex exit code: {code}. Required check results:\n{evidence}\n\n"
            "Inspect the failure, fix its root cause within the requested scope, then rerun the required checks.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-orchestrate", description="Route Codex CLI work through TypeSafe Jev.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check local prerequisites and Jev key presence")
    run = sub.add_parser("run", help="Route and run one task")
    run.add_argument("task", help="Task prompt sent to Codex")
    run.add_argument("--repo", required=True, help="Existing Git repository to edit")
    run.add_argument("--check", action="append", default=[], help="Required deterministic check; repeat for multiple")
    run.add_argument("--max-attempts", type=int, default=4)
    run.add_argument("--codex-timeout", type=int, default=1800)
    run.add_argument("--check-timeout", type=int, default=300)
    run.add_argument("--dry-run", action="store_true", help="Print an explicitly selected lane without network access")
    run.add_argument("--lane", choices=tuple(configured_lanes()), help="Required with --dry-run; selects the lane explicitly")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        failed = False
        for name, ok, detail in doctor():
            print(f"{'OK' if ok else 'MISSING'} {name}: {detail}")
            failed |= not ok
        return 1 if failed else 0
    if args.max_attempts < 1 or args.max_attempts > 5:
        parser.error("--max-attempts must be between 1 and 5")
    if args.codex_timeout < 1 or args.check_timeout < 1:
        parser.error("timeouts must be positive")
    if args.dry_run and not args.lane:
        parser.error("--dry-run requires --lane")
    if not args.dry_run and args.lane:
        parser.error("--lane is only accepted with --dry-run")
    try:
        return _run(args)
    except (OrchestratorError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
