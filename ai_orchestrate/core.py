from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

JEv_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
LANES = {
    "SMALL": ("gpt-6-luna", "low"),
    "MEDIUM": ("gpt-6-luna", "medium"),
    "HIGH": ("gpt-6-luna", "high"),
    "ESCALATE": ("gpt-6-sol", "high"),
}
EFFORTS = ("low", "medium", "high")
CONFIDENCE_THRESHOLD = 0.8


def configured_lanes() -> dict[str, tuple[str, str]]:
    luna = os.environ.get("AI_ORCHESTRATE_LUNA_MODEL", "gpt-6-luna")
    sol = os.environ.get("AI_ORCHESTRATE_SOL_MODEL", "gpt-6-sol")
    return {"SMALL": (luna, "low"), "MEDIUM": (luna, "medium"),
            "HIGH": (luna, "high"), "ESCALATE": (sol, "high")}


class OrchestratorError(Exception):
    pass


@dataclass
class Decision:
    choice: str
    confidence: float = 1.0


def jev_choice(state: dict, question: str, choices: dict[str, str], *, timeout: int = 30) -> Decision:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise OrchestratorError("TYPESAFE_API_KEY is not set. Set it before using live routing.")
    body = {
        "model": "jev-latest",
        "state": state,
        "questions": {"decision": {"type": "choice", "instructions": question, "criteria": choices}},
    }
    request = urllib.request.Request(
        JEv_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        answer = result["answers"]["decision"]
        choice = answer["choice"]
        confidence = float(answer["confidence"])
        probabilities = answer["probabilities"]
    except Exception as exc:
        # Network/server exceptions can contain response bodies; never echo them.
        raise OrchestratorError(f"Jev request or response failed ({type(exc).__name__}).") from exc
    if choice not in choices or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise OrchestratorError("Jev returned an invalid choice or confidence value.")
    if (not isinstance(probabilities, dict) or set(probabilities) != set(choices)
            or any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.02)):
        raise OrchestratorError("Jev returned invalid probabilities.")
    return Decision(choice, confidence)


def route(task: str, *, lane: str | None = None, dry_run: bool = False) -> tuple[str, str]:
    if dry_run:
        if lane not in LANES:
            raise OrchestratorError("Dry run requires --lane SMALL, MEDIUM, HIGH, or ESCALATE.")
        return configured_lanes()[lane]
    decision = jev_choice(
        {"task": task, "attempt": 0, "phase": "initial_routing"},
        "Choose the lowest-cost engineering lane sufficient to complete this task reliably.",
        {
            "SMALL": "Small localized change with obvious implementation and little uncertainty.",
            "MEDIUM": "Several related changes or moderate debugging and reasoning.",
            "HIGH": "Complex debugging, multi-component changes, or significant reasoning.",
            "ESCALATE": "Architecturally difficult, security-sensitive, or unusually risky work.",
        },
    )
    return configured_lanes()[decision.choice]


def run_checks(repo: Path, commands: list[str], *, timeout: int = 300) -> list[dict]:
    results = []
    for command in commands:
        try:
            args = split_command(command)
        except ValueError as exc:
            raise OrchestratorError(f"Invalid check command {command!r}: {exc}") from exc
        if not args or not shutil.which(args[0]):
            raise OrchestratorError(f"Check command is unavailable: {command}")
        try:
            proc = subprocess.run(args, cwd=repo, capture_output=True, text=True, timeout=timeout,
                                  check=False, env=_codex_env())
            results.append({"command": command, "returncode": proc.returncode,
                            "output": (proc.stdout + proc.stderr)[-4000:]})
        except subprocess.TimeoutExpired:
            results.append({"command": command, "returncode": 124, "output": f"Timed out after {timeout}s"})
    return results


def git_snapshot(repo: Path) -> tuple[str, str]:
    diff = subprocess.run(["git", "diff", "HEAD", "--no-ext-diff"], cwd=repo, capture_output=True, text=True, check=True).stdout
    status = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=repo,
                            capture_output=True, text=True, check=True).stdout
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo,
                               capture_output=True, text=True, check=True).stdout.splitlines()
    additions = []
    for name in untracked:
        try:
            with (repo / name).open("r", encoding="utf-8", errors="replace") as source:
                content = source.read(2000)
        except OSError:
            content = "<unreadable>"
        additions.append(f"\n--- untracked: {name} ---\n{content[:2000]}")
    return diff + "\n".join(additions), status


def split_command(command: str) -> list[str]:
    parts = shlex.split(command, posix=os.name != "nt")
    if os.name == "nt":
        parts = [part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'" else part
                 for part in parts]
    return parts


def ensure_clean_git(repo: Path) -> None:
    if not shutil.which("git"):
        raise OrchestratorError("git is not available on PATH.")
    try:
        subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=repo,
                       capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo,
                                capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        raise OrchestratorError(f"Repository must have at least one commit and usable Git metadata: {repo}") from exc
    if status.strip():
        raise OrchestratorError("Repository must have a clean worktree (including untracked files) before starting.")


def _codex_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
        env.pop(key, None)
    return env


def run_codex(repo: Path, task: str, model: str, effort: str, *, timeout: int = 1800,
              runner: Callable = subprocess.run) -> int:
    command = ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"',
               "-s", "workspace-write", "-C", str(repo), "-"]
    try:
        proc = runner(command, input=task, text=True, env=_codex_env(), timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return 124
    except OSError as exc:
        raise OrchestratorError(f"Could not start Codex CLI: {exc}") from exc
    return proc.returncode


def next_lane(model: str, effort: str) -> tuple[str, str]:
    lanes = configured_lanes()
    if model == lanes["ESCALATE"][0]:
        return model, effort
    index = EFFORTS.index(effort) if effort in EFFORTS else 0
    if index < len(EFFORTS) - 1:
        return lanes["SMALL"][0], EFFORTS[index + 1]
    return lanes["ESCALATE"]


def doctor() -> list[tuple[str, bool, str]]:
    key_present = bool(os.environ.get("TYPESAFE_API_KEY"))
    return [
        ("Python", True, "running"),
        ("Codex CLI", shutil.which("codex") is not None, "available" if shutil.which("codex") else "not found on PATH"),
        ("Git", shutil.which("git") is not None, "available" if shutil.which("git") else "not found on PATH"),
        ("TypeSafe key", key_present, "present" if key_present else "missing (live routing unavailable)"),
    ]
