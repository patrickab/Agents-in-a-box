"""Build the user prompts the engine sends to the scout, planner, coder, and reviewer agents."""

from collections.abc import Sequence
from typing import Any, Literal

from agent_sandbox.factory.models import ENVELOPE_SCHEMAS

CodeMode = Literal["initial", "retry", "rework", "breach", "hooks_fail", "gate_fail", "review_findings"]
FEEDBACK_TAIL_CHARS = 6000


def _envelope_instructions(phase: str, output_path: str) -> str:
    return (
        f"When you are done, write a JSON object to `{output_path}` (valid JSON, nothing else in that file) with exactly this shape:\n"
        f"{ENVELOPE_SCHEMAS[phase]}\n"
        'Use "status": "fail" only when you cannot make progress, and explain why in "summary".'
    )


def correction_prompt(phase: str, problem: str, output_path: str) -> str:
    """Ask the agent to repair its output envelope."""
    return (
        f"Your {phase} output could not be used: {problem}.\n"
        f"{_envelope_instructions(phase, output_path)}\n"
        "Do not change any other file."
    )


def _task_block(title: str, request: str) -> str:
    return f"# Task: {title}\n\n{request}"


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def scout_prompt(title: str, request: str, output_path: str) -> str:
    """Build the scout prompt."""
    return (
        f"{_task_block(title, request)}\n\n"
        "Explore the repository and report what matters for this task. This phase is read-only: do not modify any file "
        "except the output file.\n\n"
        f"{_envelope_instructions('scout', output_path)}"
    )


def plan_prompt(title: str, request: str, scout: dict[str, Any] | None, output_path: str) -> str:
    """Build the planner prompt, including the scout findings."""
    return (
        f"{_task_block(title, request)}\n\n{_scout_block(scout)}"
        "Write an implementation plan as ordered steps. This phase is read-only: do not modify any file except the output file.\n\n"
        f"{_envelope_instructions('plan', output_path)}"
    )


def _scout_block(scout: dict[str, Any] | None) -> str:
    if not scout:
        return ""
    return (
        f"## Scout findings\n{scout.get('summary', '')}\n\nRelevant files:\n{_bullets(scout.get('relevant_files', []))}\n\n"
        f"{scout.get('notes', '')}\n\n"
    )


def _plan_block(plan: dict[str, Any] | None) -> str:
    if not plan:
        return ""
    steps = "\n".join(f"{number}. {step}" for number, step in enumerate(plan.get("steps", []), start=1))
    return f"## Plan\n{plan.get('summary', '')}\n\n{steps}\n\n"


def _write_rules(writes: Sequence[str], protected: Sequence[str]) -> str:
    return (
        f"You may only modify files matching: {', '.join(writes) or '(none)'}. "
        f"These paths are protected and any change to them is reverted: {', '.join(protected)}. "
        "Leave the changes uncommitted; do not run git commit."
    )


def code_prompt(
    mode: CodeMode,
    *,
    title: str,
    request: str,
    scout: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    text: str,
    writes: Sequence[str],
    protected: Sequence[str],
    output_path: str,
) -> str:
    """Build the coder prompt for `mode` (initial, retry, rework, breach, hooks_fail, gate_fail, review_findings)."""
    rules = f"{_write_rules(writes, protected)}\n\n{_envelope_instructions('code', output_path)}"
    if mode == "initial":
        return f"{_task_block(title, request)}\n\n{_scout_block(scout)}{_plan_block(plan)}Implement the task.\n\n{rules}"
    openers = {
        "retry": (
            f"{_task_block(title, request)}\n\n"
            f"Your previous attempt was interrupted or blocked ({text}). Continue and finish the task."
        ),
        "rework": f"The human reviewer sent this feedback on your work:\n\n{text}\n\nUpdate the code accordingly.",
        "breach": (
            "You changed files you are not allowed to touch, and those changes were reverted:\n"
            f"{text}\n\nRedo the work without modifying them."
        ),
        "hooks_fail": f"The post-code hooks failed. Fix the cause:\n\n{text}",
        "gate_fail": f"A quality gate failed after your changes. Fix the cause:\n\n{text}",
        "review_findings": f"The reviewer found these problems. Fix each one:\n\n{text}",
    }
    return f"{openers[mode]}\n\n{rules}"


def review_prompt(
    title: str, request: str, plan: dict[str, Any] | None, changed_files: Sequence[str], diff: str, output_path: str
) -> str:
    """Build the reviewer prompt over the current diff."""
    return (
        f"{_task_block(title, request)}\n\n{_plan_block(plan)}"
        "Review the uncommitted changes the coder made for this task. This phase is read-only: do not modify any file except the "
        'output file. Set "approved" to true only when the change fully and correctly solves the task; list every blocking problem '
        'in "findings".\n\n'
        f"Changed files:\n{_bullets(changed_files)}\n\nDiff:\n```diff\n{diff}\n```\n\n"
        f"{_envelope_instructions('review', output_path)}"
    )


def failure_text(argv: Sequence[str], exit_code: int, output: str, timed_out: bool = False) -> str:
    """Describe one failed command for the coder."""
    outcome = "timed out" if timed_out else f"exited {exit_code}"
    return f"$ {' '.join(argv)}\n({outcome})\n{output[-FEEDBACK_TAIL_CHARS:]}"
