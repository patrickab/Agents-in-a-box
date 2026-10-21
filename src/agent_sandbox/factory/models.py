"""Define the factory HTTP API models and the agent output envelopes."""

from collections.abc import Iterable, Mapping
from datetime import datetime
import json
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

TaskState = Literal["backlog", "todo", "in_progress", "human_review", "rework", "blocked", "merging", "done", "cancelled"]
Phase = Literal["prepare", "scout", "plan", "code", "hooks", "gates", "review", "preview", "merge"]
AgentPhase = Literal["scout", "plan", "code", "review"]
MergeMode = Literal["auto", "review"]
MaxRounds = Annotated[int, Field(ge=1, le=20)]

TASK_ID_PATTERN = r"^[a-z0-9]{8,32}$"
AGENT_PHASES: tuple[str, ...] = get_args(AgentPhase)
PHASE_AGENTS: dict[str, str] = {"scout": "scout", "plan": "planner", "code": "coder", "review": "reviewer"}
AGENT_NAMES: tuple[str, ...] = tuple(PHASE_AGENTS.values())
TERMINAL_STATES: frozenset[str] = frozenset({"done", "cancelled"})


class Overrides(BaseModel):
    """Per-task settings that may override workflow and repo defaults."""

    model_config = ConfigDict(extra="forbid")

    merge: MergeMode | None = None
    max_rounds: MaxRounds | None = None
    preview: bool | None = None
    sandbox: bool | None = None
    models: dict[str, str] | None = None  # agent name -> model selector


class FactorySettings(BaseModel):
    """Hold the settings every task starts from before repo, workflow, and request layers apply."""

    model_config = ConfigDict(extra="forbid")

    merge: MergeMode = "review"
    max_rounds: MaxRounds = 3
    sandbox: bool = True
    preview: bool = True
    models: dict[str, str] = {}  # agent name -> model selector
    agent_timeout_s: int = Field(default=3600, ge=1, le=86400)
    command_timeout_s: int = Field(default=1800, ge=1, le=86400)

    @field_validator("models")
    @classmethod
    def _check_model_agents(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(value) - set(AGENT_NAMES))
        if unknown:
            raise ValueError(f"unknown agents {unknown}, expected {list(AGENT_NAMES)}")
        return value


class FactoryDefaults(BaseModel):
    """Carry the editable engine-wide defaults: settings and the full markdown of every agent."""

    settings: FactorySettings
    agents: dict[str, str]  # agent name -> markdown (front matter + body)


class ImageInput(BaseModel):
    """Carry one base64-encoded image attachment."""

    filename: str
    data_base64: str


class TaskCreate(BaseModel):
    """Describe a new task."""

    repo: str
    workflow: str
    title: str
    request: str
    images: list[ImageInput] = []
    overrides: Overrides = Overrides()
    start: bool = True  # False -> backlog


class MessageCreate(BaseModel):
    """Carry one user message to a waiting task."""

    text: str
    images: list[ImageInput] = []


class RepoInfo(BaseModel):
    """Name a registered repo and the main checkout clients match it by."""

    name: str
    path: str


class RepoCreate(BaseModel):
    """Carry the path of a repo main checkout to register."""

    path: str


class WorkflowInfo(BaseModel):
    """Summarize one workflow template."""

    id: str
    name: str
    phases: list[str]
    merge: str
    preview: bool


class GateResult(BaseModel):
    """Record one gate or hook command outcome."""

    argv: list[str]
    exit_code: int
    output_tail: str


class PhaseRun(BaseModel):
    """Record one phase execution of a task."""

    id: int
    phase: Phase
    round: int
    status: Literal["running", "ok", "failed"]
    started_at: datetime
    ended_at: datetime | None
    summary: str
    envelope: dict | None
    gate_results: list[GateResult] = []


class Message(BaseModel):
    """Record one user or agent message."""

    id: int
    role: Literal["user", "agent"]
    text: str
    phase_run_id: int | None
    created_at: datetime


class Preview(BaseModel):
    """Describe a running or failed task preview."""

    status: Literal["starting", "ready", "failed", "stopped"]
    url: str
    open_url: str
    error: str | None


class Task(BaseModel):
    """Describe one task and its current position in the pipeline."""

    id: str
    repo: str
    workflow: str
    title: str
    request: str
    state: TaskState
    phase: Phase | None
    round: int
    config: dict
    worktree: str | None
    branch: str | None
    base_sha: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class TaskDetail(Task):
    """Describe a task with its phase runs, messages, and preview."""

    phases: list[PhaseRun]
    messages: list[Message]
    preview: Preview | None


class Event(BaseModel):
    """Record one change notification."""

    seq: int
    task_id: str
    kind: str
    data: dict
    created_at: datetime


class ScoutEnvelope(BaseModel):
    """Validate the scout agent's output."""

    status: Literal["ok", "fail"]
    summary: str
    relevant_files: list[str] = []
    notes: str = ""


class PlanEnvelope(BaseModel):
    """Validate the planner agent's output."""

    status: Literal["ok", "fail"]
    summary: str
    steps: list[str] = []


class CodeEnvelope(BaseModel):
    """Validate the coder agent's output."""

    status: Literal["ok", "fail"]
    summary: str
    changed_files: list[str] = []
    question: str | None = None


class ReviewEnvelope(BaseModel):
    """Validate the reviewer agent's output."""

    status: Literal["ok", "fail"]
    approved: bool
    findings: list[str] = []


Envelope = ScoutEnvelope | PlanEnvelope | CodeEnvelope | ReviewEnvelope
ENVELOPE_MODELS: dict[str, type[Envelope]] = {
    "scout": ScoutEnvelope,
    "plan": PlanEnvelope,
    "code": CodeEnvelope,
    "review": ReviewEnvelope,
}


def _schema_text(annotation: Any) -> str:
    """Render a field annotation as the JSON shape shown to agents."""
    origin = get_origin(annotation)
    if origin is Literal:
        return " | ".join(json.dumps(value) for value in get_args(annotation))
    if origin in (Union, UnionType):
        return " | ".join(_schema_text(argument) for argument in get_args(annotation))
    if origin is list:
        return f"[{_schema_text(get_args(annotation)[0])}]"
    if annotation is type(None):
        return "null"
    return annotation.__name__


ENVELOPE_SCHEMAS: dict[str, str] = {
    phase: "{" + ", ".join(f'"{name}": {_schema_text(field.annotation)}' for name, field in model.model_fields.items()) + "}"
    for phase, model in ENVELOPE_MODELS.items()
}


class EnvelopeError(ValueError):
    """Report a missing or malformed agent output envelope."""


def format_validation_errors(errors: Iterable[Mapping[str, Any]]) -> str:
    """Join pydantic error dicts into `location: message` lines for API and config error text."""
    return "; ".join(f"{'.'.join(str(part) for part in issue['loc']) or 'document'}: {issue['msg']}" for issue in errors)


def parse_envelope(phase: str, raw: bytes | str | None) -> Envelope:
    """Validate raw envelope JSON for a phase, raising EnvelopeError with a correction-ready message."""
    if raw is None:
        raise EnvelopeError("the envelope file was not written")
    try:
        return ENVELOPE_MODELS[phase].model_validate_json(raw)
    except ValidationError as error:
        raise EnvelopeError(f"the envelope is invalid ({format_validation_errors(error.errors())})") from error


def envelope_summary(envelope: Envelope) -> str:
    """Return one human-readable line summarizing an envelope."""
    if isinstance(envelope, ReviewEnvelope):
        return "Approved" if envelope.approved else f"{len(envelope.findings)} finding(s)"
    return envelope.summary
