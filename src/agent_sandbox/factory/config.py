"""Load and validate engine config, repo templates, workflow files, and agent definitions."""

import copy
from importlib import resources
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
import yaml

from agent_sandbox.factory.models import (
    AGENT_NAMES,
    AGENT_PHASES,
    PHASE_AGENTS,
    AgentPhase,
    FactoryDefaults,
    FactorySettings,
    MaxRounds,
    MergeMode,
    Overrides,
    WorkflowInfo,
    format_validation_errors,
)

ENGINE_CONFIG_ENV = "AGENT_SANDBOX_FACTORY_CONFIG"
DEFAULT_ENGINE_CONFIG_PATH = Path("~/.config/agent-sandbox/factory.yaml")
REPO_CONFIG_FILE = "factory.yaml"
WORKFLOWS_DIR = ".factory/workflows"
AGENTS_DIR = ".omp/agents"
OUTPUT_DIR = ".factory/out"
SESSION_DIR = ".omp-session"
DEFAULTS_FILE = "defaults.yaml"
USER_AGENTS_DIR = "agents"
DEFAULT_SANDBOX_PROFILE = "factory-default"
REPO_PATH_PLACEHOLDER = "{repo_path}"
REPO_PATH_KEYS = ("hooks", "gates", "preview")
DEFAULT_PROTECTED = ("factory.yaml", ".factory/**", ".omp/agents/**")
SETTINGS_KEYS = ("merge", "max_rounds", "sandbox", "models", "agent_timeout_s", "command_timeout_s")
OVERRIDABLE_KEYS = ("merge", "max_rounds", "preview", "sandbox", "models")
BUILTIN_DIR = resources.files("agent_sandbox.factory") / "builtin"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FRONT_MATTER_PATTERN = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)(.*)\Z", re.DOTALL)


class ConfigError(ValueError):
    """Report an invalid or missing configuration file."""


ModelT = TypeVar("ModelT", bound=BaseModel)


class StrictModel(BaseModel):
    """Reject unknown fields so typos in config files fail loudly."""

    model_config = ConfigDict(extra="forbid")


# --- Engine config ---------------------------------------------------------------------------------


class EngineLimits(StrictModel):
    """Hold the engine-wide concurrency limits."""

    in_progress: int = Field(default=4, ge=1)


class EngineConfig(StrictModel):
    """Hold the engine process configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)  # defaults pass the validators too, so `~` expands

    listen: str = "127.0.0.1:8090"
    database_url: str
    state_dir: Path = Path("~/.local/state/agent-sandbox/factory")
    worktree_root: Path = Path("~/git/worktrees")  # task worktrees live at <worktree_root>/<repo>/<task_id>
    defaults_dir: Path = Path("~/.config/agent-sandbox/factory")  # user defaults: defaults.yaml and agents/<name>.md
    preview_ports: tuple[int, int] = (8100, 8199)
    limits: EngineLimits = EngineLimits()
    otel_endpoint: str | None = None
    omp_binary: str = "omp"

    @field_validator("state_dir", "worktree_root", "defaults_dir")
    @classmethod
    def _expand_home(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("listen")
    @classmethod
    def _check_listen(cls, value: str) -> str:
        host, separator, port = value.rpartition(":")
        if not separator or not host or not port.isdigit() or not 0 < int(port) < 65536:
            raise ValueError("listen must look like host:port")
        return value

    @field_validator("preview_ports")
    @classmethod
    def _check_ports(cls, value: tuple[int, int]) -> tuple[int, int]:
        low, high = value
        if not 0 < low <= high < 65536:
            raise ValueError("preview_ports must be [low, high] with low <= high")
        return value

    @property
    def host(self) -> str:
        return self.listen.rpartition(":")[0]

    @property
    def port(self) -> int:
        return int(self.listen.rpartition(":")[2])


def expand_env(value: Any) -> Any:
    """Replace ${VAR} and ${VAR:-default} in every string of a parsed YAML tree."""
    if isinstance(value, str):

        def substitute(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is not set")

        return _ENV_PATTERN.sub(substitute, value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def engine_config_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the engine config path from an argument, the environment, or the default."""
    chosen = explicit or os.environ.get(ENGINE_CONFIG_ENV) or DEFAULT_ENGINE_CONFIG_PATH
    return Path(chosen).expanduser()


def parse_engine_config(text: str) -> EngineConfig:
    """Parse engine config YAML with environment expansion."""
    data = _load_mapping(text, "engine config")
    return _validate(EngineConfig, expand_env(data), "engine config")


def load_engine_config(explicit: str | os.PathLike[str] | None = None) -> EngineConfig:
    """Read and validate the engine config file."""
    path = engine_config_path(explicit)
    text = _read_config_text(path, "engine config")
    return parse_engine_config(text)


def task_dir(state_dir: Path, task_id: str) -> Path:
    """Return the directory holding a task's sessions, attachments, logs, and preview data."""
    return state_dir / "tasks" / task_id


# --- Repo templates --------------------------------------------------------------------------------


class PreviewProcess(StrictModel):
    """Describe one preview process."""

    name: str
    argv: list[str] = Field(min_length=1)
    ready_url: str

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
            raise ValueError("process name must be an identifier (used in {port.<name>})")
        return value


class PreviewConfig(StrictModel):
    """Describe how to run a repo's web preview (mirrors the `preview:` block of factory.yaml)."""

    processes: list[PreviewProcess] = Field(min_length=1)
    env: dict[str, str] = {}
    database: bool = False
    url: str
    api_url: str | None = None  # Backend API base handed to the scenario as --api; defaults to the preview url.
    ready_timeout_s: int = Field(default=180, gt=0)

    @model_validator(mode="after")
    def _unique_process_names(self) -> "PreviewConfig":
        names = [process.name for process in self.processes]
        if len(set(names)) != len(names):
            raise ValueError("preview process names must be unique")
        return self


class Hooks(StrictModel):
    """Hold the lifecycle hook command lists."""

    after_create: list[list[str]] = []
    after_code: list[list[str]] = []
    before_remove: list[list[str]] = []


class RepoConfig(StrictModel):
    """Mirror a repo's optional factory.yaml: every field is optional and a missing file means all defaults."""

    main_branch: str | None = None  # None: the branch checked out in the main checkout when a task starts
    profile: str = DEFAULT_SANDBOX_PROFILE
    hooks: Hooks = Hooks()
    gates: list[list[str]] = []
    protected: list[str] = []
    preview: PreviewConfig | None = None
    merge: MergeMode | None = None
    max_rounds: MaxRounds | None = None
    sandbox: bool | None = None
    models: dict[str, str] | None = None
    agent_timeout_s: int | None = Field(default=None, ge=1, le=86400)
    command_timeout_s: int | None = Field(default=None, ge=1, le=86400)

    @model_validator(mode="after")
    def _commands_nonempty(self) -> "RepoConfig":
        commands = [*self.hooks.after_create, *self.hooks.after_code, *self.hooks.before_remove, *self.gates]
        if any(not command for command in commands):
            raise ValueError("hooks and gates must be non-empty argv lists")
        return self


class WorkflowConfig(StrictModel):
    """Mirror one `.factory/workflows/<id>.yaml`."""

    name: str
    phases: list[AgentPhase]
    max_rounds: MaxRounds | None = None
    merge: MergeMode | None = None
    preview: bool | None = None
    sandbox: bool | None = None
    models: dict[str, str] | None = None

    @field_validator("phases")
    @classmethod
    def _check_phases(cls, value: list[str]) -> list[str]:
        if "code" not in value:
            raise ValueError("phases must include code")
        if value != [phase for phase in AGENT_PHASES if phase in value]:
            raise ValueError(f"phases must be an ordered, duplicate-free subset of {list(AGENT_PHASES)}")
        return value


class AgentDef(StrictModel):
    """Hold one agent's front matter and system prompt."""

    name: str
    description: str = ""
    model: str | None = None
    thinking: str | None = None
    writes: list[str] | None = None
    output: str | None = None
    prompt: str = ""

    @field_validator("output")
    @classmethod
    def _check_output(cls, value: str | None) -> str | None:
        if value is None:
            return value
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or str(path) != value or not value.startswith(OUTPUT_DIR + "/"):
            raise ValueError(f"output must be a normalized relative path under {OUTPUT_DIR}/")
        return value


class TaskConfig(StrictModel):
    """Freeze every setting a task needs at start, independent of later repo changes."""

    repo: str
    repo_path: str
    profile: str
    main_branch: str
    worktree_root: str  # the repo's directory under the engine worktree root, task worktrees are `<worktree_root>/<task_id>`
    sandbox: bool
    merge: MergeMode
    max_rounds: MaxRounds
    preview_enabled: bool
    preview: PreviewConfig | None
    phases: list[str]
    agents: dict[str, AgentDef]  # keyed by phase
    hooks: Hooks
    gates: list[list[str]]
    protected: list[str]
    agent_timeout_s: int
    command_timeout_s: int


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return base with override merged in recursively; override values win."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_settings(user: FactorySettings, repo: RepoConfig, workflow: WorkflowConfig, overrides: Overrides) -> FactorySettings:
    """Merge packaged defaults < user defaults < factory.yaml < workflow < request overrides, then validate."""
    layers = (
        user.model_dump(),
        {key: getattr(repo, key) for key in SETTINGS_KEYS if getattr(repo, key) is not None},
        {key: getattr(workflow, key) for key in OVERRIDABLE_KEYS if getattr(workflow, key) is not None},
        overrides.model_dump(exclude_none=True),
    )
    merged: dict[str, Any] = {}
    for layer in layers:
        merged = deep_merge(merged, layer)
    return _validate(FactorySettings, merged, "merged settings")


def parse_agent_markdown(text: str, source: str) -> AgentDef:
    """Split YAML front matter from the markdown body and validate the agent definition."""
    match = _FRONT_MATTER_PATTERN.match(text)
    if match is None:
        raise ConfigError(f"{source}: missing YAML front matter")
    front = _load_mapping(match.group(1), f"{source} front matter")
    if "prompt" in front:
        raise ConfigError(f"{source}: front matter must not contain 'prompt'")
    return _validate(AgentDef, {**front, "prompt": match.group(2).strip()}, source)


def agent_output_path(agent: AgentDef, phase: str) -> str:
    """Return the repo-relative envelope path an agent writes for a phase."""
    return agent.output or f"{OUTPUT_DIR}/{phase}.json"


def agent_writes(agent: AgentDef, phase: str) -> list[str]:
    """Return the writes allowlist for an agent: code defaults to everything, other phases to nothing."""
    if agent.writes is not None:
        return agent.writes
    return ["**"] if phase == "code" else []


def load_repo_config(repo_path: Path) -> RepoConfig:
    """Read a repo's optional factory.yaml from its main checkout; a missing file means every default."""
    path = repo_path / REPO_CONFIG_FILE
    if not path.is_file():
        return RepoConfig()
    data = _load_mapping(_read_config_text(path, "repo config"), str(path))
    for key in REPO_PATH_KEYS:
        if key in data:
            data[key] = _expand_repo_path(data[key], repo_path)
    return _validate(RepoConfig, data, str(path))


def _workflow_path(repo_path: Path, workflow_id: str) -> Path:
    if not _IDENTIFIER_PATTERN.fullmatch(workflow_id):
        raise ConfigError(f"invalid workflow id {workflow_id!r}")
    return repo_path / WORKFLOWS_DIR / f"{workflow_id}.yaml"


def builtin_workflow_ids() -> list[str]:
    """List the ids of the workflows packaged with the engine."""
    return sorted(entry.name.removesuffix(".yaml") for entry in (BUILTIN_DIR / "workflows").iterdir() if entry.name.endswith(".yaml"))


def load_workflow(repo_path: Path, workflow_id: str) -> WorkflowConfig:
    """Read a workflow: the repo's `.factory/workflows/<id>.yaml` wins over the packaged workflow of the same id."""
    path = _workflow_path(repo_path, workflow_id)
    if path.is_file():
        return _validate(WorkflowConfig, _load_mapping(_read_config_text(path, "workflow"), str(path)), str(path))
    builtin = BUILTIN_DIR / "workflows" / f"{workflow_id}.yaml"
    if not builtin.is_file():
        raise ConfigError(f"unknown workflow {workflow_id!r}")
    source = f"builtin workflow {workflow_id}"
    return _validate(WorkflowConfig, _load_mapping(builtin.read_text(encoding="utf-8"), source), source)


def list_workflows(repo_path: Path, defaults_dir: Path) -> list[WorkflowInfo]:
    """Describe the packaged and repo workflows, resolving settings through user defaults and factory.yaml."""
    repo = load_repo_config(repo_path)
    user = load_user_settings(defaults_dir)
    repo_ids = {path.stem for path in (repo_path / WORKFLOWS_DIR).glob("*.yaml")}
    infos = []
    for workflow_id in sorted({*builtin_workflow_ids(), *repo_ids}):
        workflow = load_workflow(repo_path, workflow_id)
        settings = merge_settings(user, repo, workflow, Overrides())
        infos.append(
            WorkflowInfo(
                id=workflow_id,
                name=workflow.name,
                phases=list(workflow.phases),
                merge=settings.merge,
                preview=settings.preview and repo.preview is not None,
            )
        )
    return infos


def builtin_agent_text(agent_name: str) -> str:
    """Return the packaged markdown of an agent."""
    return (BUILTIN_DIR / "agents" / f"{agent_name}.md").read_text(encoding="utf-8")


def user_agent_text(defaults_dir: Path, agent_name: str) -> str:
    """Return the user's markdown for an agent, falling back to the packaged one."""
    path = defaults_dir / USER_AGENTS_DIR / f"{agent_name}.md"
    return _read_config_text(path, "agent file") if path.is_file() else builtin_agent_text(agent_name)


def load_agent(repo_path: Path, defaults_dir: Path, agent_name: str) -> AgentDef:
    """Resolve an agent as repo `.omp/agents/<name>.md`, then user default, then packaged, expanding {repo_path} in its prompt."""
    path = repo_path / AGENTS_DIR / f"{agent_name}.md"
    if path.is_file():
        text, source = _read_config_text(path, "agent file"), str(path)
    else:
        text, source = user_agent_text(defaults_dir, agent_name), f"default agent {agent_name}"
    return parse_agent_markdown(_expand_repo_path(text, repo_path), source)


# --- User defaults ---------------------------------------------------------------------------------


def load_user_settings(defaults_dir: Path) -> FactorySettings:
    """Read the user's settings layer; a missing file means the packaged defaults."""
    path = defaults_dir / DEFAULTS_FILE
    if not path.is_file():
        return FactorySettings()
    return _validate(FactorySettings, _load_mapping(_read_config_text(path, "defaults file"), str(path)), str(path))


def load_defaults(defaults_dir: Path) -> FactoryDefaults:
    """Return the effective user-or-packaged settings and agent markdown."""
    return FactoryDefaults(
        settings=load_user_settings(defaults_dir),
        agents={name: user_agent_text(defaults_dir, name) for name in AGENT_NAMES},
    )


def save_defaults(defaults_dir: Path, defaults: FactoryDefaults) -> FactoryDefaults:
    """Validate and write the user defaults; agents equal to the packaged text drop their override file."""
    unknown = sorted(set(defaults.agents) - set(AGENT_NAMES))
    if unknown:
        raise ConfigError(f"unknown agents {unknown}, expected {list(AGENT_NAMES)}")
    for name, text in defaults.agents.items():
        parse_agent_markdown(text, f"agent {name}")
    _write_atomically(defaults_dir / DEFAULTS_FILE, yaml.safe_dump(defaults.settings.model_dump(mode="json"), sort_keys=False))
    for name, text in defaults.agents.items():
        path = defaults_dir / USER_AGENTS_DIR / f"{name}.md"
        if text == builtin_agent_text(name):
            path.unlink(missing_ok=True)
        else:
            _write_atomically(path, text)
    return load_defaults(defaults_dir)


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_task_config(
    engine: EngineConfig, repo_name: str, repo_path: Path, workflow_id: str, overrides: Overrides, checked_out_branch: str | None
) -> TaskConfig:
    """Read every template layer and freeze the effective configuration for one task.

    `checked_out_branch` is the branch currently checked out in the main checkout (None when detached), the default main branch.
    """
    repo = load_repo_config(repo_path)
    workflow = load_workflow(repo_path, workflow_id)
    settings = merge_settings(load_user_settings(engine.defaults_dir), repo, workflow, overrides)
    main_branch = repo.main_branch or checked_out_branch
    if main_branch is None:
        raise ConfigError(f"{repo_path} has a detached HEAD: check out a branch or set main_branch in {REPO_CONFIG_FILE}")
    agents: dict[str, AgentDef] = {}
    for phase in workflow.phases:
        agent_name = PHASE_AGENTS[phase]
        agent = load_agent(repo_path, engine.defaults_dir, agent_name)
        agents[phase] = agent.model_copy(update={"model": settings.models.get(agent_name, agent.model)})
    return TaskConfig(
        repo=repo_name,
        repo_path=str(repo_path),
        profile=repo.profile,
        main_branch=main_branch,
        worktree_root=str(engine.worktree_root / repo_name),
        sandbox=settings.sandbox,
        merge=settings.merge,
        max_rounds=settings.max_rounds,
        preview_enabled=settings.preview and repo.preview is not None,
        preview=repo.preview,
        phases=list(workflow.phases),
        agents=agents,
        hooks=repo.hooks,
        gates=repo.gates,
        protected=[*DEFAULT_PROTECTED, *(pattern for pattern in repo.protected if pattern not in DEFAULT_PROTECTED)],
        agent_timeout_s=settings.agent_timeout_s,
        command_timeout_s=settings.command_timeout_s,
    )


def _read_config_text(path: Path, what: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read {what} {path}: {error}") from error


def _expand_repo_path(value: Any, repo_path: Path) -> Any:
    """Replace {repo_path} in every string of a parsed YAML tree."""
    if isinstance(value, str):
        return value.replace(REPO_PATH_PLACEHOLDER, str(repo_path))
    if isinstance(value, list):
        return [_expand_repo_path(item, repo_path) for item in value]
    if isinstance(value, dict):
        return {key: _expand_repo_path(item, repo_path) for key, item in value.items()}
    return value


def _load_mapping(text: str, source: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"{source}: invalid YAML: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level")
    return data


def _validate(model: type[ModelT], data: Any, source: str) -> ModelT:
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise ConfigError(f"{source}: {format_validation_errors(error.errors())}") from error
