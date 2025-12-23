from abc import ABC
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union, get_args, get_origin

import git
from llm_baseclient.config import OLLAMA_PORT
from pydantic import BaseModel, Field
import streamlit as st

from code_agents.config import (
    DOCKERTAG_AIDER,
    DOCKERTAG_CLAUDE,
    DOCKERTAG_CODEX,
    DOCKERTAG_CURSOR,
    DOCKERTAG_GEMINI,
    DOCKERTAG_OPENCODE,
    DOCKERTAG_QWEN,
    MODELS_GEMINI,
    MODELS_OLLAMA,
    MODELS_OPENAI,
    MODELS_VLLM,
)
from code_agents.sandbox import DockerSandbox

GIT_NAME = subprocess.run(["git", "config", "--global", "user.name"], capture_output=True, text=True).stdout.strip()
GIT_EMAIL = subprocess.run(["git", "config", "--global", "user.email"], capture_output=True, text=True).stdout.strip()

ENV_VARS = {
    "OLLAMA_API_BASE": f"http://host.docker.internal:{OLLAMA_PORT}",
    "GIT_AUTHOR_NAME": GIT_NAME,
    "GIT_COMMITTER_NAME": GIT_NAME,
    "GIT_AUTHOR_EMAIL": GIT_EMAIL,
    "GIT_COMMITTER_EMAIL": GIT_EMAIL,
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "OLLAMA_API_KEY": os.getenv("OLLAMA_API_KEY", ""),
}

DEFAULT_ARGS_AIDER = ["--dark-mode", "--code-theme", "inkpot", "--pretty"]


def model_selector(key: str) -> dict:
    """Create model selection dropdowns in Streamlit sidebar expanders."""
    providers = [
        ("Ollama", MODELS_OLLAMA, "ollama/"),
        ("Gemini", MODELS_GEMINI, "gemini/"),
        ("OpenAI", MODELS_OPENAI, "openai/"),
        ("VLLM", MODELS_VLLM, "hosted_vllm/"),
    ]
    available = [(name, models, prefix) for name, models, prefix in providers if models]
    model_options = [name for name, _, _ in available]
    model_configs = {name: (models, prefix) for name, models, prefix in available}

    selected_provider = st.radio(
        label="Model Provider",
        options=model_options,
        index=0,
        horizontal=True,
        key=f"model_provider_radio_{key}",
    )
    models_list, litellm_prefix = model_configs[selected_provider]

    return st.selectbox(
        label=f"Models ({selected_provider})",
        options=models_list,
        format_func=lambda model: model.replace(litellm_prefix, ""),
        key=f"{selected_provider}_model_select_{key}",
    )


class AgentCommand(BaseModel, ABC):
    """Base command model for external agent processes.

    Input:
        executable: str
           - preinstalled CLI executable name
        workspace: Path
           - absolute or relative filesystem path
           - must exist before execution
        args: list[str]
           - ordered CLI arguments
           - no executable included
        env_vars: dict[str, str] | None
           - environment variable overrides
           - merged over os.environ
    """

    executable: str = Field(..., description="CLI executable name")
    workspace: Path = Field(default_factory=Path.cwd, description="Agent workspace directory")
    args: List[str] = Field(default_factory=list, description="CLI arguments excluding executable")
    task_injection_template: List[str] = Field(default_factory=list, description="Task injection template")
    env_vars: Dict[str, str] = Field(default=ENV_VARS, description="Environment variable overrides")

    def _snake_to_kebab(self, s: str) -> str:
        """Convert snake_case to kebab-case."""
        return s.replace("_", "-")

    @classmethod
    def construct_args_from_values(cls, **field_values: Dict[str, Any]) -> List[str]:
        """Construct argument list from field values."""
        temp_instance = cls(**field_values)
        args = [
            item
            for k, v in field_values.items()
            if k not in {"executable", "workspace", "args", "env_vars", "task_injection_template"}
            for item in ([f"--{temp_instance._snake_to_kebab(k)}"] + ([] if isinstance(v, bool) else [str(v)]))
        ]
        additional_args = field_values.get("args", [])
        return args + additional_args

    def construct_args(self) -> List[str]:
        """Construct argument list."""
        fields = self.model_dump(exclude={"executable", "workspace", "args", "env_vars", "task_injection_template"})
        args = [
            item for k, v in fields.items() for item in ([f"--{self._snake_to_kebab(k)}"] + ([] if isinstance(v, bool) else [str(v)]))
        ]
        return args + self.args

    @classmethod
    def ui_define_fields(cls) -> Dict[str, Any]:
        """Generate UI elements for all pydantic fields and return their values."""
        values: Dict[str, Any] = {}
        model_fields = cls.model_fields
        base_excluded = {"executable", "workspace", "args", "env_vars", "task_injection_template"}
        base_key_prefix = cls.__name__.lower()

        def key_for(field_name: str) -> str:
            return f"{base_key_prefix}_{field_name}"

        def is_literal(t: Union[type, object]) -> bool:
            return get_origin(t) is Literal

        def literal_choices(t: Union[type, object]) -> List[Any]:
            return list(get_args(t))

        def is_list(t: Union[type, object]) -> bool:
            return get_origin(t) in (list, List)

        def inner_type(t: Union[type, object]) -> Union[type, object]:
            args = get_args(t)
            return args[0] if args else type(None)

        def ui_for_bool(name: str, description: str, default: Optional[bool]) -> bool:
            return st.checkbox(
                description,
                value=default if default is not None else False,
                key=key_for(name),
            )

        def ui_for_literal(name: str, description: str, default: Any, t: Union[type, object]) -> Any:
            options = literal_choices(t)
            index = 0
            if default is not None and default in options:
                index = options.index(default)
            return st.selectbox(
                description,
                options=options,
                index=index,
                key=key_for(name),
            )

        def ui_for_list(name: str, description: str, default: Any, t: Union[type, object]) -> List[Any]:
            inner = inner_type(t)
            if is_literal(inner):
                options = literal_choices(inner)
                default_list = default if isinstance(default, list) else []
                return st.multiselect(
                    description,
                    options=options,
                    default=default_list,
                    key=key_for(name),
                )
            text_default = ", ".join(default) if isinstance(default, list) else ""
            text_input = st.text_input(
                description,
                value=text_default,
                key=key_for(name),
            )
            return [item.strip() for item in text_input.split(",") if item.strip()]

        def ui_for_number(name: str, description: str, default: Any, t: Union[type, object]) -> Union[int, float]:
            is_int = t is int
            value = default if default is not None else (0 if is_int else 0.0)
            step = 1 if is_int else 0.1
            return st.number_input(
                description,
                value=value,
                key=key_for(name),
                step=step,
            )

        def ui_for_text(name: str, description: str, default: Any) -> str:
            return st.text_input(
                description,
                value=default if default is not None else "",
                key=key_for(name),
            )

        for field_name, field_info in model_fields.items():
            if field_name in base_excluded:
                continue

            field_type = field_info.annotation
            field_description = field_info.description or field_name
            field_default = field_info.default if field_info.default != ... else None

            if field_type is bool:
                values[field_name] = ui_for_bool(field_name, field_description, field_default)
                continue

            if is_literal(field_type):
                values[field_name] = ui_for_literal(field_name, field_description, field_default, field_type)
                continue

            if is_list(field_type):
                values[field_name] = ui_for_list(field_name, field_description, field_default, field_type)
                continue

            if field_type in (int, float):
                values[field_name] = ui_for_number(field_name, field_description, field_default, field_type)
                continue

            values[field_name] = ui_for_text(field_name, field_description, field_default)

        return values


TCommand = TypeVar("TCommand", bound=AgentCommand)


class CodeAgent(ABC, Generic[TCommand]):
    """Generic base class for Code Agents.

    Class Attributes:
        DOCKERTAG: str
           - Docker image tag for agent execution

    Input:
        repo_url: str
           - HTTPS or SSH git URL
           - points to accessible repository
        branch: str
           - existing or new branch name
           - non-empty string

    Output:
        instance: CodeAgent
           - initialized workspace path
           - ready for command execution

    Side Effects:
        - creates ~/agent_sandbox directory
        - clones or updates git repository
        - may install Python dependencies (if requirements.txt or pyproject.toml present)
    """

    DOCKERTAG: str

    def __init__(self, repo_url: str, branch: str) -> None:
        self.repo_url: str = repo_url
        self.branch: str = branch
        self.path_agent_workspace: Path = self._setup_workspace(repo_url, branch)

    def _setup_workspace(self, repo_url: str, branch: str) -> Path:
        """Setup agent workspace: clone, checkout branch, install deps.

        Input:
            repo_url: str
               - HTTPS or SSH git URL
               - parseable by gitpython
            branch: str
               - target branch name
               - used for checkout or creation

        Output:
            workspace_path: Path
               - absolute path to repo root
               - guaranteed to exist on success

        Side Effects:
            - creates ~/agent_sandbox/<repo_name>
            - runs optional dependency installation
        """
        sandbox_root: Path = Path.home() / "agent_sandbox"
        sandbox_root.mkdir(parents=True, exist_ok=True)

        repo_name: str = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        workspace: Path = sandbox_root / repo_name

        if workspace.exists() and (workspace / ".git").exists():
            subprocess.run(
                ["git", "-C", str(workspace), "pull"],
                check=False,
            )
        else:
            subprocess.run(
                ["git", "clone", "--branch", branch, repo_url, str(workspace)],
                check=False,
            )

        self._try_install_dependencies(workspace)
        return workspace

    def _try_install_dependencies(self, workspace: Path) -> None:
        """
        Best-effort dependency installation; non-fatal on failure.
        Assumes `uv` installed on system PATH.

        Input:
            workspace: Path
               - repository root path
               - must exist
        """
        requirements: Path = workspace / "requirements.txt"
        pyproject: Path = workspace / "pyproject.toml"

        cmd: Optional[List[str]] = None
        if requirements.is_file():
            cmd = ["uv", "pip", "install", "-r", str(requirements)]
        elif pyproject.is_file():
            cmd = ["uv", "pip", "install", "-e", str(workspace)]

        if not cmd:
            return

        try:
            subprocess.run(
                cmd,
                cwd=workspace,
                check=False,
            )
        except OSError:
            # ignore installation failures; agent expected to handle env issues
            return

    def _execute_agent_command(self, command: TCommand) -> None:
        """Execute the agent command in its workspace using DockerSandbox.

        Input:
            command: TCommand
               - fully configured agent command

        Side Effects:
            - runs agent inside secure Docker sandbox
        """
        agent_shell_cmd = subprocess.list2cmdline([command.executable, *command.construct_args()])

        sandbox = DockerSandbox(dockerimage_name=f"{self.__class__.DOCKERTAG}:latest")
        try:
            sandbox.run_interactive_shell(
                repo_path=str(self.path_agent_workspace), agent_cmd=agent_shell_cmd, env_vars=command.env_vars
            )
        except Exception as exc:
            st.error(f"Failed to run agent in sandbox: {exc}")
            raise

    def run(self, command: TCommand, task: Optional[str] = None) -> None:
        """Combines task with command according to agent-specific syntax.

        Input:
            task: str
               - natural language instruction
               - non-empty string
            command: TCommand
               - agent-specific command model
               - mutated with task context

        Side Effects:
            - runs agent inside secure Docker sandbox
        """
        if task:
            injected_task = [token.format(task=task) for token in command.task_injection_template]
            command.args += injected_task
        with st.spinner("Running agent in secure sandbox; interact via its UI if opened."):
            self._execute_agent_command(command)

    def get_command_class(self) -> type[TCommand]:
        """Get the command class associated with this agent using generic type resolution."""
        for cls in self.__class__.__mro__:
            orig_bases = getattr(cls, "__orig_bases__", ())
            for base in orig_bases:
                if get_origin(base) is CodeAgent:
                    args = get_args(base)
                    if args:
                        return args[0]
        raise NotImplementedError(f"Could not determine command class for {self.__class__.__name__}")

    def ui_define_command(self) -> TCommand:
        """Generic UI definition that uses pydantic model to auto-generate UI elements."""
        command_class = self.get_command_class()
        st.markdown(f"# {self.__class__.__name__} Command")
        with st.expander("", expanded=True):
            field_values = command_class.ui_define_fields()

            extra_args_str = st.text_input(
                "Extra CLI arguments (optional)",
                value="",
                key=f"{self.__class__.__name__.lower()}_extra_args",
                help="Space-separated extra arguments passed directly to the CLI",
            )
            extra_args = extra_args_str.split() if extra_args_str.strip() else []

            field_values["workspace"] = self.path_agent_workspace
            field_values["args"] = extra_args + (field_values.get("args", []))

            command = command_class(**field_values)

            with st.expander("Display Command", expanded=True):
                args = command.construct_args()
                formatted_args = "\n\t".join(args)
                st.code(f"{command.executable} {formatted_args}", language="bash")

        return command

    def get_diff(self) -> str:
        """Return git diff for workspace."""
        result = subprocess.run(
            ["git", "-C", str(self.path_agent_workspace), "diff"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout


class AiderCommand(AgentCommand):
    """Aider-specific command definition."""

    # Baseclass constants
    executable: str = "aider"
    task_injection_template: List[str] = ["--message", "{task}"]

    # Variables
    model: str = Field(..., description="Architect LLM identifier")
    editor_model: str = Field(..., description="Editor LLM identifier")
    reasoning_effort: Literal["low", "medium", "high"] = Field(default="high", description="Reasoning effort")
    edit_format: Literal["diff", "whole", "udiff"] = Field(default="diff", description="Edit format")
    map_tokens: Literal[1024, 2048, 4096, 8192] = Field(default=4096, description="Context map tokens")


class Aider(CodeAgent[AiderCommand]):
    """Aider Code Agent."""

    DOCKERTAG = DOCKERTAG_AIDER

    def ui_define_command(self) -> AiderCommand:
        """Define the Aider command with Streamlit UI - example of overridability."""
        st.markdown("# Model Control")
        with st.expander("", expanded=True):
            reasoning_effort = st.selectbox("Reasoning effort", ["low", "medium", "high"], index=2, key="aider_reasoning_effort")
            model_architect = model_selector(key="code_agent_architect")
            model_editor = model_selector(key="code_agent_editor")

        st.markdown("---")
        st.markdown("# Command Control")
        with st.expander("", expanded=False):
            edit_format = st.selectbox("Select Edit Format", ["diff", "whole", "udiff"], index=0, key="aider_edit_format")
            map_tokens = st.selectbox("Context map tokens", [1024, 2048, 4096, 8192], index=0, key="aider_map_tokens")

            common_flags = ["--architect", "--no-auto-commit", "--no-stream", "--browser", "--yes", "--cache-prompts"]
            flags = st.multiselect(
                "Common aider flags",
                options=common_flags,
                key="aider_common_flags",
                default=[common_flags[0], common_flags[1]],
                accept_new_options=True,
            )
            cmd = AiderCommand(
                workspace=self.path_agent_workspace,
                args=flags + DEFAULT_ARGS_AIDER,
                model=model_architect,
                editor_model=model_editor,
                reasoning_effort=reasoning_effort,
                edit_format=edit_format,
                map_tokens=map_tokens,
            )
            with st.expander("Display Command", expanded=True):
                args = cmd.construct_args()
                formatted_args = "\n\t".join(args)
                st.code(f"{cmd.executable} {formatted_args}", language="bash")

        return cmd


class OpenCodeCommand(AgentCommand):
    """OpenCode-specific command definition."""

    executable: str = "opencode"


class OpenCode(CodeAgent[OpenCodeCommand]):
    """OpenCode Code Agent."""

    DOCKERTAG = DOCKERTAG_OPENCODE


class GeminiCommand(AgentCommand):
    """Gemini-specific command definition."""

    executable: str = "gemini"


class Gemini(CodeAgent[GeminiCommand]):
    """Gemini Code Agent."""

    DOCKERTAG = DOCKERTAG_GEMINI


class QwenCommand(AgentCommand):
    """Qwen-specific command definition."""

    # Baseclass constants
    executable: str = "qwen"


class Qwen(CodeAgent[QwenCommand]):
    """Qwen Code Agent."""

    DOCKERTAG = DOCKERTAG_QWEN


class CodexCommand(AgentCommand):
    """Codex-specific command definition."""

    executable: str = "codex"


class Codex(CodeAgent[CodexCommand]):
    """Codex Code Agent."""

    DOCKERTAG = DOCKERTAG_CODEX


class ClaudeCodeCommand(AgentCommand):
    """Claude Code-specific command definition."""

    executable: str = "claude"  # adjust if your CLI entrypoint differs


class ClaudeCode(CodeAgent[ClaudeCodeCommand]):
    """Claude Code Agent."""

    DOCKERTAG = DOCKERTAG_CLAUDE


class CursorCommand(AgentCommand):
    """Cursor CLI-specific command definition."""

    executable: str = "cursor-agent"


class Cursor(CodeAgent[CursorCommand]):
    """Cursor CLI Code Agent."""

    DOCKERTAG = DOCKERTAG_CURSOR


# Agent registry: dynamic discovery for extensible multi-agent support
agent_subclasses = CodeAgent.__subclasses__()
agent_subclass_dict = {cls.__name__: cls for cls in agent_subclasses}
agent_subclass_names = list(agent_subclass_dict.keys())


@st.cache_resource
def get_agent(agent_type: str, repo_url: str, branch: str) -> CodeAgent[Any]:
    """Instantiate and cache agent instance."""
    return agent_subclass_dict[agent_type](repo_url, branch)


def get_remote_branches(repo_url: str) -> list[str]:
    """Extract branch names from remote repository."""
    lines = git.Git().ls_remote("--heads", repo_url).splitlines()
    return [ref.split("\t", 1)[1].replace("refs/heads/", "") for ref in lines]


def agent_controls() -> None:
    """Streamlit entrypoint for multi-agent code workspace."""

    if "selected_agent" not in st.session_state:
        st.markdown("## Agent Controls")
        selected_agent_name: str = st.selectbox("Select Code Agent", options=agent_subclass_names, key="code_agent_selector")
        repo_url = st.text_input("GitHub Repository URL", key="repo_url")
        if repo_url:
            if "branches" not in st.session_state or st.session_state.cached_repo_url != repo_url:
                st.session_state.branches = get_remote_branches(repo_url)
                st.session_state.cached_repo_url = repo_url
            branch = st.selectbox("Select Branch", options=st.session_state.branches, index=0, key="branch_selector")

        def init_agent() -> None:
            """Initialize and store agent in session state."""
            selected_agent: CodeAgent[Any] = get_agent(agent_type=selected_agent_name, repo_url=repo_url, branch=branch)
            st.session_state.selected_agent = selected_agent

        if repo_url and branch:
            st.button("Initialize Agent", key="init_agent_button", on_click=init_agent)

    else:
        execute_button = st.columns(1)
        selected_agent: CodeAgent[Any] = st.session_state.selected_agent
        repo_url = selected_agent.repo_url
        branch = selected_agent.branch
        repo_slug = "/".join(repo_url.split("/")[-2:])  # Extracts 'owner/repo'
        branch_url = f"{repo_url}/tree/{branch}"

        st.markdown("# Agent Info")
        with st.expander("", expanded=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write("**Agent**")
                st.write("**Source**")
                st.write("**Workspace**")
            with col2:
                st.markdown(f"{selected_agent.__class__.__name__}")
                st.markdown(f"[{repo_slug}]({repo_url}) / [{branch}]({branch_url})")
                st.markdown(f"`{selected_agent.path_agent_workspace}`")
            if st.button("Reset Agent", use_container_width=True):
                del st.session_state.selected_agent
                st.rerun()

        command: AgentCommand = selected_agent.ui_define_command()
        with execute_button[0]:
            st.button(
                "Execute Command",
                use_container_width=True,
                type="primary",
                on_click=lambda: selected_agent.run(command=command),
            )


def chat_interface() -> None:
    """Streamlit chat interface for code agents."""

    if "selected_agent" not in st.session_state:
        st.info("Please select and initialize an agent first.")
        return

    with st._bottom:
        task: Optional[str] = st.chat_input("Assign a task to the agent...")

    if task:
        with st.chat_message("user"):
            st.markdown(task)

        with st.chat_message("assistant"):
            selected_agent: CodeAgent[Any] = st.session_state.selected_agent
            command: AgentCommand = selected_agent.ui_define_command()

            selected_agent.run(task=task, command=command)
            diff: str = selected_agent.get_diff()
            if diff:
                st.markdown("### Git Diff")
                st.code(diff, language="diff")
            else:
                st.info("No changes detected in git diff.")


def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(page_title="Agents-in-a-Box", layout="wide")
    agent_controls()
    chat_interface()


if __name__ == "__main__":
    main()
