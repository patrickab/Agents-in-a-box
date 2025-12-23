from abc import ABC, abstractmethod
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from config import (
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
import git
from llm_baseclient.config import OLLAMA_PORT
from pydantic import BaseModel, Field
from sandbox import DockerSandbox
import streamlit as st

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
    model_options = []
    model_configs = {}

    if MODELS_OLLAMA != []:
        model_options.append("Ollama")
        model_configs["Ollama"] = (MODELS_OLLAMA, "ollama/")
    if MODELS_GEMINI != []:
        model_options.append("Gemini")
        model_configs["Gemini"] = (MODELS_GEMINI, "gemini/")
    if MODELS_OPENAI != []:
        model_options.append("OpenAI")
        model_configs["OpenAI"] = (MODELS_OPENAI, "openai/")
    if MODELS_VLLM != []:
        model_options.append("VLLM")
        model_configs["VLLM"] = (MODELS_VLLM, "hosted_vllm/")

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

    def construct_args(self) -> List[str]:
        """Construct aider argument list."""
        # This model dump will include all class values of the subclass
        # The fields is a dict of all class attributes intended for CLI
        fields = self.model_dump(exclude={"executable", "workspace", "args", "env_vars", "task_injection_template"})
        args = [
            item for k, v in fields.items() for item in ([f"--{self._snake_to_kebab(k)}"] + ([] if isinstance(v, bool) else [str(v)]))
        ]
        return args + self.args


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
        env_parts: List[str] = []
        # if command.env_vars:
        #     full_env.update(command.env_vars)
        #     for k, v in command.env_vars.items():
        #         env_parts.append(f"{k}={v}")

        arg_list: List[str] = [command.executable, *command.construct_args()]
        agent_shell_cmd = " ".join([*env_parts, subprocess.list2cmdline(arg_list)])

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

    @abstractmethod
    def ui_define_command(self) -> TCommand:
        """Define the agent command with Streamlit UI."""
        raise NotImplementedError

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
        """Define the Aider command with Streamlit UI."""
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

    def ui_define_command(self) -> OpenCodeCommand:
        """Define the OpenCode command with Streamlit UI."""
        st.markdown("# OpenCode Command")
        with st.expander("", expanded=True):
            extra_args_str = st.text_input(
                "Extra CLI arguments (optional)",
                value="",
                key="opencode_extra_args",
                help="Space-separated extra arguments passed directly to the OpenCode CLI",
            )
            extra_args = extra_args_str.split() if extra_args_str.strip() else []

            cmd = OpenCodeCommand(
                workspace=self.path_agent_workspace,
                args=extra_args,
            )

            with st.expander("Display Command", expanded=True):
                args = cmd.construct_args()
                formatted_args = "\n\t".join(args)
                st.code(f"{cmd.executable} {formatted_args}", language="bash")

        return cmd


class GeminiCommand(AgentCommand):
    """Gemini-specific command definition."""

    executable: str = "gemini"


class Gemini(CodeAgent[GeminiCommand]):
    """Gemini Code Agent."""

    DOCKERTAG = DOCKERTAG_GEMINI

    def ui_define_command(self) -> GeminiCommand:
        """Define the Gemini command with Streamlit UI."""
        st.markdown("# Gemini Command")
        with st.expander("", expanded=True):
            # Extra raw flags/args if user wants to customize
            extra_args_str = st.text_input(
                "Extra CLI arguments (optional)",
                value="",
                key="gemini_extra_args",
                help="Space-separated extra arguments passed directly to the Gemini CLI",
            )
            extra_args = extra_args_str.split() if extra_args_str.strip() else []

            cmd = GeminiCommand(
                workspace=self.path_agent_workspace,
                args=extra_args,
            )

            with st.expander("Display Command", expanded=True):
                args = cmd.construct_args()
                formatted_args = "\n\t".join(args)
                st.code(f"{cmd.executable} {formatted_args}", language="bash")

        return cmd


class QwenCommand(AgentCommand):
    """Qwen-specific command definition."""

    # Baseclass constants
    executable: str = "qwen"


class Qwen(CodeAgent[QwenCommand]):
    """Qwen Code Agent."""

    DOCKERTAG = DOCKERTAG_QWEN

    def ui_define_command(self) -> QwenCommand:
        """Define the Qwen command with Streamlit UI."""
        st.markdown("# Qwen Command")
        with st.expander("", expanded=True):
            extra_args_str = st.text_input(
                "Extra CLI arguments (optional)",
                value="",
                key="qwen_extra_args",
                help="Space-separated extra arguments passed directly to the Qwen CLI",
            )
            extra_args = extra_args_str.split() if extra_args_str.strip() else []

            cmd = QwenCommand(
                workspace=self.path_agent_workspace,
                args=extra_args,
            )

            with st.expander("Display Command", expanded=True):
                args = cmd.construct_args()
                formatted_args = "\n\t".join(args)
                st.code(f"{cmd.executable} {formatted_args}", language="bash")

        return cmd


class CodexCommand(AgentCommand):
    """Codex-specific command definition."""

    executable: str = "codex"


class Codex(CodeAgent[CodexCommand]):
    """Codex Code Agent."""

    DOCKERTAG = DOCKERTAG_CODEX

    def ui_define_command(self) -> CodexCommand:
        """Define the Codex command with Streamlit UI."""
        st.markdown("# Codex Command")
        with st.expander("", expanded=True):
            extra_args_str = st.text_input(
                "Extra CLI arguments (optional)",
                value="",
                key="codex_extra_args",
                help="Space-separated extra arguments passed directly to the Codex CLI",
            )
            extra_args = extra_args_str.split() if extra_args_str.strip() else []

            cmd = CodexCommand(
                workspace=self.path_agent_workspace,
                args=extra_args,
            )

            with st.expander("Display Command", expanded=True):
                args = cmd.construct_args()
                formatted_args = "\n\t".join(args)
                st.code(f"{cmd.executable} {formatted_args}", language="bash")

        return cmd


class ClaudeCodeCommand(AgentCommand):
    """Claude Code-specific command definition."""

    executable: str = "claude"  # adjust if your CLI entrypoint differs


class ClaudeCode(CodeAgent[ClaudeCodeCommand]):
    """Claude Code Agent."""

    DOCKERTAG = DOCKERTAG_CLAUDE

    def ui_define_command(self) -> ClaudeCodeCommand:
        """Define the Claude Code command with Streamlit UI."""
        st.markdown("# Claude Code Command")
        with st.expander("", expanded=True):
            extra_args_str = st.text_input(
                "Extra CLI arguments (optional)",
                value="",
                key="claude_code_extra_args",
                help="Space-separated extra arguments passed directly to the Claude Code CLI",
            )
            extra_args = extra_args_str.split() if extra_args_str.strip() else []

            cmd = ClaudeCodeCommand(
                workspace=self.path_agent_workspace,
                args=extra_args,
            )

            with st.expander("Display Command", expanded=True):
                args = cmd.construct_args()
                formatted_args = "\n\t".join(args)
                st.code(f"{cmd.executable} {formatted_args}", language="bash")

        return cmd


class CursorCommand(AgentCommand):
    """Cursor CLI-specific command definition."""

    executable: str = "cursor-agent"


class Cursor(CodeAgent[CursorCommand]):
    """Cursor CLI Code Agent."""

    DOCKERTAG = DOCKERTAG_CURSOR

    def ui_define_command(self) -> CursorCommand:
        """Define the Cursor CLI command with Streamlit UI."""
        st.markdown("# Cursor CLI Command")
        with st.expander("", expanded=True):
            extra_args_str = st.text_input(
                "Extra CLI arguments (optional)",
                value="",
                key="cursor_extra_args",
                help="Space-separated extra arguments passed directly to the Cursor CLI",
            )
            extra_args = extra_args_str.split() if extra_args_str.strip() else []

            cmd = CursorCommand(
                workspace=self.path_agent_workspace,
                args=extra_args,
            )

            with st.expander("Display Command", expanded=True):
                args = cmd.construct_args()
                formatted_args = "\n\t".join(args)
                st.code(f"{cmd.executable} {formatted_args}", language="bash")

        return cmd


# Agent registry: dynamic discovery for extensible multi-agent support
agent_subclasses = CodeAgent.__subclasses__()
agent_subclass_dict = {cls.__name__: cls for cls in agent_subclasses}
agent_subclass_names = list(agent_subclass_dict.keys())


@st.cache_resource
def get_agent(agent_type: str, repo_url: str, branch: str) -> CodeAgent[Any]:
    """Instantiate and cache agent instance."""
    agent_cls: type[CodeAgent[Any]] = agent_subclass_dict[agent_type]
    return agent_cls(repo_url, branch)


def get_remote_branches(repo_url: str) -> list[str]:
    """Extract branch names from remote repository."""
    return [ref.split("\t")[1].replace("refs/heads/", "") for ref in git.Git().ls_remote("--heads", repo_url).splitlines()]


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


if __name__ == "__main__":
    st.set_page_config(page_title="Agents-in-a-Box", layout="wide")
    agent_controls()
    chat_interface()
