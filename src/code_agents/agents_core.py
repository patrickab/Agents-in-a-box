"""
Architecture & Behavior

Agent Command Baseclass
    Centralized agent invocation contract
    Dynamic UI form derivation for streamlit configuration

Code Agent Baseclass
    Prepares dedicated workspaces for agents in ~/agent_workspace/<repo>.
    Launches agents in isolated Docker sandboxes mounted to workspace.
    Provides bridge between host and agent processes preserving CLI interaction.

Git-Backed Workspace Management
    Clones or updates git repositories into sandboxed workspaces.
"""

from abc import ABC
import os
from pathlib import Path
import subprocess
from typing import Generic, Literal, Optional, Type, TypeVar, get_args, get_origin

from git import GitCommandError, Repo
from pydantic import BaseModel, Field
import streamlit as st

from code_agents.lib.config import PATH_SANDBOX
from code_agents.lib.logger import get_logger
from code_agents.sandbox import DockerSandbox

OLLAMA_PORT = 11434
GIT_NAME = subprocess.run(["git", "config", "--global", "user.name"], capture_output=True, text=True).stdout.strip()
GIT_EMAIL = subprocess.run(["git", "config", "--global", "user.email"], capture_output=True, text=True).stdout.strip()
ENV_VARS = {
    "OLLAMA_API_BASE": f"http://host.docker.internal:{OLLAMA_PORT}",
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "OLLAMA_API_KEY": os.getenv("OLLAMA_API_KEY", ""),
    "GIT_AUTHOR_NAME": GIT_NAME,
    "GIT_COMMITTER_NAME": GIT_NAME,
    "GIT_AUTHOR_EMAIL": GIT_EMAIL,
    "GIT_COMMITTER_EMAIL": GIT_EMAIL,
}

logger = get_logger()


class AgentCommand(BaseModel, ABC):
    """Base command model for invoking external agent CLIs.

    Args:
        executable: Preinstalled CLI name on PATH.
        workspace: Absolute or relative working directory; must exist before run.
        args: Ordered CLI arguments excluding the executable.
        task_injection_template: Token sequence used to inject task text into args.
        env_vars: Environment overrides merged over the current process environment.

    Returns:
        Immutable configuration object used to construct a shell command line.
    """

    executable: str = Field(..., description="CLI executable name")
    workspace: Path = Field(default_factory=Path.cwd, description="Agent workspace directory")
    args: list[str] = Field(default_factory=list, description="CLI arguments excluding executable")
    task_injection_template: list[str] = Field(default_factory=list, description="Task injection template")
    env_vars: dict[str, str] = Field(default=ENV_VARS, description="Environment variable overrides")

    def _snake_to_kebab(self, s: str) -> str:
        """Convert snake_case string to kebab-case."""
        return s.replace("_", "-")

    def construct_args(self) -> list[str]:
        """Build CLI argument list from this command instance.

        Returns:
            Flat list of CLI arguments derived from model fields and extra args.
        """
        fields = self.model_dump(exclude={"executable", "workspace", "args", "env_vars", "task_injection_template"})
        fields = {k: v for k, v in fields.items() if v is not False}
        args = [
            item for k, v in fields.items() for item in ([f"--{self._snake_to_kebab(k)}"] + ([] if isinstance(v, bool) else [str(v)]))
        ]
        return args + self.args

    @classmethod
    def ui_define_fields(cls) -> dict[str, any]:
        """Render Streamlit inputs for all configurable fields and collect values.

        Returns:
            Mapping of field names to values captured from the UI.
        """
        values: dict[str, any] = {}
        base_excluded = {"executable", "workspace", "args", "env_vars", "task_injection_template"}

        for name, field in cls.model_fields.items():
            if name in base_excluded:
                continue

            key = f"{cls.__name__.lower()}_{name}"
            desc = field.description or name
            default = field.default if field.default != ... else None

            values[name] = cls._render_ui_field(desc, default, field.annotation, key)

        return values

    @staticmethod
    def _render_ui_field(desc: str, default: any, t: any, key: str) -> any:  # noqa
        """Render a Streamlit widget for a single field and return its value.

        Args:
            desc: Human-readable label for the widget.
            default: Default value used when no user input exists.
            t: Type annotation or typing construct for the field.
            key: Unique Streamlit key for widget state.

        Returns:
            Value captured from the rendered widget.
        """
        # Handle boolean fields
        if t is bool:
            return st.toggle(desc, value=default or False, key=key)

        # Handle Literal types (enums)
        origin = get_origin(t)
        if origin is Literal:
            options = list(get_args(t))
            idx = options.index(default) if default in options else 0
            return st.selectbox(desc, options=options, index=idx, key=key)

        # Handle list of Literals (multiselect)
        if origin is list:
            args = get_args(t)
            if args and get_origin(args[0]) is Literal:
                options = list(get_args(args[0]))
                return st.multiselect(desc, options=options, default=default or [], key=key)

        # Handle numeric types
        if t in (int, float):
            is_int = t is int
            val = default if default is not None else (0 if is_int else 0.0)
            step = 1 if is_int else 0.1
            return st.number_input(desc, value=val, step=step, key=key)

        # Default to text input for strings and other types
        return st.text_input(desc, value=default or "", key=key)


TCommand = TypeVar("TCommand", bound=AgentCommand)


class CodeAgent(ABC, Generic[TCommand]):
    """Base class for code agents executed inside a Docker sandbox.

    Supports two initialization modes:
    - CLI mode: no arguments; uses the current working directory as the workspace.
    - UI mode: ``branch`` and ``repo_url`` provided; prepares a sandboxed workspace
      from a Git repository.

    Class Attributes:
        DOCKERTAG: Docker image tag used for sandbox execution.

    Args:
        branch (str, optional): Git branch to check out (UI mode only).
        repo_url (str, optional): Git repository URL (UI mode only).

    Returns:
        Initialized agent instance ready for command execution.

    Note:
        In UI mode, the workspace is created or reused under ``PATH_SANDBOX`` and
        may install Python dependencies using ``uv``.
    """

    DOCKERTAG: str

    def __init__(self, repo_url: Optional[str] = None, branch: Optional[str] = None) -> None:
        """Initialize agent workspace from repo URL/branch or current directory.

        Args:
            repo_url: Remote repository URL; when omitted, use current working directory.
            branch: Target branch name when cloning/updating the repository.
        """
        if repo_url and branch:
            # Streamlit UI initialization
            self.repo_url = repo_url
            self.branch = branch
            self.path_agent_workspace = self._initialize_workspace()
            logger.info(
                "[bold green]Repository cloned & initialized",
            )
        else:
            # CLI initialization
            self.path_agent_workspace = Path.cwd()
            logger.info(
                "[bold green]Workspace ready (CLI)[/bold green] · path=[bold magenta]%s[/bold magenta]",
                self.path_agent_workspace,
            )

    def _initialize_workspace(self) -> Path:
        """Prepare workspace directory by cloning repo, checking out branch, and installing deps.

        Returns:
            Absolute path to the prepared workspace directory.
        """
        repo_name = self.repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        workspace = Path(PATH_SANDBOX) / repo_name
        workspace.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[bold magenta]Preparing workspace[/bold magenta] · repo=[bold magenta]{self.repo_url}[/bold magenta]")

        self._git_checkout(workspace)
        self._install_dependencies(workspace)

        logger.info(f"[bold magenta]Workspace prepared[/bold magenta] · dir=[bold magenta]{workspace}[/bold magenta]")
        return workspace

    def _git_checkout(self, workspace: Path) -> None:
        """Clone or update the git repository and check out the target branch.

        Args:
            workspace: Filesystem path where the repository should reside.

        Raises:
            GitCommandError: If git operations fail.
            Exception: For unexpected git-related failures.
        """
        try:
            if (workspace / ".git").exists():
                logger.info("...updating existing repo")
                repo = Repo(workspace)
                repo.git.reset("--hard")
                repo.remotes.origin.pull()
            else:
                logger.info("...cloning repo")
                repo = Repo.clone_from(self.repo_url, workspace)

            # Checkout logic
            if self.branch in repo.heads:
                logger.info(
                    "...checking out existing branch",
                )
                repo.heads[self.branch].checkout()
            else:
                try:
                    head_ref = repo.remotes.origin.refs["HEAD"].reference.name
                    default_branch = head_ref.split("/")[-1]
                except (IndexError, AttributeError, ValueError):
                    default_branch = "main"

                logger.info(
                    f"Creating branch [bold]{self.branch}[/bold] from [bold]{default_branch}[/bold]",
                )
                repo.git.checkout("-b", self.branch, f"origin/{default_branch}")

        except (GitCommandError, Exception) as e:
            logger.error(
                "[bold red]Git operation failed[/bold red] · %s",
                str(e),
            )
            raise

    def _install_dependencies(self, workspace: Path) -> None:
        """
        Assumes existing installation of uv.
        Cloned repositories are assumed to be trustworthy.
        """
        cmd = None
        if (workspace / "requirements.txt").exists():
            cmd = ["uv", "pip", "install", "-r", "requirements.txt"]
            logger.info("...installing dependencies")
        if (workspace / "pyproject.toml").exists():
            cmd = ["uv", "pip", "install", "-e", "."]
            logger.info("...installing dependencies")

        if cmd:
            try:
                subprocess.run(cmd, cwd=workspace, check=False, capture_output=True)
            except FileNotFoundError:
                logger.warning("[bold yellow]Dependency install skipped[/bold yellow]. Consider installing 'uv'.")

    def get_command_class(self) -> Type[TCommand]:
        """Resolve the concrete AgentCommand subclass from the generic parameter.

        Returns:
            Command class bound to this CodeAgent subclass.
        """
        for cls in self.__class__.__mro__:
            for base in getattr(cls, "__orig_bases__", []):
                if get_origin(base) is CodeAgent:
                    args = get_args(base)
                    if args and issubclass(args[0], AgentCommand):
                        return args[0]

    def run(
        self,
        command: Optional[TCommand] = None,
        raw_cmd: Optional[str] = None,
        task: Optional[str] = None,
    ) -> None:
        """Execute an agent command inside a Docker sandbox.

        Args:
            command: Structured AgentCommand to execute; mutually exclusive with raw_cmd.
            raw_cmd: Raw shell command string to run as-is; mutually exclusive with command.
            task: Optional task text injected via task_injection_template when using command.

        Raises:
            ValueError: If both or neither of command and raw_cmd are provided.
            Exception: If sandbox startup or agent execution fails.
        """
        if (command is None) == (raw_cmd is None):
            raise ValueError("Exactly one of `command` or `raw_cmd` must be provided")

        if command is not None:
            if task and command.task_injection_template:
                injected_task = [token.format(task=task) for token in command.task_injection_template]
                command.args.extend(injected_task)

            cmd_str = subprocess.list2cmdline([command.executable, *command.construct_args()])
        else:  # already a shell string
            cmd_str = raw_cmd

        logger.info(
            f"[bold magenta]Launching agent[/bold magenta]\n[dim]$ {cmd_str}[/dim]",
        )

        sandbox = DockerSandbox(dockerimage_name=f"{self.DOCKERTAG}:latest")
        try:
            sandbox.run_interactive_shell(
                repo_path=str(self.path_agent_workspace),
                agent_cmd=cmd_str,
                env_vars=ENV_VARS,
            )
        except Exception as exc:
            logger.error(
                f"[bold red]Agent run failed[/bold red] · {exc}",
            )
            raise

    def ui_define_command(self) -> TCommand:
        """Render Streamlit UI to configure a command instance for this agent.

        Returns:
            Command instance populated from UI inputs and workspace defaults.
        """
        command_class = self.get_command_class()
        st.markdown(f"# {self.__class__.__name__} Command")

        with st.expander("Configuration", expanded=True):
            field_values = command_class.ui_define_fields()

            extra_args_str = st.text_input(
                "Extra CLI arguments",
                value="",
                key=f"{self.__class__.__name__.lower()}_extra_args",
                help="Space-separated extra arguments passed directly to the CLI",
            )
            extra_args = extra_args_str.split() if extra_args_str.strip() else []

            # Inject system fields
            field_values["workspace"] = self.path_agent_workspace
            field_values["args"] = extra_args + field_values.get("args", [])

            command = command_class(**field_values)

            with st.expander("Preview Command", expanded=True):
                st.code("\n\t".join([command.executable, *command.construct_args()]), language="bash")

        return command

    def get_diff(self) -> str:
        """Return current git diff for the workspace against HEAD."""
        return subprocess.run(
            ["git", "-C", str(self.path_agent_workspace), "diff"], capture_output=True, text=True, check=False
        ).stdout

    def run_workspace(self) -> None:
        """Execute ./run.sh in the workspace with a clean environment."""
        try:
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)

            subprocess.run(
                ["./run.sh"],
                cwd=self.path_agent_workspace,
                env=env,
            )
        except Exception as exc:
            logger.error(
                f"[bold red]Workspace script failed[/bold red] · {exc}",
            )
            raise
