from abc import ABC
import contextlib
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
    args: list[str] = Field(default_factory=list, description="CLI arguments excluding executable")
    task_injection_template: list[str] = Field(default_factory=list, description="Task injection template")
    env_vars: dict[str, str] = Field(default=ENV_VARS, description="Environment variable overrides")

    def _snake_to_kebab(self, s: str) -> str:
        """Convert snake_case to kebab-case."""
        return s.replace("_", "-")

    @classmethod
    def construct_args_from_values(cls, **field_values: dict[str, any]) -> list[str]:
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

    def construct_args(self) -> list[str]:
        """Construct argument list."""
        fields = self.model_dump(exclude={"executable", "workspace", "args", "env_vars", "task_injection_template"})
        fields = {k: v for k, v in fields.items() if v is not False}
        args = [
            item for k, v in fields.items() for item in ([f"--{self._snake_to_kebab(k)}"] + ([] if isinstance(v, bool) else [str(v)]))
        ]
        return args + self.args

    @classmethod
    def ui_define_fields(cls) -> dict[str, any]:
        """Generate UI elements for all pydantic fields and return their values."""
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
        origin = get_origin(t)
        args = get_args(t)

        if t is bool:
            return st.toggle(desc, value=default or False, key=key)

        if origin is Literal:
            options = list(args)
            idx = options.index(default) if default in options else 0
            return st.selectbox(desc, options=options, index=idx, key=key)

        if origin in (list, list):
            inner = args[0] if args else type(None)
            if get_origin(inner) is Literal:
                options = list(get_args(inner))
                return st.multiselect(desc, options=options, default=default or [], key=key)

            text_val = ", ".join(default) if isinstance(default, list) else ""
            res = st.text_input(desc, value=text_val, key=key)
            return [x.strip() for x in res.split(",") if x.strip()]

        if t in (int, float):
            is_int = t is int
            val = default if default is not None else (0 if is_int else 0.0)
            step = 1 if is_int else 0.1
            return st.number_input(desc, value=val, step=step, key=key)

        return st.text_input(desc, value=default or "", key=key)

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

    def __init__(self, repo_url: Optional[str] = None, branch: Optional[str] = None) -> None:
        if repo_url and branch:
            # Streamlit UI initialization
            self.repo_url = repo_url
            self.branch = branch
            self.path_agent_workspace = self._initialize_workspace()
            logger.info(
                "[bold cyan]Workspace ready[/bold cyan] · repo=[bold]%s[/bold] branch=[bold]%s[/bold] path=[bold]%s[/bold]",
                self.repo_url,
                self.branch,
                self.path_agent_workspace,
            )
        else:
            # CLI initialization
            self.path_agent_workspace = Path.cwd()
            logger.info(
                "[bold cyan]Workspace ready (CLI)[/bold cyan] · path=[bold]%s[/bold]",
                self.path_agent_workspace,
            )

    def _initialize_workspace(self) -> Path:
        """Setup agent workspace: clone, checkout, install deps."""
        repo_name = self.repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        workspace = Path(PATH_SANDBOX) / repo_name
        workspace.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[bold magenta]Preparing workspace[/bold magenta] · repo=[bold]%s[/bold] dir=[bold]%s[/bold]",
            self.repo_url,
            workspace,
        )

        self._git_checkout(workspace)
        self._install_dependencies(workspace)

        logger.info(
            "[green]Workspace prepared[/green] · dir=[bold]%s[/bold]",
            workspace,
        )
        return workspace

    def _git_checkout(self, workspace: Path) -> None:
        """Setup agent workspace: clone, checkout branch, install deps."""
        try:
            if (workspace / ".git").exists():
                logger.info(
                    "[yellow]Updating existing repo[/yellow] · path=[bold]%s[/bold]",
                    workspace,
                )
                repo = Repo(workspace)
                repo.git.reset("--hard")
                repo.remotes.origin.pull()
            else:
                logger.info(
                    "[yellow]Cloning repo[/yellow] · url=[bold]%s[/bold] into [bold]%s[/bold]",
                    self.repo_url,
                    workspace,
                )
                repo = Repo.clone_from(self.repo_url, workspace)

            # Checkout logic
            if self.branch in repo.heads:
                logger.info(
                    "Checking out existing branch [bold]%s[/bold]",
                    self.branch,
                )
                repo.heads[self.branch].checkout()
            else:
                try:
                    head_ref = repo.remotes.origin.refs["HEAD"].reference.name
                    default_branch = head_ref.split("/")[-1]
                except (IndexError, AttributeError, ValueError):
                    default_branch = "main"

                logger.info(
                    "Creating branch [bold]%s[/bold] from [bold]%s[/bold]",
                    self.branch,
                    default_branch,
                )
                repo.git.checkout("-b", self.branch, f"origin/{default_branch}")

        except (GitCommandError, Exception) as e:
            logger.error(
                "[bold red]Git operation failed[/bold red] · %s",
                str(e),
            )
            st.error(f"Git operation failed: {e}")
            raise

    def _install_dependencies(self, workspace: Path) -> None:
        """Best-effort dependency installation using uv."""
        cmd = None
        if (workspace / "requirements.txt").exists():
            cmd = ["uv", "pip", "install", "-r", "requirements.txt"]
            logger.info(
                "[blue]Installing dependencies[/blue] · file=[bold]requirements.txt[/bold]"
            )
        elif (workspace / "pyproject.toml").exists():
            cmd = ["uv", "pip", "install", "-e", "."]
            logger.info(
                "[blue]Installing dependencies[/blue] · file=[bold]pyproject.toml[/bold]"
            )

        if cmd:
            try:
                subprocess.run(cmd, cwd=workspace, check=False, capture_output=True)
            except FileNotFoundError:
                logger.warning(
                    "[bold yellow]Dependency install skipped[/bold yellow] · `uv` not found on PATH"
                )

    def get_command_class(self) -> Type[TCommand]:
        """Resolve the command class from generics."""
        for cls in self.__class__.__mro__:
            for base in getattr(cls, "__orig_bases__", []):
                if get_origin(base) is CodeAgent:
                    args = get_args(base)
                    if args and issubclass(args[0], AgentCommand):
                        return args[0]
        raise NotImplementedError(f"Could not determine command class for {self.__class__.__name__}")

    def run(
        self,
        command: Optional[TCommand] = None,
        raw_cmd: Optional[str] = None,
        task: Optional[str] = None,
    ) -> None:
        """
        Unified execution entrypoint.

        - If `command` is provided: use AgentCommand (with task injection, env from command).
        - If `raw_cmd` is provided: run that shell string directly.
        Exactly one of `command` or `raw_cmd` must be provided.
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
            "[bold magenta]Launching agent[/bold magenta] · image=[bold]%s:latest[/bold]\n[dim]$ %s[/dim]",
            self.DOCKERTAG,
            cmd_str,
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
                "[bold red]Agent run failed[/bold red] · %s",
                str(exc),
            )
            st.error(f"Failed to run agent in sandbox: {exc}")
            raise

    def ui_define_command(self) -> TCommand:
        """Render UI for command configuration."""
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
        """Return git diff for workspace."""
        return subprocess.run(
            ["git", "-C", str(self.path_agent_workspace), "diff"], capture_output=True, text=True, check=False
        ).stdout

    def run_workspace(self) -> None:
        try:
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)

            logger.info(
                "[bold magenta]Running workspace script[/bold magenta] · path=[bold]%s[/bold]\n[dim]$ ./run.sh[/dim]",
                self.path_agent_workspace,
            )

            subprocess.run(
                ["./run.sh"],
                cwd=self.path_agent_workspace,
                env=env,
            )
        except Exception as exc:
            logger.error(
                "[bold red]Workspace script failed[/bold red] · %s",
                str(exc),
            )
            st.error(f"Failed to run workspace script: {exc}")
            raise
