import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any, ClassVar, List, Literal, Optional

import git
from pydantic import Field
import streamlit as st

from code_agents.agent_prompts import SYS_EMPTY, SYS_REFACTOR
from code_agents.agents_core import AgentCommand, CodeAgent
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

logger = logging.getLogger(__name__)

DEFAULT_ARGS_AIDER = ["--dark-mode", "--code-theme", "inkpot", "--pretty"]

AVAILABLE_PROMPTS = {
    "<empty prompt>": SYS_EMPTY,
    "refactor": SYS_REFACTOR,
}


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


class AiderCommand(AgentCommand):
    """Aider-specific command definition."""

    # Baseclass constants
    executable: str = "aider"
    task_injection_template: ClassVar[List[str]] = ["--message", "{task}"]

    # Variables
    model: str = Field(..., description="Architect LLM identifier")
    editor_model: str = Field(..., description="Editor LLM identifier")
    reasoning_effort: Literal["low", "medium", "high"] = Field(default="high", description="Reasoning effort")
    edit_format: Literal["diff", "whole", "udiff"] = Field(default="diff", description="Edit format")
    map_tokens: Literal[1024, 2048, 4096, 8192] = Field(default=4096, description="Context map tokens")


class Aider(CodeAgent[AiderCommand]):
    """Aider Code Agent - example for overridability."""

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

    executable: str = "claude"


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
    try:
        lines = git.Git().ls_remote("--heads", repo_url).splitlines()
        return [ref.split("\t", 1)[1].replace("refs/heads/", "") for ref in lines if "\t" in ref]
    except git.GitCommandError as e:
        logger.error("Failed to fetch remote branches for %s: %s", repo_url, e)
        return []
    except Exception:
        logger.exception("Unexpected error while fetching remote branches for %s", repo_url)
        return []


def get_changed_files(workspace: Path) -> list[Path]:
    """Return list of paths (absolute) for files changed vs HEAD."""
    result = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--name-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [workspace / f for f in files]


def sync_to_home(workspace: Path, repo_url: str) -> tuple[int, Path]:
    """
    Copy all changed files from the agent workspace to a mirror repo under the user's home.

    Source:  ~/agent_sandbox/<repo_name>/
    Target:  ~/<repo_name>/

    Returns (count_copied, target_root).
    """
    home = Path.home()
    repo_name = repo_url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    target_root = home / repo_name
    target_root.mkdir(parents=True, exist_ok=True)

    changed_files = get_changed_files(workspace)
    copied = 0

    for src in changed_files:
        if not src.is_file():
            continue
        rel = src.relative_to(workspace)
        dst = target_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    return copied, target_root


def commit_to_home(workspace: Path, repo_url: str, message: str) -> tuple[bool, str]:
    """
    Sync changed files to the user's home repo and commit them.

    Returns (success, message).
    """
    copied, target_root = sync_to_home(workspace=workspace, repo_url=repo_url)
    if copied == 0:
        return False, "No changed files to sync; nothing to commit."

    try:
        # Stage all changes
        add_result = subprocess.run(
            ["git", "-C", str(target_root), "add", "."],
            capture_output=True,
            text=True,
            check=False,
        )
        if add_result.returncode != 0:
            return False, f"git add failed:\n{add_result.stderr}"

        # Commit
        commit_result = subprocess.run(
            ["git", "-C", str(target_root), "commit", "-m", message],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_result.returncode != 0:
            # If nothing to commit, treat as non-fatal
            if "nothing to commit" in commit_result.stderr.lower():
                return False, "Nothing to commit in home repository."
            return False, f"git commit failed:\n{commit_result.stderr}"

        return True, f"Committed changes in `{target_root}`."
    except Exception as exc:
        return False, f"Commit failed: {exc}"


def agent_controls() -> None:
    """Streamlit entrypoint for multi-agent code workspace."""

    if "selected_agent" not in st.session_state:
        st.markdown("## Agent Controls")
        selected_agent_name: str = st.selectbox("Select Code Agent", options=agent_subclass_names, key="code_agent_selector")
        repo_url = st.text_input("GitHub Repository URL", key="repo_url")
        branch: Optional[str] = None

        if repo_url:
            if "branches" not in st.session_state or st.session_state.cached_repo_url != repo_url:
                with st.spinner("Fetching branches..."):
                    st.session_state.branches = get_remote_branches(repo_url)
                st.session_state.cached_repo_url = repo_url

            branches = st.session_state.branches

            # If fetching branches failed, show error and stop
            if not branches:
                st.error("Unable to fetch branches for this repository. Please check that the URL is correct and accessible.")
                return

            # Simple branch selection: user must choose an existing branch
            branch = st.selectbox(
                "Select Branch",
                options=branches,
                index=0,
                key="branch_selector",
            )

        def init_agent() -> None:
            """Initialize and store agent in session state."""
            selected_agent: CodeAgent[Any] = get_agent(agent_type=selected_agent_name, repo_url=repo_url, branch=branch)  # type: ignore[arg-type]
            st.session_state.selected_agent = selected_agent

        if repo_url and branch:
            st.button("Initialize Agent", key="init_agent_button", on_click=init_agent)

    else:

        selected_agent: CodeAgent[Any] = st.session_state.selected_agent
        repo_url = selected_agent.repo_url
        branch = selected_agent.branch
        st.markdown("# Agent Info")
        with st.expander("", expanded=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write("**Agent**")
                st.write("**Source**")
                st.write("**Workspace**")
            with col2:
                st.markdown(f"{selected_agent.__class__.__name__}")
                display_repo = repo_url.replace("git@github.com:", "github.com/").removesuffix(".git")
                st.markdown(
                    f"[{display_repo}](https://{display_repo})"
                    f" / [{branch}](https://{display_repo}/tree/{branch})\n"
                )
                st.markdown(f"`{selected_agent.path_agent_workspace}`")

        st.markdown("---")

        st.markdown("# General Control")
        with st.expander("", expanded=True):
            st.session_state.system_prompt = st.selectbox(
                "System prompt",
                list(AVAILABLE_PROMPTS.keys()),
                key="prompt_select",
            )
            col_1, col_2 = st.columns(2)

            with col_1:
                sys_prompt = AVAILABLE_PROMPTS[st.session_state.system_prompt]
                st.button(
                    "Execute Agent",
                    use_container_width=True,
                    type="secondary",
                    on_click=lambda: selected_agent.run(command=st.session_state.command, task=sys_prompt),
                )

            with col_2:
                if st.button("Reset Agent", use_container_width=True):
                    del st.session_state.selected_agent
                    st.rerun()

            if st.button("Sync Agent Workspace", use_container_width=True):
                count, target_root = sync_to_home(
                    workspace=selected_agent.path_agent_workspace,
                    repo_url=repo_url,
                )
                if count > 0:
                    st.success(f"Copied {count} file(s) to `{target_root}`.")
                else:
                    st.info("No changed files to sync.")
        st.markdown("---")

        # Define command last to render UI controls last
        command: AgentCommand = selected_agent.ui_define_command()
        st.session_state.command = command


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

            selected_agent.run(task=task+st.session_state.system_prompt, command=st.session_state.command)
            diff: str = selected_agent.get_diff()
            if diff:
                st.markdown("### Git Diff")
                st.code(diff, language="diff")

                st.markdown("### Commit Changes")
                commit_msg = st.text_input(
                    "Commit message",
                    value="",
                    key="commit_message_input",
                    placeholder="Describe your changes...",
                )
                if st.button("Commit Changes", key="commit_changes_button"):
                    if not commit_msg.strip():
                        st.warning("Please enter a commit message.")
                    else:
                        with st.spinner("Syncing and committing changes to home repository..."):
                            success, msg = commit_to_home(
                                workspace=selected_agent.path_agent_workspace,
                                repo_url=selected_agent.repo_url,
                                message=commit_msg.strip(),
                            )
                        if success:
                            st.success(msg)
                        else:
                            st.info(msg)
            else:
                st.info("No changes detected in git diff.")


def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(page_title="Agents-in-a-Box", layout="wide")
    with st.sidebar:
        agent_controls()
    chat_interface()


if __name__ == "__main__":
    main()
