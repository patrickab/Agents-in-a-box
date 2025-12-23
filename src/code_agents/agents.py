from typing import Any, ClassVar, List, Literal, Optional

import logging
import git
from pydantic import Field
import streamlit as st

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
    try:
        lines = git.Git().ls_remote("--heads", repo_url).splitlines()
        return [ref.split("\t", 1)[1].replace("refs/heads/", "") for ref in lines if "\t" in ref]
    except git.GitCommandError as e:
        logger.error("Failed to fetch remote branches for %s: %s", repo_url, e)
        return []
    except Exception as e:
        logger.exception("Unexpected error while fetching remote branches for %s", repo_url)
        return []


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

            if not st.session_state.branches:
                st.error(
                    "Unable to fetch branches for this repository. "
                    "Please check that the URL is correct and accessible."
                )
                return

            branch = st.selectbox(
                "Select Branch",
                options=st.session_state.branches,
                index=0,
                key="branch_selector",
            )

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
