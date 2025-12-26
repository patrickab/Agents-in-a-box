"""
Architecture & Behavior

Agent & Command Abstraction
    Pydantic-based command models encapsulate tool-specific parameters.
    UI is dynamically derived from command pydantic fields.
    Decouples agent configuration from execution engine.

Code Agent Runtime Binding
    Generic CodeAgent defined through
        (a) concrete *Command* subclasses
        (b) DOCKERTAG that identifies runtime environment.
    Ensures an easily extensible agent ecosystem.
"""

import logging
from typing import ClassVar, List, Literal

from pydantic import Field
import streamlit as st

from code_agents.agents_core import AgentCommand, CodeAgent
from code_agents.lib.config import (
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

# [Architect, Editor]
DEFAULT_ARCHITECT_AIDER = "openai/gpt-5.1"
DEFAULT_EDITOR_AIDER = "ollama/devstral-2:123b-cloud"
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
    model: str = Field(default=DEFAULT_ARCHITECT_AIDER, description="Architect LLM identifier")
    editor_model: str = Field(default=DEFAULT_EDITOR_AIDER, description="Editor LLM identifier")
    reasoning_effort: Literal["low", "medium", "high"] = Field(default="high", description="Reasoning effort")
    edit_format: Literal["diff", "whole", "udiff"] = Field(default="diff", description="Edit format")
    map_tokens: Literal[1024, 2048, 4096, 8192] = Field(default=1024, description="Context map tokens")


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
            if "ollama/" in model_architect or "ollama/" in model_editor:
                flags.append("--no-stream")  # Ollama's streaming is buggy

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
    edit_format: Literal["diff", "whole", "udiff"] = Field(default="diff", description="Edit format")
    map_tokens: Literal[1024, 2048, 4096, 8192] = Field(default=1024, description="Context map tokens")
    stream: bool = Field(default=False, description="Enable streaming output")


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
