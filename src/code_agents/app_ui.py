import contextlib
from pathlib import Path
import shutil
import subprocess
from typing import Any, Optional

import git
import streamlit as st

# ruff: noqa: F403
from code_agents.agents import *
from code_agents.agents_core import AgentCommand, CodeAgent
from code_agents.lib.agent_prompts import SYS_EMPTY, SYS_REFACTOR
from code_agents.lib.logger import get_logger

logger = get_logger()

AVAILABLE_PROMPTS = {
    "<empty prompt>": SYS_EMPTY,
    "refactor": SYS_REFACTOR,
}


# --------- Dynamic discovery of subclasses --------- #
agent_subclasses = CodeAgent.__subclasses__()
agent_subclass_dict = {cls.__name__: cls for cls in agent_subclasses}
agent_subclass_names = list(agent_subclass_dict.keys())


# ---------------- Git Utility Functions ---------------- #

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
    """Sync changed files to the user's home repo and commit them."""
    copied, target_root = sync_to_home(workspace=workspace, repo_url=repo_url)
    if copied == 0:
        return False, "No changed files to sync; nothing to commit."

    try:
        repo = git.Repo(target_root)
        repo.git.add(".")

        # Remove .aider* files from staging if any were added
        with contextlib.suppress(git.exc.GitCommandError):
            repo.git.execute(["git", "rm", "--cached", ".aider*", "--ignore-unmatch"])

        try:
            commit_result = repo.index.commit(message)
            return True, f"Committed changes in `{target_root}`. Commit: {commit_result.hexsha}"
        except git.exc.GitError as e:
            if "nothing to commit" in str(e).lower():
                return False, "Nothing to commit in home repository."
            return False, f"git commit failed: {e}"
    except Exception as e:
        return False, f"Commit failed: {e}"


# ---------------- Streamlit Application ---------------- #

@st.cache_resource
def get_agent(agent_type: str, repo_url: str, branch: str) -> CodeAgent[Any]:
    """Instantiate and cache agent instance."""
    return agent_subclass_dict[agent_type](repo_url, branch)


def agent_controls() -> None:
    """Streamlit sidebar for agent controls."""

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
                display_repo = display_repo.replace("https://", "")
                st.markdown(f"[{display_repo}](https://{display_repo}) / [{branch}](https://{display_repo}/tree/{branch})\n")
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
                task = None if sys_prompt == SYS_EMPTY else sys_prompt
                st.button(
                    "Execute Agent",
                    use_container_width=True,
                    type="secondary",
                    on_click=lambda: selected_agent.run(command=st.session_state.command, task=task),
                )
                st.button(
                    "Test Code",
                    use_container_width=True,
                    type="secondary",
                    on_click=lambda: selected_agent.run_workspace(),
                )

            with col_2:
                if st.button("Reset Agent", use_container_width=True):
                    del st.session_state.selected_agent
                    st.rerun()

                if st.button("Sync Code", use_container_width=True):
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

            sys_prompt = AVAILABLE_PROMPTS[st.session_state.system_prompt]

            if sys_prompt != SYS_EMPTY:
                task = "<SYSTEM PROMPT>\n" + sys_prompt + "\n</SYSTEM PROMPT>" + "\n\n" + "<USER PROMPT>\n" + task + "\n</USER PROMPT>"

            selected_agent.run(task=task, command=st.session_state.command)
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
