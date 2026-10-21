"""Run the git operations of the task pipeline: worktrees, change tracking, path enforcement, and merging."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess

from agent_sandbox.factory.config import OUTPUT_DIR, SESSION_DIR

ENGINE_OWNED_PATHS = (OUTPUT_DIR, SESSION_DIR)
BRANCH_PREFIX = "factory/"
DIFF_LIMIT_CHARS = 60_000
SNAPSHOT_INDEX_NAME = "factory-snapshot-index"
FALLBACK_IDENTITY = {
    "GIT_AUTHOR_NAME": "Agent Factory",
    "GIT_AUTHOR_EMAIL": "factory@localhost",
    "GIT_COMMITTER_NAME": "Agent Factory",
    "GIT_COMMITTER_EMAIL": "factory@localhost",
}


class GitError(RuntimeError):
    """Report a failed git command with its output."""


@dataclass(frozen=True)
class WorktreeInfo:
    """Describe a task worktree."""

    path: Path
    branch: str
    base_sha: str


@dataclass(frozen=True)
class StepOutcome:
    """Describe whether a merge step succeeded, with an error message when it did not."""

    ok: bool
    error: str = ""


def branch_name(task_id: str) -> str:
    """Return the branch name that holds a task's work."""
    return f"{BRANCH_PREFIX}{task_id}"


def run_git(
    cwd: Path | str, *args: str, input_bytes: bytes | None = None, env: dict[str, str] | None = None, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    """Run `git <args>` in cwd with prompts disabled, raising GitError on failure when check is set."""
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", **(env or {})}
    result = subprocess.run(["git", *args], cwd=cwd, input=input_bytes, capture_output=True, env=environment, check=False)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()
        raise GitError(f"git {' '.join(args[:3])} failed ({result.returncode}): {detail}")
    return result


def git_text(cwd: Path | str, *args: str, env: dict[str, str] | None = None) -> str:
    """Return stripped stdout of a git command."""
    return run_git(cwd, *args, env=env).stdout.decode(errors="replace").strip()


def _identity_env(cwd: Path | str) -> dict[str, str]:
    configured = run_git(cwd, "config", "user.email", check=False).stdout.strip()
    return {} if configured else FALLBACK_IDENTITY


# --- path matching ---


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a gitignore-like glob (`*` within a segment, `**` across segments) to a full-path regex."""
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif char == "*":
            parts.append("[^/]*")
            index += 1
        elif char == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(char))
            index += 1
    return re.compile("".join(parts))


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    """Return whether a repo-relative path matches any gitignore-style glob (`*` stops at `/`, unlike archive excludes)."""
    return any(glob_to_regex(pattern).fullmatch(path) for pattern in patterns)


def find_breaches(paths: Iterable[str], writes: Sequence[str], protected: Sequence[str]) -> list[str]:
    """Return changed paths that are protected or outside the writes allowlist."""
    return [path for path in paths if matches_any(path, protected) or not matches_any(path, writes)]


# --- worktrees ---


def add_worktree(repo: Path, worktree_root: Path, task_id: str, main_branch: str) -> WorktreeInfo:
    """Create (or reuse after a crash) the task worktree on a fresh branch off main_branch, leaving other worktrees alone."""
    path = worktree_root / task_id
    branch = branch_name(task_id)
    if path.is_dir() and (path / ".git").exists():
        return WorktreeInfo(path, branch, git_text(repo, "rev-parse", branch))
    worktree_root.mkdir(parents=True, exist_ok=True)
    branch_exists = run_git(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False).returncode == 0
    # --force re-adopts this path when a crash left it registered but missing, without pruning anyone else's stale worktrees.
    if branch_exists:
        run_git(repo, "worktree", "add", "--force", str(path), branch)
    else:
        run_git(repo, "worktree", "add", "--force", "-b", branch, str(path), main_branch)
    return WorktreeInfo(path, branch, git_text(repo, "rev-parse", branch))


def remove_worktree(repo: Path, path: Path, branch: str) -> None:
    """Delete a task worktree and its branch; tolerate already-removed pieces and never prune other worktrees."""
    run_git(repo, "worktree", "remove", "--force", str(path), check=False)
    shutil.rmtree(path, ignore_errors=True)
    run_git(repo, "branch", "-D", branch, check=False)


# --- change tracking ---


def _snapshot_index(worktree: Path) -> Path:
    return Path(git_text(worktree, "rev-parse", "--absolute-git-dir")) / SNAPSHOT_INDEX_NAME


def snapshot_tree(worktree: Path) -> str:
    """Write a tree object of the worktree (tracked and untracked, not ignored) excluding engine-owned paths."""
    index = _snapshot_index(worktree)
    env = {"GIT_INDEX_FILE": str(index)}
    if not index.exists():
        run_git(worktree, "read-tree", "HEAD", env=env)
    run_git(worktree, "add", "-A", "--", ".", env=env)
    run_git(worktree, "rm", "-r", "-q", "--cached", "--ignore-unmatch", "--", *ENGINE_OWNED_PATHS, env=env)
    return git_text(worktree, "write-tree", env=env)


def tree_changes(worktree: Path, before: str, after: str) -> list[str]:
    """List file paths that differ between two tree-ish values."""
    raw = run_git(worktree, "diff-tree", "-r", "-z", "--no-renames", "--name-only", before, after).stdout
    return [path.decode() for path in raw.split(b"\0") if path]


def changed_files(worktree: Path) -> list[str]:
    """List files differing from HEAD, including untracked ones, excluding engine-owned paths."""
    return tree_changes(worktree, "HEAD^{tree}", snapshot_tree(worktree))


def diff_text(worktree: Path, limit: int = DIFF_LIMIT_CHARS) -> str:
    """Return the worktree diff against HEAD (including untracked files), truncated to `limit` characters."""
    text = run_git(worktree, "diff", "--no-color", "--no-ext-diff", "HEAD^{tree}", snapshot_tree(worktree)).stdout.decode(
        errors="replace"
    )
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... diff truncated ({len(text) - limit} more characters)\n"


def restore_paths(worktree: Path, tree: str, paths: Iterable[str]) -> None:
    """Reset each path to its content in `tree`, deleting paths that did not exist there."""
    for relative in paths:
        target = worktree / relative
        listing = run_git(worktree, "ls-tree", "-z", tree, "--", relative).stdout
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        if not listing:
            _prune_empty_parents(target, worktree)
            continue
        meta, _, _ = listing.rstrip(b"\0").partition(b"\t")
        mode, _, sha = meta.decode().split(" ")
        content = run_git(worktree, "cat-file", "blob", sha).stdout
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "120000":
            os.symlink(content.decode(), target)
            continue
        target.write_bytes(content)
        target.chmod(0o755 if mode == "100755" else 0o644)


def _prune_empty_parents(path: Path, stop: Path) -> None:
    for parent in path.parents:
        if parent == stop or not parent.is_dir() or any(parent.iterdir()):
            return
        parent.rmdir()


# --- merging ---


def commit_changes(worktree: Path, title: str, protected: Sequence[str]) -> bool:
    """Commit every changed file that is not protected or engine-owned; return whether a commit was made."""
    files = [path for path in changed_files(worktree) if not matches_any(path, protected)]
    if not files:
        return False
    paths = b"\0".join(path.encode() for path in files)
    run_git(worktree, "--literal-pathspecs", "add", "-A", "--pathspec-from-file=-", "--pathspec-file-nul", input_bytes=paths)
    subject = " ".join(title.split()) or "task"
    run_git(worktree, "commit", "-q", "-m", f"factory: {subject}", env=_identity_env(worktree))
    return True


def rebase_onto(worktree: Path, main_branch: str) -> StepOutcome:
    """Rebase the task branch onto main_branch, aborting cleanly on conflict."""
    dirty = git_text(worktree, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        return StepOutcome(False, f"worktree has uncommitted tracked changes that cannot be merged:\n{dirty}")
    result = run_git(worktree, "rebase", main_branch, env=_identity_env(worktree), check=False)
    if result.returncode == 0:
        return StepOutcome(True)
    run_git(worktree, "rebase", "--abort", check=False)
    return StepOutcome(False, f"rebase onto {main_branch} failed:\n{(result.stdout + result.stderr).decode(errors='replace').strip()}")


def uncommit(worktree: Path, base: str) -> None:
    """Move the branch back to `base` keeping all file content uncommitted in the worktree."""
    run_git(worktree, "reset", "-q", "--mixed", base)


def is_ahead_of(repo: Path, branch: str, main_branch: str) -> bool:
    """Return whether `branch` has commits that main_branch lacks."""
    return bool(git_text(repo, "rev-list", "--max-count=1", f"{main_branch}..{branch}"))


def can_fast_forward(repo: Path, branch: str, main_branch: str) -> bool:
    """Return whether main_branch is an ancestor of branch."""
    return run_git(repo, "merge-base", "--is-ancestor", main_branch, branch, check=False).returncode == 0


def checked_out_branch(repo: Path) -> str | None:
    """Return the branch checked out in a checkout, or None when HEAD is detached."""
    return run_git(repo, "symbolic-ref", "--short", "-q", "HEAD", check=False).stdout.decode().strip() or None


def main_checkout_problem(repo: Path, main_branch: str) -> str | None:
    """Return why the main checkout cannot take a fast-forward merge (it is not on main_branch), or None when it can."""
    current = checked_out_branch(repo)
    if current != main_branch:
        return f"main checkout is on {current or 'a detached HEAD'}, not {main_branch}"
    return None


def fast_forward_main(repo: Path, branch: str, main_branch: str) -> StepOutcome:
    """Fast-forward the main checkout to `branch`; git itself refuses when local changes overlap the merge."""
    problem = main_checkout_problem(repo, main_branch)
    if problem:
        return StepOutcome(False, problem)
    result = run_git(repo, "merge", "--ff-only", branch, check=False)
    if result.returncode != 0:
        return StepOutcome(False, f"fast-forward failed:\n{(result.stdout + result.stderr).decode(errors='replace').strip()}")
    return StepOutcome(True)
