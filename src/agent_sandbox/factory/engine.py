"""Hold the engine's shared services and implement the task operations behind the HTTP API."""

import base64
import binascii
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import secrets
import threading
from typing import Any, Protocol, get_args

from agent_sandbox.factory import git_ops
from agent_sandbox.factory.config import (
    ConfigError,
    EngineConfig,
    PreviewConfig,
    TaskConfig,
    build_task_config,
    list_workflows,
    load_defaults,
    save_defaults,
    task_dir,
)
from agent_sandbox.factory.models import (
    TERMINAL_STATES,
    FactoryDefaults,
    ImageInput,
    Message,
    MessageCreate,
    Preview,
    RepoCreate,
    RepoInfo,
    Task,
    TaskCreate,
    TaskDetail,
    TaskState,
    WorkflowInfo,
)
from agent_sandbox.factory.runners import Runner, kill_task_processes, make_runner, release_sandbox
from agent_sandbox.factory.store import Store
from agent_sandbox.omp import PromptImage

LOGGER = logging.getLogger("agent_sandbox.factory")
MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
REQUEST_IMAGE_PREFIX = "request"
CANCELLABLE_STATES = tuple(state for state in get_args(TaskState) if state not in TERMINAL_STATES)


class ApiError(Exception):
    """Carry an API error kind, HTTP status, and detail message."""

    def __init__(self, kind: str, status: int, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.status = status
        self.detail = detail


def not_found(detail: str) -> ApiError:
    """Build the 404 error for a missing resource."""
    return ApiError("not_found", 404, detail)


def repo_name_for(path: Path) -> str:
    """Derive a repo name from its directory name, replacing characters the API names forbid."""
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", path.name).lstrip("_.-")
    return name or "repo"


def conflict(detail: str) -> ApiError:
    """Build the 409 error for a request the current state forbids."""
    return ApiError("conflict", 409, detail)


def invalid(detail: str) -> ApiError:
    """Build the 422 error for a malformed request."""
    return ApiError("invalid", 422, detail)


class Dispatcher(Protocol):
    """Drive the durable workflow of a task."""

    def start(self, task_id: str, config: dict[str, Any]) -> None:
        """Start the task's workflow."""

    def send(self, task_id: str, payload: dict[str, Any]) -> None:
        """Deliver a command (`message`, `approve`) to the task's waiting workflow."""

    def retry(self, task_id: str, config: dict[str, Any], entry: str) -> None:
        """Retry a blocked task: wake its waiting workflow, or start a fresh one at `entry` (`retry` or `merge`) when it ended."""

    def cancel(self, task_id: str) -> None:
        """Cancel the task's workflow and any children."""


class PreviewService(Protocol):
    """Run per-task previews (implemented by `preview.PreviewManager`)."""

    def start(self, task_id: str, worktree: Path, config: PreviewConfig) -> Preview: ...

    def stop(self, task_id: str) -> None: ...

    def status(self, task_id: str) -> Preview | None: ...

    def stop_all(self) -> None: ...


@dataclass
class Engine:
    """Bundle config, store, previews, and the workflow dispatcher."""

    config: EngineConfig
    store: Store
    previews: PreviewService
    dispatcher: Dispatcher

    # --- paths and runners ---

    def task_dir(self, task_id: str) -> Path:
        """Return the task's directory under the engine state dir."""
        return task_dir(self.config.state_dir, task_id)

    def runner_for(self, task_id: str, task_config: TaskConfig, worktree: Path) -> Runner:
        """Build the runner for a task's worktree."""
        return make_runner(task_config, task_id, worktree, self.task_dir(task_id), self.config.omp_binary)

    def task_config(self, task_id: str) -> TaskConfig:
        """Return the frozen configuration of a task."""
        raw = self.store.get_config(task_id)
        if raw is None:
            raise not_found(f"task {task_id} not found")
        return TaskConfig.model_validate(raw)

    # --- attachments ---

    def _attachments_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "attachments"

    @staticmethod
    def _decode_images(images: list[ImageInput], prefix: str) -> list[tuple[str, bytes]]:
        """Validate attachments and return their prefixed filenames with decoded content."""
        decoded: list[tuple[str, bytes]] = []
        for image in images:
            try:
                content = base64.b64decode(image.data_base64, validate=True)
                PromptImage(f"{prefix}-{image.filename}", content)
            except (binascii.Error, ValueError, TypeError) as error:
                raise invalid(f"invalid image {image.filename!r}: {error}") from error
            decoded.append((f"{prefix}-{image.filename}", content))
        if len({name for name, _ in decoded}) != len(decoded):
            raise invalid("image filenames must be unique")
        if sum(len(content) for _, content in decoded) > MAX_ATTACHMENT_BYTES:
            raise invalid(f"images must not exceed {MAX_ATTACHMENT_BYTES} bytes")
        return decoded

    def _write_images(self, task_id: str, decoded: list[tuple[str, bytes]]) -> list[str]:
        directory = self._attachments_dir(task_id)
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in decoded:
            (directory / name).write_bytes(content)
        return [name for name, _ in decoded]

    def load_images(self, task_id: str, names: list[str]) -> tuple[PromptImage, ...]:
        """Read stored attachments by filename."""
        directory = self._attachments_dir(task_id)
        return tuple(PromptImage(name, (directory / name).read_bytes()) for name in names)

    def request_image_names(self, task_id: str) -> list[str]:
        """List the attachments that came with the task request."""
        directory = self._attachments_dir(task_id)
        return sorted(path.name for path in directory.glob(f"{REQUEST_IMAGE_PREFIX}-*")) if directory.is_dir() else []

    # --- queries ---

    def repos(self) -> list[RepoInfo]:
        """List the registered repos with their main checkout paths."""
        return self.store.list_repos()

    def register_repo(self, body: RepoCreate) -> RepoInfo:
        """Register the main checkout at `path` (idempotent by resolved path) under a unique name derived from its directory."""
        path = Path(body.path)
        if not path.is_absolute():
            raise invalid(f"repo path must be absolute: {body.path!r}")
        resolved = path.resolve()
        if not resolved.is_dir():
            raise invalid(f"repo path does not exist or is not a directory: {resolved}")
        if not (resolved / ".git").is_dir():
            raise invalid(f"{resolved} is not a git main checkout (no .git directory)")
        return self.store.register_repo(str(resolved), repo_name_for(resolved))

    def _repo_path(self, repo: str) -> Path:
        info = self.store.get_repo(repo)
        if info is None:
            raise not_found(f"repo {repo!r} not found")
        return Path(info.path)

    def workflows(self, repo: str) -> list[WorkflowInfo]:
        """List the workflows a repo offers: packaged ones plus the repo's own."""
        try:
            return list_workflows(self._repo_path(repo), self.config.defaults_dir)
        except ConfigError as error:
            raise invalid(str(error)) from error

    def defaults(self) -> FactoryDefaults:
        """Return the engine-wide defaults: user settings and agents, falling back to the packaged ones."""
        try:
            return load_defaults(self.config.defaults_dir)
        except ConfigError as error:
            raise invalid(str(error)) from error

    def save_defaults(self, body: FactoryDefaults) -> FactoryDefaults:
        """Validate and persist the engine-wide defaults."""
        try:
            return save_defaults(self.config.defaults_dir, body)
        except ConfigError as error:
            raise invalid(str(error)) from error

    def get_task(self, task_id: str) -> Task:
        """Return one task or raise not_found."""
        task = self.store.get_task(task_id)
        if task is None:
            raise not_found(f"task {task_id} not found")
        return task

    def get_detail(self, task_id: str) -> TaskDetail:
        """Return a task with its phase runs, messages, and preview."""
        task = self.get_task(task_id)
        return TaskDetail(
            **task.model_dump(),
            phases=self.store.list_phases(task_id),
            messages=self.store.list_messages(task_id),
            preview=self.previews.status(task_id),
        )

    # --- commands ---

    def create_task(self, body: TaskCreate) -> Task:
        """Validate and store a new task, starting its workflow unless it goes to the backlog."""
        info = self.store.get_repo(body.repo)
        if info is None:
            raise invalid(f"unknown repo {body.repo!r}")
        repo_path = Path(info.path)
        if not repo_path.is_dir():
            raise invalid(f"repo {body.repo!r} no longer exists at {repo_path}")
        if not body.title.strip() or not body.request.strip():
            raise invalid("title and request must not be empty")
        try:
            task_config = build_task_config(
                self.config, body.repo, repo_path, body.workflow, body.overrides, git_ops.checked_out_branch(repo_path)
            )
        except ConfigError as error:
            raise invalid(str(error)) from error
        if git_ops.run_git(repo_path, "rev-parse", "--verify", "--quiet", task_config.main_branch, check=False).returncode != 0:
            raise invalid(f"branch {task_config.main_branch!r} does not exist in {repo_path}")
        task_id = secrets.token_hex(6)
        self._write_images(task_id, self._decode_images(body.images, REQUEST_IMAGE_PREFIX))
        state = "todo" if body.start else "backlog"
        task = self.store.create_task(
            task_id=task_id,
            repo=body.repo,
            workflow=body.workflow,
            title=body.title.strip(),
            request=body.request,
            state=state,
            config=task_config.model_dump(mode="json"),
        )
        if body.start:
            self.dispatcher.start(task_id, task_config.model_dump(mode="json"))
        return task

    def start_task(self, task_id: str) -> Task:
        """Move a backlog task to todo and start its workflow."""
        task = self.store.transition(task_id, ("backlog",), "todo")
        if task is None:
            raise self._wrong_state(task_id, "start", "backlog")
        self.dispatcher.start(task_id, self.task_config(task_id).model_dump(mode="json"))
        return task

    def post_message(self, task_id: str, body: MessageCreate) -> Message:
        """Send a user message to a task waiting in human_review or blocked, sending it back to rework."""
        if not body.text.strip():
            raise invalid("message text must not be empty")
        if self.store.get_task(task_id) is None:
            raise not_found(f"task {task_id} not found")
        decoded = self._decode_images(body.images, f"msg{secrets.token_hex(3)}")
        if self.store.transition(task_id, ("human_review", "blocked"), "rework", error=None) is None:
            raise self._wrong_state(task_id, "message", "human_review or blocked")
        image_names = self._write_images(task_id, decoded)
        message = self.store.add_message(task_id, "user", body.text)
        self.dispatcher.send(task_id, {"kind": "message", "text": body.text, "images": image_names})
        return message

    def approve(self, task_id: str) -> Task:
        """Approve a task in human_review and send it to the merge queue."""
        task = self.store.transition(task_id, ("human_review",), "merging", phase="merge")
        if task is None:
            raise self._wrong_state(task_id, "approve", "human_review")
        self.dispatcher.send(task_id, {"kind": "approve"})
        return task

    def retry(self, task_id: str) -> Task:
        """Retry a blocked task from its failed step, or from the merge when the block happened there."""
        current = self.get_task(task_id)
        merging = current.phase == "merge"
        task = self.store.transition(task_id, ("blocked",), "merging" if merging else "todo", error=None)
        if task is None:
            raise self._wrong_state(task_id, "retry", "blocked")
        self.dispatcher.retry(task_id, self.task_config(task_id).model_dump(mode="json"), "merge" if merging else "retry")
        return task

    def cancel(self, task_id: str) -> Task:
        """Cancel a task, stop its processes, and clean up its worktree in the background."""
        task = self.store.transition(task_id, CANCELLABLE_STATES, "cancelled")
        if task is None:
            raise self._wrong_state(task_id, "cancel", "a non-final state")
        self.dispatcher.cancel(task_id)
        kill_task_processes(task_id)
        threading.Thread(target=self._cleanup_quietly, args=(task_id,), name=f"cleanup-{task_id}", daemon=True).start()
        return task

    def restart_preview(self, task_id: str) -> Preview:
        """Restart the preview of a task waiting in human_review."""
        task = self.get_task(task_id)
        if task.state != "human_review":
            raise self._wrong_state(task_id, "restart the preview of", "human_review")
        preview_config = self.task_config(task_id).preview
        if preview_config is None or task.worktree is None:
            raise conflict(f"task {task_id} has no preview to restart")
        self.previews.stop(task_id)
        preview = self.previews.start(task_id, Path(task.worktree), preview_config)
        self.store.add_event(task_id, "preview", {"status": preview.status})
        return preview

    def _wrong_state(self, task_id: str, action: str, allowed: str) -> ApiError:
        task = self.store.get_task(task_id)
        if task is None:
            return not_found(f"task {task_id} not found")
        return conflict(f"cannot {action} task {task_id} in state {task.state}; allowed: {allowed}")

    # --- teardown ---

    def teardown_task(self, task_id: str, *, run_hooks: bool = True) -> None:
        """Stop the preview, run before_remove hooks, and delete the worktree and branch."""
        task = self.get_task(task_id)
        task_config = self.task_config(task_id)
        self.previews.stop(task_id)
        if task.worktree is not None:
            worktree = Path(task.worktree)
            if run_hooks and worktree.is_dir():
                runner = self.runner_for(task_id, task_config, worktree)
                for argv in task_config.hooks.before_remove:
                    result = runner.command(argv, task_config.command_timeout_s)
                    if result.exit_code != 0:
                        LOGGER.warning("before_remove hook %s failed for %s: %s", argv, task_id, result.output[-500:])
            git_ops.remove_worktree(Path(task_config.repo_path), worktree, task.branch or git_ops.branch_name(task_id))
        release_sandbox(task_id)

    def _cleanup_quietly(self, task_id: str) -> None:
        try:
            self.teardown_task(task_id)
        except Exception:
            LOGGER.exception("cleanup of cancelled task %s failed", task_id)
