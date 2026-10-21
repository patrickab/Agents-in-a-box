"""Model one OMP task, build its one-shot command line, and translate its output."""

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, TypeAlias

from agent_sandbox._validation import require_nonempty_string, require_optional_string
from agent_sandbox.config import (
    DEFAULT_OMP_TIMEOUT_SECONDS,
    MAX_APPEND_SYSTEM_BYTES,
    MAX_PROMPT_IMAGE_BYTES,
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
)

_LEAN_FLAGS = ("--no-lsp", "--no-skills", "--no-rules", "--no-extensions")


@dataclass(frozen=True)
class PromptImage:
    """Hold one safe prompt-image filename and its bytes."""

    filename: str
    content: bytes

    def __post_init__(self) -> None:
        name = self.filename
        if not isinstance(name, str) or name in {"", ".", ".."} or "\\" in name or "\x00" in name or name != PurePosixPath(name).name:
            raise ValueError("prompt image filename must be a safe basename")
        if not isinstance(self.content, bytes):
            raise TypeError("prompt image content must be bytes")


@dataclass(frozen=True)
class OmpTask:
    """Describe one OMP prompt and its per-run options."""

    prompt: str
    prompt_images: tuple[PromptImage, ...] = ()
    model: str | None = None
    thinking: str | None = None
    append_system: str | None = None
    lean: bool = False  # Skip LSP, skill, rule, and extension discovery.
    timeout_s: int = DEFAULT_OMP_TIMEOUT_SECONDS  # Wall-clock limit for this run.

    def __post_init__(self) -> None:
        require_nonempty_string("prompt", self.prompt)
        require_optional_string("model", self.model)
        require_optional_string("thinking", self.thinking)
        require_optional_string("append_system", self.append_system)
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, int):
            raise TypeError("timeout_s must be an integer")
        if not MIN_TIMEOUT_SECONDS <= self.timeout_s <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeout_s must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS} seconds")
        if self.append_system is not None and len(self.append_system.encode("utf-8")) > MAX_APPEND_SYSTEM_BYTES:
            raise ValueError(f"append_system must not exceed {MAX_APPEND_SYSTEM_BYTES} bytes")
        if not isinstance(self.prompt_images, tuple) or not all(isinstance(image, PromptImage) for image in self.prompt_images):
            raise TypeError("prompt_images must be a tuple of PromptImage values")
        if len({image.filename for image in self.prompt_images}) != len(self.prompt_images):
            raise ValueError("prompt image filenames must be unique")
        if sum(len(image.content) for image in self.prompt_images) > MAX_PROMPT_IMAGE_BYTES:
            raise ValueError(f"prompt images must not exceed {MAX_PROMPT_IMAGE_BYTES} bytes")


def build_omp_argv(
    omp_binary: str,
    task: OmpTask,
    *,
    session_dir: str,
    resuming: bool,
    images_dir: str,
    append_system_path: str,
) -> list[str]:
    """Build the OMP command for one task whose inputs are mounted at the given paths."""
    argv = [omp_binary, "--print", "--session-dir", session_dir]
    if task.model:
        argv.extend(["--model", task.model])
    if task.thinking:
        argv.extend(["--thinking", task.thinking])
    if task.append_system is not None:
        argv.extend(["--append-system-prompt", append_system_path])
    if task.lean:
        argv.extend(_LEAN_FLAGS)
    if resuming:
        argv.append("--continue")
    argv.extend(["--", task.prompt])
    argv.extend(f"@{images_dir}/{image.filename}" for image in task.prompt_images)
    return argv


@dataclass(frozen=True)
class StreamEvent:
    """Capture standard output or standard error from OMP."""

    name: Literal["stdout", "stderr"]
    text: str


@dataclass(frozen=True)
class ErrorEvent:
    """Capture an OMP execution failure."""

    ename: str
    evalue: str
    traceback: tuple[str, ...] = ()


OutputEvent: TypeAlias = StreamEvent | ErrorEvent


def to_output_events(exit_code: int, stdout: str, stderr: str) -> tuple[OutputEvent, ...]:
    """Translate OMP output into ordered display events."""
    events: list[OutputEvent] = []
    if stdout:
        events.append(StreamEvent(name="stdout", text=stdout))
    if stderr:
        events.append(StreamEvent(name="stderr", text=stderr))
    if exit_code != 0:
        events.append(ErrorEvent(ename="OMPExecutionError", evalue=f"omp exited with code {exit_code}"))
    return tuple(events)


def serialize_events(events: Sequence[OutputEvent]) -> list[dict[str, object]]:
    """Convert ordered events to plain mappings for caller-managed storage."""
    return [{**dataclasses.asdict(event), "type": type(event).__name__} for event in events]
