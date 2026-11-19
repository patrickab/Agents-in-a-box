"""Model one Pi task, build its one-shot JSON-mode command line, and translate its event stream."""

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
from typing import Any, Literal, TypeAlias

from agent_sandbox._validation import require_nonempty_string, require_optional_string
from agent_sandbox.config import (
    DEFAULT_PI_TIMEOUT_SECONDS,
    MAX_APPEND_SYSTEM_BYTES,
    MAX_PROMPT_IMAGE_BYTES,
    MAX_RETAINED_OUTPUT_BYTES,
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
)

_LEAN_FLAGS = ("--no-skills", "--no-context-files", "--no-extensions", "--no-prompt-templates")
_FAILED_STOP_REASONS = frozenset({"error", "aborted"})
# Records whose content decides the final text or the failure; every other record is discarded while streaming.
_RETAINED_EVENT_TYPES = frozenset({"message_end", "agent_settled"})
# Pi serializes the record type first, so this prefix classifies a record before the rest of it arrives.
_EVENT_TYPE_PREFIX = re.compile(rb'\A\s*\{\s*"type"\s*:\s*"([A-Za-z_]+)"')
_EVENT_TYPE_PROBE_BYTES = 64


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
class PiTask:
    """Describe one Pi prompt and its per-run options."""

    prompt: str
    prompt_images: tuple[PromptImage, ...] = ()
    model: str | None = None  # `<provider>/<id>` as the gateway lists it.
    thinking: str | None = None
    append_system: str | None = None
    lean: bool = False  # Skip skill, context-file, extension, and prompt-template discovery.
    timeout_s: int = DEFAULT_PI_TIMEOUT_SECONDS  # Wall-clock limit for this run.

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


def build_pi_argv(
    pi_binary: str,
    task: PiTask,
    *,
    session_dir: str,
    resuming: bool,
    images_dir: str,
    append_system_path: str,
    provider: str | None = None,
) -> list[str]:
    """Build the JSON-mode Pi command for one task whose inputs are mounted at the given paths.

    `provider` restricts model lookup to one provider, e.g. `gateway` for the seeded sandbox agent dir.
    """
    argv = [pi_binary, "--mode", "json", "--session-dir", session_dir]
    if task.model:
        if provider:
            argv.extend(["--provider", provider])
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
class PiOutcome:
    """Hold the final assistant text and the failure, if any, a Pi JSON event stream reported."""

    text: str
    error: str | None = None


class PiEventStream:
    """Parse Pi's JSONL event stream incrementally, retaining only the last assistant message.

    Records other than `message_end` and `agent_settled` are discarded as they stream, and a retained
    record longer than `max_record_bytes` is dropped, so memory stays bounded however long the run.
    """

    def __init__(self, max_record_bytes: int = MAX_RETAINED_OUTPUT_BYTES) -> None:
        self._max_record_bytes = max_record_bytes
        self._record = bytearray()
        self._classified = False
        self._skipping = False
        self._dropped_records = 0
        self._final_message: dict[str, Any] | None = None
        self._aborted = False

    def feed(self, chunk: bytes) -> None:
        """Consume the next stdout bytes; records are LF-terminated."""
        view = memoryview(chunk)
        start = 0
        while (newline := chunk.find(b"\n", start)) >= 0:
            self._append(view[start:newline])
            self._end_record()
            start = newline + 1
        self._append(view[start:])

    def finish(self) -> PiOutcome:
        """Process any unterminated last record and report the run's final text and failure."""
        self._end_record()
        message = self._final_message
        if message is None:
            dropped = f" ({self._dropped_records} oversized events dropped)" if self._dropped_records else ""
            return PiOutcome("", f"pi produced no final assistant message{dropped}")
        text = "\n".join(
            block["text"]
            for block in message.get("content") or ()
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
        stop_reason = message.get("stopReason")
        if stop_reason in _FAILED_STOP_REASONS:
            detail = message.get("errorMessage")
            return PiOutcome(
                text, detail if isinstance(detail, str) and detail else f"pi response ended with stopReason {stop_reason}"
            )
        if self._aborted:
            return PiOutcome(text, "pi run was aborted")
        return PiOutcome(text)

    def _append(self, piece: memoryview) -> None:
        if self._skipping or not piece:
            return
        self._record += piece
        if not self._classified and len(self._record) >= _EVENT_TYPE_PROBE_BYTES:
            self._classify()
        if len(self._record) > self._max_record_bytes:
            self._dropped_records += 1
            self._skip()

    def _classify(self) -> None:
        """Skip the rest of a record whose leading type Pi's final outcome never depends on."""
        self._classified = True
        match = _EVENT_TYPE_PREFIX.match(self._record)
        if match is not None and match.group(1).decode() not in _RETAINED_EVENT_TYPES:
            self._skip()

    def _skip(self) -> None:
        self._skipping = True
        self._record.clear()

    def _end_record(self) -> None:
        if not self._skipping and self._record.strip():
            if not self._classified:
                self._classify()
            if not self._skipping:
                self._handle(self._record)
        self._record.clear()
        self._classified = False
        self._skipping = False

    def _handle(self, raw: bytearray) -> None:
        try:
            record = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return
        if not isinstance(record, dict):
            return
        if record.get("type") == "agent_settled":
            self._aborted = record.get("aborted") is True
            return
        message = record.get("message")
        if record.get("type") == "message_end" and isinstance(message, dict) and message.get("role") == "assistant":
            self._final_message = message


def parse_pi_events(stdout: bytes) -> PiOutcome:
    """Parse one complete Pi JSON-mode stdout capture."""
    stream = PiEventStream()
    stream.feed(stdout)
    return stream.finish()


@dataclass(frozen=True)
class StreamEvent:
    """Capture Pi's final assistant text or its standard error."""

    name: Literal["stdout", "stderr"]
    text: str


@dataclass(frozen=True)
class ErrorEvent:
    """Capture a Pi execution failure."""

    ename: str
    evalue: str
    traceback: tuple[str, ...] = ()


OutputEvent: TypeAlias = StreamEvent | ErrorEvent


def to_output_events(exit_code: int, outcome: PiOutcome, stderr: str) -> tuple[OutputEvent, ...]:
    """Translate a Pi run into ordered display events; a failed response is an error even on exit 0."""
    events: list[OutputEvent] = []
    if outcome.text:
        events.append(StreamEvent(name="stdout", text=outcome.text))
    if stderr:
        events.append(StreamEvent(name="stderr", text=stderr))
    if exit_code != 0:
        detail = f": {outcome.error}" if outcome.error is not None else ""
        events.append(ErrorEvent(ename="PiExecutionError", evalue=f"pi exited with code {exit_code}{detail}"))
    elif outcome.error is not None:
        events.append(ErrorEvent(ename="PiExecutionError", evalue=outcome.error))
    return tuple(events)


def serialize_events(events: Sequence[OutputEvent]) -> list[dict[str, object]]:
    """Convert ordered events to plain mappings for caller-managed storage."""
    return [{**dataclasses.asdict(event), "type": type(event).__name__} for event in events]
