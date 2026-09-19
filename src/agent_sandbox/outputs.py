"""Model runtime output events.

Events are results of a runtime invocation, not durable manifest records. OMP
currently produces only ``StreamEvent`` and ``ErrorEvent``; the remaining
Jupyter-compatible event types remain available for other runtime producers.
"""

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from typing import Literal, TypeAlias

StreamName: TypeAlias = Literal["stdout", "stderr"]


@dataclass(frozen=True)
class MimeBundle:
    """Map MIME types to inline display data."""

    data: dict[str, str]


@dataclass(frozen=True)
class StreamEvent:
    """Capture standard output or standard error from a runtime."""

    name: StreamName
    text: str


@dataclass(frozen=True)
class DisplayDataEvent:
    """Capture display data."""

    bundle: MimeBundle
    display_id: str | None = None


@dataclass(frozen=True)
class ExecuteResultEvent:
    """Capture an expression result."""

    bundle: MimeBundle


@dataclass(frozen=True)
class ErrorEvent:
    """Capture an execution failure."""

    ename: str
    evalue: str
    traceback: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpdateDisplayDataEvent:
    """Update an existing display."""

    bundle: MimeBundle
    display_id: str

OmpOutputEvent: TypeAlias = StreamEvent | ErrorEvent
"""The event types currently emitted by OMP."""

OutputEvent: TypeAlias = (
    StreamEvent | DisplayDataEvent | ExecuteResultEvent | ErrorEvent | UpdateDisplayDataEvent
)
"""A runtime result event, independent of its eventual durable storage."""


def serialize_events(events: Sequence[OutputEvent]) -> list[dict[str, object]]:
    """Convert ordered runtime results to plain mappings for caller-managed storage."""
    result: list[dict[str, object]] = []
    for event in events:
        data = dataclasses.asdict(event)
        data["type"] = type(event).__name__
        result.append(data)
    return result