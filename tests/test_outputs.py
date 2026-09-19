"""Test runtime output event serialization."""

import unittest

from agent_sandbox.outputs import ErrorEvent, StreamEvent, serialize_events


class TestOutputEventSerialization(unittest.TestCase):
    """Exercise runtime result events independently from manifest storage."""

    def test_serializes_current_omp_event_types_in_order(self) -> None:
        events = (
            StreamEvent(name="stdout", text="normal output"),
            ErrorEvent(
                ename="OMPExecutionError",
                evalue="omp exited with code 7",
                traceback=("trace line",),
            ),
        )

        self.assertEqual(
            serialize_events(events),
            [
                {"name": "stdout", "text": "normal output", "type": "StreamEvent"},
                {
                    "ename": "OMPExecutionError",
                    "evalue": "omp exited with code 7",
                    "traceback": ("trace line",),
                    "type": "ErrorEvent",
                },
            ],
        )
