"""Test Pi command construction, JSON event parsing, and output event translation."""

import json
import unittest

from agent_sandbox.pi import (
    ErrorEvent,
    PiEventStream,
    PiOutcome,
    PiTask,
    PromptImage,
    StreamEvent,
    build_pi_argv,
    parse_pi_events,
    serialize_events,
    to_output_events,
)


def _argv(task: PiTask, *, resuming: bool = False, provider: str | None = None) -> list[str]:
    return build_pi_argv(
        "/opt/pi/pi",
        task,
        session_dir="/runtime/session",
        resuming=resuming,
        images_dir="/runtime/prompt-images",
        append_system_path="/runtime/append-system/APPEND_SYSTEM.md",
        provider=provider,
    )


def _jsonl(*records: dict[str, object]) -> bytes:
    return b"".join(json.dumps(record).encode() + b"\n" for record in records)


def _assistant(text: str, stop_reason: str = "stop", **extra: object) -> dict[str, object]:
    message = {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": text}]}
    return {"type": "message_end", "message": message | {"stopReason": stop_reason} | extra}


class TestBuildPiArgv(unittest.TestCase):
    def test_places_options_before_continuation_prompt_and_images(self) -> None:
        task = PiTask(
            "inspect the image",
            prompt_images=(PromptImage("photo.png", b"png"),),
            model="anthropic/claude-haiku-5-5",
            thinking="high",
            append_system="be terse",
            lean=True,
        )

        self.assertEqual(
            _argv(task, resuming=True, provider="gateway"),
            [
                "/opt/pi/pi",
                "--mode",
                "json",
                "--session-dir",
                "/runtime/session",
                "--provider",
                "gateway",
                "--model",
                "anthropic/claude-haiku-5-5",
                "--thinking",
                "high",
                "--append-system-prompt",
                "/runtime/append-system/APPEND_SYSTEM.md",
                "--no-skills",
                "--no-context-files",
                "--no-extensions",
                "--no-prompt-templates",
                "--continue",
                "--",
                "inspect the image",
                "@/runtime/prompt-images/photo.png",
            ],
        )

    def test_provider_applies_only_with_a_model(self) -> None:
        self.assertEqual(
            _argv(PiTask("--model untrusted"), provider="gateway"),
            ["/opt/pi/pi", "--mode", "json", "--session-dir", "/runtime/session", "--", "--model untrusted"],
        )
        self.assertEqual(_argv(PiTask("go", model="anthropic/x"))[5:7], ["--model", "anthropic/x"])


class TestPiEventStream(unittest.TestCase):
    def test_keeps_the_last_assistant_text_across_split_chunks(self) -> None:
        stdout = _jsonl(
            {"type": "session", "version": 3, "id": "s"},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "x" * 500}},
            _assistant("first turn", "toolUse"),
            {"type": "tool_execution_end", "toolCallId": "c", "result": {"content": [{"type": "text", "text": "y" * 500}]}},
            {"type": "message_end", "message": {"role": "toolResult", "content": []}},
            _assistant("final answer"),
            {"type": "agent_settled", "aborted": False},
        )
        stream = PiEventStream()
        for start in range(0, len(stdout), 7):
            stream.feed(stdout[start : start + 7])

        self.assertEqual(stream.finish(), PiOutcome("final answer"))

    def test_failed_or_aborted_response_is_an_error(self) -> None:
        self.assertEqual(
            parse_pi_events(_jsonl(_assistant("", "error", errorMessage="401 unauthorized"))), PiOutcome("", "401 unauthorized")
        )
        self.assertEqual(
            parse_pi_events(_jsonl(_assistant("partial", "aborted"))),
            PiOutcome("partial", "pi response ended with stopReason aborted"),
        )
        settled = _jsonl(_assistant("done"), {"type": "agent_settled", "aborted": True})
        self.assertEqual(parse_pi_events(settled), PiOutcome("done", "pi run was aborted"))

    def test_missing_final_message_is_an_error(self) -> None:
        self.assertEqual(parse_pi_events(b""), PiOutcome("", "pi produced no final assistant message"))
        self.assertEqual(parse_pi_events(b"not json\n[1]\n"), PiOutcome("", "pi produced no final assistant message"))

    def test_unterminated_last_record_and_crlf_are_parsed(self) -> None:
        self.assertEqual(parse_pi_events(_jsonl(_assistant("ok")).rstrip(b"\n")), PiOutcome("ok"))
        self.assertEqual(parse_pi_events(_jsonl(_assistant("ok")).replace(b"\n", b"\r\n")), PiOutcome("ok"))

    def test_discards_unneeded_records_and_drops_oversized_retained_ones(self) -> None:
        stream = PiEventStream(max_record_bytes=200)
        stream.feed(_jsonl({"type": "message_update", "delta": "z" * 10_000}, _assistant("fits")))
        self.assertEqual(stream.finish(), PiOutcome("fits"))

        stream = PiEventStream(max_record_bytes=200)
        stream.feed(_jsonl(_assistant("small"), _assistant("w" * 1000)))
        self.assertEqual(stream.finish(), PiOutcome("small"))

        stream = PiEventStream(max_record_bytes=200)
        stream.feed(_jsonl(_assistant("w" * 1000)))
        self.assertEqual(stream.finish(), PiOutcome("", "pi produced no final assistant message (1 oversized events dropped)"))


class TestOutputEvents(unittest.TestCase):
    def test_translates_and_serializes_in_stdout_stderr_error_order(self) -> None:
        events = to_output_events(7, PiOutcome("normal output", "boom"), "warning")

        self.assertEqual(
            serialize_events(events),
            [
                {"name": "stdout", "text": "normal output", "type": "StreamEvent"},
                {"name": "stderr", "text": "warning", "type": "StreamEvent"},
                {"ename": "PiExecutionError", "evalue": "pi exited with code 7: boom", "traceback": (), "type": "ErrorEvent"},
            ],
        )

    def test_stream_failure_is_an_error_even_on_exit_zero(self) -> None:
        self.assertEqual(to_output_events(0, PiOutcome("x"), ""), (StreamEvent("stdout", "x"),))
        self.assertEqual(
            to_output_events(0, PiOutcome("", "401 unauthorized"), ""), (ErrorEvent("PiExecutionError", "401 unauthorized"),)
        )
        self.assertEqual(to_output_events(1, PiOutcome(""), ""), (ErrorEvent("PiExecutionError", "pi exited with code 1"),))


if __name__ == "__main__":
    unittest.main()
