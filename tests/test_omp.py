"""Test OMP command construction and output event translation."""

import unittest

from agent_sandbox.omp import ErrorEvent, OmpTask, PromptImage, StreamEvent, build_omp_argv, serialize_events, to_output_events


def _argv(task: OmpTask, *, resuming: bool = False) -> list[str]:
    return build_omp_argv(
        "/usr/local/bin/omp",
        task,
        session_dir="/runtime/session",
        resuming=resuming,
        images_dir="/runtime/prompt-images",
        append_system_path="/runtime/append-system/APPEND_SYSTEM.md",
    )


class TestBuildOmpArgv(unittest.TestCase):
    def test_places_options_before_continuation_prompt_and_images(self) -> None:
        task = OmpTask(
            "inspect the image",
            prompt_images=(PromptImage("photo.png", b"png"),),
            model="ollama gemma4",
            thinking="high",
            append_system="be terse",
            lean=True,
        )

        self.assertEqual(
            _argv(task, resuming=True),
            [
                "/usr/local/bin/omp",
                "--print",
                "--session-dir",
                "/runtime/session",
                "--model",
                "ollama gemma4",
                "--thinking",
                "high",
                "--append-system-prompt",
                "/runtime/append-system/APPEND_SYSTEM.md",
                "--no-lsp",
                "--no-skills",
                "--no-rules",
                "--no-extensions",
                "--continue",
                "--",
                "inspect the image",
                "@/runtime/prompt-images/photo.png",
            ],
        )

    def test_treats_option_looking_prompt_as_task_text(self) -> None:
        self.assertEqual(
            _argv(OmpTask("--model untrusted")),
            ["/usr/local/bin/omp", "--print", "--session-dir", "/runtime/session", "--", "--model untrusted"],
        )


class TestOutputEvents(unittest.TestCase):
    def test_translates_and_serializes_in_stdout_stderr_error_order(self) -> None:
        events = to_output_events(7, "normal output", "warning")

        self.assertEqual(
            serialize_events(events),
            [
                {"name": "stdout", "text": "normal output", "type": "StreamEvent"},
                {"name": "stderr", "text": "warning", "type": "StreamEvent"},
                {"ename": "OMPExecutionError", "evalue": "omp exited with code 7", "traceback": (), "type": "ErrorEvent"},
            ],
        )

    def test_successful_silent_run_has_no_events(self) -> None:
        self.assertEqual(to_output_events(0, "", ""), ())
        self.assertIsInstance(to_output_events(1, "", "")[0], ErrorEvent)
        self.assertEqual(to_output_events(0, "x", ""), (StreamEvent("stdout", "x"),))


if __name__ == "__main__":
    unittest.main()
