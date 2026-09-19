import unittest

from agent_sandbox.omp_runner import build_omp_argv


class TestBuildOmpArgv(unittest.TestCase):
    def test_includes_optional_model_and_thinking_before_prompt(self) -> None:
        argv = build_omp_argv(
            "/usr/local/bin/omp",
            "inspect the image",
            "/runtime/session",
            resuming=True,
            image_paths=("/runtime/prompt-images/photo.png",),
            model="ollama gemma4",
            thinking="high",
            append_system_path="/runtime/append-system/APPEND_SYSTEM.md",
        )

        self.assertEqual(
            argv,
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
                "--continue",
                "--",
                "inspect the image",
                "@/runtime/prompt-images/photo.png",
            ],
        )

    def test_places_lean_flags_before_continuation_and_prompt(self) -> None:
        argv = build_omp_argv(
            "/usr/local/bin/omp",
            "inspect the image",
            "/runtime/session",
            resuming=True,
            model="ollama gemma4",
            thinking="high",
            lean=True,
        )

        self.assertEqual(
            argv,
            [
                "/usr/local/bin/omp",
                "--print",
                "--session-dir",
                "/runtime/session",
                "--model",
                "ollama gemma4",
                "--thinking",
                "high",
                "--no-lsp",
                "--no-skills",
                "--no-rules",
                "--no-extensions",
                "--continue",
                "--",
                "inspect the image",
            ],
        )

    def test_treats_option_looking_prompt_as_task_text(self) -> None:
        argv = build_omp_argv(
            "/usr/local/bin/omp",
            "--model untrusted",
            "/runtime/session",
            resuming=False,
        )

        self.assertEqual(
            argv,
            [
                "/usr/local/bin/omp",
                "--print",
                "--session-dir",
                "/runtime/session",
                "--",
                "--model untrusted",
            ],
        )
