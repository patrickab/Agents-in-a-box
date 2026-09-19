"""Build one-shot OMP commands and translate their output."""

from agent_sandbox.outputs import ErrorEvent, StreamEvent

_LEAN_FLAGS = ("--no-lsp", "--no-skills", "--no-rules", "--no-extensions")


def build_omp_argv(
    omp_binary: str,
    prompt: str,
    session_dir: str,
    resuming: bool,
    image_paths: tuple[str, ...] = (),
    model: str | None = None,
    thinking: str | None = None,
    append_system_path: str | None = None,
    lean: bool = False,
) -> list[str]:
    """Build the OMP command for one prompt and its transient image files."""
    argv = [omp_binary, "--print", "--session-dir", session_dir]
    if model:
        argv.extend(["--model", model])
    if thinking:
        argv.extend(["--thinking", thinking])
    if append_system_path is not None:
        argv.extend(["--append-system-prompt", append_system_path])
    if lean:
        # Skip discovery the caller's workflow does not use.
        argv.extend(_LEAN_FLAGS)
    if resuming:
        argv.append("--continue")
    argv.append("--")
    argv.append(prompt)
    argv.extend(f"@{path}" for path in image_paths)
    return argv


def to_output_events(exit_code: int, stdout: str, stderr: str) -> list:
    """Translate OMP output into ordered display events."""
    events = []
    if stdout:
        events.append(StreamEvent(name="stdout", text=stdout))
    if stderr:
        events.append(StreamEvent(name="stderr", text=stderr))
    if exit_code != 0:
        events.append(
            ErrorEvent(
                ename="OMPExecutionError",
                evalue=f"omp exited with code {exit_code}",
                traceback=(),
            )
        )
    return events
