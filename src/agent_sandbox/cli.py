"""Run the agent-sandbox command-line interface."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import uuid

from agent_sandbox import (
    AgentSandboxError,
    OmpTask,
    Sandbox,
    SandboxInvocation,
    WorkspaceArchive,
    manifest_from_dict,
    manifest_to_dict,
)
from agent_sandbox.doctor import run_doctor
from agent_sandbox.omp import serialize_events


def _cmd_doctor(args: argparse.Namespace) -> int:
    results = run_doctor(args.profile)
    for r in results:
        mark = "OK  " if r.passed else "FAIL"
        print(f"[{mark}] {r.name}: {r.detail}")
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n{len(failed)} check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


def _require_output_parent(path: str) -> None:
    parent = Path(path).parent
    if not parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {parent}")


def _cmd_omp(args: argparse.Namespace) -> int:
    """Run OMP through the sandbox and persist its workspace, manifest, and events."""
    _require_output_parent(args.workspace_out)
    _require_output_parent(args.manifest_out)
    active_manifest = None
    if args.manifest_in:
        with open(args.manifest_in, encoding="utf-8") as source:
            active_manifest = manifest_from_dict(json.load(source))
    invocation = SandboxInvocation(
        slot_key=args.slot_key,
        profile_name=args.profile,
        run_id=args.run_id,
        task=OmpTask(args.prompt, model=args.model, thinking=args.thinking, append_system=args.append_system, lean=args.lean),
        active_manifest=active_manifest,
        workspace=WorkspaceArchive.from_file(args.workspace_in) if args.workspace_in else None,
    )
    with Sandbox().run(invocation) as result:
        shutil.copyfile(result.workspace.path, args.workspace_out)
        with open(args.manifest_out, "w", encoding="utf-8") as target:
            json.dump(manifest_to_dict(result.next_manifest), target, indent=2)
            target.write("\n")
    print(f"status: {result.status}")
    print(f"summary: {result.summary}")
    print(json.dumps(serialize_events(result.outputs), separators=(",", ":")))
    return 0 if result.status == "completed" else 1


def _cmd_factory_serve(args: argparse.Namespace) -> int:
    """Run the agent factory engine."""
    try:
        from agent_sandbox.factory.server import serve
    except ImportError as error:
        raise ValueError(f"the factory engine needs the 'factory' extra: pip install 'agent-sandbox[factory]' ({error})") from error
    serve(args.config)
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agent-sandbox", description="Rootless-Docker/gVisor sandbox runtime.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Verify a profile's runtime prerequisites.")
    doctor_parser.add_argument("--profile", required=True, help="Profile name to verify.")
    doctor_parser.set_defaults(func=_cmd_doctor)

    omp_parser = subparsers.add_parser("omp", help="Run OMP through a profiled sandbox.")
    omp_parser.add_argument("prompt")
    omp_parser.add_argument("--profile", required=True)
    omp_parser.add_argument("--workspace-out", required=True)
    omp_parser.add_argument("--manifest-out", required=True)
    omp_parser.add_argument("--workspace-in")
    omp_parser.add_argument("--manifest-in")
    omp_parser.add_argument("--slot-key", default="cli")
    omp_parser.add_argument("--run-id", default=str(uuid.uuid4()))
    omp_parser.add_argument("--model")
    omp_parser.add_argument("--thinking")
    omp_parser.add_argument("--append-system")
    omp_parser.add_argument("--lean", action="store_true")
    omp_parser.set_defaults(func=_cmd_omp)

    factory_parser = subparsers.add_parser("factory", help="Run the agent factory engine.")
    factory_subparsers = factory_parser.add_subparsers(dest="factory_command", required=True)
    serve_parser = factory_subparsers.add_parser("serve", help="Serve the factory HTTP API and run its workflows.")
    serve_parser.add_argument(
        "--config", help="Engine config path (default: $AGENT_SANDBOX_FACTORY_CONFIG or ~/.config/agent-sandbox/factory.yaml)."
    )
    serve_parser.set_defaults(func=_cmd_factory_serve)

    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except (OSError, ValueError, AgentSandboxError) as error:
        parser.exit(2, f"{parser.prog} {args.command}: error: {error}\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
