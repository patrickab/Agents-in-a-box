"""Run the agent-sandbox command-line interface."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

from agent_sandbox import Sandbox, SandboxInvocation, manifest_from_dict, manifest_to_dict
from agent_sandbox.doctor import run_doctor
from agent_sandbox.outputs import serialize_events


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


def _sha256_file(path: str) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: str):
    """Load one strictly validated JSON manifest."""
    with open(path, encoding="utf-8") as source:
        return manifest_from_dict(json.load(source))


def _require_output_parent(path: str) -> None:
    """Require an existing parent directory for an output path."""
    parent = Path(path).parent
    if not parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {parent}")


def _cmd_omp(args: argparse.Namespace) -> int:
    """Run OMP through the native sandbox seam and persist its result."""
    _require_output_parent(args.workspace_out)
    _require_output_parent(args.manifest_out)
    active_manifest = _load_manifest(args.manifest_in) if args.manifest_in else None
    workspace_sha256 = _sha256_file(args.workspace_in) if args.workspace_in and not active_manifest else None
    invocation = SandboxInvocation(
        slot_key=args.slot_key,
        profile_name=args.profile,
        prompt=args.prompt,
        run_id=args.run_id,
        active_manifest=active_manifest,
        workspace_archive_path=args.workspace_in,
        workspace_archive_sha256=workspace_sha256,
        model=args.model,
        thinking=args.thinking,
        append_system=args.append_system,
        lean=args.lean,
    )
    result = Sandbox().run(invocation)
    try:
        shutil.copyfile(result.capture_path, args.workspace_out)
        with open(args.manifest_out, "w", encoding="utf-8") as target:
            json.dump(manifest_to_dict(result.next_manifest), target, indent=2)
            target.write("\n")
    finally:
        result.release()
    print(f"status: {result.status}")
    print(f"summary: {result.summary}")
    print(json.dumps(serialize_events(result.outputs), separators=(",", ":")))
    return 0 if result.status == "completed" else 1


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

    args = parser.parse_args(argv)
    if args.command == "omp" and args.manifest_in and not args.workspace_in:
        parser.error("omp: --manifest-in requires --workspace-in")
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()