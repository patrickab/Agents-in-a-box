"""Codec for storage-neutral durable sandbox records.

Manifests contain stable metadata and durable asset identifiers only. Runtime-local
paths and identifiers belong to the execution layer, not this record.
"""

from collections.abc import Mapping
import dataclasses
from pathlib import PurePosixPath, PureWindowsPath
import re
from types import MappingProxyType
from typing import Any, Literal

import yaml
from yaml.constructor import ConstructorError

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXECUTION_STATUSES = frozenset({"completed", "failed"})
_MANIFEST_FIELDS = frozenset(
    {"schema_version", "manifest_id", "profile", "runtime_fingerprint", "workspace", "outputs", "omp_sessions", "execution"}
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in mapping:
                    raise ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}",
                        key_node.start_mark,
                    )
                mapping[key] = self.construct_object(value_node, deep=deep)
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "mapping keys must be scalar",
                    key_node.start_mark,
                ) from error
        return mapping


def _require_identifier(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_asset_id(field: str, value: object, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    identifier = _require_identifier(field, value)
    path = PurePosixPath(identifier)
    if (
        "\\" in identifier
        or path.is_absolute()
        or PureWindowsPath(identifier).is_absolute()
        or any(component in {".", ".."} for component in identifier.split("/"))
    ):
        raise ValueError(f"{field} must be a durable asset ID, never a host path")
    return identifier


@dataclasses.dataclass(frozen=True)
class WorkspaceRef:
    """Reference the durable snapshot backing a workspace."""

    snapshot_asset_id: str | None
    sha256: str

    def __post_init__(self) -> None:
        _require_asset_id("snapshot_asset_id", self.snapshot_asset_id, allow_none=True)
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("workspace sha256 must be a 64-character lowercase hexadecimal digest")


@dataclasses.dataclass(frozen=True)
class Artifact:
    """Reference a durable artifact emitted by an execution."""

    asset_id: str
    mime_type: str

    def __post_init__(self) -> None:
        _require_asset_id("asset_id", self.asset_id)
        _require_identifier("mime_type", self.mime_type)


@dataclasses.dataclass(frozen=True)
class Outputs:
    """Reference durable storage for an execution's events and artifacts."""

    events_asset_id: str | None
    artifacts: tuple[Artifact, ...]

    def __post_init__(self) -> None:
        _require_asset_id("events_asset_id", self.events_asset_id, allow_none=True)
        if not isinstance(self.artifacts, tuple) or not all(isinstance(artifact, Artifact) for artifact in self.artifacts):
            raise ValueError("artifacts must be a tuple of Artifact values")


@dataclasses.dataclass(frozen=True)
class OmpSessionState:
    """Reference the durable state of one OMP session."""

    session_id: str
    state_asset_id: str | None

    def __post_init__(self) -> None:
        _require_identifier("session_id", self.session_id)
        _require_asset_id("state_asset_id", self.state_asset_id, allow_none=True)


@dataclasses.dataclass(frozen=True)
class Execution:
    """Record durable metadata for one execution; empty ``created_at`` means legacy unknown time."""

    run_id: str
    status: str
    exit_code: int | None
    created_at: str

    def __post_init__(self) -> None:
        _require_identifier("run_id", self.run_id)
        if not isinstance(self.status, str) or self.status not in _EXECUTION_STATUSES:
            raise ValueError(f"execution status must be one of {', '.join(sorted(_EXECUTION_STATUSES))}")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("execution exit_code must be an integer or null")
        if not isinstance(self.created_at, str):
            raise ValueError("execution created_at must be a string")


@dataclasses.dataclass(frozen=True)
class SandboxManifest:
    """Storage-neutral durable record for one sandbox execution."""

    schema_version: Literal[1]
    manifest_id: str
    profile: str
    runtime_fingerprint: str
    workspace: WorkspaceRef
    outputs: Outputs
    omp_sessions: Mapping[str, OmpSessionState]
    execution: Execution

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        _require_identifier("manifest_id", self.manifest_id)
        _require_identifier("profile", self.profile)
        _require_identifier("runtime_fingerprint", self.runtime_fingerprint)
        if not isinstance(self.workspace, WorkspaceRef):
            raise ValueError("workspace must be a WorkspaceRef")
        if not isinstance(self.outputs, Outputs):
            raise ValueError("outputs must be an Outputs")
        if not isinstance(self.omp_sessions, Mapping):
            raise ValueError("omp_sessions must be a mapping")
        sessions = {}
        for session_name, session_state in self.omp_sessions.items():
            _require_identifier("omp session name", session_name)
            if not isinstance(session_state, OmpSessionState):
                raise ValueError("omp_sessions values must be OmpSessionState values")
            sessions[session_name] = session_state
        if not isinstance(self.execution, Execution):
            raise ValueError("execution must be an Execution")
        object.__setattr__(self, "omp_sessions", MappingProxyType(sessions))

    def with_workspace_asset_id(self, asset_id: str | None) -> "SandboxManifest":
        """Return this manifest bound to a durable workspace asset."""
        workspace = WorkspaceRef(snapshot_asset_id=asset_id, sha256=self.workspace.sha256)
        return dataclasses.replace(self, workspace=workspace)


def manifest_to_dict(manifest: SandboxManifest) -> dict[str, Any]:
    """Encode a storage-neutral manifest as its strict mapping schema."""
    if not isinstance(manifest, SandboxManifest):
        raise ValueError("manifest must be a SandboxManifest")
    return {
        "schema_version": manifest.schema_version,
        "manifest_id": manifest.manifest_id,
        "profile": manifest.profile,
        "runtime_fingerprint": manifest.runtime_fingerprint,
        "workspace": {
            "snapshot_asset_id": manifest.workspace.snapshot_asset_id,
            "sha256": manifest.workspace.sha256,
        },
        "outputs": {
            "events_asset_id": manifest.outputs.events_asset_id,
            "artifacts": [{"asset_id": artifact.asset_id, "mime_type": artifact.mime_type} for artifact in manifest.outputs.artifacts],
        },
        "omp_sessions": {
            name: {"session_id": state.session_id, "state_asset_id": state.state_asset_id}
            for name, state in manifest.omp_sessions.items()
        },
        "execution": {
            "run_id": manifest.execution.run_id,
            "status": manifest.execution.status,
            "exit_code": manifest.execution.exit_code,
            "created_at": manifest.execution.created_at,
        },
    }


def to_yaml(manifest: SandboxManifest) -> str:
    """Encode a durable manifest with the version-1 YAML field layout."""
    return yaml.safe_dump(manifest_to_dict(manifest), sort_keys=False)


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_exact_fields(value: object, fields: frozenset[str], context: str) -> Mapping[str, Any]:
    mapping = _require_mapping(value, context)
    missing = fields - mapping.keys()
    if missing:
        raise ValueError(f"{context} missing required field {sorted(missing)[0]!r}")
    unknown = mapping.keys() - fields
    if unknown:
        raise ValueError(f"{context} has unsupported field {sorted(unknown, key=repr)[0]!r}")
    return mapping


def _dict_to_workspace_ref(value: object) -> WorkspaceRef:
    data = _require_exact_fields(value, frozenset({"snapshot_asset_id", "sha256"}), "workspace")
    return WorkspaceRef(snapshot_asset_id=data["snapshot_asset_id"], sha256=data["sha256"])


def _dict_to_artifact(value: object) -> Artifact:
    data = _require_exact_fields(value, frozenset({"asset_id", "mime_type"}), "artifact")
    return Artifact(asset_id=data["asset_id"], mime_type=data["mime_type"])


def _dict_to_outputs(value: object) -> Outputs:
    data = _require_exact_fields(value, frozenset({"events_asset_id", "artifacts"}), "outputs")
    artifacts_data = data["artifacts"]
    if not isinstance(artifacts_data, list):
        raise ValueError("outputs.artifacts must be a list")
    return Outputs(
        events_asset_id=data["events_asset_id"],
        artifacts=tuple(_dict_to_artifact(artifact) for artifact in artifacts_data),
    )


def _dict_to_omp_session_state(value: object) -> OmpSessionState:
    data = _require_exact_fields(value, frozenset({"session_id", "state_asset_id"}), "omp session")
    return OmpSessionState(session_id=data["session_id"], state_asset_id=data["state_asset_id"])


def _dict_to_execution(value: object) -> Execution:
    data = _require_exact_fields(value, frozenset({"run_id", "status", "exit_code", "created_at"}), "execution")
    return Execution(
        run_id=data["run_id"],
        status=data["status"],
        exit_code=data["exit_code"],
        created_at=data["created_at"],
    )


def manifest_from_dict(value: object) -> SandboxManifest:
    """Decode and strictly validate a storage-neutral manifest mapping."""
    root = _require_exact_fields(value, _MANIFEST_FIELDS, "manifest root")
    sessions_data = _require_mapping(root["omp_sessions"], "omp_sessions")
    sessions = {name: _dict_to_omp_session_state(state) for name, state in sessions_data.items()}
    return SandboxManifest(
        schema_version=root["schema_version"],
        manifest_id=root["manifest_id"],
        profile=root["profile"],
        runtime_fingerprint=root["runtime_fingerprint"],
        workspace=_dict_to_workspace_ref(root["workspace"]),
        outputs=_dict_to_outputs(root["outputs"]),
        omp_sessions=sessions,
        execution=_dict_to_execution(root["execution"]),
    )


def from_yaml(text: str) -> SandboxManifest:
    """Decode and strictly validate a version-1 durable manifest."""
    if not isinstance(text, str):
        raise ValueError("Manifest YAML must be text")
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Manifest YAML is invalid: {error}") from error
    return manifest_from_dict(data)
