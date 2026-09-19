"""Load and validate sandbox runtime profiles."""

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
import posixpath
import re

import yaml
from yaml.constructor import ConstructorError

from agent_sandbox.config import PROFILE_SEARCH_PATH, RUNTIME_POLICY_DESCRIPTOR

_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUNTIME_OWNED_TARGET = "/runtime"
_REQUIRED_PROFILE_FIELDS = frozenset({"name", "image", "workdir", "mounts", "workspace", "omp_binary"})
_OPTIONAL_PROFILE_FIELDS = frozenset({"python_interpreters", "env_passthrough", "host_services"})


class ProfileError(Exception):
    """Report an invalid or unavailable profile."""


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


@dataclass(frozen=True)
class Mount:
    """Declare one bind mount."""

    source: str
    target: str
    mode: str


@dataclass(frozen=True)
class WorkspaceConfig:
    """Declare managed workspace settings."""

    mode: str


@dataclass(frozen=True)
class Profile:
    """Declare the image, mounts, and runtime policy."""

    name: str
    image: str
    workdir: str
    mounts: tuple[Mount, ...]
    workspace: WorkspaceConfig
    omp_binary: str
    python_interpreters: tuple[tuple[str, str], ...] = ()
    env_passthrough: tuple[str, ...] = ()
    host_services: tuple[tuple[str, int], ...] = ()

    @cached_property
    def runtime_fingerprint(self) -> str:
        """Return the canonical runtime identity without reading secret values."""
        payload = {
            "runtime_policy": RUNTIME_POLICY_DESCRIPTOR,
            "profile": {
                "name": self.name,
                "image": self.image,
                "workdir": self.workdir,
                "mounts": sorted(
                    (
                        {"source": mount.source, "target": mount.target, "mode": mount.mode}
                        for mount in self.mounts
                    ),
                    key=lambda mount: (mount["source"], mount["target"], mount["mode"]),
                ),
                "workspace": {"mode": self.workspace.mode},
                "omp_binary": self.omp_binary,
                "python_interpreters": dict(sorted(self.python_interpreters)),
                "env_passthrough": sorted(self.env_passthrough),
                "host_services": dict(sorted(self.host_services)),
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _is_safe_profile_name(name: object) -> bool:
    return isinstance(name, str) and bool(_PROFILE_NAME_PATTERN.fullmatch(name))


def _is_absolute_path(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("/")
        and all(component not in {".", ".."} for component in value.split("/"))
    )


def _require_absolute_path(name: str, field: str, value: object) -> str:
    if not _is_absolute_path(value):
        raise ProfileError(f"profile '{name}' {field} must be an absolute path without traversal")
    return value


def _require_exact_fields(
    name: str, value: object, fields: frozenset[str], context: str, *, optional_fields: frozenset[str] = frozenset()
) -> dict:
    if not isinstance(value, dict):
        raise ProfileError(f"profile '{name}' {context} must be a mapping")
    missing = fields - optional_fields - value.keys()
    if missing:
        field = sorted(missing)[0]
        raise ProfileError(f"profile '{name}' {context} missing required field '{field}'")
    unknown = value.keys() - fields
    if unknown:
        field = sorted(unknown, key=repr)[0]
        raise ProfileError(f"profile '{name}' {context} has unsupported field {field!r}")
    return value


def _normalized_path(path: str) -> str:
    return posixpath.normpath(path)


def _mount_targets_conflict(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _find_profile_path(name: str) -> Path:
    """Return the first profile file found in search-path order."""
    candidates = [Path(directory) / f"{name}.yaml" for directory in PROFILE_SEARCH_PATH]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise ProfileError(f"profile '{name}' not found; searched: {searched}")
    return path


def _read_profile_yaml(path: Path, name: str) -> object:
    """Read profile YAML while translating parser failures to ProfileError."""
    try:
        with path.open("r", encoding="utf-8") as profile_file:
            return yaml.load(profile_file, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ProfileError(f"profile '{name}' contains invalid YAML: {error}") from error


def _profile_from_mapping(name: str, data: object) -> Profile:
    """Validate a parsed profile mapping and construct its immutable model."""
    root = _require_exact_fields(
        name,
        data,
        _REQUIRED_PROFILE_FIELDS | _OPTIONAL_PROFILE_FIELDS,
        "root",
        optional_fields=_OPTIONAL_PROFILE_FIELDS,
    )
    if not _is_safe_profile_name(root["name"]):
        raise ProfileError(f"profile '{name}' name must be a safe profile name")
    if root["name"] != name:
        raise ProfileError(f"profile '{name}' name field mismatch: got '{root['name']}'")

    image = root["image"]
    if not isinstance(image, str) or not image:
        raise ProfileError(f"profile '{name}' image must be a nonempty string")
    workdir = _require_absolute_path(name, "workdir", root["workdir"])
    omp_binary = _require_absolute_path(name, "omp_binary", root["omp_binary"])

    workspace_data = _require_exact_fields(name, root["workspace"], frozenset({"mode"}), "workspace")
    if workspace_data["mode"] != "managed":
        raise ProfileError(f"profile '{name}' workspace mode must be 'managed'")
    workspace = WorkspaceConfig(mode="managed")

    mounts_data = root["mounts"]
    if not isinstance(mounts_data, list):
        raise ProfileError(f"profile '{name}' mounts must be a list")
    mounts = []
    normalized_targets: list[tuple[str, str]] = []
    for index, mount_data in enumerate(mounts_data):
        mount = _require_exact_fields(name, mount_data, frozenset({"source", "target", "mode"}), f"mount {index}")
        source = _require_absolute_path(name, f"mount {index} source", mount["source"])
        target = _require_absolute_path(name, f"mount {index} target", mount["target"])
        mode = mount["mode"]
        if not isinstance(mode, str) or mode not in {"ro", "rw"}:
            raise ProfileError(f"profile '{name}' mount target '{target}' has invalid mode {mode!r} (must be 'ro' or 'rw')")
        normalized_target = _normalized_path(target)
        if normalized_target == _RUNTIME_OWNED_TARGET or normalized_target.startswith(_RUNTIME_OWNED_TARGET + "/"):
            raise ProfileError(f"profile '{name}' mount target '{target}' is owned by the runtime")
        if normalized_target == _normalized_path(workdir):
            raise ProfileError(f"profile '{name}' mount target '{target}' must not replace the managed workspace root")
        for prior_target, prior_raw_target in normalized_targets:
            if _mount_targets_conflict(normalized_target, prior_target):
                raise ProfileError(
                    f"profile '{name}' mount target '{target}' conflicts with mount target '{prior_raw_target}'"
                )
        normalized_targets.append((normalized_target, target))
        mounts.append(Mount(source=source, target=target, mode=mode))

    python_data = root.get("python_interpreters", {})
    if not isinstance(python_data, dict):
        raise ProfileError(f"profile '{name}' python_interpreters must be a mapping")
    python_interpreters = []
    for label, interpreter_path in python_data.items():
        if not _is_safe_profile_name(label):
            raise ProfileError(f"profile '{name}' python interpreter labels must be safe names")
        python_interpreters.append((label, _require_absolute_path(name, f"python_interpreters.{label}", interpreter_path)))

    env_passthrough_data = root.get("env_passthrough", [])
    if not isinstance(env_passthrough_data, list) or not all(isinstance(value, str) for value in env_passthrough_data):
        raise ProfileError(f"profile '{name}' env_passthrough must be a list of strings")
    if not all(_ENVIRONMENT_NAME_PATTERN.fullmatch(value) for value in env_passthrough_data):
        raise ProfileError(f"profile '{name}' env_passthrough must contain environment variable names")
    if len(set(env_passthrough_data)) != len(env_passthrough_data):
        raise ProfileError(f"profile '{name}' env_passthrough must not contain duplicates")

    host_services_data = root.get("host_services", {})
    if not isinstance(host_services_data, dict):
        raise ProfileError(f"profile '{name}' host_services must be a mapping")
    host_services = []
    for service_name, port in host_services_data.items():
        if not isinstance(service_name, str) or not _ENVIRONMENT_NAME_PATTERN.fullmatch(service_name):
            raise ProfileError(f"profile '{name}' host_services must use environment variable names")
        if type(port) is not int or not 0 < port < 65536:
            raise ProfileError(f"profile '{name}' host_services must map environment variable names to host ports")
        host_services.append((service_name, port))

    return Profile(
        name=name,
        image=image,
        workdir=workdir,
        mounts=tuple(mounts),
        workspace=workspace,
        omp_binary=omp_binary,
        python_interpreters=tuple(sorted(python_interpreters)),
        env_passthrough=tuple(env_passthrough_data),
        host_services=tuple(sorted(host_services)),
    )


def load_profile(name: str) -> Profile:
    """Load and validate the named profile."""
    if not _is_safe_profile_name(name):
        raise ProfileError("profile name must be a safe profile name")
    path = _find_profile_path(name)
    return _profile_from_mapping(name, _read_profile_yaml(path, name))
