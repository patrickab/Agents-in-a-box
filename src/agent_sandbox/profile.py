"""Load and validate sandbox runtime profiles."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
from pathlib import Path
import posixpath
import re
from types import MappingProxyType
from typing import Any, Literal

import yaml

from agent_sandbox._validation import is_safe_name, load_strict_yaml, require_exact_fields, require_mapping
from agent_sandbox.config import PROFILE_SEARCH_PATH, RUNTIME_POLICY_DESCRIPTOR
from agent_sandbox.errors import ProfileError

_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUNTIME_OWNED_TARGET = "/runtime"
_OPTIONAL_FIELDS = frozenset({"python_interpreters", "env_passthrough", "host_services"})
_PROFILE_FIELDS = frozenset({"name", "image", "workdir", "mounts", "omp_binary"}) | _OPTIONAL_FIELDS


@dataclass(frozen=True)
class Mount:
    """Declare one bind mount."""

    source: str
    target: str
    mode: Literal["ro", "rw"]

    @property
    def read_only(self) -> bool:
        return self.mode == "ro"


@dataclass(frozen=True)
class Profile:
    """Declare the image, mounts, and runtime policy."""

    name: str
    image: str
    workdir: str
    mounts: tuple[Mount, ...]
    omp_binary: str
    python_interpreters: Mapping[str, str] = field(default_factory=dict)  # Label -> absolute interpreter path.
    env_passthrough: tuple[str, ...] = ()
    host_services: Mapping[str, int] = field(default_factory=dict)  # Environment variable -> host port.

    def __post_init__(self) -> None:
        object.__setattr__(self, "python_interpreters", MappingProxyType(dict(sorted(self.python_interpreters.items()))))
        object.__setattr__(self, "host_services", MappingProxyType(dict(sorted(self.host_services.items()))))

    def workdir_mount_paths(self) -> tuple[str, ...]:
        """Return the workdir-relative paths of mounts inside the workdir; they are neither restored nor captured."""
        prefix = self.workdir.rstrip("/") + "/"
        return tuple(sorted({mount.target[len(prefix) :] for mount in self.mounts if mount.target.startswith(prefix)}))

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
                    ({"source": mount.source, "target": mount.target, "mode": mount.mode} for mount in self.mounts),
                    key=lambda mount: (mount["source"], mount["target"], mount["mode"]),
                ),
                "omp_binary": self.omp_binary,
                "python_interpreters": dict(self.python_interpreters),
                "env_passthrough": sorted(self.env_passthrough),
                "host_services": dict(self.host_services),
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def load_profile(name: str) -> Profile:
    """Load and validate the named profile from the first search-path directory that has it."""
    if not is_safe_name(name):
        raise ProfileError("profile name must be a safe profile name")
    candidates = [Path(directory) / f"{name}.yaml" for directory in PROFILE_SEARCH_PATH]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise ProfileError(f"profile '{name}' not found; searched: {', '.join(map(str, candidates))}")
    try:
        with path.open("r", encoding="utf-8") as profile_file:
            return _parse_profile(name, load_strict_yaml(profile_file))
    except yaml.YAMLError as error:
        raise ProfileError(f"profile '{name}' contains invalid YAML: {error}") from error
    except ValueError as error:
        raise ProfileError(f"profile '{name}' {error}") from error


def _parse_profile(name: str, data: object) -> Profile:
    root = require_exact_fields(data, _PROFILE_FIELDS, "root", optional=_OPTIONAL_FIELDS)
    if not is_safe_name(root["name"]):
        raise ValueError("name must be a safe profile name")
    if root["name"] != name:
        raise ValueError(f"name field mismatch: got '{root['name']}'")
    image = root["image"]
    if not isinstance(image, str) or not image:
        raise ValueError("image must be a nonempty string")
    workdir = _require_absolute_path("workdir", root["workdir"])
    return Profile(
        name=name,
        image=image,
        workdir=workdir,
        mounts=_parse_mounts(root["mounts"], workdir),
        omp_binary=_require_absolute_path("omp_binary", root["omp_binary"]),
        python_interpreters=_parse_python_interpreters(root.get("python_interpreters", {})),
        env_passthrough=_parse_env_passthrough(root.get("env_passthrough", [])),
        host_services=_parse_host_services(root.get("host_services", {})),
    )


def _require_absolute_path(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.startswith("/") or any(part in {".", ".."} for part in value.split("/")):
        raise ValueError(f"{field} must be an absolute path without traversal")
    return value


def _targets_conflict(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _parse_mounts(data: object, workdir: str) -> tuple[Mount, ...]:
    if not isinstance(data, list):
        raise ValueError("mounts must be a list")
    mounts: list[Mount] = []
    for index, mount_data in enumerate(data):
        mount = require_exact_fields(mount_data, frozenset({"source", "target", "mode"}), f"mount {index}")
        source = _require_absolute_path(f"mount {index} source", mount["source"])
        target = _require_absolute_path(f"mount {index} target", mount["target"])
        mode = mount["mode"]
        if not isinstance(mode, str) or mode not in {"ro", "rw"}:
            raise ValueError(f"mount target '{target}' has invalid mode {mode!r} (must be 'ro' or 'rw')")
        normalized = posixpath.normpath(target)
        if normalized == _RUNTIME_OWNED_TARGET or normalized.startswith(_RUNTIME_OWNED_TARGET + "/"):
            raise ValueError(f"mount target '{target}' is owned by the runtime")
        if normalized == posixpath.normpath(workdir):
            raise ValueError(f"mount target '{target}' must not replace the managed workspace root")
        for prior in mounts:
            if _targets_conflict(normalized, posixpath.normpath(prior.target)):
                raise ValueError(f"mount target '{target}' conflicts with mount target '{prior.target}'")
        mounts.append(Mount(source=source, target=target, mode=mode))
    return tuple(mounts)


def _parse_python_interpreters(data: object) -> dict[str, str]:
    interpreters = require_mapping(data, "python_interpreters")
    if not all(is_safe_name(label) for label in interpreters):
        raise ValueError("python interpreter labels must be safe names")
    return {label: _require_absolute_path(f"python_interpreters.{label}", path) for label, path in interpreters.items()}


def _parse_env_passthrough(data: object) -> tuple[str, ...]:
    if not isinstance(data, list) or not all(isinstance(value, str) for value in data):
        raise ValueError("env_passthrough must be a list of strings")
    if not all(_ENVIRONMENT_NAME_PATTERN.fullmatch(value) for value in data):
        raise ValueError("env_passthrough must contain environment variable names")
    if len(set(data)) != len(data):
        raise ValueError("env_passthrough must not contain duplicates")
    return tuple(data)


def _parse_host_services(data: object) -> dict[str, int]:
    services: Mapping[Any, Any] = require_mapping(data, "host_services")
    for service_name, port in services.items():
        if not isinstance(service_name, str) or not _ENVIRONMENT_NAME_PATTERN.fullmatch(service_name):
            raise ValueError("host_services must use environment variable names")
        if type(port) is not int or not 0 < port < 65536:
            raise ValueError("host_services must map environment variable names to host ports")
    return dict(services)
