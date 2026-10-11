"""Share strict validation helpers for profile, manifest, and invocation records."""

from collections.abc import Mapping
import re
from typing import IO, Any

import yaml
from yaml.constructor import ConstructorError

SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate and non-scalar mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in mapping:
                    raise ConstructorError(
                        "while constructing a mapping", node.start_mark, f"found duplicate key {key!r}", key_node.start_mark
                    )
                mapping[key] = self.construct_object(value_node, deep=deep)
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping", node.start_mark, "mapping keys must be scalar", key_node.start_mark
                ) from error
        return mapping


def load_strict_yaml(source: str | IO[str]) -> object:
    """Parse safe YAML, rejecting duplicate mapping keys."""
    return yaml.load(source, Loader=_UniqueKeyLoader)


def is_safe_name(value: object) -> bool:
    return isinstance(value, str) and bool(SAFE_NAME_PATTERN.fullmatch(value))


def require_nonempty_string(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def require_optional_string(field: str, value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field} must be a string or None")
    return value


def require_sha256(field: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{field} must be a 64-character lowercase hexadecimal digest")
    return value


def require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def require_exact_fields(
    value: object, fields: frozenset[str], context: str, *, optional: frozenset[str] = frozenset()
) -> Mapping[str, Any]:
    """Require a mapping with every non-optional field and no unknown field."""
    mapping = require_mapping(value, context)
    missing = fields - optional - mapping.keys()
    if missing:
        raise ValueError(f"{context} missing required field {sorted(missing)[0]!r}")
    unknown = mapping.keys() - fields
    if unknown:
        raise ValueError(f"{context} has unsupported field {sorted(unknown, key=repr)[0]!r}")
    return mapping
