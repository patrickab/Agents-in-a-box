"""Validate, hash, and read workspace tar archives without extracting them on the host."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import PurePosixPath
import posixpath
import tarfile

from agent_sandbox._validation import require_nonempty_string, require_sha256
from agent_sandbox.config import MAX_WORKSPACE_ARCHIVE_BYTES
from agent_sandbox.errors import ArchiveError

_HASH_CHUNK_BYTES = 1 << 20
_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class WorkspaceArchive:
    """Pin one local workspace tar to the SHA-256 its bytes must match."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        require_nonempty_string("workspace archive path", self.path)
        require_sha256("workspace archive sha256", self.sha256)

    @classmethod
    def from_file(cls, path: str) -> "WorkspaceArchive":
        """Validate a local archive and pin it to its current digest."""
        return cls(path, verify_archive(path))


def is_safe_member_path(path: str) -> bool:
    return bool(path) and not path.startswith("/") and ".." not in PurePosixPath(path).parts


def validate_member(member: tarfile.TarInfo) -> None:
    """Accept only regular files, directories, and links that stay inside the archive root."""
    if not is_safe_member_path(member.name):
        raise ArchiveError(f"workspace archive contains unsafe member path: {member.name!r}")
    if member.issym() or member.islnk():
        target = member.linkname
        base = posixpath.dirname(member.name) if member.issym() else ""
        if not target or target.startswith("/") or not is_safe_member_path(posixpath.normpath(posixpath.join(base, target))):
            raise ArchiveError(f"workspace archive contains unsafe link target: {target!r}")
    elif not member.isfile() and not member.isdir():
        raise ArchiveError(f"workspace archive contains unsupported member type: {member.name!r}")


def verify_archive(path: str) -> str:
    """Validate every member of a bounded tar and return the SHA-256 of its bytes."""
    try:
        if os.path.getsize(path) > MAX_WORKSPACE_ARCHIVE_BYTES:
            raise ArchiveError(f"workspace archive exceeds {MAX_WORKSPACE_ARCHIVE_BYTES} byte limit")
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                validate_member(member)
        digest = hashlib.sha256()
        size = 0
        with open(path, "rb") as source:
            while chunk := source.read(_HASH_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_WORKSPACE_ARCHIVE_BYTES:
                    raise ArchiveError(f"workspace archive exceeds {MAX_WORKSPACE_ARCHIVE_BYTES} byte limit")
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError(f"Refusing workspace archive: {exc}") from exc


def _require_member_path(relative_path: object) -> str:
    if not isinstance(relative_path, str):
        raise TypeError("relative_path must be a string")
    if (
        "\\" in relative_path
        or "\x00" in relative_path
        or relative_path in {"", "."}
        or relative_path != str(PurePosixPath(relative_path))
        or not is_safe_member_path(relative_path)
    ):
        raise ValueError("relative_path must be a normalized relative POSIX path")
    return relative_path


def read_member(archive_path: str, relative_path: str, max_bytes: int) -> bytes:
    """Read one bounded regular file from a validated workspace archive."""
    path = _require_member_path(relative_path)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be a positive integer")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            match = None
            for member in archive:
                validate_member(member)
                if member.name.removeprefix("./") != path:
                    continue
                if match is not None:
                    raise ArchiveError(f"workspace archive contains duplicate member: {path!r}")
                match = member
            if match is None:
                raise ArchiveError(f"workspace archive does not contain {path!r}")
            if not match.isfile():
                raise ArchiveError(f"workspace archive member is not a regular file: {path!r}")
            if match.size > max_bytes:
                raise ArchiveError(f"workspace archive member exceeds {max_bytes} byte limit")
            source = archive.extractfile(match)
            content = bytearray()
            while chunk := source.read(min(_READ_CHUNK_BYTES, max_bytes - len(content) + 1)):
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise ArchiveError(f"workspace archive member exceeds {max_bytes} byte limit")
            if len(content) != match.size:
                raise ArchiveError(f"workspace archive member is truncated: {path!r}")
            return bytes(content)
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ArchiveError(f"Refusing workspace archive: {exc}") from exc
