"""Validate, hash, build, and read workspace tar archives, and safely extract them onto the host."""

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from fnmatch import fnmatchcase
import hashlib
import os
from pathlib import PurePosixPath
import posixpath
import shutil
import tarfile
from typing import BinaryIO

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

    @classmethod
    def from_directory(cls, source_dir: str, archive_path: str, exclude: Sequence[str] = (".git",)) -> "WorkspaceArchive":
        """Tar a host directory minus excluded top-level paths or globs, keeping symlinks as links, and pin it."""
        patterns = _normalize_exclusions(exclude)
        if not os.path.isdir(source_dir):
            raise ArchiveError(f"workspace source is not a directory: {source_dir!r}")
        try:
            _write_directory_archive(source_dir, archive_path, patterns)
            return cls.from_file(archive_path)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(archive_path)
            raise


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
                if archived_member_name(member.name) != path:
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


def _normalize_exclusions(exclude: Sequence[str]) -> tuple[str, ...]:
    """Return exclusion patterns as normalized relative POSIX paths or globs."""
    if isinstance(exclude, str):
        raise TypeError("exclude must be a sequence of patterns, not a string")
    patterns = []
    for pattern in exclude:
        require_nonempty_string("exclude pattern", pattern)
        normalized = posixpath.normpath(pattern)
        if normalized == "." or not is_safe_member_path(normalized):
            raise ValueError(f"exclude pattern must be a relative path or glob inside the workspace: {pattern!r}")
        patterns.append(normalized)
    return tuple(patterns)


def _is_excluded(relative_path: str, patterns: Sequence[str]) -> bool:
    """Match a workspace-relative path, or any of its ancestors, against fnmatch patterns in which `*` also crosses `/`."""
    parts = relative_path.split("/")
    return any(fnmatchcase("/".join(parts[:depth]), pattern) for depth in range(1, len(parts) + 1) for pattern in patterns)


def archived_member_name(member_name: str) -> str:
    """Strip the leading ./ that tar-created archives carry."""
    return posixpath.normpath(member_name)


def _write_directory_archive(source_dir: str, archive_path: str, patterns: Sequence[str]) -> None:
    """Stream one directory tree into an owner-only tar, stopping at the workspace size limit."""

    def add_tree(archive: tarfile.TarFile, output: BinaryIO, directory: str, relative_directory: str) -> None:
        with os.scandir(directory) as scanner:
            entries = sorted(scanner, key=lambda entry: entry.name)
        for entry in entries:
            relative_path = f"{relative_directory}/{entry.name}" if relative_directory else entry.name
            if _is_excluded(relative_path, patterns):
                continue
            is_directory = entry.is_dir(follow_symlinks=False)
            if not (is_directory or entry.is_file(follow_symlinks=False) or entry.is_symlink()):
                continue  # Sockets and FIFOs cannot be restored.
            if entry.inode() == archive_inode and os.path.samefile(entry.path, archive_path):
                continue  # The archive being written lives inside its own source.
            member = archive.gettarinfo(entry.path, arcname=f"./{relative_path}")
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            if member.isfile():
                with open(entry.path, "rb") as source:
                    archive.addfile(member, source)
            else:
                archive.addfile(member)
            if output.tell() > MAX_WORKSPACE_ARCHIVE_BYTES:
                raise ArchiveError(f"workspace archive exceeds {MAX_WORKSPACE_ARCHIVE_BYTES} byte limit")
            if is_directory:
                add_tree(archive, output, entry.path, relative_path)

    try:
        descriptor = os.open(archive_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        archive_inode = os.fstat(descriptor).st_ino
        with os.fdopen(descriptor, "wb") as output, tarfile.open(fileobj=output, mode="w") as archive:
            add_tree(archive, output, source_dir, "")
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError(f"Failed to archive workspace directory: {exc}") from exc


def extract_archive(archive_path: str, target_dir: str, exclude: Sequence[str] = ()) -> None:
    """Safely extract a validated archive into target_dir, deleting files it lacks except excluded patterns."""
    patterns = _normalize_exclusions(exclude)
    verify_archive(archive_path)
    try:
        os.makedirs(target_dir, exist_ok=True)
        root = os.path.realpath(target_dir)
        kept: set[str] = set()
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                name = archived_member_name(member.name)
                if name == "." or _is_excluded(name, patterns):
                    continue
                _extract_member(archive, member, name, root)
                kept.update(str(parent) for parent in PurePosixPath(name).parents if str(parent) != ".")
                kept.add(name)
        _prune_untracked(root, "", kept, patterns)
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ArchiveError(f"Failed to extract workspace archive: {exc}") from exc


def _remove_path(path: str) -> None:
    """Remove a file, link, or directory tree without following symlinks."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def _extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, name: str, root: str) -> None:
    """Write one validated member, replacing whatever occupies its path and refusing to write through symlinks."""
    parent = root
    for part in PurePosixPath(name).parts[:-1]:
        parent = os.path.join(parent, part)
        if os.path.islink(parent):
            raise ArchiveError(f"workspace archive member {name!r} would be written through a symlink")
        if os.path.lexists(parent) and not os.path.isdir(parent):
            os.unlink(parent)
        os.makedirs(parent, exist_ok=True)
    destination = os.path.join(root, name)
    if member.isdir():
        if os.path.lexists(destination) and (os.path.islink(destination) or not os.path.isdir(destination)):
            _remove_path(destination)
        os.makedirs(destination, exist_ok=True)
        os.chmod(destination, (member.mode & 0o777) | 0o700)
        return
    if os.path.lexists(destination):
        _remove_path(destination)
    if member.issym():
        os.symlink(member.linkname, destination)
    elif member.islnk():
        source = os.path.join(root, archived_member_name(member.linkname))
        if os.path.islink(source) or not os.path.isfile(source):
            raise ArchiveError(f"workspace archive hard link target is not an extracted regular file: {member.linkname!r}")
        os.link(source, destination)
    else:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, member.mode & 0o777)
        with os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(archive.extractfile(member), output, _READ_CHUNK_BYTES)
        os.chmod(destination, member.mode & 0o777)
        os.utime(destination, (member.mtime, member.mtime))


def _prune_untracked(root: str, relative_directory: str, kept: set[str], patterns: Sequence[str]) -> None:
    """Delete entries absent from the archive, keeping excluded paths and the directories that lead to them."""
    with os.scandir(os.path.join(root, relative_directory)) as scanner:
        entries = list(scanner)
    for entry in entries:
        relative_path = f"{relative_directory}/{entry.name}" if relative_directory else entry.name
        if _is_excluded(relative_path, patterns):
            continue
        if entry.is_dir(follow_symlinks=False):
            _prune_untracked(root, relative_path, kept, patterns)
            if relative_path not in kept and not os.listdir(entry.path):
                os.rmdir(entry.path)
        elif relative_path not in kept:
            os.unlink(entry.path)
