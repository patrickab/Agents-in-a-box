"""Test workspace archive validation, pinning, and bounded member reads."""

import hashlib
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from agent_sandbox.archive import WorkspaceArchive, read_member, verify_archive
from agent_sandbox.errors import ArchiveError


def _write_archive(*entries: tuple[tarfile.TarInfo, bytes]) -> str:
    descriptor, path = tempfile.mkstemp(suffix=".tar")
    os.close(descriptor)
    with tarfile.open(path, "w") as archive:
        for member, content in entries:
            if member.isfile():
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            else:
                archive.addfile(member)
    return path


def _member(name: str, member_type: bytes = tarfile.REGTYPE, linkname: str = "") -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = member_type
    member.linkname = linkname
    return member


class TestVerifyArchive(unittest.TestCase):
    def tearDown(self) -> None:
        for path in getattr(self, "_paths", ()):
            Path(path).unlink(missing_ok=True)

    def _archive(self, *entries: tuple[tarfile.TarInfo, bytes]) -> str:
        path = _write_archive(*entries)
        self._paths = [*getattr(self, "_paths", ()), path]
        return path

    def test_pins_a_safe_archive_to_its_byte_digest(self) -> None:
        path = self._archive(
            (_member("dir", tarfile.DIRTYPE), b""),
            (_member("dir/file.txt"), b"ok"),
            (_member("dir/link", tarfile.SYMTYPE, "file.txt"), b""),
        )

        pinned = WorkspaceArchive.from_file(path)

        self.assertEqual(pinned, WorkspaceArchive(path, hashlib.sha256(Path(path).read_bytes()).hexdigest()))

    def test_rejects_unsafe_members(self) -> None:
        for member, message in (
            (_member("../escape"), "unsafe member path"),
            (_member("/absolute"), "unsafe member path"),
            (_member("pipe", tarfile.FIFOTYPE), "unsupported member type"),
            (_member("absolute-link", tarfile.SYMTYPE, "/outside"), "unsafe link target"),
            (_member("nested/escape-link", tarfile.SYMTYPE, "../../outside"), "unsafe link target"),
            (_member("hard-link", tarfile.LNKTYPE, "../outside"), "unsafe link target"),
        ):
            with self.subTest(member=member.name), self.assertRaisesRegex(ArchiveError, message):
                verify_archive(self._archive((member, b"")))

    def test_rejects_oversized_and_unreadable_archives(self) -> None:
        path = self._archive((_member("file.txt"), b"x" * 10))
        with (
            patch("agent_sandbox.archive.MAX_WORKSPACE_ARCHIVE_BYTES", 10),
            self.assertRaisesRegex(ArchiveError, "exceeds 10 byte limit"),
        ):
            verify_archive(path)
        with self.assertRaisesRegex(ArchiveError, "Refusing workspace archive"):
            verify_archive(path + ".missing")


class TestReadMember(unittest.TestCase):
    def _read(
        self, entries: tuple[tuple[tarfile.TarInfo, bytes], ...], relative_path: str = "result.txt", max_bytes: int = 1024
    ) -> bytes:
        path = _write_archive(*entries)
        try:
            return read_member(path, relative_path, max_bytes)
        finally:
            os.unlink(path)

    def test_reads_regular_and_nested_members_with_conventional_prefix(self) -> None:
        entries = ((_member("./result.txt"), b"root"), (_member("./nested/result.txt"), b"nested"))
        self.assertEqual(self._read(entries), b"root")
        self.assertEqual(self._read(entries, "nested/result.txt"), b"nested")

    def test_rejects_invalid_paths_and_limits(self) -> None:
        entries = ((_member("result.txt"), b"result"),)
        for relative_path in ("../result.txt", "/result.txt", "./result.txt", "", "."):
            with self.subTest(relative_path=relative_path), self.assertRaises(ValueError):
                self._read(entries, relative_path)
        for limit, error in ((0, ValueError), (-1, ValueError), (1.0, TypeError), (True, TypeError)):
            with self.subTest(limit=limit), self.assertRaises(error):
                self._read(entries, max_bytes=limit)  # type: ignore[arg-type]

    def test_rejects_missing_duplicate_nonregular_unsafe_and_oversized_members(self) -> None:
        for entries, message in (
            (((_member("other.txt"), b"x"),), "does not contain"),
            (((_member("result.txt"), b"first"), (_member("result.txt"), b"second")), "duplicate"),
            (((_member("result.txt", tarfile.DIRTYPE), b""),), "regular file"),
            (((_member("result.txt", tarfile.SYMTYPE, "target.txt"), b""),), "regular file"),
            (((_member("result.txt", tarfile.FIFOTYPE), b""),), "unsupported member type"),
            (((_member("../result.txt"), b""),), "unsafe member path"),
            (((_member("result.txt"), b"toolong"),), "exceeds 3 byte limit"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ArchiveError, message):
                self._read(entries, max_bytes=3)

    def test_rejects_truncated_member(self) -> None:
        path = _write_archive((_member("result.txt"), b"result"))
        os.truncate(path, 513)
        try:
            with self.assertRaises(ArchiveError):
                read_member(path, "result.txt", 1024)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
