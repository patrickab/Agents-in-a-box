"""Test workspace archive validation, pinning, and bounded member reads."""

import hashlib
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from agent_sandbox.archive import WorkspaceArchive, extract_archive, read_member, verify_archive
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


class _DirectoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = Path(scratch.name)
        self.source = self.scratch / "source"
        self.target = self.scratch / "target"
        self.source.mkdir()
        self.target.mkdir()

    def _archive(self, exclude: tuple[str, ...] = (".git",)) -> WorkspaceArchive:
        return WorkspaceArchive.from_directory(str(self.source), str(self.scratch / "workspace.tar"), exclude=exclude)

    def _tree(self, root: Path) -> dict[str, str]:
        entries = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                entries[relative] = f"-> {os.readlink(path)}"
            else:
                entries[relative] = "dir" if path.is_dir() else path.read_text()
        return entries


class TestFromDirectory(_DirectoryTestCase):
    def test_archives_relative_members_skips_git_by_default_and_keeps_symlinks_and_modes(self) -> None:
        (self.source / "src").mkdir()
        (self.source / "src" / "a.txt").write_text("a")
        (self.source / "run.sh").write_text("#!/bin/sh\n")
        (self.source / "run.sh").chmod(0o755)
        (self.source / "alias").symlink_to("src/a.txt")
        (self.source / ".git").mkdir()
        (self.source / ".git" / "HEAD").write_text("ref")
        (self.source / "src" / ".git").write_text("nested gitfile")

        pinned = self._archive()

        with tarfile.open(pinned.path) as archive:
            members = {member.name: member for member in archive}
        self.assertEqual(sorted(members), ["./alias", "./run.sh", "./src", "./src/.git", "./src/a.txt"])
        self.assertTrue(members["./alias"].issym())
        self.assertEqual(members["./alias"].linkname, "src/a.txt")
        self.assertEqual(members["./run.sh"].mode & 0o777, 0o755)
        self.assertEqual((members["./src/a.txt"].uid, members["./src/a.txt"].gid), (0, 0))
        self.assertEqual(oct(os.stat(pinned.path).st_mode & 0o777), "0o600")
        self.assertEqual(pinned, WorkspaceArchive.from_file(pinned.path))

    def test_exclude_accepts_nested_paths_and_globs_and_ignores_sockets(self) -> None:
        (self.source / ".factory" / "out").mkdir(parents=True)
        (self.source / ".factory" / "out" / "scout.json").write_text("{}")
        (self.source / ".factory" / "keep.yaml").write_text("keep")
        (self.source / "a.pyc").write_bytes(b"x")
        (self.source / "pkg").mkdir()
        (self.source / "pkg" / "b.pyc").write_bytes(b"x")
        (self.source / "pkg" / "b.py").write_text("b")
        os.mkfifo(self.source / "pipe")

        pinned = self._archive((".factory/out", "*.pyc"))

        with tarfile.open(pinned.path) as archive:
            self.assertEqual(sorted(archive.getnames()), ["./.factory", "./.factory/keep.yaml", "./pkg", "./pkg/b.py"])

    def test_escaping_symlink_is_refused_and_leaves_no_archive(self) -> None:
        (self.source / "escape").symlink_to("../outside")
        archive_path = self.scratch / "workspace.tar"

        with self.assertRaisesRegex(ArchiveError, "unsafe link target"):
            WorkspaceArchive.from_directory(str(self.source), str(archive_path))

        self.assertFalse(archive_path.exists())

    def test_size_cap_aborts_and_removes_the_partial_archive(self) -> None:
        (self.source / "big.bin").write_bytes(b"x" * 4096)
        archive_path = self.scratch / "workspace.tar"

        with (
            patch("agent_sandbox.archive.MAX_WORKSPACE_ARCHIVE_BYTES", 1024),
            self.assertRaisesRegex(ArchiveError, "exceeds 1024 byte limit"),
        ):
            WorkspaceArchive.from_directory(str(self.source), str(archive_path))

        self.assertFalse(archive_path.exists())


class TestExtractArchive(_DirectoryTestCase):
    def test_roundtrip_reproduces_the_tree_and_preserves_excluded_git(self) -> None:
        (self.source / "src").mkdir()
        (self.source / "src" / "a.txt").write_text("a")
        (self.source / "empty").mkdir()
        (self.source / "link").symlink_to("src/a.txt")
        (self.source / "script.sh").write_text("run")
        (self.source / "script.sh").chmod(0o755)
        (self.target / ".git").mkdir()
        (self.target / ".git" / "HEAD").write_text("ref")

        extract_archive(self._archive().path, str(self.target), exclude=(".git",))

        self.assertEqual(
            self._tree(self.target),
            {
                ".git": "dir",
                ".git/HEAD": "ref",
                "empty": "dir",
                "link": "-> src/a.txt",
                "script.sh": "run",
                "src": "dir",
                "src/a.txt": "a",
            },
        )
        self.assertEqual(os.stat(self.target / "script.sh").st_mode & 0o777, 0o755)

    def test_files_absent_from_the_archive_are_deleted_except_excluded_patterns(self) -> None:
        (self.source / "keep.txt").write_text("new")
        (self.target / "keep.txt").write_text("old")
        (self.target / "stale.txt").write_text("stale")
        (self.target / "stale_dir" / "deep").mkdir(parents=True)
        (self.target / "stale_dir" / "deep" / "f.txt").write_text("stale")
        (self.target / "stale_link").symlink_to("keep.txt")
        (self.target / ".factory" / "out").mkdir(parents=True)
        (self.target / ".factory" / "out" / "plan.json").write_text("{}")
        (self.target / ".factory" / "drop.txt").write_text("drop")
        (self.target / "cache.pyc").write_bytes(b"x")

        extract_archive(self._archive().path, str(self.target), exclude=(".factory/out", "*.pyc"))

        self.assertEqual(
            self._tree(self.target),
            {".factory": "dir", ".factory/out": "dir", ".factory/out/plan.json": "{}", "cache.pyc": "x", "keep.txt": "new"},
        )

    def test_excluded_members_in_the_archive_do_not_overwrite_the_target(self) -> None:
        (self.source / ".git").mkdir()
        (self.source / ".git" / "HEAD").write_text("from archive")
        (self.target / ".git").mkdir()
        (self.target / ".git" / "HEAD").write_text("real")

        extract_archive(self._archive(()).path, str(self.target), exclude=(".git",))

        self.assertEqual((self.target / ".git" / "HEAD").read_text(), "real")

    def test_replaces_entries_whose_type_changed(self) -> None:
        (self.source / "was_dir").write_text("file now")
        (self.source / "was_file" / "inner").mkdir(parents=True)
        (self.source / "was_link").write_text("file now")
        (self.target / "was_dir" / "old").mkdir(parents=True)
        (self.target / "was_file").write_text("file before")
        outside = self.scratch / "outside.txt"
        outside.write_text("untouched")
        (self.target / "was_link").symlink_to(outside)

        extract_archive(self._archive(()).path, str(self.target))

        self.assertEqual(
            self._tree(self.target), {"was_dir": "file now", "was_file": "dir", "was_file/inner": "dir", "was_link": "file now"}
        )
        self.assertEqual(outside.read_text(), "untouched")

    def test_rejects_unsafe_archives_before_touching_the_target(self) -> None:
        (self.target / "precious.txt").write_text("precious")
        for member, message in (
            (_member("../escape.txt"), "unsafe member path"),
            (_member("escape", tarfile.SYMTYPE, "../../outside"), "unsafe link target"),
            (_member("pipe", tarfile.FIFOTYPE), "unsupported member type"),
        ):
            with self.subTest(member=member.name):
                path = _write_archive((_member("fine.txt"), b"fine"), (member, b""))
                self.addCleanup(Path(path).unlink, missing_ok=True)
                with self.assertRaisesRegex(ArchiveError, message):
                    extract_archive(path, str(self.target))
                self.assertEqual(self._tree(self.target), {"precious.txt": "precious"})
                self.assertFalse((self.scratch / "escape.txt").exists())

    def test_refuses_to_write_through_a_symlink_that_leaves_the_target(self) -> None:
        outside = self.scratch / "outside"
        outside.mkdir()
        (self.target / "link").symlink_to(outside)
        path = _write_archive((_member("link/evil.txt"), b"evil"))
        self.addCleanup(Path(path).unlink, missing_ok=True)

        with self.assertRaisesRegex(ArchiveError, "through a symlink"):
            extract_archive(path, str(self.target))

        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
