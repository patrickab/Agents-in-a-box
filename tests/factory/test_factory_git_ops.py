"""Test worktrees, change tracking, path enforcement, and merging against a real temporary git repo."""

from pathlib import Path
import shutil

from factory_helpers import git, write
import pytest

from agent_sandbox.factory import git_ops

PROTECTED = ["factory.yaml", ".factory/**", ".omp/agents/**"]


@pytest.fixture
def worktree(repo: Path, tmp_path: Path) -> Path:
    info = git_ops.add_worktree(repo, tmp_path / "worktrees", "task00000001", "main")
    return info.path


def test_add_worktree_is_idempotent_and_remove_cleans_up(repo: Path, tmp_path: Path) -> None:
    first = git_ops.add_worktree(repo, tmp_path / "wt", "task00000001", "main")
    second = git_ops.add_worktree(repo, tmp_path / "wt", "task00000001", "main")

    assert first == second
    assert first.branch == "factory/task00000001"
    assert first.base_sha == git(repo, "rev-parse", "main").strip()
    assert (first.path / "hello.txt").is_file()

    git_ops.remove_worktree(repo, first.path, first.branch)
    assert not first.path.exists()
    assert "factory/task00000001" not in git(repo, "branch", "--list")
    assert "task00000001" not in git(repo, "worktree", "list")


def test_worktrees_lost_by_a_crash_are_readopted_and_removed_without_pruning_unrelated_ones(repo: Path, tmp_path: Path) -> None:
    git(repo, "worktree", "add", "-q", "-b", "elsewhere", str(tmp_path / "elsewhere"))
    shutil.rmtree(tmp_path / "elsewhere")  # Someone else's stale registration, not ours to prune.
    first = git_ops.add_worktree(repo, tmp_path / "wt", "task00000003", "main")
    shutil.rmtree(first.path)  # A crash lost the directory but left it registered.

    again = git_ops.add_worktree(repo, tmp_path / "wt", "task00000003", "main")
    assert again.path.is_dir() and again.branch == first.branch
    shutil.rmtree(again.path)
    git_ops.remove_worktree(repo, again.path, again.branch)

    listing = git(repo, "worktree", "list")
    assert "task00000003" not in listing and "elsewhere" in listing
    assert "factory/task00000003" not in git(repo, "branch", "--list")


def test_changed_files_include_untracked_and_ignore_engine_owned(worktree: Path) -> None:
    write(worktree / "hello.txt", "changed\n")
    write(worktree / "new/dir/file.py", "x = 1\n")
    write(worktree / ".factory/out/code.json", "{}")
    write(worktree / ".omp-session/s.jsonl", "{}")

    assert sorted(git_ops.changed_files(worktree)) == ["hello.txt", "new/dir/file.py"]
    assert "+changed" in git_ops.diff_text(worktree)
    assert "new/dir/file.py" in git_ops.diff_text(worktree)


@pytest.mark.parametrize(
    ("path", "patterns", "expected"),
    [
        ("src/a.py", ["**"], True),
        ("a.py", ["**/*.py"], True),
        ("src/deep/a.py", ["src/**"], True),
        ("src/deep/a.py", ["src/*.py"], False),
        ("factory.yaml", ["factory.yaml"], True),
        (".factory/workflows/x.yaml", [".factory/**"], True),
        ("other/.factory/x", [".factory/**"], False),
        ("a.txt", [], False),
    ],
)
def test_glob_matching(path: str, patterns: list[str], expected: bool) -> None:
    assert git_ops.matches_any(path, patterns) is expected


def test_allowlist_breach_is_reverted_to_the_pre_turn_state(worktree: Path) -> None:
    write(worktree / "src/keep.py", "legit = 1\n")
    before = git_ops.snapshot_tree(worktree)

    # The "agent turn": legit change, an edit to a protected file, a new protected file, a deleted tracked file, and a mode change.
    write(worktree / "src/keep.py", "legit = 2\n")
    write(worktree / "factory.yaml", "gates: [[rm, -rf, /]]\n")
    write(worktree / ".omp/agents/evil.md", "---\nname: evil\n---\n")
    (worktree / "hello.txt").unlink()
    after = git_ops.snapshot_tree(worktree)
    changed = git_ops.tree_changes(worktree, before, after)
    breaches = git_ops.find_breaches(changed, ["**"], PROTECTED)

    assert sorted(changed) == [".omp/agents/evil.md", "factory.yaml", "hello.txt", "src/keep.py"]
    assert sorted(breaches) == [".omp/agents/evil.md", "factory.yaml"]

    git_ops.restore_paths(worktree, before, breaches)
    assert not (worktree / ".omp/agents/evil.md").exists()
    assert sorted(path.name for path in (worktree / ".omp/agents").iterdir()) == ["coder.md"]
    assert "gates: [[rm" not in (worktree / "factory.yaml").read_text()
    assert (worktree / "src/keep.py").read_text() == "legit = 2\n"
    assert not (worktree / "hello.txt").exists()


def test_restore_paths_recovers_deleted_files_and_executable_bit(worktree: Path) -> None:
    script = worktree / "run.sh"
    write(script, "#!/bin/sh\n")
    script.chmod(0o755)
    before = git_ops.snapshot_tree(worktree)

    script.unlink()
    (worktree / "hello.txt").write_text("vandalized\n")
    git_ops.restore_paths(worktree, before, ["run.sh", "hello.txt"])

    assert script.read_text() == "#!/bin/sh\n" and script.stat().st_mode & 0o111
    assert (worktree / "hello.txt").read_text() == "hello\n"


def test_commit_changes_skips_protected_and_engine_owned(worktree: Path, repo: Path) -> None:
    write(worktree / "feature.py", "ok = True\n")
    write(worktree / "factory.yaml", "tampered: true\n")
    write(worktree / ".factory/out/code.json", "{}")

    assert git_ops.commit_changes(worktree, "Add\nfeature", PROTECTED) is True

    assert git(worktree, "log", "-1", "--format=%s").strip() == "factory: Add feature"
    assert git(worktree, "show", "--name-only", "--format=", "HEAD").split() == ["feature.py"]
    assert git_ops.commit_changes(worktree, "again", PROTECTED) is False


def test_merge_refuses_a_main_checkout_on_another_branch(worktree: Path, repo: Path) -> None:
    write(worktree / "feature.py", "ok = True\n")
    git_ops.commit_changes(worktree, "Feature", PROTECTED)
    git(repo, "checkout", "-q", "-b", "side")

    outcome = git_ops.fast_forward_main(repo, "factory/task00000001", "main")

    assert not outcome.ok and "not main" in outcome.error
