"""Test engine config, template parsing, the settings and agent precedence, and the user defaults."""

from pathlib import Path

from factory_helpers import write
import pytest

from agent_sandbox.factory.config import (
    DEFAULT_PROTECTED,
    ConfigError,
    EngineConfig,
    RepoConfig,
    WorkflowConfig,
    build_task_config,
    builtin_agent_text,
    expand_env,
    load_agent,
    load_repo_config,
    load_workflow,
    merge_settings,
    parse_agent_markdown,
)
from agent_sandbox.factory.models import FactorySettings, Overrides


def repo_config(**fields: object) -> RepoConfig:
    return RepoConfig(**fields)


def engine_config(tmp_path: Path) -> EngineConfig:
    return EngineConfig(database_url="postgresql://x/y", worktree_root=tmp_path / "worktrees", defaults_dir=tmp_path / "defaults")


def workflow(**fields: object) -> WorkflowConfig:
    return WorkflowConfig(name="w", phases=["code"], **fields)


def test_expand_env_substitutes_defaults_and_rejects_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORY_TEST_VALUE", "abc")
    monkeypatch.delenv("FACTORY_TEST_MISSING", raising=False)
    assert expand_env({"a": ["x-${FACTORY_TEST_VALUE}"], "b": "${FACTORY_TEST_MISSING:-fallback}", "c": 3}) == {
        "a": ["x-abc"],
        "b": "fallback",
        "c": 3,
    }
    with pytest.raises(ConfigError, match="FACTORY_TEST_MISSING"):
        expand_env("${FACTORY_TEST_MISSING}")


def test_settings_precedence_is_packaged_user_repo_workflow_overrides() -> None:
    packaged = FactorySettings()
    assert merge_settings(packaged, repo_config(), workflow(), Overrides()) == packaged

    user = FactorySettings(
        merge="auto", max_rounds=7, sandbox=False, models={"coder": "user-model", "scout": "user-scout", "reviewer": "user-review"}
    )
    repo = repo_config(max_rounds=5, sandbox=True, models={"coder": "repo-model", "reviewer": "repo-review"}, agent_timeout_s=60)
    flow = workflow(max_rounds=2, merge="review", models={"coder": "flow-model"})

    only_user = merge_settings(user, repo_config(), workflow(), Overrides())
    assert (only_user.merge, only_user.max_rounds, only_user.sandbox) == ("auto", 7, False)

    with_repo = merge_settings(user, repo, workflow(), Overrides())
    assert (with_repo.merge, with_repo.max_rounds, with_repo.sandbox, with_repo.agent_timeout_s) == ("auto", 5, True, 60)
    assert with_repo.models == {"coder": "repo-model", "scout": "user-scout", "reviewer": "repo-review"}

    with_workflow = merge_settings(user, repo, flow, Overrides())
    assert (with_workflow.merge, with_workflow.max_rounds) == ("review", 2)
    assert with_workflow.models == {"coder": "flow-model", "scout": "user-scout", "reviewer": "repo-review"}

    overridden = merge_settings(
        user, repo, flow, Overrides(max_rounds=1, merge="auto", sandbox=False, preview=False, models={"coder": "mine"})
    )
    assert (overridden.sandbox, overridden.max_rounds, overridden.merge, overridden.preview) == (False, 1, "auto", False)
    assert overridden.models == {"coder": "mine", "scout": "user-scout", "reviewer": "repo-review"}


@pytest.mark.parametrize(
    "text",
    [
        "no front matter",
        "---\ndescription: missing name\n---\nbody",
        "---\nname: x\noutput: ../escape.json\n---\nbody",
        "---\nname: x\noutput: src/out.json\n---\nbody",
        "---\nname: x\nprompt: smuggled\n---\nbody",
    ],
)
def test_parse_agent_markdown_rejects_invalid(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_agent_markdown(text, "agent.md")


def test_build_task_config_freezes_effective_settings(repo: Path, tmp_path: Path) -> None:
    write(repo / ".factory/workflows/full.yaml", "name: Full\nphases: [code]\nmax_rounds: 2\nmerge: auto\npreview: false\n")
    write(
        repo / "factory.yaml",
        (repo / "factory.yaml").read_text() + "profile: demo\nprotected: [secrets/**]\nmodels: {coder: repo-model}\n",
    )

    task = build_task_config(
        engine_config(tmp_path), "demo", repo, "full", Overrides(models={"coder": "override-model"}, sandbox=True), "main"
    )

    assert (task.merge, task.max_rounds, task.preview_enabled, task.sandbox, task.profile) == ("auto", 2, False, True, "demo")
    assert task.agents["code"].model == "override-model"
    assert task.protected == [*DEFAULT_PROTECTED, "secrets/**"]
    assert task.agents["code"].prompt == "You are the coder."
    assert task.worktree_root == str(tmp_path / "worktrees" / "demo")


def test_main_branch_defaults_to_the_checked_out_branch_and_factory_yaml_wins(bare_repo: Path, tmp_path: Path) -> None:
    engine = engine_config(tmp_path)

    assert build_task_config(engine, "bare", bare_repo, "code", Overrides(), "trunk").main_branch == "trunk"

    write(bare_repo / "factory.yaml", "main_branch: release\n")
    assert build_task_config(engine, "bare", bare_repo, "code", Overrides(), "trunk").main_branch == "release"
    assert build_task_config(engine, "bare", bare_repo, "code", Overrides(), None).main_branch == "release"


def test_detached_head_without_main_branch_is_an_error(bare_repo: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="detached"):
        build_task_config(engine_config(tmp_path), "bare", bare_repo, "code", Overrides(), None)


def test_agent_precedence_is_packaged_user_repo(bare_repo: Path, tmp_path: Path) -> None:
    defaults_dir = tmp_path / "defaults"

    assert load_agent(bare_repo, defaults_dir, "coder").prompt == parse_agent_markdown(builtin_agent_text("coder"), "coder").prompt

    write(defaults_dir / "agents/coder.md", "---\nname: coder\n---\nUser coder.\n")
    assert load_agent(bare_repo, defaults_dir, "coder").prompt == "User coder."
    assert load_agent(bare_repo, defaults_dir, "scout").name == "scout"  # untouched agents stay packaged

    write(bare_repo / ".omp/agents/coder.md", "---\nname: coder\n---\nRepo coder in {repo_path}.\n")
    assert load_agent(bare_repo, defaults_dir, "coder").prompt == f"Repo coder in {bare_repo}."
    assert load_agent(bare_repo, defaults_dir, "scout").name == "scout"


def test_workflow_precedence_is_packaged_then_repo_with_the_same_id(bare_repo: Path) -> None:
    assert load_workflow(bare_repo, "code").name == "Code"

    write(bare_repo / ".factory/workflows/code.yaml", "name: My code\nphases: [scout, code]\nmax_rounds: 9\n")
    write(bare_repo / ".factory/workflows/custom.yaml", "name: Custom\nphases: [code]\n")

    assert (load_workflow(bare_repo, "code").name, load_workflow(bare_repo, "code").max_rounds) == ("My code", 9)
    assert load_workflow(bare_repo, "custom").name == "Custom"


def test_repo_path_placeholder_is_expanded_in_hooks_gates_and_preview(repo: Path) -> None:
    write(
        repo / "factory.yaml",
        """
        main_branch: main
        sandbox: false
        hooks:
          after_create: [[uv, sync, --project, "{repo_path}"]]
        gates:
          - [uv, run, --project, "{repo_path}", pytest]
        preview:
          processes:
            - {name: web, argv: [serve, "{repo_path}/web", "{port.web}"], ready_url: "http://x:{port.web}/"}
          env: {DATA: "{repo_path}/data"}
          url: "http://x:{port.web}"
          api_url: "{repo_path}/api"
        """,
    )

    config = load_repo_config(repo)

    root = str(repo)
    assert config.hooks.after_create == [["uv", "sync", "--project", root]]
    assert config.gates == [["uv", "run", "--project", root, "pytest"]]
    assert config.preview is not None
    assert config.preview.processes[0].argv == ["serve", f"{root}/web", "{port.web}"]
    assert config.preview.env == {"DATA": f"{root}/data"}
    assert config.preview.api_url == f"{root}/api"
