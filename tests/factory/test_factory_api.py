"""Test the /v1 API: task creation, error shapes, and state-transition conflicts (needs the factory test database)."""

import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from factory_helpers import FakePreviews, RecordingDispatcher, git
from fastapi.testclient import TestClient
import psycopg
import pytest

from agent_sandbox.factory.api import create_app
from agent_sandbox.factory.config import EngineConfig
from agent_sandbox.factory.engine import Engine
from agent_sandbox.factory.store import Store

TASK_BODY = {"repo": "demo", "workflow": "code", "title": "Add feature", "request": "please"}


@pytest.fixture
def dispatcher() -> RecordingDispatcher:
    return RecordingDispatcher()


@pytest.fixture
def previews() -> FakePreviews:
    return FakePreviews()


@pytest.fixture
def store(database_url: str) -> Store:
    return Store(database_url)


@pytest.fixture
def client(
    repo: Path, tmp_path: Path, database_url: str, store: Store, dispatcher: RecordingDispatcher, previews: FakePreviews
) -> Iterator[TestClient]:
    config = EngineConfig(
        database_url=database_url,
        state_dir=tmp_path / "state",
        worktree_root=tmp_path / "worktrees",
        defaults_dir=tmp_path / "defaults",
    )
    store.register_repo(str(repo), "demo")
    with TestClient(create_app(Engine(config, store, previews, dispatcher)), raise_server_exceptions=False) as test_client:
        yield test_client


def create(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post("/v1/tasks", json={**TASK_BODY, **overrides})
    assert response.status_code == 200, response.text
    return response.json()


def force_state(store: Store, task_id: str, state: str, **fields: Any) -> None:
    with psycopg.connect(store.database_url) as connection:
        connection.execute("UPDATE tasks SET state = %s WHERE id = %s", (state, task_id))
    if fields:
        store.update_task(task_id, **fields)


def assert_error(response: Any, status: int, kind: str) -> None:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error", "detail"} and body["error"] == kind and body["detail"]


def test_create_starts_a_workflow_and_freezes_config_without_prompts(client: TestClient, dispatcher: RecordingDispatcher) -> None:
    task = create(client)

    assert task["state"] == "todo"
    assert ("start", task["id"], "demo") in dispatcher.calls
    assert "prompt" not in task["config"]["agents"]["code"]
    detail = client.get(f"/v1/tasks/{task['id']}").json()
    assert detail["phases"] == [] and detail["messages"] == [] and detail["preview"] is None


def test_create_backlog_then_start_once(client: TestClient, dispatcher: RecordingDispatcher) -> None:
    task = create(client, start=False)
    assert task["state"] == "backlog" and not dispatcher.calls

    assert client.post(f"/v1/tasks/{task['id']}/start").json()["state"] == "todo"
    assert dispatcher.calls == [("start", task["id"], "demo")]
    assert_error(client.post(f"/v1/tasks/{task['id']}/start"), 409, "conflict")


@pytest.mark.parametrize(
    "body",
    [
        {"repo": "other"},
        {"workflow": "missing"},
        {"workflow": "../factory"},
        {"title": "  "},
        {"overrides": {"max_rounds": 0}},
        {"overrides": {"gates": [["rm"]]}},
        {"images": [{"filename": "../x.png", "data_base64": "aGk="}]},
        {"images": [{"filename": "x.png", "data_base64": "***"}]},
    ],
)
def test_create_rejects_invalid_requests(client: TestClient, body: dict[str, Any]) -> None:
    assert_error(client.post("/v1/tasks", json={**TASK_BODY, **body}), 422, "invalid")


def test_unknown_and_malformed_task_ids_are_404(client: TestClient) -> None:
    assert_error(client.get("/v1/tasks/abcdefabcdef"), 404, "not_found")
    assert_error(client.get("/v1/tasks/NOT-AN-ID"), 404, "not_found")
    assert_error(client.post("/v1/tasks/abcdefabcdef/approve"), 404, "not_found")
    assert_error(client.post("/v1/tasks/abcdefabcdef/messages", json={"text": "x"}), 404, "not_found")
    assert_error(client.get("/v1/nothing-here"), 404, "not_found")


ACTION_ALLOWED_STATES = {
    "approve": {"human_review"},
    "retry": {"blocked"},
    "start": {"backlog"},
    "cancel": {"backlog", "todo", "in_progress", "human_review", "rework", "blocked", "merging"},
    "preview/restart": {"human_review"},
}
ALL_STATES = ["backlog", "todo", "in_progress", "human_review", "rework", "blocked", "merging", "done", "cancelled"]


@pytest.mark.parametrize("action", sorted(ACTION_ALLOWED_STATES))
def test_every_action_returns_409_outside_its_allowed_states(
    client: TestClient, store: Store, dispatcher: RecordingDispatcher, action: str
) -> None:
    task_id = create(client, start=False)["id"]
    for state in ALL_STATES:
        force_state(store, task_id, state)
        dispatcher.calls.clear()
        response = client.post(f"/v1/tasks/{task_id}/{action}")
        if state in ACTION_ALLOWED_STATES[action]:
            if action == "preview/restart":
                assert_error(response, 409, "conflict")
                assert "no preview" in response.json()["detail"]  # state accepted; this workflow has no preview configured
            else:
                assert response.status_code == 200, (action, state, response.text)
            force_state(store, task_id, state)
        else:
            assert_error(response, 409, "conflict")
            assert state in response.json()["detail"]
            assert not dispatcher.calls
            assert store.get_task(task_id).state == state


def test_messages_are_only_accepted_in_human_review_or_blocked(
    client: TestClient, store: Store, dispatcher: RecordingDispatcher
) -> None:
    task_id = create(client, start=False)["id"]
    for state in ALL_STATES:
        if state in {"human_review", "blocked"}:
            continue
        force_state(store, task_id, state)
        assert_error(client.post(f"/v1/tasks/{task_id}/messages", json={"text": "hi"}), 409, "conflict")
    assert not dispatcher.calls and store.list_messages(task_id) == []


def test_message_moves_to_rework_and_reaches_the_workflow(
    client: TestClient, store: Store, dispatcher: RecordingDispatcher, tmp_path: Path
) -> None:
    task_id = create(client, start=False)["id"]
    force_state(store, task_id, "human_review")
    data = base64.b64encode(b"img").decode()

    response = client.post(
        f"/v1/tasks/{task_id}/messages", json={"text": "fix the title", "images": [{"filename": "a.png", "data_base64": data}]}
    )

    assert response.status_code == 200 and response.json()["role"] == "user" and response.json()["text"] == "fix the title"
    assert store.get_task(task_id).state == "rework"
    (call,) = dispatcher.calls
    assert call[0] == "send" and call[2]["kind"] == "message" and call[2]["text"] == "fix the title"
    (stored_name,) = call[2]["images"]
    assert (tmp_path / "state/tasks" / task_id / "attachments" / stored_name).read_bytes() == b"img"
    assert_error(client.post(f"/v1/tasks/{task_id}/messages", json={"text": "again"}), 409, "conflict")
    assert_error(client.post(f"/v1/tasks/{task_id}/messages", json={"text": "   "}), 422, "invalid")


def test_approve_moves_to_merging_exactly_once(client: TestClient, store: Store, dispatcher: RecordingDispatcher) -> None:
    task_id = create(client, start=False)["id"]
    force_state(store, task_id, "human_review")

    assert client.post(f"/v1/tasks/{task_id}/approve").json()["state"] == "merging"
    assert_error(client.post(f"/v1/tasks/{task_id}/approve"), 409, "conflict")
    assert dispatcher.calls == [("send", task_id, {"kind": "approve"})]


def test_retry_resumes_work_or_merge_depending_on_where_it_blocked(
    client: TestClient, store: Store, dispatcher: RecordingDispatcher
) -> None:
    task_id = create(client, start=False)["id"]
    force_state(store, task_id, "blocked", phase="gates", error="gates failed")
    assert client.post(f"/v1/tasks/{task_id}/retry").json()["state"] == "todo"
    assert store.get_task(task_id).error is None

    force_state(store, task_id, "blocked", phase="merge", error="main dirty")
    assert client.post(f"/v1/tasks/{task_id}/retry").json()["state"] == "merging"
    assert [call[2] for call in dispatcher.calls] == ["retry", "merge"]


def test_events_are_ordered_filtered_and_paged(client: TestClient) -> None:
    first = create(client)
    second = create(client, start=False)

    events = client.get("/v1/events").json()
    assert [event["seq"] for event in events] == sorted(event["seq"] for event in events)
    assert {event["task_id"] for event in events} == {first["id"], second["id"]}
    assert all(event["kind"] == "task.created" for event in events)
    after = events[0]["seq"]
    assert [event["task_id"] for event in client.get(f"/v1/events?after={after}").json()] == [second["id"]]
    assert client.get("/v1/events?repo=other").json() == []
    assert_error(client.get("/v1/events?after=-1"), 422, "invalid")


def test_latest_event_seq_is_the_newest_seq_or_zero(client: TestClient) -> None:
    assert client.get("/v1/events/latest").json() == {"seq": 0}

    create(client)
    create(client, start=False)

    assert client.get("/v1/events/latest").json() == {"seq": client.get("/v1/events").json()[-1]["seq"]}


def test_message_with_an_invalid_image_leaves_the_task_state_alone(client: TestClient, store: Store) -> None:
    task_id = create(client, start=False)["id"]
    force_state(store, task_id, "human_review")
    bad_image = {"filename": "shot.png", "data_base64": "***"}

    assert_error(client.post(f"/v1/tasks/{task_id}/messages", json={"text": "hello", "images": [bad_image]}), 422, "invalid")

    assert store.get_task(task_id).state == "human_review"


def test_register_repo_is_idempotent_by_resolved_path(client: TestClient, bare_repo: Path, tmp_path: Path) -> None:
    link = tmp_path / "link-to-bare"
    link.symlink_to(bare_repo)

    first = client.post("/v1/repos", json={"path": str(bare_repo)})
    again = client.post("/v1/repos", json={"path": str(link)})

    assert first.status_code == 200 and first.json() == {"name": "bare", "path": str(bare_repo.resolve())}
    assert again.json() == first.json()
    assert [repo["name"] for repo in client.get("/v1/repos").json()] == ["bare", "demo"]


def test_register_repo_suffixes_clashing_names_and_sanitizes(client: TestClient, tmp_path: Path) -> None:
    paths = [tmp_path / "a" / "app", tmp_path / "b" / "app", tmp_path / "c" / "app", tmp_path / "d" / "_my repo!"]
    for path in paths:
        (path / ".git").mkdir(parents=True)

    names = [client.post("/v1/repos", json={"path": str(path)}).json()["name"] for path in paths]

    assert names == ["app", "app-2", "app-3", "my-repo-"]


@pytest.mark.parametrize("kind", ["relative", "missing", "file", "not_git", "linked_worktree"])
def test_register_repo_rejects_invalid_paths(client: TestClient, tmp_path: Path, repo: Path, kind: str) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    file = tmp_path / "file.txt"
    file.write_text("x")
    linked = tmp_path / "linked"
    git(repo, "worktree", "add", "-q", "-b", "side", str(linked))
    path = {
        "relative": "some/relative/dir",
        "missing": str(tmp_path / "missing"),
        "file": str(file),
        "not_git": str(plain),
        "linked_worktree": str(linked),
    }[kind]

    assert_error(client.post("/v1/repos", json={"path": path}), 422, "invalid")
    assert [entry["name"] for entry in client.get("/v1/repos").json()] == ["demo"]


def test_put_defaults_persists_and_feeds_new_tasks(client: TestClient, tmp_path: Path) -> None:
    current = client.get("/v1/defaults").json()
    edited_coder = current["agents"]["coder"] + "\nBe careful."
    body = {
        "settings": {**current["settings"], "merge": "auto", "max_rounds": 5, "models": {"coder": "my-model"}},
        "agents": {"coder": edited_coder},
    }

    saved = client.put("/v1/defaults", json=body)

    assert saved.status_code == 200
    assert saved.json()["settings"]["merge"] == "auto" and saved.json()["agents"]["coder"] == edited_coder
    assert saved.json()["agents"]["scout"] == current["agents"]["scout"]
    assert client.get("/v1/defaults").json() == saved.json()
    assert (tmp_path / "defaults/agents/coder.md").read_text() == edited_coder
    assert not (tmp_path / "defaults/agents/scout.md").exists()
    task = create(client)
    assert (task["config"]["max_rounds"], task["config"]["agents"]["code"]["model"]) == (5, "my-model")
    assert task["config"]["merge"] == "auto"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body["settings"].update(max_rounds=0),
        lambda body: body["settings"].update(max_rounds=21),
        lambda body: body["settings"].update(merge="yolo"),
        lambda body: body["settings"].update(unknown_key=1),
        lambda body: body["settings"].update(models={"deployer": "x"}),
        lambda body: body["settings"].update(agent_timeout_s=0),
        lambda body: body["agents"].update(coder="no front matter"),
    ],
)
def test_put_defaults_rejects_invalid_bodies_and_writes_nothing(client: TestClient, tmp_path: Path, mutate: Any) -> None:
    body = client.get("/v1/defaults").json()
    body["agents"] = {"coder": body["agents"]["coder"] + "\nedited"}
    mutate(body)

    assert_error(client.put("/v1/defaults", json=body), 422, "invalid")

    assert not (tmp_path / "defaults").exists()
