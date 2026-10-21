"""Serve the factory engine's HTTP API under /v1."""

import logging
import re
from typing import Annotated, get_args

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent_sandbox.factory.engine import ApiError, Engine, invalid, not_found
from agent_sandbox.factory.models import (
    TASK_ID_PATTERN,
    Event,
    FactoryDefaults,
    Message,
    MessageCreate,
    Preview,
    RepoCreate,
    RepoInfo,
    Task,
    TaskCreate,
    TaskDetail,
    TaskState,
    WorkflowInfo,
    format_validation_errors,
)

LOGGER = logging.getLogger("agent_sandbox.factory")
_TASK_ID = re.compile(TASK_ID_PATTERN)
_STATES = frozenset(get_args(TaskState))


def _error(status: int, kind: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": kind, "detail": detail})


def _task_id(raw: str) -> str:
    if not _TASK_ID.fullmatch(raw):
        raise not_found(f"task {raw} not found")
    return raw


def create_app(engine: Engine) -> FastAPI:
    """Build the FastAPI application over an engine."""
    app = FastAPI(title="Agent factory", docs_url=None, redoc_url=None, openapi_url=None)
    router = APIRouter(prefix="/v1")

    @app.exception_handler(ApiError)
    def _api_error(_: Request, error: ApiError) -> JSONResponse:
        return _error(error.status, error.kind, error.detail)

    @app.exception_handler(RequestValidationError)
    def _validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        return _error(422, "invalid", format_validation_errors(error.errors()))

    @app.exception_handler(StarletteHTTPException)
    def _http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        return _error(error.status_code, "not_found" if error.status_code == 404 else "invalid", str(error.detail))

    @app.exception_handler(Exception)
    def _unexpected(_: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("unhandled error in the factory API", exc_info=error)
        return _error(500, "internal", f"{type(error).__name__}: {error}")

    @router.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @router.get("/repos")
    def repos() -> list[RepoInfo]:
        return engine.repos()

    @router.post("/repos")
    def register_repo(body: RepoCreate) -> RepoInfo:
        return engine.register_repo(body)

    @router.get("/defaults")
    def defaults() -> FactoryDefaults:
        return engine.defaults()

    @router.put("/defaults")
    def save_defaults(body: FactoryDefaults) -> FactoryDefaults:
        return engine.save_defaults(body)

    @router.get("/repos/{repo}/workflows")
    def workflows(repo: str) -> list[WorkflowInfo]:
        return engine.workflows(repo)

    @router.post("/tasks")
    def create_task(body: TaskCreate) -> Task:
        return engine.create_task(body)

    @router.get("/tasks")
    def list_tasks(repo: str | None = None, state: str | None = None) -> list[Task]:
        if state is not None and state not in _STATES:
            raise invalid(f"unknown state {state!r}")
        return engine.store.list_tasks(repo, state)

    @router.get("/tasks/{task_id}")
    def get_task(task_id: str) -> TaskDetail:
        return engine.get_detail(_task_id(task_id))

    @router.get("/events")
    def events(after: Annotated[int, Query(ge=0)] = 0, repo: str | None = None) -> list[Event]:
        return engine.store.list_events(after, repo)

    @router.get("/events/latest")
    def latest_event() -> dict[str, int]:
        return {"seq": engine.store.latest_event_seq()}

    @router.post("/tasks/{task_id}/start")
    def start(task_id: str) -> Task:
        return engine.start_task(_task_id(task_id))

    @router.post("/tasks/{task_id}/messages")
    def post_message(task_id: str, body: MessageCreate) -> Message:
        return engine.post_message(_task_id(task_id), body)

    @router.post("/tasks/{task_id}/approve")
    def approve(task_id: str) -> Task:
        return engine.approve(_task_id(task_id))

    @router.post("/tasks/{task_id}/cancel")
    def cancel(task_id: str) -> Task:
        return engine.cancel(_task_id(task_id))

    @router.post("/tasks/{task_id}/retry")
    def retry(task_id: str) -> Task:
        return engine.retry(_task_id(task_id))

    @router.post("/tasks/{task_id}/preview/restart")
    def restart_preview(task_id: str) -> Preview:
        return engine.restart_preview(_task_id(task_id))

    app.include_router(router)
    return app
