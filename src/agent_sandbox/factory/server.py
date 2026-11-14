"""Start the factory engine: database schema, DBOS, previews, and the uvicorn HTTP server."""

import logging
import os
import signal
import sys
import threading

import uvicorn

from agent_sandbox.factory.api import create_app
from agent_sandbox.factory.config import load_engine_config
from agent_sandbox.factory.engine import Engine
from agent_sandbox.factory.preview import PreviewManager
from agent_sandbox.factory.store import Store
from agent_sandbox.factory.workflows import DbosDispatcher, bind_engine, launch_dbos, shutdown_dbos


def serve(config_path: str | os.PathLike[str] | None = None) -> None:
    """Run the engine until interrupted."""
    config = load_engine_config(config_path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.state_dir.mkdir(parents=True, exist_ok=True)
    store = Store(config.database_url)
    store.ensure_schema()
    previews = PreviewManager(config.preview_ports, config.state_dir, config.database_url)
    engine = Engine(config, store, previews, DbosDispatcher())
    bind_engine(engine)
    launch_dbos(config.database_url, config.limits.in_progress, config.otel_endpoint)
    # In the background: each preview takes its readiness wait, the API must not wait for them.
    threading.Thread(target=engine.restore_previews, name="restore-previews", daemon=True).start()
    # uvicorn shuts down on SIGTERM, then raises it again with this handler back in place: the default action would end the
    # process before the cleanup below, leaving every preview process running.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        uvicorn.run(create_app(engine), host=config.host, port=config.port, log_level="info")
    finally:
        previews.stop_all()
        shutdown_dbos()
        bind_engine(None)
