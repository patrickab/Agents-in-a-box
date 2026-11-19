"""Build the credential-free Pi agent-dir seed a sandboxed Pi runs from.

The seed holds the host's `settings.json` (only `defaultTools` and `packages`, defaulting to a gateway
model) and a generated `models.json` whose single provider `gateway` reaches pi-host's OpenAI-compatible gateway with the
gateway-only token. The host's installed packages (`npm/`) are bound read-only beside it. Nothing
else of `~/.pi/agent` (never `auth.json`, sessions, or extensions) enters a container.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import urllib.error
import urllib.request

from agent_sandbox.config import HOST_SERVICE_ADDRESS
from agent_sandbox.errors import ContainerRuntimeError

GATEWAY_PROVIDER = "gateway"
# Profile host service whose port is pi-host's gateway; containers reach it at HOST_SERVICE_ADDRESS.
GATEWAY_HOST_SERVICE = "PI_GATEWAY_URL"
GATEWAY_TOKEN_ENV = "PI_GATEWAY_TOKEN"
_KEPT_SETTINGS = ("defaultTools", "packages")
_THINKING_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")
_MODEL_INPUTS = frozenset({"text", "image"})
_MODELS_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class AgentSeed:
    """Locate one private seed directory and the host package tree bound read-only at `<agent dir>/npm`."""

    directory: Path  # settings.json and models.json; owned by the caller, removed by remove().
    npm_dir: Path | None

    def remove(self) -> None:
        """Delete the seed directory."""
        shutil.rmtree(self.directory, ignore_errors=True)

    def __enter__(self) -> "AgentSeed":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.remove()


def host_agent_dir() -> Path:
    """Return the host Pi agent dir the seed is built from."""
    return Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent")


def gateway_token_path() -> Path:
    """Return the gateway-only token file pi-host creates."""
    if explicit := os.environ.get("PI_GATEWAY_TOKEN_FILE"):
        return Path(explicit)
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "gigachad-bot" / "pi-gateway.token"


def read_gateway_token() -> str:
    """Read the gateway-only token; containers never receive pi-host's full token."""
    path = gateway_token_path()
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ContainerRuntimeError(f"cannot read pi gateway token {path}: {error}") from error
    if not token:
        raise ContainerRuntimeError(f"pi gateway token {path} is empty")
    return token


def gateway_environment(agent_dir: str, token: str) -> dict[str, str]:
    """Return the environment a sandboxed Pi needs: its agent dir, the gateway token, and no network chatter."""
    return {"PI_CODING_AGENT_DIR": agent_dir, GATEWAY_TOKEN_ENV: token, "PI_OFFLINE": "1"}


def fetch_gateway_models(port: int, token: str) -> list[dict[str, Any]]:
    """List the models pi-host's gateway serves, from the host side."""
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=_MODELS_TIMEOUT_SECONDS) as response:
            body = json.load(response)
    except (OSError, ValueError) as error:
        raise ContainerRuntimeError(f"cannot list pi gateway models on port {port}: {error}") from error
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise ContainerRuntimeError("pi gateway model list has no data array")
    return [model for model in data if isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"]]


def gateway_models_config(models: list[dict[str, Any]], port: int) -> dict[str, Any]:
    """Build `models.json` declaring the gateway as one OpenAI-compatible provider reached from a container."""
    return {
        "providers": {
            GATEWAY_PROVIDER: {
                "baseUrl": f"http://{HOST_SERVICE_ADDRESS}:{port}/v1",
                "api": "openai-completions",
                "apiKey": f"${GATEWAY_TOKEN_ENV}",
                "compat": {
                    "supportsDeveloperRole": True,
                    "supportsReasoningEffort": True,
                    "supportsStore": False,
                    "thinkingFormat": "openai",
                    "maxTokensField": "max_completion_tokens",
                },
                "models": [_model_entry(model) for model in models],
            }
        }
    }


def _model_entry(model: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one gateway model listing into a Pi `models.json` model."""
    reasoning = model.get("reasoning") is True
    inputs = [kind for kind in model.get("input") or () if kind in _MODEL_INPUTS]
    entry: dict[str, Any] = {
        "id": model["id"],
        "name": model.get("name") if isinstance(model.get("name"), str) else model["id"],
        "reasoning": reasoning,
        "input": inputs or ["text"],
    }
    for source, target in (("context_window", "contextWindow"), ("max_tokens", "maxTokens")):
        if type(model.get(source)) is int and model[source] > 0:
            entry[target] = model[source]
    if reasoning:
        supported = set(model.get("thinking_levels") or ())
        entry["thinkingLevelMap"] = {level: level if level in supported else None for level in _THINKING_LEVELS}
    return entry


def _host_settings(source: Path) -> dict[str, Any]:
    """Read the host Pi settings, empty when the file is missing."""
    try:
        settings = json.loads((source / "settings.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise ContainerRuntimeError(f"cannot read {source / 'settings.json'}: {error}") from error
    if not isinstance(settings, dict):
        raise ContainerRuntimeError(f"{source / 'settings.json'} must hold a JSON object")
    return settings


def seed_settings(host_settings: Mapping[str, Any], models: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep the host settings a sandboxed Pi needs and default it to a gateway model.

    The default model is the host default as a gateway id when the gateway serves it, else the first
    anthropic model, else the first model, so a Pi started without a model never picks an uncredentialed provider.
    """
    settings = {key: host_settings[key] for key in _KEPT_SETTINGS if key in host_settings}
    settings["defaultProvider"] = GATEWAY_PROVIDER
    ids = [model["id"] for model in models]
    host_default = f"{host_settings.get('defaultProvider')}/{host_settings.get('defaultModel')}"
    anthropic = [model_id for model_id in ids if model_id.startswith("anthropic/")]
    candidates = [host_default] if host_default in ids else anthropic or ids
    if candidates:
        settings["defaultModel"] = candidates[0]
    return settings


def build_agent_seed(gateway_port: int, *, token: str | None = None, source: Path | None = None) -> AgentSeed:
    """Write a fresh seed directory for a Pi reaching the gateway on `gateway_port`; the caller removes it."""
    source = source or host_agent_dir()
    models = fetch_gateway_models(gateway_port, token or read_gateway_token())
    settings = seed_settings(_host_settings(source), models)
    directory = Path(tempfile.mkdtemp(prefix="agent-sandbox-pi-agent-"))
    try:
        (directory / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        (directory / "models.json").write_text(json.dumps(gateway_models_config(models, gateway_port), indent=2), encoding="utf-8")
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    npm_dir = source / "npm"
    return AgentSeed(directory, npm_dir if npm_dir.is_dir() else None)
