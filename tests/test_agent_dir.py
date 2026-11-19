"""Test the credential-free Pi agent-dir seed and the gateway token lookup."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent_sandbox.agent_dir import (
    build_agent_seed,
    gateway_environment,
    gateway_models_config,
    gateway_token_path,
    read_gateway_token,
    seed_settings,
)
from agent_sandbox.errors import ContainerRuntimeError

_HAIKU = {
    "id": "anthropic/claude-haiku-5-5",
    "name": "Claude Haiku 5.5",
    "context_window": 200000,
    "max_tokens": 64000,
    "input": ["text", "image"],
    "reasoning": True,
    "thinking_levels": ["low", "medium", "high"],
}
_PLAIN = {"id": "openai/plain", "reasoning": False, "thinking_levels": [], "input": ["text", "audio"]}


class TestModelsConfig(unittest.TestCase):
    def test_declares_one_gateway_provider_reached_through_the_host_service_alias(self) -> None:
        config = gateway_models_config([_HAIKU, _PLAIN], 8301)

        provider = config["providers"]["gateway"]
        self.assertEqual(list(config["providers"]), ["gateway"])
        self.assertEqual(
            (provider["baseUrl"], provider["api"], provider["apiKey"]),
            ("http://10.200.200.1:8301/v1", "openai-completions", "$PI_GATEWAY_TOKEN"),
        )
        haiku, plain = provider["models"]
        self.assertEqual(
            haiku,
            {
                "id": "anthropic/claude-haiku-5-5",
                "name": "Claude Haiku 5.5",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 200000,
                "maxTokens": 64000,
                "thinkingLevelMap": {"minimal": None, "low": "low", "medium": "medium", "high": "high", "xhigh": None, "max": None},
            },
        )
        self.assertEqual(plain, {"id": "openai/plain", "name": "openai/plain", "reasoning": False, "input": ["text"]})


class TestSeedSettings(unittest.TestCase):
    def test_keeps_tools_and_packages_and_defaults_to_the_gateway(self) -> None:
        host = {
            "defaultTools": ["+codemode"],
            "packages": ["npm:x"],
            "theme": "dark",
            "defaultProvider": "openai",
            "defaultModel": "plain",
        }

        self.assertEqual(
            seed_settings(host, [_HAIKU, _PLAIN]),
            {"defaultTools": ["+codemode"], "packages": ["npm:x"], "defaultProvider": "gateway", "defaultModel": "openai/plain"},
        )

    def test_falls_back_to_the_first_anthropic_then_any_model(self) -> None:
        self.assertEqual(seed_settings({"defaultModel": "gone"}, [_PLAIN, _HAIKU])["defaultModel"], "anthropic/claude-haiku-5-5")
        self.assertEqual(seed_settings({}, [_PLAIN])["defaultModel"], "openai/plain")
        self.assertEqual(seed_settings({}, []), {"defaultProvider": "gateway"})


class TestBuildAgentSeed(unittest.TestCase):
    def test_writes_settings_and_models_only_and_points_at_host_packages(self) -> None:
        with tempfile.TemporaryDirectory() as host:
            source = Path(host)
            (source / "settings.json").write_text(json.dumps({"packages": ["npm:x"], "theme": "dark"}))
            (source / "auth.json").write_text("{}")
            (source / "sessions").mkdir()
            (source / "npm").mkdir()
            with (
                patch("agent_sandbox.agent_dir.fetch_gateway_models", return_value=[_HAIKU]) as fetch,
                build_agent_seed(8300, token="gateway-token", source=source) as seed,
            ):
                self.assertEqual(sorted(path.name for path in seed.directory.iterdir()), ["models.json", "settings.json"])
                self.assertEqual(json.loads((seed.directory / "settings.json").read_text())["packages"], ["npm:x"])
                self.assertEqual(json.loads((seed.directory / "models.json").read_text()), gateway_models_config([_HAIKU], 8300))
                self.assertEqual(seed.npm_dir, source / "npm")
                directory = seed.directory
            fetch.assert_called_once_with(8300, "gateway-token")
            self.assertFalse(directory.exists())

    def test_environment_carries_agent_dir_token_and_offline_mode(self) -> None:
        self.assertEqual(
            gateway_environment("/runtime/pi-agent", "t"),
            {"PI_CODING_AGENT_DIR": "/runtime/pi-agent", "PI_GATEWAY_TOKEN": "t", "PI_OFFLINE": "1"},
        )


class TestGatewayToken(unittest.TestCase):
    def test_reads_the_configured_or_state_token_file(self) -> None:
        with tempfile.TemporaryDirectory() as state:
            with patch.dict(os.environ, {"XDG_STATE_HOME": state}, clear=False):
                os.environ.pop("PI_GATEWAY_TOKEN_FILE", None)
                self.assertEqual(gateway_token_path(), Path(state, "gigachad-bot", "pi-gateway.token"))
            token_file = Path(state, "token")
            token_file.write_text("secret\n")
            with patch.dict(os.environ, {"PI_GATEWAY_TOKEN_FILE": str(token_file)}):
                self.assertEqual(read_gateway_token(), "secret")
                token_file.write_text("\n")
                with self.assertRaisesRegex(ContainerRuntimeError, "is empty"):
                    read_gateway_token()
                token_file.unlink()
                with self.assertRaisesRegex(ContainerRuntimeError, "cannot read pi gateway token"):
                    read_gateway_token()


if __name__ == "__main__":
    unittest.main()
