"""Test profile loading and validation."""

from dataclasses import replace
import os
import tempfile
import unittest
from unittest.mock import patch

import yaml

from agent_sandbox.profile import Mount, Profile, ProfileError, WorkspaceConfig, load_profile


def _profile_data(name: str = "test_profile") -> dict:
    return {
        "name": name,
        "image": "test:image",
        "workdir": "/workspace",
        "mounts": [],
        "workspace": {"mode": "managed"},
        "omp_binary": "/usr/local/bin/omp",
    }


class TestProfileLoading(unittest.TestCase):
    """Exercise profile loading at its untrusted YAML boundary."""

    def _load(self, data: dict, name: str = "test_profile") -> Profile:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/{name}.yaml"
            with open(path, "w", encoding="utf-8") as profile_file:
                yaml.safe_dump(data, profile_file, sort_keys=False)
            with patch("agent_sandbox.profile.PROFILE_SEARCH_PATH", [tmpdir]):
                return load_profile(name)

    def _load_yaml(self, text: str, name: str = "test_profile") -> Profile:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/{name}.yaml"
            with open(path, "w", encoding="utf-8") as profile_file:
                profile_file.write(text)
            with patch("agent_sandbox.profile.PROFILE_SEARCH_PATH", [tmpdir]):
                return load_profile(name)

    def test_load_gigachad_profile_and_runtime_fingerprint(self) -> None:
        """Keep the shipped trusted profile valid and fingerprinted."""
        profile = load_profile("gigachad")

        self.assertIsInstance(profile.env_passthrough, tuple)
        self.assertIn("OPENAI_API_KEY", profile.env_passthrough)
        self.assertIn("/root/.omp", [mount.target for mount in profile.mounts])
        self.assertNotIn("/home/sandbox/.omp", [mount.target for mount in profile.mounts])
        self.assertEqual(profile.host_services, (("OLLAMA_HOST", 11434),))
        self.assertRegex(profile.runtime_fingerprint, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(profile.runtime_fingerprint, load_profile("gigachad").runtime_fingerprint)

    def test_load_profile_prefers_earliest_search_directory(self) -> None:
        """Use the first matching profile from the configured search path."""
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, image in ((first, "first:image"), (second, "second:image")):
                with open(f"{directory}/preferred.yaml", "w", encoding="utf-8") as profile_file:
                    yaml.safe_dump({**_profile_data("preferred"), "image": image}, profile_file)

            with patch("agent_sandbox.profile.PROFILE_SEARCH_PATH", [first, second]):
                profile = load_profile("preferred")

        self.assertEqual(profile.image, "first:image")

    def test_optional_profile_collections_default_to_empty_tuples(self) -> None:
        profile = self._load(_profile_data())

        self.assertEqual(profile.python_interpreters, ())
        self.assertEqual(profile.env_passthrough, ())
        self.assertEqual(profile.host_services, ())

    def test_rejects_unsafe_profile_names_before_file_lookup(self) -> None:
        for name in ("", "../gigachad", "gigachad.yaml", "name/child"):
            with self.subTest(name=name), self.assertRaisesRegex(ProfileError, "safe profile name"):
                load_profile(name)

    def test_rejects_unknown_root_fields_and_invalid_required_values(self) -> None:
        cases = (
            (lambda data: data.update({"extra": True}), "unsupported field"),
            (lambda data: data.update({"image": ""}), "image must be a nonempty string"),
            (lambda data: data.update({"workdir": "workspace"}), "workdir must be an absolute path"),
            (lambda data: data.update({"omp_binary": "/usr/local/../bin/omp"}), "omp_binary must be an absolute path"),
            (lambda data: data.update({"workspace": {"mode": "host"}}), "workspace mode must be 'managed'"),
        )
        for mutate, error in cases:
            with self.subTest(error=error):
                data = _profile_data()
                mutate(data)
                with self.assertRaisesRegex(ProfileError, error):
                    self._load(data)

    def test_rejects_invalid_and_conflicting_mounts(self) -> None:
        cases = (
            ([{"source": "/host/tool", "target": "/opt/tool", "mode": "other"}], "invalid mode"),
            ([{"source": "host/tool", "target": "/opt/tool", "mode": "ro"}], "source must be an absolute path"),
            ([{"source": "/host/tool", "target": "opt/tool", "mode": "ro"}], "target must be an absolute path"),
            (
                [
                    {"source": "/host/a", "target": "/opt/tool", "mode": "ro"},
                    {"source": "/host/b", "target": "/opt/tool", "mode": "ro"},
                ],
                "conflicts",
            ),
            (
                [
                    {"source": "/host/a", "target": "/opt", "mode": "ro"},
                    {"source": "/host/b", "target": "/opt/tool", "mode": "ro"},
                ],
                "conflicts",
            ),
            ([{"source": "/host/a", "target": "/runtime/active", "mode": "ro"}], "owned by the runtime"),
            ([{"source": "/host/a", "target": "/workspace", "mode": "ro"}], "must not replace the managed workspace"),
            ([{"source": "/host/a", "target": "/opt/tool", "mode": "ro", "extra": True}], "unsupported field"),
        )
        for mounts, error in cases:
            with self.subTest(error=error):
                data = _profile_data()
                data["mounts"] = mounts
                with self.assertRaisesRegex(ProfileError, error):
                    self._load(data)

    def test_rejects_invalid_interpreters_environment_and_host_services(self) -> None:
        cases = (
            ("python_interpreters", {"unsafe/name": "/opt/python"}, "interpreter labels"),
            ("python_interpreters", {"venv": "opt/python"}, "must be an absolute path"),
            ("env_passthrough", ["INVALID-NAME"], "environment variable names"),
            ("env_passthrough", ["OPENAI_API_KEY", "OPENAI_API_KEY"], "must not contain duplicates"),
            ("host_services", {"INVALID-NAME": 11434}, "environment variable names"),
            ("host_services", {"OLLAMA_HOST": True}, "host ports"),
        )
        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                data = _profile_data()
                data[field] = value
                with self.assertRaisesRegex(ProfileError, error):
                    self._load(data)

    def test_rejects_duplicate_yaml_keys(self) -> None:
        text = """
name: test_profile
name: duplicate
image: test:image
workdir: /workspace
mounts: []
workspace: {mode: managed}
omp_binary: /usr/local/bin/omp
"""
        with self.assertRaisesRegex(ProfileError, "duplicate key"):
            self._load_yaml(text)

    def test_runtime_fingerprint_covers_profile_fields_and_policy_not_secret_values(self) -> None:
        profile = Profile(
            name="test_profile",
            image="test:image",
            workdir="/workspace",
            mounts=(Mount("/host/a", "/opt/a", "ro"), Mount("/host/b", "/opt/b", "rw")),
            workspace=WorkspaceConfig("managed"),
            omp_binary="/usr/local/bin/omp",
            python_interpreters=(("venv", "/opt/venv/bin/python"),),
            env_passthrough=("OPENAI_API_KEY", "TAVILY_API_KEY"),
            host_services=(("OLLAMA_HOST", 11434),),
        )
        variants = (
            replace(profile, name="other_profile"),
            replace(profile, image="other:image"),
            replace(profile, workdir="/project"),
            replace(profile, mounts=(Mount("/host/c", "/opt/c", "ro"),)),
            replace(profile, workspace=WorkspaceConfig("other")),
            replace(profile, omp_binary="/opt/omp"),
            replace(profile, python_interpreters=(("base", "/opt/base/python"),)),
            replace(profile, env_passthrough=("GEMINI_API_KEY",)),
            replace(profile, host_services=(("SERVICE_HOST", 8080),)),
        )

        fingerprint = profile.runtime_fingerprint
        self.assertEqual(len({fingerprint, *(item.runtime_fingerprint for item in variants)}), 10)
        reordered = replace(
            profile,
            mounts=tuple(reversed(profile.mounts)),
            python_interpreters=tuple(reversed(profile.python_interpreters)),
            env_passthrough=tuple(reversed(profile.env_passthrough)),
            host_services=tuple(reversed(profile.host_services)),
        )
        self.assertEqual(reordered.runtime_fingerprint, fingerprint)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-value-that-must-not-affect-the-fingerprint"}):
            self.assertEqual(profile.runtime_fingerprint, fingerprint)
        with patch("agent_sandbox.profile.RUNTIME_POLICY_DESCRIPTOR", "agent-sandbox-runtime-policy-v5"):
            self.assertNotEqual(replace(profile).runtime_fingerprint, fingerprint)


if __name__ == "__main__":
    unittest.main()
