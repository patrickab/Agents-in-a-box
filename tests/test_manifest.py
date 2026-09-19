"""Test durable manifest serialization contracts."""

import unittest

import yaml

from agent_sandbox.manifest import (
    Artifact,
    Execution,
    OmpSessionState,
    Outputs,
    SandboxManifest,
    WorkspaceRef,
    from_yaml,
    manifest_from_dict,
    manifest_to_dict,
    to_yaml,
)


def _manifest_data() -> dict:
    return {
        "schema_version": 1,
        "manifest_id": "manifest-asset-001",
        "profile": "gigachad",
        "runtime_fingerprint": "sha256:runtime",
        "workspace": {"snapshot_asset_id": "workspace-asset-001", "sha256": "a" * 64},
        "outputs": {
            "events_asset_id": "events-asset-001",
            "artifacts": [{"asset_id": "artifact-asset-001", "mime_type": "text/plain"}],
        },
        "omp_sessions": {"main": {"session_id": "omp-session-001", "state_asset_id": "omp-state-asset-001"}},
        "execution": {"run_id": "run-001", "status": "completed", "exit_code": 0, "created_at": "2026-09-19T12:00:00Z"},
    }


class TestManifestSerialization(unittest.TestCase):
    """Exercise the public YAML manifest boundary."""

    def _from_data(self, data: dict) -> SandboxManifest:
        return from_yaml(yaml.safe_dump(data, sort_keys=False))

    def _manifest(self, sessions: dict[str, OmpSessionState] | None = None) -> SandboxManifest:
        return SandboxManifest(
            schema_version=1,
            manifest_id="manifest-asset-001",
            profile="gigachad",
            runtime_fingerprint="sha256:runtime",
            workspace=WorkspaceRef(snapshot_asset_id="workspace-asset-001", sha256="a" * 64),
            outputs=Outputs(
                events_asset_id="events-asset-001",
                artifacts=(Artifact(asset_id="artifact-asset-001", mime_type="text/plain"),),
            ),
            omp_sessions=sessions
            if sessions is not None
            else {"main": OmpSessionState(session_id="omp-session-001", state_asset_id="omp-state-asset-001")},
            execution=Execution(run_id="run-001", status="completed", exit_code=0, created_at="2026-09-19T12:00:00Z"),
        )

    def test_yaml_round_trip_preserves_complete_manifest_layout(self) -> None:
        manifest = self._manifest()

        serialized = to_yaml(manifest)

        self.assertEqual(
            serialized,
            """schema_version: 1
manifest_id: manifest-asset-001
profile: gigachad
runtime_fingerprint: sha256:runtime
workspace:
  snapshot_asset_id: workspace-asset-001
  sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
outputs:
  events_asset_id: events-asset-001
  artifacts:
  - asset_id: artifact-asset-001
    mime_type: text/plain
omp_sessions:
  main:
    session_id: omp-session-001
    state_asset_id: omp-state-asset-001
execution:
  run_id: run-001
  status: completed
  exit_code: 0
  created_at: '2026-09-19T12:00:00Z'
""",
        )
        self.assertEqual(from_yaml(serialized), manifest)

    def test_mapping_round_trip_is_storage_neutral(self) -> None:
        manifest = self._manifest()

        data = manifest_to_dict(manifest)

        self.assertEqual(data, _manifest_data())
        self.assertEqual(manifest_from_dict(data), manifest)

    def test_mapping_codec_rejects_unknown_fields(self) -> None:
        data = _manifest_data()
        data["extra"] = True

        with self.assertRaisesRegex(ValueError, "root has unsupported field"):
            manifest_from_dict(data)

    def test_yaml_root_must_be_a_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "root must be a mapping"):
            from_yaml("- not\n- a manifest\n")

    def test_schema_rejects_unknown_or_missing_fields_at_every_mapping(self) -> None:
        cases = (
            (lambda data: data.update({"extra": True}), "root has unsupported field"),
            (lambda data: data.pop("profile"), "root missing required field"),
            (lambda data: data["workspace"].update({"extra": True}), "workspace has unsupported field"),
            (lambda data: data["outputs"]["artifacts"][0].pop("mime_type"), "artifact missing required field"),
            (lambda data: data["omp_sessions"]["main"].update({"extra": True}), "omp session has unsupported field"),
            (lambda data: data["execution"].update({"extra": True}), "execution has unsupported field"),
        )
        for mutate, error in cases:
            with self.subTest(error=error):
                data = _manifest_data()
                mutate(data)
                with self.assertRaisesRegex(ValueError, error):
                    self._from_data(data)

    def test_schema_rejects_invalid_nested_mapping_and_list_shapes(self) -> None:
        cases = (
            (lambda data: data.update({"workspace": []}), "workspace must be a mapping"),
            (lambda data: data["outputs"].update({"artifacts": {}}), "outputs.artifacts must be a list"),
            (lambda data: data.update({"omp_sessions": []}), "omp_sessions must be a mapping"),
            (lambda data: data.update({"execution": []}), "execution must be a mapping"),
        )
        for mutate, error in cases:
            with self.subTest(error=error):
                data = _manifest_data()
                mutate(data)
                with self.assertRaisesRegex(ValueError, error):
                    self._from_data(data)

    def test_schema_requires_version_one_and_lowercase_workspace_hashes(self) -> None:
        cases = (
            (lambda data: data.update({"schema_version": 2}), "schema_version must be 1"),
            (lambda data: data.update({"schema_version": True}), "schema_version must be 1"),
            (lambda data: data["workspace"].update({"sha256": "A" * 64}), "64-character lowercase"),
            (lambda data: data["workspace"].update({"sha256": "a" * 63}), "64-character lowercase"),
        )
        for mutate, error in cases:
            with self.subTest(error=error):
                data = _manifest_data()
                mutate(data)
                with self.assertRaisesRegex(ValueError, error):
                    self._from_data(data)

    def test_schema_requires_nonempty_identifiers_and_storage_neutral_asset_ids(self) -> None:
        data = _manifest_data()
        data["manifest_id"] = " "
        with self.assertRaisesRegex(ValueError, "manifest_id must be a nonempty string"):
            self._from_data(data)

        for path in ("/tmp/workspace.tar", "../workspace.tar", "assets/../workspace.tar", r"C:\\workspace.tar"):
            with self.subTest(path=path):
                data = _manifest_data()
                data["workspace"]["snapshot_asset_id"] = path
                with self.assertRaisesRegex(ValueError, "durable asset ID"):
                    self._from_data(data)

    def test_execution_statuses_and_exit_codes_are_strict(self) -> None:
        for status in ("completed", "failed"):
            with self.subTest(status=status):
                data = _manifest_data()
                data["execution"]["status"] = status
                self.assertEqual(self._from_data(data).execution.status, status)
        for status in ("running", "succeeded", "", None):
            with self.subTest(status=status):
                data = _manifest_data()
                data["execution"]["status"] = status
                with self.assertRaisesRegex(ValueError, "execution status"):
                    self._from_data(data)
        for exit_code in (True, "0"):
            with self.subTest(exit_code=exit_code):
                data = _manifest_data()
                data["execution"]["exit_code"] = exit_code
                with self.assertRaisesRegex(ValueError, "integer or null"):
                    self._from_data(data)
        data = _manifest_data()
        data["execution"]["exit_code"] = None
        self.assertIsNone(self._from_data(data).execution.exit_code)
        data = _manifest_data()
        data["execution"]["created_at"] = ""
        self.assertEqual(self._from_data(data).execution.created_at, "")
        for created_at in (None, 0):
            with self.subTest(created_at=created_at):
                data = _manifest_data()
                data["execution"]["created_at"] = created_at
                with self.assertRaisesRegex(ValueError, "created_at must be a string"):
                    self._from_data(data)

    def test_asset_references_reject_host_paths_in_direct_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "durable asset ID"):
            WorkspaceRef(snapshot_asset_id="/tmp/workspace.tar", sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "durable asset ID"):
            Artifact(asset_id="../artifact", mime_type="text/plain")
        with self.assertRaisesRegex(ValueError, "durable asset ID"):
            OmpSessionState(session_id="main", state_asset_id="assets/../state")

    def test_session_mapping_is_copied_and_immutable(self) -> None:
        sessions = {"main": OmpSessionState(session_id="omp-session-001", state_asset_id=None)}
        manifest = self._manifest(sessions)

        sessions["later"] = OmpSessionState(session_id="omp-session-002", state_asset_id=None)

        self.assertEqual(tuple(manifest.omp_sessions), ("main",))
        with self.assertRaises(TypeError):
            manifest.omp_sessions["new"] = OmpSessionState(session_id="omp-session-003", state_asset_id=None)  # type: ignore[index]

    def test_workspace_asset_binding_returns_a_new_manifest(self) -> None:
        manifest = self._manifest()

        bound = manifest.with_workspace_asset_id("workspace-asset-002")

        self.assertEqual(bound.workspace, WorkspaceRef(snapshot_asset_id="workspace-asset-002", sha256="a" * 64))
        self.assertEqual(bound.schema_version, manifest.schema_version)
        self.assertEqual(bound.profile, manifest.profile)
        self.assertEqual(bound.runtime_fingerprint, manifest.runtime_fingerprint)
        self.assertEqual(bound.outputs, manifest.outputs)
        self.assertEqual(bound.omp_sessions, manifest.omp_sessions)
        self.assertEqual(bound.execution, manifest.execution)
        self.assertEqual(bound.manifest_id, manifest.manifest_id)
        self.assertEqual(manifest.workspace.snapshot_asset_id, "workspace-asset-001")
        with self.assertRaisesRegex(ValueError, "durable asset ID"):
            manifest.with_workspace_asset_id("/tmp/workspace.tar")

    def test_duplicate_yaml_keys_are_rejected(self) -> None:
        text = """
schema_version: 1
schema_version: 1
manifest_id: manifest-asset-001
profile: gigachad
runtime_fingerprint: sha256:runtime
workspace: {snapshot_asset_id: workspace-asset-001, sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}
outputs: {events_asset_id: null, artifacts: []}
omp_sessions: {}
execution: {run_id: run-001, status: completed, exit_code: 0, created_at: '2026-09-19T12:00:00Z'}
"""
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            from_yaml(text)


if __name__ == "__main__":
    unittest.main()
