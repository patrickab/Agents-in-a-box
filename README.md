# Agents-in-a-Box

**A sandbox runtime for coding agents.** Runs [Oh My Pi (OMP)](https://github.com/can1357/oh-my-pi) in disposable containers using rootless Docker and gVisor, with a Python API and CLI.

---

## 🎯 Purpose

Coding agents typically run directly on the host machine with full user privileges

By default this means:
*   **Arbitrary code generation & execution**
*   **Nearly unrestricted filesystem access**
*   **Unrestricted network access**

While coding agents typically ask for permission or filter auto-approval for harmful commands, they can technically send any user-accessible (private) data to any publicly accessible domain, download & install arbitrary software in the userspace or execute arbitrary code.

- **Fresh containers:** each run starts clean and is removed afterward.
- **Explicit access:** profiles define host mounts, environment variables, and local services such as Ollama.
- **Workspace continuity:** verified archives and session manifests let work continue across runs without keeping a container alive.
- **Isolated Python:** run small scripts with a profile-defined interpreter and networking disabled.
 
---
 
## Security architecture: defense in depth


[Rootless Docker](https://docs.docker.com/engine/security/rootless/) runs without host root privileges. [gVisor](https://github.com/google/gvisor) adds a user-space kernel between the agent and the host. Containers have a read-only root filesystem, bounded writable temporary storage, all Linux capabilities dropped, and `no-new-privileges` enabled.

Profiles are trusted administrator policy. Prefer read-only mounts and grant only the secrets and services a task needs. Workspace archives are checked for unsafe paths, links, and file types. Host mounts inside the workspace are excluded from captures.

```mermaid
flowchart TD
    App[Host application or CLI] --> Runtime[Sandbox runtime]
    Profile["Trusted profile<br/>mounts, environment, host services"] --> Runtime

    subgraph Isolation[Rootless Docker and gVisor]
        subgraph Container[Fresh container with read-only root]
            OMP[OMP process]
            Workspace[Managed workspace]
            Home[Ephemeral OMP home]
            OMP <--> Workspace
            OMP --> Home
        end
    end

    Runtime -->|create, run, remove| Container
    Archive[Verified workspace archive] -->|restore| Workspace
    Workspace -->|capture archive and manifest| App
    Mounts[Profile-declared host paths] <-->|read-only or read-write mounts| OMP
    OMP <-->|bridge networking| Network["Declared host services and network<br/>no enforced egress allowlist"]
```

---
 
## Setup

Requires Python 3.10+, `uv`, a user systemd session, and `curl`, `tar`, `zstd`, and `sha512sum`. The setup script supports Debian, Ubuntu, and Arch Linux on `x86_64` and `aarch64`.

```bash
uv sync
./scripts/setup_agent_sandbox.sh
```

Setup installs or reuses rootless Docker, verifies and configures gVisor, builds the sandbox image, and configures access to declared host services. It uses `sudo` for host-level setup when needed.

Setup targets the current user's Docker socket at `/run/user/<uid>/docker.sock`. It leaves any rootful Docker service running and preserves unrelated settings in `~/.config/docker/daemon.json`, backing up the file before adding the `runsc` runtime.

Before running, copy the [bundled profile](src/agent_sandbox/profiles/gigachad.yaml) to an administrator-controlled directory as `example.yaml`. Set its `name` to `example` and adjust its absolute paths, mounts, and credentials for your machine. Mount the actual OMP executable, not a launcher such as a `mise` shim.

```bash
export AGENT_SANDBOX_PROFILE_PATH="/absolute/path/to/profiles"
uv run agent-sandbox doctor --profile example
```

Doctor checks Docker, gVisor, the image, mounts, OMP, interpreters, instructions, and declared services. It stops at the first failing stage.

<details>
<summary>Profile fields and local model access</summary>

Profiles are strict YAML files named `<profile>.yaml`. `AGENT_SANDBOX_PROFILE_PATH` accepts colon-separated directories, searched before bundled profiles.

| Field | Purpose |
| --- | --- |
| `name`, `image`, `workdir` | Profile identity, Docker image, and absolute working directory |
| `mounts` | Explicit host paths with container targets and `ro` or `rw` access |
| `omp_binary` | Path to the OMP executable inside the container |
| `python_interpreters` | Optional labels mapped to interpreter paths inside the container |
| `env_passthrough` | Optional host environment variables to forward |
| `host_services` | Optional environment-variable names mapped to host service ports |

For local inference, `host_services: {OLLAMA_HOST: 11434}` exposes Ollama as `http://10.200.200.1:11434`. The host service stays bound to `127.0.0.1`. Setup creates the loopback alias and per-port systemd proxies, defaulting to port `11434`.

To expose additional profile-declared ports, or configure an existing installation:

```bash
HOST_SERVICE_PORTS="11434 8080" ./scripts/setup_host_services.sh
```

</details>

---

## Usage

### CLI

Run a task and save its workspace and session manifest. Output directories must already exist.

```bash
mkdir -p artifacts
uv run agent-sandbox omp "Summarize the current workspace." \
  --profile example \
  --workspace-out artifacts/workspace.tar \
  --manifest-out artifacts/manifest.json
```

To continue the session, supply both saved files:

```bash
uv run agent-sandbox omp "Continue the workspace." \
  --profile example \
  --workspace-in artifacts/workspace.tar \
  --manifest-in artifacts/manifest.json \
  --workspace-out artifacts/next-workspace.tar \
  --manifest-out artifacts/next-manifest.json
```

Use `--model`, `--thinking`, `--append-system`, and `--lean` for per-run options. See `uv run agent-sandbox omp --help` for all flags. The command prints status, summary, and output events. Exit codes are `0` for completion, `1` for a non-completed result after saving outputs, and `2` for invalid inputs or sandbox errors.

### Python API

The library is the primary integration for applications that manage their own requests and storage.

```python
from agent_sandbox import OmpTask, Sandbox, SandboxInvocation

sandbox = Sandbox(namespace="tenant-a")
invocation = SandboxInvocation(
    slot_key="workspace-42",
    profile_name="example",
    run_id="request-001",
    task=OmpTask("Summarize the current workspace.", lean=True),
)

with sandbox.run(invocation) as result:
    print(result.status, result.summary)
    print(result.outputs)
    # Persist result.workspace and result.next_manifest here.
    # The local workspace archive is deleted when this block exits.
```

The caller owns storage. Save the archive at `result.workspace.path` with its `sha256` and the returned `next_manifest` before releasing the run.

<details>
<summary>Restoring workspaces and managing results</summary>

- **Restore files:** pass `workspace=WorkspaceArchive(path, sha256)` with a caller-downloaded archive. `WorkspaceArchive.from_file(path)` validates a local archive and records its digest.
- **Resume the session:** also pass the previous manifest as `active_manifest`. Its archive digest, profile, and image must match.
- **Read an artifact:** `result.read_workspace_file()` safely reads a bounded regular file from the captured workspace while the result is retained.
- **Release:** exiting the context manager, or calling `result.release()`, removes the temporary capture and cached result. `slot_key` and `run_id` provide in-process idempotency within one `Sandbox` instance.
- **Task options:** `OmpTask` accepts `prompt_images` (a tuple of `PromptImage`), `model`, `thinking`, `append_system`, `lean`, and `timeout_s` (seconds, 1 to 86400, default 300).
- **Errors:** sandbox failures derive from `AgentSandboxError`. Invalid invocation inputs raise `ValueError` or `TypeError`.

</details>


---


### Isolated Python

Choose an interpreter label declared in the profile, not a host executable path. Each script runs in a fresh, read-only, network-disabled container.

```python
from agent_sandbox import Sandbox

output = Sandbox().run_python_script(
    "venv",
    "print('hello from the sandbox')",
    timeout=30,
    profile_name="example",
)
print(output)
```

### Workspace directories

`WorkspaceArchive.from_directory` tars a host directory into a pinned archive that a `SandboxInvocation` can restore, and `extract_archive` writes a captured workspace back into a directory.

```python
from agent_sandbox import WorkspaceArchive, extract_archive

workspace = WorkspaceArchive.from_directory("/path/to/worktree", "/tmp/in.tar", exclude=(".git", "AGENTS.md"))
# Run a SandboxInvocation with workspace=workspace, then write the captured workspace back:
extract_archive(result.workspace.path, "/path/to/worktree", exclude=(".git", "AGENTS.md"))
```

- `from_directory` writes relative members, skips `exclude` patterns (a top-level path or an fnmatch glob in which `*` also crosses `/`, matched against each path and its ancestors), and keeps symlinks as links. Links that leave the workspace are refused.
- `extract_archive` validates the archive, writes it into the directory, and deletes files the archive lacks except `exclude` patterns. It never writes through symlinks.
- Exclude `profile.workdir_mount_paths()` (profile mounts that live inside the workdir, such as `AGENTS.md`): those paths are not restored or captured, so extracting would otherwise delete the host copy.

<details>
<summary>Runtime limits</summary>

| Boundary | Limit |
| --- | --- |
| OMP execution | 300 seconds by default; `OmpTask(timeout_s=...)` accepts 1 to 86400 |
| Prompt images | 32 MiB combined |
| Retained stdout and stderr | 4 MiB combined; a larger output fails the run |
| Workspace archive | 512 MiB |
| `append_system` | 64 KiB |
| `/runtime` and `/tmp` tmpfs | 1 GiB and 64 MiB respectively |
| Container CPU, memory, and PIDs | No cgroup limits |

</details>

---

## Agent factory

`agent_sandbox.factory` is an optional service that turns task requests into reviewed, merged code. It runs a durable [DBOS](https://docs.dbos.dev) pipeline per task: a git worktree, optional scout and plan turns, coder rounds with hooks, gates, and an optional review, a live preview, a human review stop, and a fast-forward merge. Agent turns run inside the sandbox (`sandbox: true`) or directly on the host. Hooks, gates, previews, and preview scenarios always run on the host in the task worktree with your user's privileges: the sandbox image has no project toolchain, and `node_modules` and `.venv` never enter the sandbox archive. So `sandbox: true` isolates the agent's own shell, not the code it writes once the pipeline executes it. Repos need no config files: register a git main checkout (`POST /v1/repos`) and every task runs with packaged defaults, which you edit engine-wide through `GET/PUT /v1/defaults`.

```bash
uv sync --extra factory          # DBOS, FastAPI, uvicorn, psycopg
export FACTORY_DATABASE_URL=postgresql://user:password@127.0.0.1:5432/factory
uv run agent-sandbox factory serve [--config PATH]
```

The engine listens on `127.0.0.1:8090` by default. Killing it mid-run is safe: on restart DBOS resumes every interrupted workflow from its last finished step, the interrupted agent turn re-runs with `--continue`, and a leftover host `omp` process from the crashed engine is killed first.

### Engine config

`~/.config/agent-sandbox/factory.yaml` (override with `--config` or `AGENT_SANDBOX_FACTORY_CONFIG`). `${VAR}` and `${VAR:-default}` are expanded from the environment.

```yaml
listen: 127.0.0.1:8090
database_url: ${FACTORY_DATABASE_URL}   # DBOS system database and the engine tables (tasks, phase_runs, messages, events)
state_dir: ~/.local/state/agent-sandbox/factory   # sessions, archives, logs, attachments per task
worktree_root: ~/git/worktrees          # task worktrees live at <worktree_root>/<repo>/<task_id>
defaults_dir: ~/.config/agent-sandbox/factory   # user defaults: defaults.yaml (settings) and agents/<name>.md
preview_ports: [8100, 8199]
limits:
  in_progress: 4                        # tasks working at once; read at startup, so changing it needs an engine restart
otel_endpoint: null                     # OTLP traces endpoint for DBOS, optional
omp_binary: omp                         # host omp for `sandbox: false` runs
```

### Repos, defaults, and templates

Repos live in the engine database (`repos(name, path, created_at)`), registered at runtime with `POST /v1/repos {"path": "/abs/main/checkout"}`. Registration is idempotent by resolved path; the path must be absolute and a git main checkout (a `.git` directory, not a linked worktree); the name is the directory name sanitized to `[A-Za-z0-9][A-Za-z0-9_.-]*` and suffixed `-2`, `-3` on a clash.

The engine reads the layers below when a task starts and freezes the result into the task.

- Built-in defaults packaged in `agent_sandbox.factory`: settings (`merge: review`, `max_rounds: 3`, `sandbox: true`, `preview: true`, `models: {}`, `agent_timeout_s: 3600`, `command_timeout_s: 1800`), the agents `scout`, `planner`, `coder`, `reviewer` (the phases `scout`, `plan`, `code`, `review` use them in that order), and the workflows `code`, `scout-code`, `scout-plan-code`, `scout-plan-code-review`. The default sandbox profile is the bundled `factory-default`.
- User defaults in `defaults_dir`, edited with `GET/PUT /v1/defaults` (`{"settings": {...}, "agents": {"<name>": "<markdown>"}}`): `defaults.yaml` overrides the settings, `agents/<name>.md` overrides a built-in agent. `PUT` validates every setting and agent (unknown agent names are rejected with 422) and writes atomically; an agent saved with its built-in text drops its override file.
- Optional `factory.yaml` in the repo's main checkout, every field optional: `main_branch` (default: the branch checked out in the main checkout when the task starts; a detached HEAD is an error), `profile` (sandbox profile, default `factory-default`), `sandbox`, `hooks` (`after_create`, `after_code`, `before_remove`, each a list of argv lists), `gates` (argv lists run in order until one fails), `protected` globs (`factory.yaml`, `.factory/**`, `.omp/agents/**` are always protected), optional `preview:` block (`processes` with `name`, `argv`, `ready_url`, plus `env`, `database`, `url`, `api_url`, `ready_timeout_s`), and optional `merge`, `max_rounds`, `models`, `agent_timeout_s` (default 3600), `command_timeout_s` (default 1800). Unknown keys are rejected: the task queue concurrency is an engine setting (`limits.in_progress`), not a repo setting.
- Placeholders: `{repo_path}` (the repo's main checkout) is expanded in `hooks`, `gates`, and every `preview` field when a task starts. Preview fields also expand `{port.<process name>}`, `{database_url}` (with `database: true`), `{data_dir}`, `{worktree}`, and `{task_id}` in process `argv` and `ready_url`, `env` values, `url`, and `api_url`. `api_url` (default: `url`) is the backend API base the engine passes to the coder-written preview scenario as `--api`.
- `.factory/workflows/<id>.yaml` (the same id overrides a built-in workflow): `name`, `phases` (ordered subset of `scout, plan, code, review`, `code` required), `max_rounds`, `merge` (`auto` or `review`), `preview`, `sandbox`, `models`.
- `.omp/agents/<name>.md` (overrides the user and built-in agent of the same name; names are fixed to `scout`, `planner`, `coder`, `reviewer`): YAML front matter (`name`, `description`, `model`, `thinking`, `writes` globs, `output`) and a markdown body sent as the system prompt. The coder may write `**` by default; every other agent is read-only and its changes are discarded.

Settings merge as built-in defaults, then user defaults, then `factory.yaml`, then the workflow, then the request `overrides`, and are validated afterwards. Agents resolve as repo `.omp/agents/<name>.md`, then user default, then built-in. Requests can only override `merge`, `max_rounds`, `preview`, `sandbox`, and `models`.

Agents report through a JSON envelope in `.factory/out/<phase>.json` (the engine tells each agent the exact shape). A missing or malformed envelope gets one correction turn with `--continue`, then the task blocks. Writes outside an agent's allowlist or to protected paths are reverted, the agent is told, and the attempt costs a round. `.factory/out/` is engine-owned: it is never committed and never counts as a breach, and sandboxed turns exchange everything in it with the worktree except the JSON envelopes (those are read from the sandbox capture).

### Pipeline and states

Task states: `backlog`, `todo`, `in_progress`, `human_review`, `rework`, `blocked`, `merging`, `done`, `cancelled`. Worktrees live at `<worktree_root>/<repo>/<task_id>` on branch `factory/<task_id>` and stay uncommitted until the merge. A message in `human_review` or `blocked` starts a rework turn, approval starts the merge. The merge commits the changed files (never protected or engine-owned paths), rebases onto `main_branch`, reruns the gates, and runs `git merge --ff-only` in the main checkout, which must be on `main_branch`. Uncommitted edits in that checkout are left alone unless they overlap the merge, in which case git refuses and its message becomes the block reason. `retry` tries again: it wakes the task's workflow, or, when that workflow itself ended in error, starts a fresh one that resumes at the coder phase (or the merge) in the same worktree and session. Cancelling stops the work, the preview, and removes the worktree and branch.

### HTTP API

JSON, snake_case, under `/v1`. Errors are `{"error": "not_found" | "conflict" | "invalid", "detail": str}` with 404, 409, 422.

| Endpoint | Allowed task states |
| --- | --- |
| `GET /health`, `GET /repos` (`[{"name", "path"}]`, `path` = main checkout), `POST /repos` (`path`; returns the `{"name", "path"}` entry), `GET /repos/{repo}/workflows` | |
| `GET /defaults`, `PUT /defaults` (`settings`, `agents`: name to full markdown) | |
| `POST /tasks` (`repo`, `workflow`, `title`, `request`, `images`, `overrides`, `start`), `GET /tasks?repo=&state=`, `GET /tasks/{id}` | |
| `GET /events?after=<seq>&repo=` (at most 500, ordered by `seq`), `GET /events/latest` (`{"seq": int}`, `0` when empty; start pollers here) | |
| `POST /tasks/{id}/start` | `backlog` |
| `POST /tasks/{id}/messages` (`text`, `images`) | `human_review`, `blocked` |
| `POST /tasks/{id}/approve`, `POST /tasks/{id}/preview/restart` | `human_review` |
| `POST /tasks/{id}/retry` | `blocked` |
| `POST /tasks/{id}/cancel` | any state except `done` and `cancelled` |

### Development

`uv sync` installs the `factory` extra through the dev group. Database-backed tests need a Postgres database they may reset: `FACTORY_TEST_DATABASE_URL=postgresql://user:password@127.0.0.1:5432/factory_test uv run pytest tests/factory`. Without it those tests are skipped.
