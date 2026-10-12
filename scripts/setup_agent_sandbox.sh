#!/usr/bin/env bash
# Install rootless Docker, the latest gVisor runtime, and sandbox prerequisites on Debian or Arch Linux
# (including Arch derivatives such as Omarchy).
# It then runs setup_host_services.sh to bridge loopback-only host services (Ollama) into containers.
# This script does not remove Docker packages, disable system services, or edit shell startup files.

set -euo pipefail

RUNSC_PATH="$HOME/.local/bin/runsc"
DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
export DOCKER_HOST
# DOCKER_CONTEXT otherwise takes precedence over DOCKER_HOST.
unset DOCKER_CONTEXT
CONFIG_FILE="$HOME/.config/docker/daemon.json"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command '$1' is not installed."
}

rootless_docker_available() {
    docker info --format '{{json .SecurityOptions}}' 2>/dev/null | grep -qi rootless
}

install_rootless_docker_packages() {
    require_command sudo
    [[ -r /etc/os-release ]] || fail "Cannot identify the Linux distribution because /etc/os-release is unavailable."
    . /etc/os-release

    if [[ "$ID" == debian || "$ID" == ubuntu ]]; then
        require_command apt-get
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-rootless-extras uidmap
    elif [[ "$ID" == arch || " ${ID_LIKE:-} " == *" arch "* ]]; then
        local aur_helper
        if command -v yay >/dev/null 2>&1; then
            aur_helper=yay
        elif command -v paru >/dev/null 2>&1; then
            aur_helper=paru
        else
            fail "Arch Linux requires the docker-rootless-extras AUR package. Install yay or paru, then rerun this script."
        fi
        "$aur_helper" -S --needed --noconfirm docker docker-rootless-extras slirp4netns
    else
        fail "Unsupported Linux distribution '$ID' (ID_LIKE='${ID_LIKE:-}'). Supported distributions are Debian, Ubuntu, and Arch Linux or an Arch derivative."
    fi
}

for command in python3 curl sha512sum tar zstd; do
    require_command "$command"
done
GVISOR_ARCH="$(uname -m)"
case "$GVISOR_ARCH" in
    x86_64|aarch64) ;;
    *) fail "Unsupported architecture: $GVISOR_ARCH." ;;
esac

printf '%s\n' 'Ensuring a rootless Docker daemon is available...'
if rootless_docker_available; then
    printf '%s\n' 'Using the existing rootless Docker daemon.'
else
    install_rootless_docker_packages
    for kind in uid gid; do
        if ! awk -F: -v user="$USER" '$1 == user && $3 >= 65536 { found = 1 } END { exit !found }' "/etc/sub$kind"; then
            sudo usermod "--add-sub${kind}s" 100000-165535 "$USER"
        fi
    done

    rootless_setup_tool="$(command -v dockerd-rootless-setuptool.sh || true)"
    if [[ -z "$rootless_setup_tool" && -x /usr/share/docker.io/contrib/dockerd-rootless-setuptool.sh ]]; then
        rootless_setup_tool=/usr/share/docker.io/contrib/dockerd-rootless-setuptool.sh
    fi

    if [[ -n "$rootless_setup_tool" ]]; then
        if systemctl is-active --quiet docker.service; then
            printf '%s\n' 'A rootful Docker service is active. Leaving it running and installing the rootless daemon alongside it.'
            "$rootless_setup_tool" install --force
        else
            "$rootless_setup_tool" install
        fi
        systemctl --user start docker
    elif [[ -f /usr/lib/systemd/user/docker.service || -f "$HOME/.config/systemd/user/docker.service" ]]; then
        # Arch's docker-rootless-extras package ships the user unit but not the
        # Debian-style setup helper.
        printf '%s\n' 'Using the rootless Docker user service provided by the installed package.'
        systemctl --user daemon-reload
        systemctl --user enable --now docker.service
    else
        fail "Docker was installed but its rootless setup tool and user service are unavailable."
    fi
    rootless_docker_available || fail "Rootless Docker is unavailable at $DOCKER_HOST."
fi

temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT
mkdir -p "$(dirname "$RUNSC_PATH")"
archive_name=gvisor.tar.zstd
base_url="https://storage.googleapis.com/gvisor/releases/release/latest/${GVISOR_ARCH}"
printf 'Installing the latest gVisor release to %s...\n' "$(dirname "$RUNSC_PATH")"
curl --fail --location --silent --show-error "$base_url/$archive_name" --output "$temporary_dir/$archive_name"
curl --fail --location --silent --show-error "$base_url/$archive_name.sha512" --output "$temporary_dir/$archive_name.sha512"
(
    cd "$temporary_dir"
    sha512sum --check --status "$archive_name.sha512"
) || fail "gVisor checksum verification failed."
tar --extract --zstd --file "$temporary_dir/$archive_name" --directory "$(dirname "$RUNSC_PATH")"
[[ -x "$RUNSC_PATH" ]] || fail "The gVisor archive did not install an executable runsc at $RUNSC_PATH."

mkdir -p "$(dirname "$CONFIG_FILE")"
if [[ -f "$CONFIG_FILE" ]]; then
    backup_file="${CONFIG_FILE}.agent-sandbox.bak.$(date +%Y%m%d%H%M%S)"
    cp -p "$CONFIG_FILE" "$backup_file"
    printf 'Backed up Docker configuration to %s.\n' "$backup_file"
fi

python3 - "$CONFIG_FILE" "$RUNSC_PATH" <<'PY'
import json
import os
import sys
import tempfile

config_path, runsc_path = sys.argv[1:]
if os.path.exists(config_path):
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
else:
    config = {}
if not isinstance(config, dict):
    raise SystemExit(f"Docker configuration must be a JSON object: {config_path}")
runtimes = config.setdefault("runtimes", {})
if not isinstance(runtimes, dict):
    raise SystemExit(f"Docker configuration field 'runtimes' must be an object: {config_path}")
runtimes["runsc"] = {
    "path": runsc_path,
    "runtimeArgs": ["--ignore-cgroups"],
}
fd, temporary_path = tempfile.mkstemp(dir=os.path.dirname(config_path), prefix="daemon.json.", text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    os.replace(temporary_path, config_path)
finally:
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)
PY

printf '%s\n' 'Configured runsc without CPU, memory, or PID cgroup limits.'

printf '%s\n' 'Restarting the current user Docker service...'
systemctl --user restart docker || fail "Could not restart the user Docker service. Restart it manually, then rerun this script."
rootless_docker_available || fail "Rootless Docker did not become available after restart."
docker info --format '{{json .Runtimes}}' | grep -q '"runsc"' || fail "Docker did not register the runsc runtime."

printf '%s\n' 'Rootless Docker and gVisor runsc are registered.'
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t agent-sandbox:trixie -f "$repository_root/docker/Dockerfile" "$repository_root"
printf '%s\n' 'Built agent-sandbox:trixie.'
"$repository_root/scripts/setup_host_services.sh"
printf '%s\n' 'Run agent-sandbox doctor with a profile to verify container startup and profile prerequisites.'
