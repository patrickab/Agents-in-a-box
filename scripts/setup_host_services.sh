#!/usr/bin/env bash
# Let sandbox containers reach selected loopback-only host services (default: Ollama on 11434).
#
# Rootless Docker runs slirp4netns with --disable-host-loopback, so containers cannot reach
# 127.0.0.1. Enabling host loopback would expose every local daemon (databases, app servers)
# to the agent. Instead this adds a dedicated lo alias and proxies only the listed ports from
# it to 127.0.0.1, so the services themselves stay loopback-bound.
#
# Idempotent. Uses sudo once for the persistent alias unit. Usage:
#   ./scripts/setup_host_services.sh                   # proxy port 11434
#   HOST_SERVICE_PORTS="11434 8080" ./scripts/setup_host_services.sh

set -euo pipefail

# Must match HOST_SERVICE_ADDRESS in src/agent_sandbox/config.py.
HOST_SERVICE_ADDRESS="10.200.200.1"
# Must cover every port under the profile's host_services.
HOST_SERVICE_PORTS="${HOST_SERVICE_PORTS:-11434}"
ALIAS_UNIT=/etc/systemd/system/agent-sandbox-host-alias.service
USER_UNIT_DIR="$HOME/.config/systemd/user"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

command -v sudo >/dev/null 2>&1 || fail "sudo is required to install $ALIAS_UNIT."
command -v systemctl >/dev/null 2>&1 || fail "systemd is required."
systemctl --user show-environment >/dev/null 2>&1 || fail "A user systemd session is required."
proxyd="$(ls /usr/lib/systemd/systemd-socket-proxyd /lib/systemd/systemd-socket-proxyd 2>/dev/null | head -n1 || true)"
[[ -n "$proxyd" ]] || fail "systemd-socket-proxyd is unavailable."

printf 'Installing persistent loopback alias %s...\n' "$HOST_SERVICE_ADDRESS"
sudo tee "$ALIAS_UNIT" >/dev/null <<EOF
[Unit]
Description=Loopback alias for agent-sandbox host services
After=network-pre.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=ip addr replace ${HOST_SERVICE_ADDRESS}/32 dev lo
ExecStop=ip addr del ${HOST_SERVICE_ADDRESS}/32 dev lo

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable agent-sandbox-host-alias.service
sudo systemctl restart agent-sandbox-host-alias.service

mkdir -p "$USER_UNIT_DIR"
for port in $HOST_SERVICE_PORTS; do
    [[ "$port" =~ ^[0-9]+$ ]] && (( port > 0 && port < 65536 )) || fail "Invalid port '$port'."
    printf 'Proxying %s:%s -> 127.0.0.1:%s...\n' "$HOST_SERVICE_ADDRESS" "$port" "$port"
    cat >"$USER_UNIT_DIR/agent-sandbox-host-$port.socket" <<EOF
[Unit]
Description=agent-sandbox bridge for host port $port

[Socket]
ListenStream=${HOST_SERVICE_ADDRESS}:$port
FreeBind=true

[Install]
WantedBy=sockets.target
EOF
    cat >"$USER_UNIT_DIR/agent-sandbox-host-$port.service" <<EOF
[Unit]
Description=agent-sandbox bridge for host port $port

[Service]
ExecStart=$proxyd 127.0.0.1:$port
EOF
done
systemctl --user daemon-reload
for port in $HOST_SERVICE_PORTS; do
    systemctl --user enable "agent-sandbox-host-$port.socket"
    systemctl --user restart "agent-sandbox-host-$port.socket"
done

ip -4 addr show dev lo | grep -q "inet ${HOST_SERVICE_ADDRESS}/32" || fail "Alias $HOST_SERVICE_ADDRESS is not on lo."
for port in $HOST_SERVICE_PORTS; do
    # Probing the alias would always succeed: the socket-activated proxy accepts before dialing out.
    if ss -Hltn "sport = :$port" | grep -qE "(127\.0\.0\.1|0\.0\.0\.0|\*|\[::\]):$port\b"; then
        printf 'OK   %s:%s forwards to a listening 127.0.0.1:%s\n' "$HOST_SERVICE_ADDRESS" "$port" "$port"
    else
        printf 'WARN %s:%s is bridged, but nothing listens on 127.0.0.1:%s yet\n' "$HOST_SERVICE_ADDRESS" "$port" "$port"
    fi
done
printf '%s\n' 'Verify from inside the sandbox with: uv run agent-sandbox doctor --profile <profile>'
