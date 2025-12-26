#!/bin/bash
# Configure a rootless Docker + gVisor sandbox on Debian/Ubuntu hosts.
#
# Behavior:
# - Installs required system packages via apt.
# - Configures /etc/subuid and /etc/subgid for the current user.
# - Downloads and verifies gVisor runsc, installs to /usr/local/bin.
# - Installs Docker CE with rootless extras if missing.
# - Runs dockerd-rootless-setuptool.sh and configures DOCKER_HOST/PATH.
# - Writes ~/.config/docker/daemon.json to register runsc with --ignore-cgroups.
# - Restarts the user-level docker service and verifies rootless + runsc.
#
# Assumptions:
# - Running on a modern Debian/Ubuntu system with sudo access.
# - User shell is bash or zsh; PATH/DOCKER_HOST are persisted in the shell rc file.
# - Host and container architectures must match (amd64/x86_64 or arm64/aarch64).
#
# Side effects:
# - Modifies /etc/subuid and /etc/subgid.
# - Installs or removes Docker-related packages.
# - Enables systemd user lingering for the current user.
# - Creates or overwrites ~/.config/docker/daemon.json.

# Exit immediately if a command exits with a non-zero status
set -e

# [ADD] Disclaimer about sandbox setup and assumptions
echo -e "${YELLOW}[Disclaimer]${NC} This script sets up a Rootless Docker + gVisor sandbox."
echo -e " • Ensure 'aider' (and other tools) are installed in one of the host's mounted bin dirs (e.g. /usr/local/bin)."
echo -e " • Verify host & container share the same CPU architecture (x86_64 vs aarch64)."
echo -e " • This script is tested on modern Debian/Ubuntu systems. Use at your own risk!"
echo

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}[+] Starting Secure Sandbox Setup (Rootless Docker + gVisor)...${NC}"

# ---------------------------------------------------------
# 1. System Preparation & Dependencies
# ---------------------------------------------------------
echo -e "${BLUE}[1/5] Installing system dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    uidmap \
    dbus-user-session \
    fuse-overlayfs \
    slirp4netns \
    jq \
    curl \
    wget \
    iptables \
    git \
    ca-certificates \
    gnupg \
    lsb-release

# ---------------------------------------------------------
# 2. Configure Subordinate UIDs/GIDs (CRITICAL FIX)
# ---------------------------------------------------------
echo -e "${BLUE}[2/5] Configuring Subordinate UIDs/GIDs...${NC}"

# Rootless Docker requires the user to have a range of UIDs/GIDs in /etc/subuid and /etc/subgid
if ! grep -q "^$USER:" /etc/subuid; then
    echo -e "${YELLOW}    Adding subuid entry for $USER...${NC}"
    echo "$USER:100000:65536" | sudo tee -a /etc/subuid
else
    echo -e "${GREEN}    Subuid entry exists.${NC}"
fi

if ! grep -q "^$USER:" /etc/subgid; then
    echo -e "${YELLOW}    Adding subgid entry for $USER...${NC}"
    echo "$USER:100000:65536" | sudo tee -a /etc/subgid
else
    echo -e "${GREEN}    Subgid entry exists.${NC}"
fi

# ---------------------------------------------------------
# 3. Install gVisor (runsc)
# ---------------------------------------------------------
echo -e "${BLUE}[3/5] Installing gVisor (runsc)...${NC}"

# Map dpkg architecture (amd64) to gVisor URL format (x86_64)
ARCH=$(dpkg --print-architecture)
case "$ARCH" in
    amd64) GVISOR_ARCH="x86_64" ;;
    arm64) GVISOR_ARCH="aarch64" ;;
    *) echo -e "${RED}Unsupported architecture: $ARCH${NC}"; exit 1 ;;
esac

URL="https://storage.googleapis.com/gvisor/releases/release/latest/${GVISOR_ARCH}"

# Download runsc binary
wget -q "${URL}/runsc" -O runsc
wget -q "${URL}/runsc.sha512" -O runsc.sha512

# Robust checksum verification
EXPECTED_HASH=$(awk '{print $1}' runsc.sha512)
echo "$EXPECTED_HASH  runsc" | sha512sum -c - --status

if [ $? -eq 0 ]; then
    echo -e "${GREEN}    Checksum verified.${NC}"
else
    echo -e "${RED}    Checksum failed! Exiting.${NC}"
    rm runsc runsc.sha512
    exit 1
fi

# Make executable and move to global path
chmod a+x runsc
sudo mv runsc /usr/local/bin/runsc
rm runsc.sha512

echo -e "${GREEN}    gVisor installed to /usr/local/bin/runsc${NC}"

# ---------------------------------------------------------
# 4. Install Rootless Docker
# ---------------------------------------------------------
echo -e "${BLUE}[4/5] Installing Rootless Docker...${NC}"

# Remove conflicting legacy packages
echo -e "    Cleaning up potential conflicting packages..."
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
    sudo apt-get remove -y $pkg >/dev/null 2>&1 || true
done

# Disable system-wide docker if it exists
if systemctl is-active --quiet docker; then
    sudo systemctl disable --now docker.service docker.socket || true
fi

# Install Official Docker CE + Rootless Extras if missing
if ! command -v dockerd-rootless-setuptool.sh >/dev/null 2>&1; then
    echo -e "${YELLOW}    Installing official Docker packages...${NC}"
    curl -fsSL https://get.docker.com | sh
    sudo apt-get install -y -qq docker-ce-rootless-extras
fi

# Enable Linger (Required for Rootless Docker to stay running)
sudo loginctl enable-linger "$USER"

# Run the setup tool
echo -e "    Running Rootless Setup Tool..."
dockerd-rootless-setuptool.sh install --force

# Determine Shell for Path Export
SHELL_CONFIG=""
if [[ "$SHELL" == */zsh ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == */bash ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    SHELL_CONFIG="$HOME/.bashrc"
fi

# Export variables for current session
export PATH=/home/$USER/bin:$PATH
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock

# Add to shell config for persistence
if ! grep -q "export PATH=/home/$USER/bin:\$PATH" "$SHELL_CONFIG"; then
    echo "" >> "$SHELL_CONFIG"
    echo '# Rootless Docker Config' >> "$SHELL_CONFIG"
    echo 'export PATH=/home/$USER/bin:$PATH' >> "$SHELL_CONFIG"
    echo 'export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock' >> "$SHELL_CONFIG"
    echo -e "${GREEN}    Added Docker exports to $SHELL_CONFIG${NC}"
fi

# ---------------------------------------------------------
# 5. Configure Docker Daemon for gVisor
# ---------------------------------------------------------
echo -e "${BLUE}[5/5] Configuring Docker Daemon to use gVisor...${NC}"

CONFIG_DIR="$HOME/.config/docker"
CONFIG_FILE="$CONFIG_DIR/daemon.json"

mkdir -p "$CONFIG_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo '{}' > "$CONFIG_FILE"
fi

# Configure runsc with ignore-cgroups (critical for rootless)
jq '
  .runtimes.runsc.path = "/usr/local/bin/runsc" |
  .runtimes.runsc.runtimeArgs = ["--ignore-cgroups"]
' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"

echo -e "${GREEN}    Daemon configuration updated at $CONFIG_FILE${NC}"

# ---------------------------------------------------------
# 6. Restart & Verify
# ---------------------------------------------------------
echo -e "${BLUE}[+] Restarting Docker and Verifying...${NC}"

systemctl --user restart docker

# Wait for socket
sleep 5

# Verification Checks
DOCKER_ROOTLESS=$(docker info -f '{{.SecurityOptions}}' 2>/dev/null | grep "rootless" || echo "")
DOCKER_RUNTIMES=$(docker info -f '{{.Runtimes}}' 2>/dev/null | grep "runsc" || echo "")

if [[ -n "$DOCKER_ROOTLESS" && -n "$DOCKER_RUNTIMES" ]]; then
    echo -e "${GREEN}SUCCESS! Environment is ready.${NC}"
    echo -e "  - Rootless Mode: ${GREEN}Active${NC}"
    echo -e "  - gVisor Runtime: ${GREEN}Registered${NC}"
    echo -e "${YELLOW}IMPORTANT: Run 'source $SHELL_CONFIG' or restart your terminal to use docker.${NC}"
    echo -e "Test it with: docker run --rm --runtime=runsc hello-world"

    # [ADD] Usage note for interactive sandbox with mounted host binaries
    echo
    echo -e "${BLUE}[+] To run an interactive sandbox with mounted host binaries, you can do:${NC}"
    echo -e "${GREEN}docker run --runtime=runsc --init --user=1000:1000 -w /app \\"
    echo -e "    -v /usr/bin:/usr/bin:ro \\"
    echo -e "    -v /usr/lib:/usr/lib:ro \\"
    echo -e "    -v /lib:/lib:ro \\"
    echo -e "    -v /usr/local/bin:/usr/local/bin:ro \\"
    echo -e "    -it agent-sandbox:latest \\"
    echo -e "    bash${NC}"
    echo
    echo -e "That command provides a shell with read-only access to host binaries (like 'aider')."
    echo -e "Enjoy your fully isolated rootless + gVisor sandbox!"
else
    echo -e "${RED}WARNING: Verification failed.${NC}"
    echo "Rootless status (Expected 'rootless'): $DOCKER_ROOTLESS"
    echo "Runtimes found (Expected 'runsc'): $DOCKER_RUNTIMES"
    echo "Check 'systemctl --user status docker' for logs."
    exit 1
fi
