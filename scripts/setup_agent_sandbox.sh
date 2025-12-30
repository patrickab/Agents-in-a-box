#!/bin/bash
# Configure a rootless Docker + gVisor sandbox on Debian/Ubuntu hosts.
#
# Behavior:
# - Installs required system packages via apt.
# - Configures /etc/subuid and /etc/subgid for the current user.
# - Downloads and verifies gVisor runsc, installs to /usr/local/bin.
# - Installs Docker CE with rootless extras if missing.
# - Runs dockerd-rootless-setuptool.sh.
# - Writes ~/.config/docker/daemon.json to register runsc with --ignore-cgroups.
# - Restarts the user-level docker service and verifies rootless + runsc.
#
# Constraints:
# - NO modification of .bashrc, .zshrc, or .profile.
# - NO global environment variable persistence.

# Exit immediately if a command exits with a non-zero status
set -e

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
# 2. Configure Subordinate UIDs/GIDs
# ---------------------------------------------------------
echo -e "${BLUE}[2/5] Configuring Subordinate UIDs/GIDs...${NC}"

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

ARCH=$(dpkg --print-architecture)
case "$ARCH" in
    amd64) GVISOR_ARCH="x86_64" ;;
    arm64) GVISOR_ARCH="aarch64" ;;
    *) echo -e "${RED}Unsupported architecture: $ARCH${NC}"; exit 1 ;;
esac

URL="https://storage.googleapis.com/gvisor/releases/release/latest/${GVISOR_ARCH}"

wget -q "${URL}/runsc" -O runsc
wget -q "${URL}/runsc.sha512" -O runsc.sha512

EXPECTED_HASH=$(awk '{print $1}' runsc.sha512)
echo "$EXPECTED_HASH  runsc" | sha512sum -c - --status

if [ $? -eq 0 ]; then
    chmod a+x runsc
    sudo mv runsc /usr/local/bin/runsc
    rm runsc.sha512
    echo -e "${GREEN}    gVisor installed to /usr/local/bin/runsc${NC}"
else
    echo -e "${RED}    Checksum failed! Exiting.${NC}"
    rm runsc runsc.sha512
    exit 1
fi

# ---------------------------------------------------------
# 4. Install Rootless Docker
# ---------------------------------------------------------
echo -e "${BLUE}[4/5] Installing Rootless Docker...${NC}"

# Remove conflicting legacy packages
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
    sudo apt-get remove -y $pkg >/dev/null 2>&1 || true
done

# Disable system-wide docker
if systemctl is-active --quiet docker; then
    sudo systemctl disable --now docker.service docker.socket || true
fi

# Install Official Docker CE + Rootless Extras
if ! command -v dockerd-rootless-setuptool.sh >/dev/null 2>&1; then
    echo -e "${YELLOW}    Installing official Docker packages...${NC}"
    curl -fsSL https://get.docker.com | sh
    sudo apt-get install -y -qq docker-ce-rootless-extras
fi

# Enable Linger
sudo loginctl enable-linger "$USER"

# Run the setup tool
echo -e "    Running Rootless Setup Tool..."
dockerd-rootless-setuptool.sh install --force

# ---------------------------------------------------------
# 5. Configure Docker Daemon for gVisor
# ---------------------------------------------------------
echo -e "${BLUE}[5/5] Configuring Docker Daemon to use gVisor...${NC}"

CONFIG_DIR="$HOME/.config/docker"
CONFIG_FILE="$CONFIG_DIR/daemon.json"

mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_FILE" ]; then echo '{}' > "$CONFIG_FILE"; fi

# Configure runsc with ignore-cgroups
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
sleep 5

# Define temporary variables for verification (NOT exported to shell config)
TEMP_DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
TEMP_PATH="$HOME/bin:$PATH"

# Verify using explicit environment
DOCKER_ROOTLESS=$(DOCKER_HOST="$TEMP_DOCKER_HOST" PATH="$TEMP_PATH" docker info -f '{{.SecurityOptions}}' 2>/dev/null | grep "rootless" || echo "")
DOCKER_RUNTIMES=$(DOCKER_HOST="$TEMP_DOCKER_HOST" PATH="$TEMP_PATH" docker info -f '{{.Runtimes}}' 2>/dev/null | grep "runsc" || echo "")

if [[ -n "$DOCKER_ROOTLESS" && -n "$DOCKER_RUNTIMES" ]]; then
    echo -e "${GREEN}SUCCESS! Environment is ready.${NC}"
    echo -e "  - Rootless Mode: ${GREEN}Active${NC}"
    echo -e "  - gVisor Runtime: ${GREEN}Registered${NC}"
    echo -e "${YELLOW}NOTE: No changes were made to your .bashrc/.zshrc.${NC}"
    echo -e "      Your Python sandbox module must inject DOCKER_HOST/PATH manually."
else
    echo -e "${RED}WARNING: Verification failed.${NC}"
    echo "Rootless status (Expected 'rootless'): $DOCKER_ROOTLESS"
    echo "Runtimes found (Expected 'runsc'): $DOCKER_RUNTIMES"
    echo "Check 'systemctl --user status docker' for logs."
    exit 1
fi