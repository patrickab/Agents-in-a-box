# Agent-in-a-Box

**Agent-in-a-Box** is a security-first orchestration layer that wraps arbitrary CLI-based AI Agents into a unified, sandboxed runtime. It allows developers to deploy AI agents against local repositories with zero operational risk to the host operating system. After execution the agents container is destroyed & only the user-specified directory remains on the system for inspection.

Supports:
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- [Qwen Code](https://github.com/QwenLM/qwen-code)
- [Aider](https://aider.chat/)

---

## 🎯 Purpose

Most state-of-the-art coding agents—such as **Aider**, **Gemini CLI**, and **Qwen Code**—are typically designed for frictionless adoption, operating directly on the host machine with the full privileges of the active user.

In their default configuration, these agents possess:
*   **Unrestricted Filesystem Access:** Agents can read, modify, or delete any file accessible to the user, extending far beyond the target repository.
*   **Arbitrary Code Execution:** The ability to generate and execute scripts (Python, Bash, etc.) directly on the host OS.

**Agent-in-a-Box** mitigates this operational risk by decoupling the agent from the host. It functions as a containment layer, providing a unified, ergonomic interface to orchestrate these tools within a heavily restricted runtime.

---

## 🛡️ Security Architecture

### **Zero-Trust Execution Policy**

Any agentic system is treated as potentially malicious.

### 1. Rootless Infrastructure
Excludes any potential of container breakout attacks by removing root privileges from the container.
*   **Unprivileged Daemon:** The Docker Engine runs without root privileges on the host.
*   **Identity Mapping:** Utilizes user namespaces to map the container's `root` user to a non-privileged user on the host. Even if an attacker breaks out of the container, they find themselves with zero permissions on the host machine (no read, no read, no execute permissions)

### 2. Kernel-Level Isolation (gVisor)
Standard containers share the host kernel, leaving a surface for syscall exploits. This project utilizes **gVisor (runsc)** to close this gap.
*   **Sandboxed System Calls:** gVisor intercepts application system calls and handles them in a distinct, user-space kernel.
*   **Host Protection:** This acts as a robust "firewall" between the untrusted application and the actual host kernel, preventing deep-system exploits.

### 3. Network Perimeter Control
*   **Traffic Segregation:** Containers are placed on isolated bridge networks.
*   **Exfiltration Prevention:** Strict rules limit the container's ability to communicate with internal networks or move laterally to other services.

---

## 🏗️ System Architecture

The core philosophy is **Isolation by Default**. The system wraps any CLI agent into an abstract `CodeAgent` class. When a user provides a GitHub URL via the Streamlit interface, the system orchestrates the following pipeline:

1.  **Workspace Provisioning:** A temporary directory (`~/agent_sandbox/<repository>`) is created on the host.
2.  **Ephemeral Runtime:** A Docker container is launched using a pre-baked, immutable image specific to the chosen agent.
3.  **Bind-Mounting:** The host workspace is mounted into the container.
    *   **Read/Write:** The agent can modify files within the sandbox.
    *   **Persistence:** When the container is destroyed, artifacts (code changes) remain on the host for inspection.
4.  **Interactive Session:** The container launches in interactive mode, bridging the user's terminal to the sandboxed agent.

### Key Guarantees
*   **Filesystem Integrity:** The agent has **no access** to the host filesystem outside the specific bind-mounted sandbox.
*   **Side-Effect Containment:** Arbitrary code execution (e.g., `rm -rf /`) or network calls (e.g., `curl malicious.site`) are contained entirely within the disposable container.
*   **Cross-Agent Isolation:** Each agent runs on a dedicated Docker image defined in `dockerconfig`. Vulnerabilities in "Agent A" cannot contaminate the environment of "Agent B."

---

## 🔌 Extensibility

Agent-in-a-Box is designed as a framework, not a tool. Adding a new CLI agent requires zero changes to the core logic.

*   **Logic:** Inherit from the abstract `CodeAgent` base class.
*   **Configuration:** Define a **Pydantic model** for the agent's specific command-line arguments.
*   **Interface:** Create a minimal Streamlit UI component to populate that model.

The system supports both GUI-driven execution and headless Python scripting via the Pydantic command models.

---

## 🚀 Usage

### 1. Infrastructure Setup
Initialize the security layer (Rootless Docker & gVisor configuration).
```bash
./scripts/setup-agent-sandbox.sh
```

---

### 2. The Bakery (Image Build)
Compile the immutable Docker images for your configured agents.
```bash
python src/docker/dockerimage-bakery.py
```

---

### 3. Run
Launch the Streamlit interface or execute the Python entry point to start a session.

---

## Local Inference/Service Setup

Any local inference service (Ollama, etc.) must listen on all interfaces (0.0.0.0) (127.0.0.1) so Docker containers can access it via host.docker.internal.

**Systemd Configuration (Recommended)**
``` bash
sudo systemctl edit ollama.service

# --- Add to configuration
# ... /etc/systemd/system/ollama.service.d/.#override.conffefd312e8a766f95
# ---

[Service]
Environment="[SERVICE_HOST_VAR]=0.0.0.0:[PORT]"
```

**Apply Changes**
``` bash
# Reload systemd and restart service
sudo systemctl daemon-reload
sudo systemctl restart [SERVICE_NAME]

# Verify service is listening on all interfaces
sudo ss -tlnp | grep [PORT]  # Should show *:[PORT] not 127.0.0.1:[PORT]
```

**Why This is Required**
 - Docker containers possess their own local network - therefore they cant easily reach 127.0.0.1 (localhost) of the host
 - host.docker.internal maps to host's network, but only if service listens on external interfaces
 - Setting [SERVICE_HOST_VAR]=0.0.0.0:[PORT] makes the service accessible to Docker containers

**Security Implications**

This setup is safe when using the recommended systemd configuration because:

1. Local Network Only: Service is only accessible within your trusted local network (not exposed to internet)
2. API Authentication: Ollama requires API keys for access, preventing unauthorized usage
3. Container Isolation: Docker containers are still isolated by gVisor and security policies
4. Firewall Protection: Can be further secured with ufw to restrict access to Docker subnet only
Risk Level: Equivalent to other local development tools (Docker, databases, etc.) that bind to 0.0.0.0 for container access.
