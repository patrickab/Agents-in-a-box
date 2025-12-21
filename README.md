# Agent-in-a-Box

**Agent-in-a-Box** is a security-first orchestration layer that wraps arbitrary CLI-based AI Agents into a unified, sandboxed runtime. It allows developers to deploy untrusted AI agents against local repositories with zero risk to the host operating system.

---

## 🛡️ Security Architecture

Operates under **Zero-Trust Execution Policy** based on the **"Assume Breach" Principle**: *Any agentic system is treated as potentially malicious.*

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

### 2. The Bakery (Image Build)
Compile the immutable Docker images for your configured agents.
```bash
python src/docker/dockerimage-bakery.py
```

### 3. Run
Launch the Streamlit interface or execute the Python entry point to start a session.
