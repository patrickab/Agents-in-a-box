# Agent-in-a-Box

**Agent-in-a-Box** is a secure orchestration layer that wraps arbitrary agents into a unified sandboxed, ephemeral runtime. It allows developers to deploy agents against local repositories with **rootless containerization**, **kernel-level isolation**, and **strict network perimeter control**.

**Supported Agents:**
*   [Aider](https://aider.chat/)
*   [Gemini CLI](https://github.com/google-gemini/gemini-cli)
*   [Qwen Code](https://github.com/QwenLM/qwen-code)

---

## 🎯 Purpose

State-of-the-art coding agents are designed for friction-free adoption, typically running directly on the host machine with the full privileges of the active user.

Default configuration:
*   **Unrestricted Filesystem Access:** Agents can read, modify, or delete any file accessible to the user.
*   **Arbitrary Code Execution:** Generate & execute scripts (Python, Bash).
*   **Network Access:** Allow arbitrary network traffic - While the supported Agents always ask for permission before contacting a website, they technically have the capability to send any data to any publicly accessible domain. They can also install arbitrary software in your binaries.

**Agent-in-a-Box** mitigates this operational risk by decoupling the agent from the host. It functions as a containment layer, providing a unified, ergonomic interface to orchestrate these tools within a heavily restricted runtime with supervised network traffic.

---

## 🛡️ Security Architecture

### **Zero-Trust Execution Policy**

Agent-in-a-Box treats any AI agent as an untrusted entity with intent of malicious code execution. It enforces a **Zero-Trust** policy using a multi-layered defense strategy.

### 1. Rootless Infrastructure
Excludes any potential of container breakout attacks by removing root privileges from the container.
*   **Unprivileged Daemon:** The Docker Engine runs without root privileges on the host.
*   **Identity Mapping:** Utilizes user namespaces to map the container's `root` user to a non-privileged user on the host. Even in the event of a container breakout, the attacker finds themselves with zero permissions on the host filesystem.

### 2. Kernel-Level Isolation (gVisor)
Standard containers share the host kernel, leaving a surface for syscall exploits. This project integrates **gVisor (runsc)** to virtualize the system call interface.
*   **Interception of System Calls:** gVisor handles application systemcalls in a distinct, user-space kernel. Any system call with system-level privileges is automatically rejected.
*   **Attack Surface Reduction:** This creates a robust boundary between the untrusted application and the actual host kernel, preventing deep-system kernel exploits.

### 3. Network Perimeter Control
*   **Traffic Segregation:** Containers are isolated on strict bridge networks.
*   **Exfiltration Prevention:** Firewall rules limit the container's ability to scan internal networks or move laterally to other services.

### 4. Secure Host Bridge (Digital Air-Gap)
*   **Loopback Isolation:** Host-Container communication is routed via a static loopback alias (`10.200.200.1`). Traffic is air-gapped from physical interfaces (Wi-Fi/Eth), preventing accidental LAN exposure.
*   **Deterministic Firewalling:** Access is strictly controlled via UFW rules linking the Docker subnet to the static alias, independent of host network changes.

---

## 🏗️ System Architecture

The core philosophy is **Isolation by Default**. The system wraps any CLI agent into an abstract `CodeAgent` class.

1.  **Workspace Provisioning:** A temporary directory (`~/agent_sandbox/<repo>`) is created on the host.
2.  **Ephemeral Runtime:** A Docker container is launched using a pre-baked, immutable image specific to the chosen agent.
3.  **Bind-Mounting:** The host workspace is mounted into the container.
    *   *Persistence:* Code changes are written to the mounted directory.
    *   *Containment:* The agent cannot access any path outside this specific mount.
4.  **Interactive Session:** The container launches in interactive mode, bridging the user's terminal to the sandboxed agent.
5.  **Cleanup:** Upon exit, the container is destroyed. Only the modified code artifacts remain on the host.

### Key Guarantees
*   **Filesystem Integrity:** The agent has **no access** to the host filesystem outside the specific bind-mounted sandbox.
*   **Side-Effect Containment:** Arbitrary code execution (e.g., `rm -rf /`) or network calls (e.g., `curl malicious.site`) are contained entirely within the disposable container.
*   **Cross-Agent Isolation:** Each agent runs on a dedicated Docker image defined in `dockerconfig`. Vulnerabilities in "Agent A" cannot contaminate the environment of "Agent B."

---

## 🔌 Extensibility

Agent-in-a-Box is designed as a framework, not a tool. Adding a new CLI agent requires zero changes to the core logic.

*   **Step 1:** Inherit from the abstract `CodeAgent` base class.
*   **Step 2:** Define a **Pydantic model** for the agent's specific command-line arguments.
*   **Step 3:** Create a minimal Streamlit UI component to populate that model.

The system supports both GUI-driven execution and headless Python scripting via the Pydantic command models.

---

## 💻 Local Inference & Secure Bridging

To allow the sandboxed agent to communicate with local LLM servers (e.g., Ollama) without exposing the host to the local area network (LAN), we utilize a **Logical Air-Gap**.

### The Loopback Strategy
Instead of binding services to `0.0.0.0` (all interfaces), we create a static alias on the loopback interface.

1.  **Traffic Isolation:** We assign a static IP (`10.200.200.1`) to the host's loopback device. This IP is not routable from physical interfaces (Wi-Fi/Ethernet).
2.  **Deterministic Firewalling:** We use UFW to strictly allow traffic *only* from the Docker subnet (`172.17.0.0/16`) to this specific alias.

**Network Flow:**
`Container (172.17.x.x)` $\to$ `Host Loopback Alias (10.200.200.1)` $\to$ `Ollama Service`

This ensures that while the container can access the LLM, the LLM service remains invisible to the outside world.

---

## 🚀 Usage

### 1. Infrastructure Setup
Initialize the security layer (Rootless Docker context & gVisor configuration).

```bash
./scripts/setup-agent-sandbox.sh
```

### 2. Image Build Pipeline
Compile the immutable Docker images for your configured agents.

```bash
python src/docker/dockerimage-bakery.py
```

### 3. Network Configuration (Optional for Local LLMs)
If running local inference, establish the secure loopback bridge:

```bash
# 1. Create Loopback Alias
sudo ip addr add 10.200.200.1/32 dev lo

# 2. Configure Firewall (Allow Docker Subnet -> Alias)
sudo ufw allow from 172.17.0.0/16 to 10.200.200.1 port 11434 proto tcp comment 'Secure Sandbox Bridge'
sudo ufw reload
```

### 4. Run
Launch the orchestration UI:

```bash
streamlit run src/app.py
```
