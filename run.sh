#!/bin/bash
if [ -z "${SECURE_LOOPBACK_IP}" ]; then
  echo "Error: SECURE_LOOPBACK_IP is not set in environment. Suggested value: 10.200.200.1" >&2
  exit 1
fi
sudo ip addr add ${SECURE_LOOPBACK_IP}/32 dev lo
uv sync
uv run streamlit run src/code_agents/app_ui.py
