#!/bin/bash
sudo ip addr add 10.200.200.1/32 dev lo
uv sync
uv run streamlit run src/code_agents/app_ui.py
