#!/bin/sh
uv lock --upgrade
uv sync
streamlit run  src/code_agents/agents.py
