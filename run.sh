#!/bin/bash
# Verify the Gigachad profile prerequisites.
uv sync
uv run agent-sandbox doctor --profile gigachad
