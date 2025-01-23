#!/bin/bash
uv venv
uv pip install torch
uv sync --no-build-isolation
