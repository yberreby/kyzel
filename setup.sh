#!/bin/bash
uv venv
uv pip install torch setuptools
uv sync --no-build-isolation
