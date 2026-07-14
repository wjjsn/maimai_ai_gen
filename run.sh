#!/usr/bin/env bash
set -euo pipefail

bash test.sh
uv run src/full_check.py
