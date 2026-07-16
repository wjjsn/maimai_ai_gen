#!/usr/bin/env bash
set -euo pipefail

uv run src/config.py
uv run python -c 'import sys; sys.path.insert(0, "src"); from maidata_parser import _self_check; _self_check()'
uv run src/dataset.py
uv run python -c 'import sys; sys.path.insert(0, "src"); from chart_metrics import _self_check; _self_check()'
uv run src/model.py
uv run python -c 'import sys; sys.path.insert(0, "src"); from infer import _self_check; _self_check()'
uv run python -c 'import sys; sys.path.insert(0, "src"); from train import _self_check; _self_check()'
