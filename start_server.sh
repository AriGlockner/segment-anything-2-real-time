#!/usr/bin/env bash
set -euo pipefail # Exits on error, unset var use, or pipe failure

# 
PY="$HOME/miniconda3/envs/sam2/bin/python"

"$PY" sam2/benchmark.py
"$PY" demo/main.py