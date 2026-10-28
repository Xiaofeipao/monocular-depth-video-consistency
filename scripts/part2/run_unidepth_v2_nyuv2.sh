#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

export PYTHONPATH="$project_root/UniDepth:$project_root/ml-depth-pro/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled

exec conda run --no-capture-output -n aiaa3201-unidepth \
  python pipelines/part2/run_unidepth_v2_nyuv2.py "$@"
