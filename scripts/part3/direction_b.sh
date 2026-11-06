#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/aiaa3201-matplotlib}"
args=("$@")
run_env=aiaa3201
if [[ "${args[0]:-}" == predict-development ]]; then
  for ((i=0;i<${#args[@]};i++)); do
    if [[ "${args[i]}" == --model && "${args[i+1]:-}" == unidepth_v2 ]]; then
      run_env=aiaa3201-unidepth
    fi
  done
fi
exec conda run --no-capture-output -n "$run_env" python pipelines/part3/direction_b.py "$@"
