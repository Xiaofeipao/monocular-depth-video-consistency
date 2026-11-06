#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/aiaa3201-matplotlib}"
exec conda run --no-capture-output -n aiaa3201 python pipelines/part3_studies/direction_c.py "$@"
