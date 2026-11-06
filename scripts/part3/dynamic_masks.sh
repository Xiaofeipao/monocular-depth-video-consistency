#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${TMPDIR:-$ROOT/.project_cache/tmp}"
mkdir -p "$TMPDIR"
exec conda run --no-capture-output -n aiaa3201 python -m pipelines.part3.dynamic_masks "$@"
