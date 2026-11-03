#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
exec conda run --no-capture-output -n aiaa3201-unidepth \
  python pipelines/part2/analyze_scale_boundary.py "$@"
