#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

conda run -n aiaa3201 python pipelines/part2/run_depth_pro_nyuv2.py "$@"
