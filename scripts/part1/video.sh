#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/aiaa3201-part1b-mpl-${UID}}"
export PYTHONDONTWRITEBYTECODE=1
exec conda run --no-capture-output -n aiaa3201 python pipelines/part1/video_clip.py "$@"
