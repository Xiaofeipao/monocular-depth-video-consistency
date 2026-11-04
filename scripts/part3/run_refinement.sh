#!/usr/bin/env bash
# Run from the repository root, inside the configured Python environment.
set -euo pipefail
exec python -m pipelines.part3.refine_nyuv2 "$@"
