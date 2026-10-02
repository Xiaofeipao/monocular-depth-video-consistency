#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p \
    "${CHECKPOINT_ROOT}/depth_pro" \
    "${CHECKPOINT_ROOT}/depth_anything_v2" \
    "${CHECKPOINT_ROOT}/torchvision/hub/checkpoints" \
    "${CHECKPOINT_ROOT}/video_depth_anything"

download() {
    local url="$1"
    local destination="$2"
    local partial="${destination}.part"
    if [[ -s "${destination}" ]]; then
        echo "Already present: ${destination}"
        return
    fi
    echo "Downloading ${url}"
    wget --continue --tries=10 --timeout=30 --output-document="${partial}" "${url}"
    mv "${partial}" "${destination}"
}

download \
    "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt" \
    "${CHECKPOINT_ROOT}/depth_pro/depth_pro.pt"
download \
    "${HF_ENDPOINT}/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" \
    "${CHECKPOINT_ROOT}/depth_anything_v2/depth_anything_v2_vits.pth"
download \
    "${HF_ENDPOINT}/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth" \
    "${CHECKPOINT_ROOT}/video_depth_anything/video_depth_anything_vits.pth"
download \
    "${HF_ENDPOINT}/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth" \
    "${CHECKPOINT_ROOT}/video_depth_anything/metric_video_depth_anything_vits.pth"
download \
    "https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth" \
    "${CHECKPOINT_ROOT}/torchvision/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth"

(
    cd "${CHECKPOINT_ROOT}"
    find . -type f \( -name '*.pth' -o -name '*.pt' \) \
        | sort \
        | xargs sha256sum \
        > checksums.sha256
)

echo "Checkpoint downloads prepared under ${CHECKPOINT_ROOT}"
