#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/data"
DOWNLOAD_ROOT="${DATA_ROOT}/downloads"
MIDDLEBURY_ROOT="${DATA_ROOT}/middlebury"
NYUV2_ROOT="${DATA_ROOT}/nyuv2"
MANIFEST_ROOT="${DATA_ROOT}/manifests"
NYUV2_EXPECTED_BYTES=2972037809

mkdir -p "${DOWNLOAD_ROOT}" "${MIDDLEBURY_ROOT}" "${NYUV2_ROOT}" "${MANIFEST_ROOT}"

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
    "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip" \
    "${DOWNLOAD_ROOT}/MiddEval3-data-Q.zip"
download \
    "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip" \
    "${DOWNLOAD_ROOT}/MiddEval3-GT0-Q.zip"
# The course's canonical source is:
# https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
# It is heavily throttled from this HPC. This public mirror contains a single
# file with the canonical name and exact canonical byte length. We validate the
# extracted file below and record both archive and MAT checksums.
if [[ ! -s "${NYUV2_ROOT}/nyu_depth_v2_labeled.mat" ]]; then
    download \
        "https://www.kaggle.com/api/v1/datasets/download/wesleypan/nyu-depth-v2-labeled-mat" \
        "${DOWNLOAD_ROOT}/nyu-depth-v2-labeled-mat.zip"
    unzip -j -o \
        "${DOWNLOAD_ROOT}/nyu-depth-v2-labeled-mat.zip" \
        "nyu_depth_v2_labeled.mat" \
        -d "${NYUV2_ROOT}"
fi
download \
    "https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat" \
    "${NYUV2_ROOT}/splits.mat"

actual_nyuv2_bytes="$(stat -c '%s' "${NYUV2_ROOT}/nyu_depth_v2_labeled.mat")"
if [[ "${actual_nyuv2_bytes}" != "${NYUV2_EXPECTED_BYTES}" ]]; then
    echo "Unexpected NYUv2 MAT size: ${actual_nyuv2_bytes}; expected ${NYUV2_EXPECTED_BYTES}" >&2
    exit 1
fi

if [[ ! -s "${MIDDLEBURY_ROOT}/MiddEval3/trainingQ/Adirondack/im0.png" ]]; then
    unzip -q "${DOWNLOAD_ROOT}/MiddEval3-data-Q.zip" -d "${MIDDLEBURY_ROOT}"
fi
if [[ ! -s "${MIDDLEBURY_ROOT}/MiddEval3/trainingQ/Adirondack/disp0GT.pfm" ]]; then
    unzip -q "${DOWNLOAD_ROOT}/MiddEval3-GT0-Q.zip" -d "${MIDDLEBURY_ROOT}"
fi

(
    cd "${PROJECT_ROOT}"
    sha256sum \
        "data/downloads/MiddEval3-data-Q.zip" \
        "data/downloads/MiddEval3-GT0-Q.zip" \
        "data/downloads/nyu-depth-v2-labeled-mat.zip" \
        "data/nyuv2/nyu_depth_v2_labeled.mat" \
        "data/nyuv2/splits.mat"
) > "${MANIFEST_ROOT}/downloads.sha256"

echo "Data downloads prepared under ${DATA_ROOT}"
