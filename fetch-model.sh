#!/usr/bin/env bash
# Fetch RVM models for mattemodel.
#   ./fetch-model.sh cuda      # ONNX RVM (resnet50 fp16) for the cuda backend
#   ./fetch-model.sh vulkan    # nihui ncnn RVM (resnet50 + mobilenetv3) for vulkan
#   ./fetch-model.sh all
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/models"; mkdir -p "$DEST"
WHICH="${1:-cuda}"

fetch_vulkan() {
    local base="https://github.com/nihui/ncnn-android-rvm/raw/master/app/src/main/assets"
    for n in rvm_resnet50 rvm_mobilenetv3; do
        for ext in ncnn.param ncnn.bin; do
            echo "fetching $n.$ext …" >&2
            curl -sL --retry 3 -o "$DEST/$n.$ext" "$base/$n.$ext"
        done
    done
}
fetch_cuda() {
    # RVM resnet50 fp16 ONNX (same weights the camera daemon uses) — official release.
    local url="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp16.onnx"
    echo "fetching rvm_resnet50_fp16.onnx …" >&2
    curl -sL --retry 3 -o "$DEST/rvm_resnet50_fp16.onnx" "$url"
}

case "$WHICH" in
    cuda)   fetch_cuda ;;
    vulkan) fetch_vulkan ;;
    all)    fetch_cuda; fetch_vulkan ;;
    *) echo "usage: $0 [cuda|vulkan|all]" >&2; exit 1 ;;
esac
echo "done -> $DEST" >&2
