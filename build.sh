#!/usr/bin/env bash
# Build mattemodel-server for a chosen backend.
#   ./build.sh           # cuda (ONNX Runtime, default)
#   ./build.sh vulkan    # ncnn-Vulkan
#   ./build.sh cuda
#
# cuda  needs system ONNX Runtime (pkg-config libonnxruntime) + OpenCV.
# vulkan needs ncnn built with -DNCNN_VULKAN=ON, installed to ~/programming/ncnn/install
#        (sibling checkout; built + installed here if missing) + OpenCV.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${1:-cuda}"
[ "$BACKEND" = cuda ] || [ "$BACKEND" = vulkan ] || { echo "usage: $0 [cuda|vulkan]" >&2; exit 1; }

CMAKE_ARGS=(-DBACKEND="$BACKEND" -DCMAKE_BUILD_TYPE=Release)

if [ "$BACKEND" = vulkan ]; then
    NCNN_DIR="${NCNN_DIR:-$HOME/programming/ncnn}"
    NCNN_INSTALL="${NCNN_INSTALL:-$NCNN_DIR/install}"
    if [ ! -f "$NCNN_INSTALL/lib/cmake/ncnn/ncnnConfig.cmake" ]; then
        echo "ncnn install not found — building ncnn (Vulkan)…" >&2
        [ -d "$NCNN_DIR" ] || git clone --depth 1 --recurse-submodules --shallow-submodules \
            https://github.com/Tencent/ncnn "$NCNN_DIR"
        cmake -S "$NCNN_DIR" -B "$NCNN_DIR/build" -DNCNN_VULKAN=ON -DNCNN_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
        cmake --build "$NCNN_DIR/build" -j"$(nproc)"
        cmake --install "$NCNN_DIR/build" --prefix "$NCNN_INSTALL"
    fi
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$NCNN_INSTALL")
fi

cmake -S "$HERE" -B "$HERE/build" "${CMAKE_ARGS[@]}"
cmake --build "$HERE/build" -j"$(nproc)"
echo "$BACKEND" > "$HERE/build/backend.txt"   # launcher reads this to pick the default model
echo "built ($BACKEND): $HERE/build/mattemodel-server" >&2
