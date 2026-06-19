#!/usr/bin/env bash
# mattemodel — system-wide RVM matting daemon. Launches mattemodel-server with
# one RVM model warm on the GPU. Any app POSTs an image to
# http://$MM_HOST:$MM_PORT/matte and gets an alpha matte (RGBA cutout / gray) back.
# The backend (cuda|vulkan) is fixed at build time; this picks the matching model.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$HERE/config.env" ] && . "$HERE/config.env"

MM_SERVER_BIN="${MM_SERVER_BIN:-$HERE/build/mattemodel-server}"
export MM_HOST="${MM_HOST:-127.0.0.1}"
export MM_PORT="${MM_PORT:-48460}"

# Which backend was built? (build.sh writes this.) Default cuda.
BACKEND="${MM_BACKEND:-$( [ -f "$HERE/build/backend.txt" ] && cat "$HERE/build/backend.txt" || echo cuda )}"

# Pick the default model for the backend: cuda → .onnx, vulkan → ncnn prefix.
if [ "$BACKEND" = vulkan ]; then
    DEFAULT_MODEL="$HERE/models/rvm_resnet50"
else
    DEFAULT_MODEL="$HERE/models/rvm_resnet50_fp16.onnx"
fi
MM_MODEL="${MM_MODEL:-$DEFAULT_MODEL}"

[ -x "$MM_SERVER_BIN" ] || { echo "mattemodel-server not built at $MM_SERVER_BIN — run ./build.sh [$BACKEND]" >&2; exit 1; }

echo "mattemodel: ${MM_MODEL##*/} on $MM_HOST:$MM_PORT (backend=$BACKEND)" >&2
exec "$MM_SERVER_BIN" "$MM_MODEL"
