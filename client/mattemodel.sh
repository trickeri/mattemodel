#!/usr/bin/env bash
# Minimal shell client for mattemodel.
#   mattemodel.sh <image> [out.png] [cutout|alpha]
#   mattemodel.sh --status | --load | --unload | --park | --activate
set -euo pipefail
BASE="${MM_HTTP_URL:-http://${MM_HOST:-127.0.0.1}:${MM_PORT:-48460}}"

case "${1:-}" in
  --status)  curl -s "$BASE/status"; echo ;;
  --load|--unload|--park|--activate)
             curl -s -X POST -d '' "$BASE/${1#--}"; echo ;;
  "")        echo "usage: $0 <image> [out.png] [cutout|alpha] | --status|--load|--unload|--park|--activate" >&2; exit 1 ;;
  *)
    img="$1"; out="${2:-matte.png}"; fmt="${3:-cutout}"
    curl -s -F "image=@${img}" "$BASE/matte?format=${fmt}" -o "$out"
    echo "wrote $out (format=$fmt)" ;;
esac
