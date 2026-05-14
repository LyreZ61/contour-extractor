#!/usr/bin/env bash
# Batch-run contour.py over examples/test/*.jpg → examples/test_out/*.png
# Auto-select rembg model from filename hint: portrait → u2net_human_seg, else isnet-general-use
set -euo pipefail

IN_DIR="examples/test"
OUT_DIR="examples/test_out"
mkdir -p "$OUT_DIR"

PY=".venv/bin/python"

for img in "$IN_DIR"/*.jpg; do
  base=$(basename "$img" .jpg)
  out_transp="$OUT_DIR/${base}_transparent.png"
  out_white="$OUT_DIR/${base}_white.png"

  case "$base" in
    *portrait*|*human*) model="u2net_human_seg" ;;
    *)                  model="isnet-general-use" ;;
  esac

  echo "▸ $base [$model]"
  "$PY" contour.py "$img" "$out_transp" --model "$model" >/dev/null
  "$PY" contour.py "$img" "$out_white"  --model "$model" --background white >/dev/null
done

echo "done. $(ls "$OUT_DIR"/*.png 2>/dev/null | wc -l) outputs in $OUT_DIR"
