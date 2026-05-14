#!/usr/bin/env bash
# Fetch the 15-image test set (Unsplash hotlinks) listed in examples/test/urls.txt.
# Photos used under Unsplash License (free, no attribution required).
set -euo pipefail

cd "$(dirname "$0")/.."
in_dir="examples/test"
mkdir -p "$in_dir"

while IFS='=' read -r name url; do
  [ -z "$name" ] && continue
  out="$in_dir/${name}.jpg"
  if [ -f "$out" ]; then
    echo "= $name (exists)"
    continue
  fi
  echo "↓ $name"
  curl -sSL --fail -o "$out" "$url" || { echo "  FAIL: $name"; rm -f "$out"; }
done < "$in_dir/urls.txt"

echo "done. $(ls "$in_dir"/*.jpg 2>/dev/null | wc -l) images in $in_dir"
