#!/usr/bin/env bash
set -euo pipefail

cd /hy-tmp/work/Wan2.2

echo "== start $(date -Is) =="
df -h /hy-tmp

archives=(
  "prompt001-033.tar.gz"
  "prompt034-100.tar.gz"
)

for archive in "${archives[@]}"; do
  echo "== downloading ${archive} $(date -Is) =="
  oss cp "oss://datasets/${archive}" "/hy-tmp/${archive}"
  echo "== finished ${archive} $(date -Is) =="
  ls -lh "/hy-tmp/${archive}"
  df -h /hy-tmp
done

echo "== all done $(date -Is) =="
ls -lh /hy-tmp/prompt001-033.tar.gz /hy-tmp/prompt034-100.tar.gz
df -h /hy-tmp
