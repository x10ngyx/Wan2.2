#!/usr/bin/env bash
set -euo pipefail

cd /hy-tmp/work/Wan2.2

root="/hy-tmp/openvid_100_seacache_trace_data"
mkdir -p "${root}"

echo "== start $(date -Is) =="
df -h /hy-tmp

archives=(
  "/hy-tmp/prompt001-033.tar.gz"
  "/hy-tmp/prompt034-100.tar.gz"
)

for archive in "${archives[@]}"; do
  echo "== checking ${archive} $(date -Is) =="
  test -s "${archive}"
  echo "== extracting ${archive} -> ${root} $(date -Is) =="
  tar -xzf "${archive}" -C "${root}"
  echo "== extracted ${archive} $(date -Is) =="
  df -h /hy-tmp
done

echo "== organizing links $(date -Is) =="
mkdir -p "${root}/sources" "${root}/shards"
find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'wan22_openvid_prompts*_seacache_trace_*' | sort | while read -r source_dir; do
  ln -sfn "${source_dir}" "${root}/sources/$(basename "${source_dir}")"
  find "${source_dir}" -mindepth 1 -maxdepth 1 -type d -name 'shard*_prompts*' | sort | while read -r shard_dir; do
    ln -sfn "${shard_dir}" "${root}/shards/$(basename "${shard_dir}")"
  done
done

ln -sfn "${root}" experiment_results/openvid_100_seacache_trace_data

echo "== summary $(date -Is) =="
echo "root=${root}"
echo "source_dirs=$(find "${root}" -mindepth 1 -maxdepth 1 -type d -name 'wan22_openvid_prompts*_seacache_trace_*' | wc -l)"
echo "shard_dirs=$(find "${root}/shards" -mindepth 1 -maxdepth 1 -type l | wc -l)"
echo "baseline_videos=$(find "${root}" -path '*/baseline/*.mp4' -type f | wc -l)"
echo "seacache_videos=$(find "${root}" -path '*/seacache/th_*/*.mp4' -type f | wc -l)"
echo "summary_csv=$(find "${root}" -path '*/results/summary.csv' -type f | wc -l)"
du -sh "${root}"
df -h /hy-tmp
echo "== all done $(date -Is) =="
