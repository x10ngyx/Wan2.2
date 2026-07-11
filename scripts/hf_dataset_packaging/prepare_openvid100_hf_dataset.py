#!/usr/bin/env python3
"""Prepare a Hugging Face-ready OpenVid-100 SeaCache trace dataset.

The source experiment was produced in multiple machine shards. This script
normalizes it into public dataset semantics:

  prompt_001..prompt_100
  threshold_0p10..threshold_0p80
  one training row per prompt/threshold/denoising step

Large files are staged as symlinks. The companion pack script dereferences
those symlinks while creating OSS transfer archives.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


THRESHOLD_LABELS = [
    "th_0p10",
    "th_0p15",
    "th_0p20",
    "th_0p25",
    "th_0p30",
    "th_0p40",
    "th_0p50",
    "th_0p60",
    "th_0p70",
    "th_0p80",
]


TEXT_PLACEHOLDER = """---
pretty_name: Wan2.2 OpenVid-100 SeaCache Trace Dataset
task_categories:
- text-to-video
- feature-extraction
tags:
- Wan2.2
- SeaCache
- diffusion
- video-generation
- inference-acceleration
---

# Wan2.2 OpenVid-100 SeaCache Trace Dataset

This dataset contains Wan2.2 T2V-A14B SeaCache inference traces for 100
OpenVid prompts. It is organized as 50,000 step-level training rows:

`100 prompts * 10 SeaCache thresholds * 50 denoising steps`.

The public layout is normalized to `prompt_001` through `prompt_100` and
`threshold_0p10` through `threshold_0p80`. Internal experiment-machine shard
names are not part of the dataset interface.

## Contents

- `data/train-*.parquet`: step-level training rows.
- `data/candidates.parquet`: one row per prompt/threshold inference.
- `data/prompts.parquet`: one row per prompt.
- `data/videos.parquet`: baseline and SeaCache video metadata.
- `media/`: baseline and SeaCache videos.
- `traces/`: raw step input tensors (`.pt`) including latent tensors and
  per-step timing/model inputs.
- `artifacts/`: logs, commands, ffprobe JSON, PSNR records, and raw result
  tables.
- `checksums/`: archive and file integrity manifests.
- `scripts/`: restore, verify, and upload helper scripts.

## Step-Level Rows

Each training row corresponds to one prompt, one SeaCache threshold, and one
denoising step. Important columns include:

- `prompt_id`, `prompt_key`, `prompt`, `source_id`
- `threshold`, `threshold_key`, `step_index`
- `is_reused`, `is_recomputed`, `reuse_decision`
- `branch_reuse_keys`, `branch_recompute_keys`
- `cache_metric_by_branch`
- `raw_latent_path`
- `compute_elapsed_seconds`, `speedup`, `mean_psnr`
- `candidate_video_path`, `baseline_video_path`

The raw latent tensor is stored in the referenced `.pt` step file under key
`latent`. Each `.pt` file also contains timestep, scheduler sigma, Wan model
stage, time embeddings, and context references.

## Generation Setup

- model: Wan2.2 T2V-A14B
- task: `t2v-A14B`
- size: `832*480`
- frame count: `45`
- sample steps: `50`
- sample solver: `dpm++`
- seed: `42`
- cache method: SeaCache timestep cache
- thresholds: `0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80`
- quality metric: FFmpeg PSNR against the same prompt/seed/shape no-cache baseline
- speed metric: compute-only inference elapsed time

## Integrity

After download or restore from OSS archives, run:

```bash
python scripts/verify_dataset.py .
```

For archive transfer verification, run:

```bash
sha256sum -c checksums/archive_sha256s.txt
```
"""


@dataclass(frozen=True)
class Paths:
    source_root: Path
    staging_root: Path


def rel_to_source(paths: Paths, value: str | float | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        return None
    p = paths.source_root / value
    if not p.exists() and not p.is_symlink():
        return None
    try:
        return p.resolve(strict=True)
    except FileNotFoundError:
        return p


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def force_symlink(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_branch_map(value: str | float | None) -> dict[str, list[Any]]:
    if not isinstance(value, str) or not value.strip():
        return {}
    out: dict[str, list[Any]] = {}
    for part in value.split(" | "):
        if ":" not in part:
            continue
        key_text, val_text = part.split(":", 1)
        try:
            key = ast.literal_eval(key_text)
            val = ast.literal_eval(val_text)
        except Exception:
            continue
        if isinstance(key, tuple):
            key_name = "/".join(str(x) for x in key)
        else:
            key_name = str(key)
        out[key_name] = list(val) if isinstance(val, list) else []
    return out


def branch_steps(branch_map: dict[str, list[Any]]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for branch, values in branch_map.items():
        for item in values:
            step = item[0] if isinstance(item, tuple) and item else item
            if isinstance(step, int):
                out.setdefault(step, []).append(branch)
    return out


def branch_metrics(branch_map: dict[str, list[Any]], step_index: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for branch, values in branch_map.items():
        for item in values:
            if isinstance(item, tuple) and len(item) >= 2 and item[0] == step_index:
                try:
                    out[branch] = float(item[1])
                except Exception:
                    pass
    return out


def safe_float(value: Any) -> float | None:
    try:
        if value == "" or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value == "" or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def copy_table_files(paths: Paths) -> None:
    tables_src = paths.source_root / "data" / "tables"
    tables_dst = paths.staging_root / "data" / "source_tables"
    tables_dst.mkdir(parents=True, exist_ok=True)
    for src in sorted(tables_src.glob("*")):
        if src.is_file():
            shutil.copy2(src, tables_dst / src.name)
    meta_src = paths.source_root / "data" / "metadata"
    meta_dst = paths.staging_root / "metadata"
    if meta_src.exists():
        shutil.copytree(meta_src, meta_dst, dirs_exist_ok=True)


def write_dataframe(df: pd.DataFrame, base: Path, shard_rows: int | None = None) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if shard_rows is None or len(df) <= shard_rows:
        parquet_path = base.with_suffix(".parquet")
        jsonl_path = base.with_suffix(".jsonl")
        df.to_parquet(parquet_path, index=False)
        df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        written.extend([str(parquet_path), str(jsonl_path)])
        return written
    stem = base.name
    total = (len(df) + shard_rows - 1) // shard_rows
    for idx in range(total):
        sub = df.iloc[idx * shard_rows : (idx + 1) * shard_rows]
        name = f"{stem}-{idx:05d}-of-{total:05d}"
        parquet_path = base.parent / f"{name}.parquet"
        jsonl_path = base.parent / f"{name}.jsonl"
        sub.to_parquet(parquet_path, index=False)
        sub.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
        written.extend([str(parquet_path), str(jsonl_path)])
    return written


def build(paths: Paths) -> dict[str, Any]:
    if paths.staging_root.exists():
        raise SystemExit(f"staging root already exists: {paths.staging_root}")
    paths.staging_root.mkdir(parents=True)
    copy_table_files(paths)

    summary_csv = paths.source_root / "data" / "tables" / "summary.csv"
    prompts_csv = paths.source_root / "data" / "tables" / "prompts.csv"
    summary = pd.read_csv(summary_csv)
    prompts = pd.read_csv(prompts_csv)

    expected_candidates = 100 * len(THRESHOLD_LABELS)
    if len(summary) != expected_candidates:
        raise SystemExit(f"expected {expected_candidates} candidate rows, found {len(summary)}")

    if set(summary["threshold_label"].unique()) != set(THRESHOLD_LABELS):
        raise SystemExit("threshold labels do not match expected SeaCache sweep")

    prompt_rows: list[dict[str, Any]] = []
    for row in prompts.to_dict(orient="records"):
        sample_index = int(row["sample_index"])
        prompt_id = sample_index + 1
        prompt_key = f"prompt_{prompt_id:03d}"
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_key": prompt_key,
                "sample_id": row["sample_id"],
                "sample_index": int(row["sample_index"]),
                "source_id": row["source_id"],
                "prompt": row["prompt"],
                "source_video": row["source_video"],
                "source_video_relative_path": row["source_video_relative_path"],
                "content_group": row["content_group"],
                "portrait_group": row["portrait_group"],
                "motion_group": row["motion_group"],
                "baseline_video_path": f"media/baseline/{prompt_key}.mp4",
            }
        )

    candidate_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []

    seen_baseline_video: set[int] = set()
    missing_files: list[str] = []

    for row in summary.sort_values(["sample_index", "threshold"]).to_dict(orient="records"):
        sample_index = int(row["sample_index"])
        prompt_id = sample_index + 1
        prompt_key = f"prompt_{prompt_id:03d}"
        threshold_label = row["threshold_label"]
        threshold_key = threshold_label.replace("th_", "threshold_")
        threshold = float(row["threshold"])
        sample_id = row["sample_id"]
        candidate_id = f"{prompt_key}__{threshold_key}"

        baseline_video = f"media/baseline/{prompt_key}.mp4"
        seacache_video = f"media/seacache/{threshold_key}/{prompt_key}.mp4"
        baseline_log = f"artifacts/logs/baseline/{prompt_key}.log"
        seacache_log = f"artifacts/logs/seacache/{threshold_key}/{prompt_key}.log"
        baseline_command = f"artifacts/commands/baseline/{prompt_key}.sh"
        seacache_command = f"artifacts/commands/seacache/{threshold_key}/{prompt_key}.sh"
        baseline_ffprobe = f"artifacts/ffprobe/baseline/{prompt_key}.json"
        seacache_ffprobe = f"artifacts/ffprobe/seacache/{threshold_key}/{prompt_key}.json"
        psnr_json = f"artifacts/psnr/{threshold_key}/{prompt_key}.json"
        psnr_log = f"artifacts/psnr/{threshold_key}/{prompt_key}.log"

        links = [
            (row.get("baseline_video"), baseline_video),
            (row.get("seacache_video"), seacache_video),
            (row.get("baseline_log"), baseline_log),
            (row.get("seacache_log"), seacache_log),
            (row.get("baseline_command"), baseline_command),
            (row.get("seacache_command"), seacache_command),
            (row.get("baseline_ffprobe_path"), baseline_ffprobe),
            (row.get("seacache_ffprobe_path"), seacache_ffprobe),
            (row.get("psnr_json"), psnr_json),
            (row.get("psnr_log"), psnr_log),
        ]
        for src_rel, dst_rel in links:
            src = rel_to_source(paths, src_rel)
            if src is None:
                missing_files.append(f"{src_rel} -> {dst_rel}")
                continue
            force_symlink(src, paths.staging_root / dst_rel)

        if prompt_id not in seen_baseline_video:
            video_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_key": prompt_key,
                    "threshold": None,
                    "threshold_key": None,
                    "video_type": "baseline",
                    "path": baseline_video,
                    "ffprobe_path": baseline_ffprobe,
                }
            )
            seen_baseline_video.add(prompt_id)
        video_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_key": prompt_key,
                "threshold": threshold,
                "threshold_key": threshold_key,
                "video_type": "seacache",
                "path": seacache_video,
                "ffprobe_path": seacache_ffprobe,
            }
        )

        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "prompt_id": prompt_id,
                "prompt_key": prompt_key,
                "sample_id": sample_id,
                "source_id": row.get("source_id"),
                "threshold": threshold,
                "threshold_key": threshold_key,
                "cache_method": row.get("cache_method"),
                "timestep_cache": row.get("timestep_cache"),
                "timestep_threshold": safe_float(row.get("timestep_threshold")),
                "baseline_elapsed_seconds": safe_float(row.get("baseline_elapsed_seconds")),
                "compute_elapsed_seconds": safe_float(row.get("seacache_elapsed_seconds")),
                "speedup": safe_float(row.get("speedup")),
                "mean_psnr": safe_float(row.get("mean_psnr")),
                "min_psnr": safe_float(row.get("min_psnr")),
                "max_psnr": safe_float(row.get("max_psnr")),
                "psnr_frames": safe_int(row.get("psnr_frames")),
                "decoded_frames_total": safe_int(row.get("decoded_frames_total")),
                "excluded_perfect_frames": safe_int(row.get("excluded_perfect_frames")),
                "seacache_reuse_count": safe_int(row.get("seacache_reuse_count")),
                "seacache_recompute_count": safe_int(row.get("seacache_recompute_count")),
                "seacache_reuse_branch_call_count": safe_int(row.get("seacache_reuse_branch_call_count")),
                "seacache_recompute_branch_call_count": safe_int(row.get("seacache_recompute_branch_call_count")),
                "baseline_video_path": baseline_video,
                "candidate_video_path": seacache_video,
                "baseline_log_path": baseline_log,
                "candidate_log_path": seacache_log,
                "baseline_command_path": baseline_command,
                "candidate_command_path": seacache_command,
                "baseline_ffprobe_path": baseline_ffprobe,
                "candidate_ffprobe_path": seacache_ffprobe,
                "psnr_json_path": psnr_json,
                "psnr_log_path": psnr_log,
                "status": "completed",
            }
        )

        skipping = parse_branch_map(row.get("seacache_skipping_path"))
        recompute = parse_branch_map(row.get("seacache_recompute_path"))
        rel_l1 = parse_branch_map(row.get("seacache_rel_l1_path"))
        accumulated_rel_l1 = parse_branch_map(row.get("seacache_accumulated_rel_l1_path"))
        skip_steps = branch_steps(skipping)
        recompute_steps = branch_steps(recompute)

        seacache_step_dir_src = rel_to_source(paths, row.get("seacache_step_inputs"))
        baseline_step_dir_src = rel_to_source(paths, row.get("baseline_step_inputs"))
        if seacache_step_dir_src is None:
            missing_files.append(f"{row.get('seacache_step_inputs')} -> traces/seacache/{threshold_key}/{prompt_key}")
        else:
            for name in ["meta.pt", "context.pt", "trace_done.pt"]:
                src = seacache_step_dir_src / name
                if src.exists():
                    force_symlink(src, paths.staging_root / f"traces/seacache/{threshold_key}/{prompt_key}/{name}")
            for step in range(50):
                src = seacache_step_dir_src / f"step_{step:03d}.pt"
                dst = paths.staging_root / f"traces/seacache/{threshold_key}/{prompt_key}/step_{step:03d}.pt"
                if src.exists():
                    force_symlink(src, dst)
                else:
                    missing_files.append(str(src))

        if baseline_step_dir_src is not None and threshold_label == THRESHOLD_LABELS[0]:
            for name in ["meta.pt", "context.pt", "trace_done.pt"]:
                src = baseline_step_dir_src / name
                if src.exists():
                    force_symlink(src, paths.staging_root / f"traces/baseline/{prompt_key}/{name}")
            for step in range(50):
                src = baseline_step_dir_src / f"step_{step:03d}.pt"
                if src.exists():
                    force_symlink(src, paths.staging_root / f"traces/baseline/{prompt_key}/step_{step:03d}.pt")

        for step in range(50):
            reused_branches = sorted(skip_steps.get(step, []))
            recompute_branches = sorted(recompute_steps.get(step, []))
            if reused_branches and recompute_branches:
                decision = "mixed"
            elif reused_branches:
                decision = "reuse"
            elif recompute_branches:
                decision = "recompute"
            else:
                decision = "unknown"
            train_rows.append(
                {
                    "row_id": f"{candidate_id}__step_{step:03d}",
                    "candidate_id": candidate_id,
                    "prompt_id": prompt_id,
                    "prompt_key": prompt_key,
                    "sample_id": sample_id,
                    "source_id": row.get("source_id"),
                    "prompt": row.get("prompt"),
                    "content_group": row.get("content_group"),
                    "portrait_group": row.get("portrait_group"),
                    "motion_group": row.get("motion_group"),
                    "threshold": threshold,
                    "threshold_key": threshold_key,
                    "step_index": step,
                    "num_steps": 50,
                    "cache_method": "seacache",
                    "is_reused": bool(reused_branches),
                    "is_recomputed": bool(recompute_branches),
                    "reuse_decision": decision,
                    "branch_reuse_keys": json.dumps(reused_branches),
                    "branch_recompute_keys": json.dumps(recompute_branches),
                    "cache_metric_by_branch": json.dumps(branch_metrics(rel_l1, step), sort_keys=True),
                    "accumulated_cache_metric_by_branch": json.dumps(branch_metrics(accumulated_rel_l1, step), sort_keys=True),
                    "raw_latent_path": f"traces/seacache/{threshold_key}/{prompt_key}/step_{step:03d}.pt",
                    "trace_context_path": f"traces/seacache/{threshold_key}/{prompt_key}/context.pt",
                    "trace_meta_path": f"traces/seacache/{threshold_key}/{prompt_key}/meta.pt",
                    "baseline_trace_path": f"traces/baseline/{prompt_key}/step_{step:03d}.pt",
                    "latent_key": "latent",
                    "latent_shape": "[16, 12, 60, 104]",
                    "latent_dtype": "torch.float16",
                    "baseline_video_path": baseline_video,
                    "candidate_video_path": seacache_video,
                    "baseline_log_path": baseline_log,
                    "candidate_log_path": seacache_log,
                    "psnr_json_path": psnr_json,
                    "psnr_log_path": psnr_log,
                    "compute_elapsed_seconds": safe_float(row.get("seacache_elapsed_seconds")),
                    "baseline_elapsed_seconds": safe_float(row.get("baseline_elapsed_seconds")),
                    "speedup": safe_float(row.get("speedup")),
                    "mean_psnr": safe_float(row.get("mean_psnr")),
                    "min_psnr": safe_float(row.get("min_psnr")),
                    "max_psnr": safe_float(row.get("max_psnr")),
                    "status": "completed",
                }
            )

    if missing_files:
        write_json(paths.staging_root / "checksums" / "missing_files.json", missing_files)

    prompt_df = pd.DataFrame(prompt_rows).sort_values("prompt_id")
    candidate_df = pd.DataFrame(candidate_rows).sort_values(["prompt_id", "threshold"])
    train_df = pd.DataFrame(train_rows).sort_values(["prompt_id", "threshold", "step_index"])
    video_df = pd.DataFrame(video_rows).sort_values(["prompt_id", "video_type", "threshold_key"], na_position="first")

    written = []
    written += write_dataframe(prompt_df, paths.staging_root / "data" / "prompts")
    written += write_dataframe(candidate_df, paths.staging_root / "data" / "candidates")
    written += write_dataframe(video_df, paths.staging_root / "data" / "videos")
    written += write_dataframe(train_df, paths.staging_root / "data" / "train", shard_rows=5000)

    (paths.staging_root / "README.md").write_text(TEXT_PLACEHOLDER, encoding="utf-8")

    manifest = {
        "dataset_name": "wan22_openvid100_seacache_trace",
        "source_root": str(paths.source_root),
        "public_prompt_count": int(prompt_df["prompt_id"].nunique()),
        "thresholds": [float(x.replace("th_0p", "0.")) for x in THRESHOLD_LABELS],
        "candidate_rows": len(candidate_df),
        "training_rows": len(train_df),
        "baseline_video_rows": int((video_df["video_type"] == "baseline").sum()),
        "seacache_video_rows": int((video_df["video_type"] == "seacache").sum()),
        "sample_steps": 50,
        "layout_policy": "Public paths use prompt_001..prompt_100 and threshold_0pXX names; source machine shard names are hidden from the public layout.",
        "large_file_policy": "Large files are staged as symlinks and should be archived with tar -h or the pack script.",
        "index_files": [str(Path(p).relative_to(paths.staging_root)) for p in written],
        "missing_file_count": len(missing_files),
    }
    write_json(paths.staging_root / "dataset_manifest.json", manifest)

    # Checksums for lightweight metadata only. Archive checksums are generated by pack script.
    checksum_lines: list[str] = []
    for p in sorted(paths.staging_root.rglob("*")):
        if p.is_file() and not p.is_symlink() and p.stat().st_size < 1024 * 1024 * 64:
            rel = p.relative_to(paths.staging_root)
            checksum_lines.append(f"{sha256_file(p)}  {rel}\n")
    checksums_dir = paths.staging_root / "checksums"
    checksums_dir.mkdir(parents=True, exist_ok=True)
    (checksums_dir / "metadata_sha256s.txt").write_text("".join(checksum_lines), encoding="utf-8")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/hy-tmp/openvid_100_seacache_trace_data")
    parser.add_argument("--staging-root", default="/hy-tmp/hf_staging/wan22_openvid100_seacache_trace")
    args = parser.parse_args()
    manifest = build(Paths(Path(args.source_root).resolve(), Path(args.staging_root)))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
