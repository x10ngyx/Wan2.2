#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path("/hy-tmp/openvid_100_seacache_trace_data")
DATA = ROOT / "data"
SHARDS = ROOT / "shards"

THRESHOLDS = ("0p10", "0p15", "0p20", "0p25", "0p30", "0p40", "0p50", "0p60", "0p70", "0p80")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return
        link.unlink()
    link.symlink_to(target)


def ensure_optional_link(link: Path, target: Path) -> str:
    if target.exists():
        ensure_link(link, target)
        return rel(link)
    if link.is_symlink() or link.exists():
        link.unlink()
    return ""


def sample_index(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[-1]


def threshold_suffix(label: str) -> str:
    return label.removeprefix("th_")


def main() -> None:
    if not SHARDS.exists():
        raise SystemExit(f"missing shard symlink directory: {SHARDS}")

    for subdir in (
        "baseline/videos",
        "baseline/logs",
        "baseline/ffprobe",
        "baseline/commands",
        "baseline/step_inputs",
        "seacache/videos",
        "seacache/logs",
        "seacache/ffprobe",
        "seacache/psnr",
        "seacache/commands",
        "seacache/step_inputs",
        "tables",
        "metadata",
    ):
        (DATA / subdir).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    prompts: dict[str, dict[str, str]] = {}

    for summary_path in sorted(ROOT.glob("shards/*/results/summary.csv")):
        shard_dir = summary_path.parent.parent.resolve()
        with summary_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row["sample_id"]
                idx = sample_index(sample_id)
                th_label = row["threshold_label"]
                th = threshold_suffix(th_label)

                prompts.setdefault(
                    sample_id,
                    {
                        "sample_id": sample_id,
                        "sample_index": idx,
                        "source_id": row.get("source_id", ""),
                        "prompt": row.get("prompt", ""),
                        "source_video": row.get("source_video", ""),
                        "source_video_relative_path": row.get("source_video_relative_path", ""),
                        "content_group": row.get("content_group", ""),
                        "portrait_group": row.get("portrait_group", ""),
                        "motion_group": row.get("motion_group", ""),
                    },
                )

                baseline_video = DATA / "baseline/videos" / f"{sample_id}.mp4"
                baseline_log = DATA / "baseline/logs" / f"{sample_id}.log"
                baseline_ffprobe = DATA / "baseline/ffprobe" / f"{sample_id}.json"
                baseline_command = DATA / "baseline/commands" / f"{sample_id}.sh"
                baseline_step_inputs = DATA / "baseline/step_inputs" / sample_id

                seacache_video = DATA / "seacache/videos" / th_label / f"{sample_id}.mp4"
                seacache_log = DATA / "seacache/logs" / th_label / f"{sample_id}.log"
                seacache_ffprobe = DATA / "seacache/ffprobe" / th_label / f"{sample_id}.json"
                seacache_psnr = DATA / "seacache/psnr" / th_label / f"{sample_id}.json"
                seacache_psnr_log = DATA / "seacache/psnr" / th_label / f"{sample_id}.log"
                seacache_psnr_ffmpeg = DATA / "seacache/psnr" / th_label / f"{sample_id}.ffmpeg_psnr.log"
                seacache_command = DATA / "seacache/commands" / th_label / f"{sample_id}.sh"
                seacache_step_inputs = DATA / "seacache/step_inputs" / th_label / sample_id

                ensure_link(baseline_video, shard_dir / "baseline" / f"{sample_id}.mp4")
                ensure_link(baseline_log, shard_dir / "logs" / f"baseline_{sample_id}.log")
                ensure_link(baseline_ffprobe, shard_dir / "ffprobe" / f"baseline_{sample_id}.json")
                ensure_link(baseline_command, shard_dir / "commands" / f"baseline_{sample_id}.sh")
                ensure_link(baseline_step_inputs, shard_dir / "step_inputs" / "baseline" / sample_id)

                ensure_link(seacache_video, shard_dir / "seacache" / th_label / f"{sample_id}.mp4")
                ensure_link(seacache_log, shard_dir / "logs" / f"seacache_{th_label}_{sample_id}.log")
                ensure_link(seacache_ffprobe, shard_dir / "ffprobe" / f"seacache_{th_label}_{sample_id}.json")
                ensure_link(seacache_psnr, shard_dir / "psnr" / th_label / f"{sample_id}.json")
                psnr_log_value = ensure_optional_link(seacache_psnr_log, shard_dir / "psnr" / th_label / f"{sample_id}.log")
                psnr_ffmpeg_value = ensure_optional_link(
                    seacache_psnr_ffmpeg,
                    shard_dir / "psnr" / th_label / f"{sample_id}.json.ffmpeg_psnr.log",
                )
                ensure_link(seacache_command, shard_dir / "commands" / f"seacache_{th_label}_{sample_id}.sh")
                ensure_link(seacache_step_inputs, shard_dir / "step_inputs" / "seacache" / th_label / sample_id)

                row = dict(row)
                row["sample_index"] = idx
                row["cache_method"] = "seacache"
                row["timestep_cache"] = "seacache"
                row["timestep_threshold"] = row["threshold"]
                row["video_root"] = rel(DATA)
                row["baseline_video"] = rel(baseline_video)
                row["seacache_video"] = rel(seacache_video)
                row["baseline_log"] = rel(baseline_log)
                row["seacache_log"] = rel(seacache_log)
                row["baseline_ffprobe_path"] = rel(baseline_ffprobe)
                row["seacache_ffprobe_path"] = rel(seacache_ffprobe)
                row["psnr_json"] = rel(seacache_psnr)
                row["psnr_log"] = psnr_log_value
                row["psnr_ffmpeg_log"] = psnr_ffmpeg_value
                row["baseline_command"] = rel(baseline_command)
                row["seacache_command"] = rel(seacache_command)
                row["baseline_step_inputs"] = rel(baseline_step_inputs)
                row["seacache_step_inputs"] = rel(seacache_step_inputs)
                row["source_summary_csv"] = rel(summary_path.resolve())
                rows.append(row)

    rows.sort(key=lambda r: (int(r["sample_index"]), float(r["threshold"])))

    summary_csv = DATA / "tables" / "summary.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (DATA / "tables" / "summary.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prompt_rows = [prompts[k] for k in sorted(prompts, key=lambda x: int(sample_index(x)))]
    with (DATA / "tables" / "prompts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prompt_rows[0]))
        writer.writeheader()
        writer.writerows(prompt_rows)
    with (DATA / "tables" / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for row in prompt_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "root": str(ROOT),
        "data_dir": rel(DATA),
        "layout": "flat training-data symlink view; no shard or source split is required for consumers",
        "sample_count": len(prompt_rows),
        "candidate_count": len(rows),
        "threshold_labels": [f"th_{x}" for x in THRESHOLDS],
        "baseline_videos": rel(DATA / "baseline/videos"),
        "seacache_videos": rel(DATA / "seacache/videos"),
        "tables": {
            "summary_csv": rel(summary_csv),
            "summary_jsonl": rel(DATA / "tables/summary.jsonl"),
            "prompts_csv": rel(DATA / "tables/prompts.csv"),
            "prompts_jsonl": rel(DATA / "tables/prompts.jsonl"),
        },
        "paths_are_relative_to": str(ROOT),
    }
    with (DATA / "metadata" / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
