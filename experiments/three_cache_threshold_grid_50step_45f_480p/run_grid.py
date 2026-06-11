#!/usr/bin/env python3
import argparse
import ast
import csv
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wan
from wan.block_group_cache import BlockGroupCacheConfig
from wan.cfg_cache import CFGCacheConfig
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.timestep_cache import ZeusThresholdTimestepCacheConfig
from wan.utils.utils import save_video


BASELINE_COMPUTE_SECONDS = 522.603


def parse_prompts(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    data = ast.literal_eval("{" + text + "}")
    return [prompt.replace("\n", " ").strip() for prompt in data["prompts"]]


def parse_float_list(value: str):
    return [float(item) for item in value.split()]


def threshold_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def combo_name(timestep_threshold: float, block_threshold: float,
               cfg_threshold: float) -> str:
    return (
        f"ts_{threshold_label(timestep_threshold)}"
        f"__bg_{threshold_label(block_threshold)}"
        f"__cfg_{threshold_label(cfg_threshold)}")


def resolve_tool(name: str) -> str:
    candidates = [
        shutil.which(name),
        f"/hy-tmp/miniconda3/envs/Wan2.2/bin/{name}",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit(f"{name} not found")


def run_checked(cmd, stdout_path: Path = None):
    if stdout_path is None:
        return subprocess.run(cmd, check=True)
    with stdout_path.open("w", encoding="utf-8") as f:
        return subprocess.run(
            cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=True)


def read_elapsed(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    if not matches:
        raise ValueError(f"Missing inference_compute_elapsed_seconds in {log_path}")
    return float(matches[-1])


def read_psnr(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stream_metadata(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["streams"][0]


def is_complete(video_path: Path, ffprobe_path: Path, psnr_path: Path,
                time_path: Path) -> bool:
    return (
        video_path.is_file() and video_path.stat().st_size > 0 and
        ffprobe_path.is_file() and ffprobe_path.stat().st_size > 0 and
        psnr_path.is_file() and psnr_path.stat().st_size > 0 and
        time_path.is_file() and time_path.stat().st_size > 0)


def timestep_config(threshold: float):
    return ZeusThresholdTimestepCacheConfig(
        enabled=True,
        acc_range=(8, 47),
        caching_mode="reuse_interp",
        max_interval=6,
        threshold=threshold,
        metric="rel_l1",
        eps=1e-6,
    )


def block_group_config(threshold: float):
    return BlockGroupCacheConfig(
        enabled=True,
        group_size=5,
        threshold=threshold,
        metric="pooled_rel_l1",
        start=0.1,
        end=0.9,
        max_reuse=3,
        eps=1e-6,
    )


def cfg_config(threshold: float):
    return CFGCacheConfig(
        enabled=True,
        start=0.1,
        end=0.9,
        threshold=threshold,
        max_reuse=3,
        eps=1e-6,
        metric="timestep_modulated_input_rel_l1",
        force_uncond_recompute_on_miss=False,
    )


def write_candidate_config(path: Path, prompt_id: str, name: str,
                           timestep_threshold: float, block_threshold: float,
                           cfg_threshold: float):
    config = {
        "name": name,
        "prompt_id": prompt_id,
        "seed": 42,
        "size": "832*480",
        "frame_num": 45,
        "sample_steps": 50,
        "sample_solver": "dpm++",
        "offload_model": True,
        "convert_model_dtype": True,
        "timestep_cache": {
            "enabled": True,
            "kind": "zeus-threshold",
            "threshold": timestep_threshold,
            "acc_range": [8, 47],
            "caching_mode": "reuse_interp",
            "max_interval": 6,
            "metric": "rel_l1",
            "eps": 1e-6,
        },
        "block_cache": {
            "enabled": True,
            "kind": "block-group",
            "threshold": block_threshold,
            "group_size": 5,
            "metric": "pooled_rel_l1",
            "start": 0.1,
            "end": 0.9,
            "max_reuse": 3,
            "eps": 1e-6,
        },
        "cfg_cache": {
            "enabled": True,
            "kind": "threshold",
            "threshold": cfg_threshold,
            "metric": "timestep_modulated_input_rel_l1",
            "start": 0.1,
            "end": 0.9,
            "max_reuse": 3,
            "eps": 1e-6,
            "force_uncond_recompute_on_miss": False,
        },
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_results(exp_root: Path, rows):
    if not rows:
        return
    summary_csv = exp_root / "results" / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "num_completed": len(rows),
        "baseline_elapsed_seconds": BASELINE_COMPUTE_SECONDS,
        "rows": rows,
    }
    (exp_root / "results" / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--ckpt-dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt-file", default=str(ROOT / "prompt.txt"))
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--baseline-video", required=True)
    parser.add_argument("--baseline-ffprobe", default="")
    parser.add_argument(
        "--timestep-thresholds", default="0.001 0.005 0.02 0.60")
    parser.add_argument(
        "--block-thresholds", default="0.001 0.015 0.03 1.00")
    parser.add_argument(
        "--cfg-thresholds", default="0.001 0.02 0.03 1.00")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--combo-limit", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/hy-tmp/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/hy-tmp/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/hy-tmp/hf-cache/hub")

    exp_root = Path(args.exp_root)
    dirs = [
        "videos", "logs", "commands", "ffprobe", "psnr", "results",
        "failed", "baseline"
    ]
    for dirname in dirs:
        (exp_root / dirname).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    prompts = parse_prompts(Path(args.prompt_file))
    if not 1 <= args.prompt_index <= len(prompts):
        raise SystemExit(
            f"--prompt-index must be in [1, {len(prompts)}], got {args.prompt_index}")
    prompt = prompts[args.prompt_index - 1]
    prompt_id = f"{args.prompt_index:02d}"

    baseline_video = exp_root / "baseline" / f"prompt_{prompt_id}.mp4"
    if not baseline_video.exists():
        shutil.copy2(args.baseline_video, baseline_video)
    if args.baseline_ffprobe:
        baseline_ffprobe = exp_root / "ffprobe" / f"baseline_prompt_{prompt_id}.json"
        if not baseline_ffprobe.exists():
            shutil.copy2(args.baseline_ffprobe, baseline_ffprobe)

    timestep_thresholds = parse_float_list(args.timestep_thresholds)
    block_thresholds = parse_float_list(args.block_thresholds)
    cfg_thresholds = parse_float_list(args.cfg_thresholds)
    candidates = list(itertools.product(
        timestep_thresholds, block_thresholds, cfg_thresholds))
    if args.combo_limit > 0:
        candidates = candidates[:args.combo_limit]

    ffprobe = resolve_tool("ffprobe")
    python_bin = sys.executable
    compute_psnr = ROOT / "experiments/zeus_timestep_cache_50step_45f_480p/compute_psnr.py"

    experiment_config = {
        "exp_root": str(exp_root),
        "ckpt_dir": args.ckpt_dir,
        "prompt_file": args.prompt_file,
        "prompt_index": args.prompt_index,
        "prompt_id": prompt_id,
        "baseline_video": str(baseline_video),
        "baseline_compute_seconds": BASELINE_COMPUTE_SECONDS,
        "timestep_thresholds": timestep_thresholds,
        "block_thresholds": block_thresholds,
        "cfg_thresholds": cfg_thresholds,
        "num_candidates": len(candidates),
        "resume_existing": args.resume_existing,
        "force_uncond_recompute_on_miss": False,
    }
    (exp_root / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2), encoding="utf-8")

    cfg = WAN_CONFIGS["t2v-A14B"]
    logging.info("Creating one WanT2V pipeline for %d threshold combinations.",
                 len(candidates))
    pipeline = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )

    rows = []
    summary_csv = exp_root / "results" / "summary.csv"
    if args.resume_existing and summary_csv.is_file():
        with summary_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    completed_names = {row["candidate"] for row in rows}
    for index, (ts_th, bg_th, cfg_th) in enumerate(candidates, start=1):
        name = combo_name(ts_th, bg_th, cfg_th)
        video_path = exp_root / "videos" / f"{name}_prompt_{prompt_id}.mp4"
        log_path = exp_root / "logs" / f"{name}_prompt_{prompt_id}.log"
        time_path = exp_root / "logs" / f"{name}_prompt_{prompt_id}.time"
        config_path = exp_root / "commands" / f"{name}_prompt_{prompt_id}.config.json"
        ffprobe_path = exp_root / "ffprobe" / f"{name}_prompt_{prompt_id}.json"
        psnr_path = exp_root / "psnr" / f"{name}_prompt_{prompt_id}.json"

        write_candidate_config(
            config_path, prompt_id, name, ts_th, bg_th, cfg_th)

        if (args.resume_existing and name in completed_names and
                is_complete(video_path, ffprobe_path, psnr_path, time_path)):
            logging.info("Skipping completed candidate %s (%d/%d)",
                         name, index, len(candidates))
            continue

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)
        try:
            logging.info(
                "Starting candidate %s (%d/%d): timestep=%s block=%s cfg=%s",
                name, index, len(candidates), ts_th, bg_th, cfg_th)
            start = time.perf_counter()
            video = pipeline.generate(
                prompt,
                size=SIZE_CONFIGS["832*480"],
                frame_num=45,
                shift=12.0,
                sample_solver="dpm++",
                sampling_steps=50,
                guide_scale=(3.0, 4.0),
                seed=42,
                offload_model=True,
                timestep_cache_config=timestep_config(ts_th),
                block_cache_config=None,
                block_group_cache_config=block_group_config(bg_th),
                cfg_cache_config=cfg_config(cfg_th),
            )
            generation_wall = time.perf_counter() - start
            logging.info("candidate_generation_wall_elapsed_seconds=%.3f",
                         generation_wall)
            logging.info("Saving generated video to %s", video_path)
            save_video(
                tensor=video[None],
                save_file=str(video_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            logging.info("Finished candidate %s", name)
        except Exception as exc:
            failed_path = exp_root / "failed" / f"{name}_prompt_{prompt_id}.failed"
            failed_path.write_text(
                f"name={name}\nstatus=1\nerror={exc!r}\nlog={log_path}\n",
                encoding="utf-8",
            )
            logging.exception("Candidate %s failed", name)
            raise
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
            torch.cuda.empty_cache()

        elapsed = read_elapsed(log_path)
        time_path.write_text(f"elapsed_seconds={elapsed:.3f}\n", encoding="utf-8")
        run_checked([
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
            "-of", "json",
            str(video_path),
        ], stdout_path=ffprobe_path)
        run_checked([
            python_bin,
            str(compute_psnr),
            "--reference", str(baseline_video),
            "--candidate", str(video_path),
            "--output", str(psnr_path),
        ])

        psnr = read_psnr(psnr_path)
        meta = stream_metadata(ffprobe_path)
        row = {
            "prompt_id": prompt_id,
            "candidate": name,
            "timestep_threshold": ts_th,
            "block_group_threshold": bg_th,
            "cfg_threshold": cfg_th,
            "cfg_force_uncond_recompute_on_miss": False,
            "baseline_elapsed_seconds": BASELINE_COMPUTE_SECONDS,
            "elapsed_seconds": elapsed,
            "speedup": BASELINE_COMPUTE_SECONDS / elapsed if elapsed else "",
            "mean_psnr": psnr["mean_psnr"],
            "min_psnr": psnr["min_psnr"],
            "max_psnr": psnr["max_psnr"],
            "psnr_frames": psnr["frames"],
            "video_path": str(video_path),
            "log_path": str(log_path),
            "config_path": str(config_path),
            "psnr_path": str(psnr_path),
            "ffprobe_width": meta["width"],
            "ffprobe_height": meta["height"],
            "ffprobe_frames": meta["nb_frames"],
            "ffprobe_fps": meta["r_frame_rate"],
            "ffprobe_duration": meta["duration"],
        }
        rows = [r for r in rows if r["candidate"] != name]
        rows.append(row)
        write_results(exp_root, rows)
        logging.info("Archived candidate %s: elapsed=%.3f speedup=%.3f psnr=%.3f",
                     name, elapsed, BASELINE_COMPUTE_SECONDS / elapsed,
                     psnr["mean_psnr"])

    logging.info("Threshold grid complete: %d candidates archived.", len(rows))


if __name__ == "__main__":
    main()
