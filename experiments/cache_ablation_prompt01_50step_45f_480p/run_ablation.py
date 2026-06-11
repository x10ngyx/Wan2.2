#!/usr/bin/env python3
import argparse
import ast
import csv
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
        return subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=True)


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


def timestep_config(enabled: bool):
    if not enabled:
        return None
    return ZeusThresholdTimestepCacheConfig(
        enabled=True,
        acc_range=(8, 47),
        caching_mode="reuse_interp",
        max_interval=6,
        threshold=0.02,
        metric="rel_l1",
        eps=1e-6,
    )


def block_group_config(enabled: bool):
    if not enabled:
        return None
    return BlockGroupCacheConfig(
        enabled=True,
        group_size=5,
        threshold=0.03,
        metric="pooled_rel_l1",
        start=0.1,
        end=0.9,
        max_reuse=3,
        eps=1e-6,
    )


def cfg_config(enabled: bool):
    if not enabled:
        return None
    return CFGCacheConfig(
        enabled=True,
        start=0.1,
        end=0.9,
        threshold=0.03,
        max_reuse=3,
        eps=1e-6,
        metric="timestep_modulated_input_rel_l1",
        force_uncond_recompute_on_miss=False,
    )


def candidate_matrix():
    return [
        ("timestep_only", True, False, False),
        ("block_only", False, True, False),
        ("cfg_only", False, False, True),
        ("timestep_block", True, True, False),
        ("timestep_cfg", True, False, True),
        ("block_cfg", False, True, True),
        ("all_three", True, True, True),
    ]


def write_candidate_config(path: Path, name: str, use_timestep: bool, use_block: bool, use_cfg: bool):
    config = {
        "name": name,
        "prompt_id": "01",
        "seed": 42,
        "size": "832*480",
        "frame_num": 45,
        "sample_steps": 50,
        "sample_solver": "dpm++",
        "offload_model": True,
        "convert_model_dtype": True,
        "timestep_cache": {
            "enabled": use_timestep,
            "kind": "zeus-threshold",
            "threshold": 0.02,
            "acc_range": [8, 47],
            "caching_mode": "reuse_interp",
            "max_interval": 6,
        },
        "block_cache": {
            "enabled": use_block,
            "kind": "block-group",
            "threshold": 0.03,
            "group_size": 5,
            "metric": "pooled_rel_l1",
            "start": 0.1,
            "end": 0.9,
            "max_reuse": 3,
        },
        "cfg_cache": {
            "enabled": use_cfg,
            "kind": "threshold",
            "threshold": 0.03,
            "metric": "timestep_modulated_input_rel_l1",
            "start": 0.1,
            "end": 0.9,
            "max_reuse": 3,
            "force_uncond_recompute_on_miss": False,
        },
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--ckpt-dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt-file", default=str(ROOT / "prompt.txt"))
    parser.add_argument("--baseline-video", required=True)
    parser.add_argument("--baseline-ffprobe", default="")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/hy-tmp/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/hy-tmp/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/hy-tmp/hf-cache/hub")

    exp_root = Path(args.exp_root)
    dirs = ["videos", "logs", "commands", "ffprobe", "psnr", "results", "failed", "baseline"]
    for dirname in dirs:
        (exp_root / dirname).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    prompt = parse_prompts(Path(args.prompt_file))[0]
    baseline_video = exp_root / "baseline" / "prompt_01.mp4"
    shutil.copy2(args.baseline_video, baseline_video)
    if args.baseline_ffprobe:
        shutil.copy2(args.baseline_ffprobe, exp_root / "ffprobe" / "baseline_prompt_01.json")

    ffprobe = resolve_tool("ffprobe")
    python_bin = sys.executable
    compute_psnr = ROOT / "experiments/zeus_timestep_cache_50step_45f_480p/compute_psnr.py"

    experiment_config = {
        "exp_root": str(exp_root),
        "ckpt_dir": args.ckpt_dir,
        "prompt_file": args.prompt_file,
        "baseline_video": str(baseline_video),
        "baseline_compute_seconds": BASELINE_COMPUTE_SECONDS,
        "candidates": [name for name, _, _, _ in candidate_matrix()],
    }
    (exp_root / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2), encoding="utf-8")

    cfg = WAN_CONFIGS["t2v-A14B"]
    logging.info("Creating one WanT2V pipeline for all ablation candidates.")
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
    for name, use_timestep, use_block, use_cfg in candidate_matrix():
        video_path = exp_root / "videos" / f"{name}_prompt_01.mp4"
        log_path = exp_root / "logs" / f"{name}_prompt_01.log"
        time_path = exp_root / "logs" / f"{name}_prompt_01.time"
        config_path = exp_root / "commands" / f"{name}_prompt_01.config.json"
        ffprobe_path = exp_root / "ffprobe" / f"{name}_prompt_01.json"
        psnr_path = exp_root / "psnr" / f"{name}_prompt_01.json"

        write_candidate_config(config_path, name, use_timestep, use_block, use_cfg)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)
        status = 0
        try:
            logging.info("Starting ablation candidate %s", name)
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
                timestep_cache_config=timestep_config(use_timestep),
                block_cache_config=None,
                block_group_cache_config=block_group_config(use_block),
                cfg_cache_config=cfg_config(use_cfg),
            )
            generation_wall = time.perf_counter() - start
            logging.info("candidate_generation_wall_elapsed_seconds=%.3f", generation_wall)
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
            status = 1
            failed_path = exp_root / "failed" / f"{name}_prompt_01.failed"
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

        if status != 0:
            break

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
        rows.append({
            "prompt_id": "01",
            "candidate": name,
            "timestep_cache": use_timestep,
            "block_group_cache": use_block,
            "cfg_cache": use_cfg,
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
        })

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
        logging.info("Archived candidate %s: elapsed=%.3f speedup=%.3f psnr=%.3f",
                     name, elapsed, BASELINE_COMPUTE_SECONDS / elapsed, psnr["mean_psnr"])

    logging.info("Ablation complete: %d candidates", len(rows))


if __name__ == "__main__":
    main()
