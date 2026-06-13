#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_ROOT = "/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625"
DEFAULT_THRESHOLDS = "0.05 0.10 0.20"
BASELINE_COMPUTE_SECONDS = 522.603


def read_prompts(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    data = ast.literal_eval("{" + text + "}")
    return [prompt.replace("\n", " ").strip() for prompt in data["prompts"]]


def label_for(threshold: str):
    return f"sea_full_accum_bg_{threshold.replace('.', 'p')}"


def read_elapsed(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def read_time_file(path: Path):
    if not path.exists():
        return None
    match = re.search(
        r"elapsed_seconds=([0-9.]+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    return float(match.group(1)) if match else None


def write_failed(root: Path, label: str, fields):
    failed = root / "failed" / f"{label}_prompt_01.failed"
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text(
        "\n".join(f"{key}={value}" for key, value in fields.items()) + "\n",
        encoding="utf-8",
    )


def save_command(path: Path, cmd, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"cd {REPO_ROOT}"]
    for key in ["HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE"]:
        lines.append(f"export {key}={env.get(key, '')}")
    lines.append(" ".join(subprocess.list2cmdline([item]) for item in cmd))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ffprobe(ffprobe_bin: str, video: Path, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration",
        "-of", "json",
        str(video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video}: {proc.stderr.strip()}")


def run_psnr(python_bin: str, reference: Path, candidate: Path, output: Path, log_path: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(REPO_ROOT / "experiments" / "zeus_timestep_cache_50step_45f_480p" / "compute_psnr.py"),
        "--reference", str(reference),
        "--candidate", str(candidate),
        "--output", str(output),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"PSNR failed for {candidate}; see {log_path}")


def candidate_complete(video: Path, time_file: Path, ffprobe_json: Path, psnr_json: Path):
    return all(path.exists() and path.stat().st_size > 0 for path in [
        video,
        time_file,
        ffprobe_json,
        psnr_json,
    ])


def summarize(exp_root: Path, thresholds):
    rows = []
    for threshold in thresholds:
        label = label_for(threshold)
        time_file = exp_root / "logs" / f"{label}_prompt_01.time"
        psnr_file = exp_root / "psnr" / f"{label}_prompt_01.json"
        ffprobe_file = exp_root / "ffprobe" / f"{label}_prompt_01.json"
        video_path = exp_root / "videos" / f"{label}_prompt_01.mp4"
        log_path = exp_root / "logs" / f"{label}_prompt_01.log"
        if not candidate_complete(video_path, time_file, ffprobe_file, psnr_file):
            continue
        elapsed = read_time_file(time_file)
        psnr = json.loads(psnr_file.read_text(encoding="utf-8"))
        meta = json.loads(ffprobe_file.read_text(encoding="utf-8"))["streams"][0]
        rows.append({
            "prompt_id": "01",
            "candidate": label,
            "block_cache": "block-group",
            "block_group_decision": "accumulated",
            "block_group_metric": "sea_full_rel_l1",
            "block_threshold": threshold,
            "block_group_size": "5",
            "block_group_ret_steps": "1",
            "block_group_cutoff_steps": "1",
            "baseline_elapsed_seconds": f"{BASELINE_COMPUTE_SECONDS:.3f}",
            "elapsed_seconds": f"{elapsed:.3f}" if elapsed is not None else "",
            "speedup": f"{BASELINE_COMPUTE_SECONDS / elapsed:.6f}" if elapsed else "",
            "mean_psnr": psnr.get("mean_psnr", ""),
            "min_psnr": psnr.get("min_psnr", ""),
            "max_psnr": psnr.get("max_psnr", ""),
            "psnr_frames": psnr.get("frames", ""),
            "ffprobe_width": meta.get("width", ""),
            "ffprobe_height": meta.get("height", ""),
            "ffprobe_frames": meta.get("nb_frames") or meta.get("nb_read_frames", ""),
            "ffprobe_fps": meta.get("r_frame_rate", ""),
            "ffprobe_duration": meta.get("duration", ""),
            "video_path": str(video_path),
            "log_path": str(log_path),
            "psnr_path": str(psnr_file),
            "ffprobe_path": str(ffprobe_file),
        })

    results_dir = exp_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (results_dir / "summary.json").write_text(
        json.dumps({"num_completed": len(rows), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--baseline_root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--ffprobe_bin", default="/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe")
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt_file", default=str(REPO_ROOT / "prompt.txt"))
    parser.add_argument("--resume_existing", action="store_true")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    for dirname in ["baseline", "videos", "logs", "commands", "ffprobe", "psnr", "results", "failed"]:
        (exp_root / dirname).mkdir(parents=True, exist_ok=True)

    thresholds = args.thresholds.split()
    prompt = read_prompts(Path(args.prompt_file))[0]
    baseline_root = Path(args.baseline_root)
    baseline_video = exp_root / "baseline" / "prompt_01.mp4"
    shutil.copy2(baseline_root / "baseline" / "prompt_01.mp4", baseline_video)
    baseline_ffprobe = baseline_root / "ffprobe" / "baseline_prompt_01.json"
    if baseline_ffprobe.exists():
        shutil.copy2(baseline_ffprobe, exp_root / "ffprobe" / "baseline_prompt_01.json")
    (exp_root / "logs" / "baseline_prompt_01.time").write_text(
        f"elapsed_seconds={BASELINE_COMPUTE_SECONDS:.3f}\n",
        encoding="utf-8",
    )

    config = {
        "exp_root": str(exp_root),
        "baseline_root": str(baseline_root),
        "thresholds": thresholds,
        "task": "t2v-A14B",
        "size": "832*480",
        "frame_num": 45,
        "sample_steps": 50,
        "sample_solver": "dpm++",
        "base_seed": 42,
        "block_cache": "block-group",
        "block_group_decision": "accumulated",
        "block_group_metric": "sea_full_rel_l1",
        "block_group_size": 5,
        "block_group_ret_steps": 1,
        "block_group_cutoff_steps": 1,
        "process_model": "one generate.py subprocess per threshold to isolate OOM",
    }
    (exp_root / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2))

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/hy-tmp/hf-cache")
    env.setdefault("TRANSFORMERS_CACHE", "/hy-tmp/hf-cache")
    env.setdefault("HF_HUB_CACHE", "/hy-tmp/hf-cache/hub")

    for threshold in thresholds:
        label = label_for(threshold)
        video = exp_root / "videos" / f"{label}_prompt_01.mp4"
        log_path = exp_root / "logs" / f"{label}_prompt_01.log"
        time_file = exp_root / "logs" / f"{label}_prompt_01.time"
        ffprobe_json = exp_root / "ffprobe" / f"{label}_prompt_01.json"
        psnr_json = exp_root / "psnr" / f"{label}_prompt_01.json"
        psnr_log = exp_root / "psnr" / f"{label}_prompt_01.psnr.log"
        command_path = exp_root / "commands" / f"{label}_prompt_01.sh"

        if args.resume_existing and candidate_complete(video, time_file, ffprobe_json, psnr_json):
            print(f"Skipping completed {label}")
            continue

        cmd = [
            args.python_bin,
            str(REPO_ROOT / "generate.py"),
            "--task", "t2v-A14B",
            "--ckpt_dir", args.ckpt_dir,
            "--prompt", prompt,
            "--size", "832*480",
            "--frame_num", "45",
            "--sample_steps", "50",
            "--sample_solver", "dpm++",
            "--sample_shift", "12.0",
            "--base_seed", "42",
            "--offload_model", "True",
            "--convert_model_dtype",
            "--timestep_cache", "none",
            "--block_cache", "block-group",
            "--block_group_size", "5",
            "--block_group_threshold", threshold,
            "--block_group_metric", "sea_full_rel_l1",
            "--block_group_decision", "accumulated",
            "--block_group_ret_steps", "1",
            "--block_group_cutoff_steps", "1",
            "--block_group_max_reuse", "50",
            "--cfg_cache", "none",
            "--save_file", str(video),
        ]
        save_command(command_path, cmd, env)
        print(f"Running {label}")
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        wall = time.perf_counter() - start
        if proc.returncode != 0:
            write_failed(exp_root, label, {
                "candidate": label,
                "threshold": threshold,
                "returncode": proc.returncode,
                "wall_elapsed_seconds": f"{wall:.3f}",
                "log_path": log_path,
            })
            print(f"FAILED {label} returncode={proc.returncode}; see {log_path}")
            break

        elapsed = read_elapsed(log_path)
        if elapsed is None:
            write_failed(exp_root, label, {
                "candidate": label,
                "threshold": threshold,
                "returncode": proc.returncode,
                "wall_elapsed_seconds": f"{wall:.3f}",
                "error": "missing inference_compute_elapsed_seconds",
                "log_path": log_path,
            })
            print(f"FAILED {label}: missing compute elapsed; see {log_path}")
            break
        time_file.write_text(f"elapsed_seconds={elapsed:.3f}\nwall_elapsed_seconds={wall:.3f}\n", encoding="utf-8")
        run_ffprobe(args.ffprobe_bin, video, ffprobe_json)
        run_psnr(args.python_bin, baseline_video, video, psnr_json, psnr_log)
        rows = summarize(exp_root, thresholds)
        print(f"Archived {label}; completed_rows={len(rows)}")

    rows = summarize(exp_root, thresholds)
    print(f"Completed pilot rows={len(rows)}")


if __name__ == "__main__":
    main()
