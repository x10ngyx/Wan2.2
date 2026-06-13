#!/usr/bin/env python3
import argparse
import ast
import contextlib
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASELINE_ROOT = "/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625"
DEFAULT_FFPROBE = "/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe"
DEFAULT_CANDIDATES = "threshold:0.02 threshold:0.03 sea-threshold:0.10 sea-threshold:0.20 sea-threshold:0.30"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def read_prompts(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    data = ast.literal_eval("{" + text + "}")
    return [prompt.replace("\n", " ").strip() for prompt in data["prompts"]]


def candidate_label(method: str, threshold: str) -> str:
    return f"{method.replace('-', '_')}_th_{threshold.replace('.', 'p')}"


def parse_candidates(text: str):
    candidates = []
    for item in text.split():
        method, threshold = item.split(":", 1)
        if method not in {"threshold", "sea-threshold"}:
            raise ValueError(f"Unsupported cfg candidate method: {method}")
        candidates.append((method, threshold, candidate_label(method, threshold)))
    return candidates


def parse_elapsed(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    if not matches:
        return None
    return float(matches[-1])


def read_time_file(path: Path):
    if not path.exists():
        return None
    match = re.search(r"elapsed_seconds=([0-9.]+)", path.read_text(encoding="utf-8", errors="replace"))
    return float(match.group(1)) if match else None


def maybe_completed(video: Path, time_file: Path, ffprobe_json: Path, psnr_json: Path):
    return all(path.exists() and path.stat().st_size > 0 for path in [video, time_file, ffprobe_json, psnr_json])


def write_failed(root: Path, name: str, fields):
    failed = root / "failed" / f"{name}.txt"
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text("\n".join(f"{k}={v}" for k, v in fields.items()) + "\n", encoding="utf-8")


@contextlib.contextmanager
def run_log_context(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    old_handlers = list(logger.handlers)
    old_level = logger.level
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    with log_path.open("a", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)):
            with contextlib.redirect_stderr(Tee(sys.stderr, stream)):
                try:
                    yield
                finally:
                    handler.flush()
                    handler.close()
                    logger.handlers = old_handlers
                    logger.setLevel(old_level)


def copy_baseline_artifacts(args, exp_root: Path):
    source_root = Path(args.baseline_root)
    baseline_video = exp_root / "baseline" / "prompt_01.mp4"
    source_video = source_root / "baseline" / "prompt_01.mp4"
    source_ffprobe = source_root / "ffprobe" / "baseline_prompt_01.json"
    if not source_video.exists() or source_video.stat().st_size == 0:
        raise FileNotFoundError(f"Missing baseline video: {source_video}")
    if not source_ffprobe.exists() or source_ffprobe.stat().st_size == 0:
        raise FileNotFoundError(f"Missing baseline ffprobe: {source_ffprobe}")
    baseline_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_video, baseline_video)
    shutil.copy2(source_ffprobe, exp_root / "ffprobe" / "baseline_prompt_01.json")
    baseline_time = exp_root / "logs" / "baseline_prompt_01.time"
    baseline_time.write_text(f"elapsed_seconds={args.baseline_compute_seconds}\n", encoding="utf-8")
    return baseline_video, baseline_time


def run_ffprobe(ffprobe_bin: str, video: Path, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration",
        "-of", "json",
        str(video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        output.write_text(proc.stdout, encoding="utf-8")
        raise RuntimeError(f"ffprobe failed for {video}: {proc.stderr.strip()}")
    output.write_text(proc.stdout, encoding="utf-8")


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


def save_command_record(path: Path, argv, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"cd {REPO_ROOT}", " ".join(subprocess.list2cmdline([item]) for item in argv)]
    lines.extend(f"# {k}={v}" for k, v in extra.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_cfg_config(args, method: str, threshold: str):
    from wan.cfg_cache import CFGCacheConfig, SeaCFGCacheConfig

    if method == "threshold":
        return CFGCacheConfig(
            enabled=True,
            start=args.cfg_start,
            end=args.cfg_end,
            threshold=float(threshold),
            max_reuse=args.cfg_max_reuse,
            eps=args.cfg_eps,
            metric=args.cfg_metric,
            force_uncond_recompute_on_miss=False,
        )
    return SeaCFGCacheConfig(
        enabled=True,
        threshold=float(threshold),
        eps=args.cfg_eps,
        power_exp=args.cfg_sea_power_exp,
        power_const=args.cfg_sea_power_const,
        norm_mode=args.cfg_sea_norm_mode,
        ret_steps=args.cfg_ret_steps,
        cutoff_steps=args.cfg_cutoff_steps,
        force_uncond_recompute_on_miss=False,
    )


def create_pipeline(args, cfg):
    import wan

    return wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=args.convert_model_dtype,
    )


def generate_one(args, pipeline, cfg, prompt: str, output: Path, log_path: Path, cfg_cache_config):
    import torch
    from wan.configs import SIZE_CONFIGS
    from wan.utils.utils import save_video

    with run_log_context(log_path):
        logging.info(f"Input prompt: {prompt}")
        logging.info(f"Generating video to {output}")
        logging.info(f"cfg_cache_config={cfg_cache_config}")
        start = time.perf_counter()
        video = pipeline.generate(
            prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            timestep_cache_config=None,
            block_cache_config=None,
            block_group_cache_config=None,
            cfg_cache_config=cfg_cache_config,
        )
        wall = time.perf_counter() - start
        logging.info(f"generation_wall_elapsed_seconds={wall:.3f}")
        logging.info(f"Saving generated video to {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        save_video(
            tensor=video[None],
            save_file=str(output),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        del video
        torch.cuda.empty_cache()
        logging.info("Finished.")


def summarize(exp_root: Path, candidates):
    baseline_time = read_time_file(exp_root / "logs" / "baseline_prompt_01.time")
    rows = []
    for method, threshold, label in candidates:
        time_file = exp_root / "logs" / f"{label}_prompt_01.time"
        psnr_file = exp_root / "psnr" / f"{label}_prompt_01.json"
        log_path = exp_root / "logs" / f"{label}_prompt_01.log"
        video_path = exp_root / "videos" / f"{label}_prompt_01.mp4"
        elapsed = read_time_file(time_file)
        psnr = json.loads(psnr_file.read_text(encoding="utf-8"))
        rows.append({
            "prompt_id": "01",
            "candidate": label,
            "cfg_method": method,
            "cfg_threshold": threshold,
            "baseline_elapsed_seconds": baseline_time,
            "candidate_elapsed_seconds": elapsed,
            "speedup": baseline_time / elapsed if baseline_time and elapsed else None,
            "mean_psnr": psnr.get("mean_psnr"),
            "min_psnr": psnr.get("min_psnr"),
            "max_psnr": psnr.get("max_psnr"),
            "psnr_frames": psnr.get("frames"),
            "video_path": str(video_path),
            "log_path": str(log_path),
            "psnr_path": str(psnr_file),
        })
    output = exp_root / "results" / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (exp_root / "results" / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print(output.read_text(encoding="utf-8"), end="")


def cpu_validate(args):
    candidates = parse_candidates(args.candidates)
    missing = []
    baseline_root = Path(args.baseline_root)
    for rel in [
        Path("baseline") / "prompt_01.mp4",
        Path("ffprobe") / "baseline_prompt_01.json",
    ]:
        path = baseline_root / rel
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
    print(json.dumps({
        "status": "ok" if not missing else "missing_baseline_artifacts",
        "candidates": candidates,
        "candidate_count": len(candidates),
        "single_process_pipeline_load": True,
        "baseline_root": str(baseline_root),
        "missing": missing,
    }, indent=2))
    if missing:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description="Prompt-01 cfg-cache-only comparison runner")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt_file", default=str(REPO_ROOT / "prompt.txt"))
    parser.add_argument("--baseline_root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--baseline_compute_seconds", type=float, default=522.603)
    parser.add_argument("--exp_root", default=None)
    parser.add_argument("--task", default="t2v-A14B")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_solver", default="dpm++", choices=["dpm++", "unipc"])
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, nargs=2, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--offload_model", type=lambda value: value.lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--no_convert_model_dtype", dest="convert_model_dtype", action="store_false")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--cpu_validate", action="store_true")
    parser.add_argument("--ffprobe_bin", default=DEFAULT_FFPROBE)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--cfg_start", type=float, default=0.1)
    parser.add_argument("--cfg_end", type=float, default=0.9)
    parser.add_argument("--cfg_metric", default="timestep_modulated_input_rel_l1")
    parser.add_argument("--cfg_max_reuse", type=int, default=3)
    parser.add_argument("--cfg_eps", type=float, default=1e-6)
    parser.add_argument("--cfg_sea_power_exp", type=float, default=3.0)
    parser.add_argument("--cfg_sea_power_const", type=float, default=1.0)
    parser.add_argument("--cfg_sea_norm_mode", default="mean", choices=["mean", "peak"])
    parser.add_argument("--cfg_ret_steps", type=int, default=1)
    parser.add_argument("--cfg_cutoff_steps", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/hy-tmp/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/hy-tmp/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/hy-tmp/hf-cache/hub")

    if args.cpu_validate:
        cpu_validate(args)
        return

    import torch
    import wan
    from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS

    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_{stamp}"
    exp_root = Path(args.exp_root)
    candidates = parse_candidates(args.candidates)

    cfg = WAN_CONFIGS[args.task]
    if args.task != "t2v-A14B":
        raise SystemExit("This runner currently supports t2v-A14B only")
    if args.size not in SUPPORTED_SIZES[args.task]:
        raise SystemExit(f"Unsupported size {args.size} for {args.task}")
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
        if not isinstance(args.sample_guide_scale, tuple):
            args.sample_guide_scale = tuple(args.sample_guide_scale)
    else:
        args.sample_guide_scale = tuple(args.sample_guide_scale)

    for subdir in ["baseline", "videos", "logs", "commands", "ffprobe", "psnr", "results", "failed"]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)
    subprocess.run(["nvidia-smi"], stdout=(exp_root / "gpu.txt").open("w", encoding="utf-8"), stderr=subprocess.STDOUT)
    baseline_video, _baseline_time = copy_baseline_artifacts(args, exp_root)

    env = vars(args).copy()
    env.update({
        "experiment_root": str(exp_root),
        "baseline_video": str(baseline_video),
        "candidates": args.candidates,
        "timestep_cache": "none",
        "block_cache": "none",
    })
    (exp_root / "experiment_config.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")

    prompts = read_prompts(Path(args.prompt_file))
    prompt = prompts[0]
    print(json.dumps({k: str(v) for k, v in env.items()}, indent=2))
    print("Creating WanT2V pipeline once for cfg-cache prompt-01 batch run")
    with run_log_context(exp_root / "logs" / "pipeline_init.log"):
        pipeline = create_pipeline(args, cfg)

    for method, threshold, label in candidates:
        output = exp_root / "videos" / f"{label}_prompt_01.mp4"
        log_path = exp_root / "logs" / f"{label}_prompt_01.log"
        time_file = exp_root / "logs" / f"{label}_prompt_01.time"
        ffprobe_json = exp_root / "ffprobe" / f"{label}_prompt_01.json"
        psnr_json = exp_root / "psnr" / f"{label}_prompt_01.json"
        psnr_log = exp_root / "psnr" / f"{label}_prompt_01.log"
        save_command_record(
            exp_root / "commands" / f"{label}_prompt_01.sh",
            sys.argv,
            {
                "method": "cfg-cache-only",
                "cfg_method": method,
                "cfg_threshold": threshold,
                "prompt_index": "01",
                "seed": args.base_seed,
                "output": output,
                "timestep_cache": "none",
                "block_cache": "none",
            },
        )
        if args.resume_existing and maybe_completed(output, time_file, ffprobe_json, psnr_json):
            print(f"Skipping existing {label}")
            continue
        print(f"Running {label} prompt 01 seed {args.base_seed}")
        try:
            if not (args.resume_existing and output.exists() and output.stat().st_size > 0 and time_file.exists() and time_file.stat().st_size > 0):
                cfg_cache_config = make_cfg_config(args, method, threshold)
                generate_one(args, pipeline, cfg, prompt, output, log_path, cfg_cache_config)
                elapsed = parse_elapsed(log_path)
                time_file.write_text(f"elapsed_seconds={elapsed if elapsed is not None else ''}\n", encoding="utf-8")
                torch.cuda.empty_cache()
            if not (args.resume_existing and ffprobe_json.exists() and ffprobe_json.stat().st_size > 0):
                run_ffprobe(args.ffprobe_bin, output, ffprobe_json)
            if not (args.resume_existing and psnr_json.exists() and psnr_json.stat().st_size > 0):
                run_psnr(args.python_bin, baseline_video, output, psnr_json, psnr_log)
        except Exception as exc:
            write_failed(exp_root, f"{label}_prompt_01", {
                "method": "cfg-cache-only",
                "cfg_method": method,
                "cfg_threshold": threshold,
                "prompt_index": "01",
                "seed": args.base_seed,
                "status": "exception",
                "error": repr(exc),
                "log": log_path,
            })
            raise

    summarize(exp_root, candidates)
    print(f"Completed experiment: {exp_root}")


if __name__ == "__main__":
    main()
