#!/usr/bin/env python3
import argparse
import ast
import contextlib
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


DEFAULT_THRESHOLDS = "0.05 0.10 0.20 0.30 0.50"
DEFAULT_BASELINE_ROOT = "/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427"
DEFAULT_FFPROBE = "/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe"


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
    prompts = data["prompts"]
    if not isinstance(prompts, list):
        raise ValueError("prompt.txt must contain a prompts list")
    return [prompt.replace("\n", " ").strip() for prompt in prompts]


def threshold_label(value: str) -> str:
    return f"th_{value.replace('.', 'p').replace('-', '_')}"


def parse_elapsed(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    if not matches:
        return None
    return float(matches[-1])


def read_time_file(path: Path):
    if not path.exists():
        return None
    match = re.search(r"elapsed_seconds=([0-9.]+)", path.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    return float(match.group(1))


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


def run_psnr(python_bin: str, tools_dir: Path, reference: Path, candidate: Path, output: Path, log_path: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(tools_dir / "compute_psnr.py"),
        "--reference", str(reference),
        "--candidate", str(candidate),
        "--output", str(output),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"PSNR failed for {candidate}; see {log_path}")


def save_command_record(path: Path, root_dir: Path, argv, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"cd {root_dir}", " ".join(subprocess.list2cmdline([item]) for item in argv)]
    lines.extend(f"# {k}={v}" for k, v in extra.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_completed(video: Path, time_file: Path, ffprobe_json: Path, psnr_json: Path | None = None):
    required = [video, time_file, ffprobe_json]
    if psnr_json is not None:
        required.append(psnr_json)
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def copy_baseline_artifacts(args, prompt_index: str, exp_root: Path):
    source_root = Path(args.baseline_root)
    baseline_video = exp_root / "baseline" / f"prompt_{prompt_index}.mp4"
    baseline_ffprobe = exp_root / "ffprobe" / f"baseline_prompt_{prompt_index}.json"
    baseline_time = exp_root / "logs" / f"baseline_prompt_{prompt_index}.time"
    baseline_log = exp_root / "logs" / f"baseline_prompt_{prompt_index}.log"

    source_video = source_root / "baseline" / f"prompt_{prompt_index}.mp4"
    source_ffprobe = source_root / "ffprobe" / f"baseline_prompt_{prompt_index}.json"
    source_time = source_root / "logs" / f"baseline_prompt_{prompt_index}.time"
    source_log = source_root / "logs" / f"baseline_prompt_{prompt_index}.log"
    for source, dest in [
        (source_video, baseline_video),
        (source_ffprobe, baseline_ffprobe),
        (source_time, baseline_time),
        (source_log, baseline_log),
    ]:
        if not source.exists() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing reusable baseline artifact: {source}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size == 0:
            shutil.copy2(source, dest)
    return baseline_video, baseline_ffprobe, baseline_time, baseline_log


def make_seacache_config(args, threshold: str):
    from wan.timestep_cache import SeaCacheTimestepCacheConfig

    return SeaCacheTimestepCacheConfig(
        enabled=True,
        threshold=float(threshold),
        num_steps=args.seacache_num_steps,
        use_ret_steps=args.seacache_use_ret_steps,
        power_exp=args.seacache_power_exp,
        power_const=args.seacache_power_const,
        eps=args.seacache_eps,
        norm_mode=args.seacache_norm_mode,
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


def generate_one(args, pipeline, cfg, prompt: str, seed: int, output: Path, log_path: Path, cache_config):
    import torch
    from wan.configs import SIZE_CONFIGS
    from wan.utils.utils import save_video

    with run_log_context(log_path):
        logging.info(f"Input prompt: {prompt}")
        logging.info(f"Generating video to {output}")
        logging.info(f"timestep_cache_config={cache_config}")
        start = time.perf_counter()
        video = pipeline.generate(
            prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=seed,
            offload_model=args.offload_model,
            timestep_cache_config=cache_config,
            block_cache_config=None,
            block_group_cache_config=None,
            cfg_cache_config=None,
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


def cpu_validate(args):
    prompts = read_prompts(Path(args.prompt_file))
    selected = prompts[args.prompt_start:]
    if args.prompt_limit > 0:
        selected = selected[:args.prompt_limit]
    thresholds = args.thresholds.split()
    if not selected:
        raise SystemExit("No prompts selected")
    if not thresholds:
        raise SystemExit("No thresholds selected")
    baseline_root = Path(args.baseline_root)
    missing = []
    if not args.generate_baseline:
        for offset, _prompt in enumerate(selected):
            prompt_index = f"{args.prompt_start + offset + 1:02d}"
            for rel in [
                Path("baseline") / f"prompt_{prompt_index}.mp4",
                Path("ffprobe") / f"baseline_prompt_{prompt_index}.json",
                Path("logs") / f"baseline_prompt_{prompt_index}.time",
            ]:
                path = baseline_root / rel
                if not path.exists() or path.stat().st_size == 0:
                    missing.append(str(path))
    result = {
        "status": "ok" if not missing else "missing_baseline_artifacts",
        "prompt_count": len(selected),
        "thresholds": thresholds,
        "generate_baseline": args.generate_baseline,
        "baseline_root": str(baseline_root),
        "missing": missing,
        "expected_candidate_runs": len(selected) * len(thresholds),
        "single_process_pipeline_load": True,
    }
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description="Single-process SeaCache T2V batch runner")
    parser.add_argument("--root_dir", default="/hy-tmp/work/Wan2.2")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt_file", default="/hy-tmp/work/Wan2.2/prompt.txt")
    parser.add_argument("--baseline_root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--exp_root", default=None)
    parser.add_argument("--task", default="t2v-A14B")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_solver", default="dpm++", choices=["dpm++", "unipc"])
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, nargs=2, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--prompt_limit", type=int, default=1)
    parser.add_argument("--prompt_start", type=int, default=0)
    parser.add_argument("--offload_model", type=lambda value: value.lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--no_convert_model_dtype", dest="convert_model_dtype", action="store_false")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--generate_baseline", action="store_true")
    parser.add_argument("--cpu_validate", action="store_true")
    parser.add_argument("--ffprobe_bin", default=DEFAULT_FFPROBE)
    parser.add_argument("--seacache_num_steps", type=int, default=None)
    parser.add_argument("--seacache_use_ret_steps", action="store_true", default=False)
    parser.add_argument("--seacache_power_exp", type=float, default=3.0)
    parser.add_argument("--seacache_power_const", type=float, default=1.0)
    parser.add_argument("--seacache_eps", type=float, default=1e-16)
    parser.add_argument("--seacache_norm_mode", default="mean", choices=["mean", "peak"])
    args = parser.parse_args()

    if args.cpu_validate:
        cpu_validate(args)
        return

    import torch
    import wan
    from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS

    root_dir = Path(args.root_dir)
    tools_dir = root_dir / "experiments" / "zeus_timestep_cache_50step_45f_480p"
    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_seacache_50step_45f_480p_{stamp}"
    exp_root = Path(args.exp_root)
    thresholds = args.thresholds.split()

    cfg = WAN_CONFIGS[args.task]
    if args.task != "t2v-A14B":
        raise SystemExit("This batch runner currently supports t2v-A14B only")
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

    prompts = read_prompts(Path(args.prompt_file))
    if args.prompt_start:
        prompts = prompts[args.prompt_start:]
    if args.prompt_limit > 0:
        prompts = prompts[:args.prompt_limit]
    if not prompts:
        raise SystemExit("No prompts selected")

    for subdir in ["baseline", "seacache", "thresholds", "logs", "commands", "ffprobe", "psnr", "results", "failed"]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)
    for threshold in thresholds:
        label = threshold_label(threshold)
        (exp_root / "seacache" / label).mkdir(parents=True, exist_ok=True)
        (exp_root / "psnr" / label).mkdir(parents=True, exist_ok=True)
        (exp_root / "thresholds" / f"{label}.env").write_text(f"threshold={threshold}\n", encoding="utf-8")

    env = {
        "experiment_root": exp_root,
        "root_dir": root_dir,
        "python_bin": args.python_bin,
        "ckpt_dir": args.ckpt_dir,
        "prompt_file": args.prompt_file,
        "baseline_root": args.baseline_root,
        "generate_baseline": args.generate_baseline,
        "ffprobe_bin": args.ffprobe_bin,
        "task": args.task,
        "size": args.size,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "sample_solver": args.sample_solver,
        "sample_shift": args.sample_shift,
        "sample_guide_scale": args.sample_guide_scale,
        "base_seed": args.base_seed,
        "thresholds": args.thresholds,
        "prompt_start": args.prompt_start,
        "prompt_limit": args.prompt_limit,
        "resume_existing": args.resume_existing,
        "seacache_num_steps": args.seacache_num_steps,
        "seacache_use_ret_steps": args.seacache_use_ret_steps,
        "seacache_power_exp": args.seacache_power_exp,
        "seacache_power_const": args.seacache_power_const,
        "seacache_eps": args.seacache_eps,
        "seacache_norm_mode": args.seacache_norm_mode,
        "block_cache": "none",
        "cfg_cache": "none",
    }
    (exp_root / "experiment.env").write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
    subprocess.run(["nvidia-smi"], stdout=(exp_root / "gpu.txt").open("w", encoding="utf-8"), stderr=subprocess.STDOUT)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    print(json.dumps({k: str(v) for k, v in env.items()}, indent=2))
    print("Creating WanT2V pipeline once for SeaCache batch run")
    pipeline_log = exp_root / "logs" / "pipeline_init.log"
    with run_log_context(pipeline_log):
        pipeline = create_pipeline(args, cfg)

    for offset, prompt in enumerate(prompts):
        prompt_index = f"{args.prompt_start + offset + 1:02d}"
        seed = args.base_seed
        baseline_video = exp_root / "baseline" / f"prompt_{prompt_index}.mp4"
        baseline_log = exp_root / "logs" / f"baseline_prompt_{prompt_index}.log"
        baseline_time = exp_root / "logs" / f"baseline_prompt_{prompt_index}.time"
        baseline_ffprobe = exp_root / "ffprobe" / f"baseline_prompt_{prompt_index}.json"
        save_command_record(
            exp_root / "commands" / f"baseline_prompt_{prompt_index}.sh",
            root_dir,
            sys.argv,
            {"method": "baseline", "prompt_index": prompt_index, "seed": seed, "output": baseline_video},
        )
        if args.generate_baseline:
            if args.resume_existing and maybe_completed(baseline_video, baseline_time, baseline_ffprobe):
                print(f"Skipping existing generated baseline prompt {prompt_index}")
            else:
                print(f"Running baseline prompt {prompt_index} seed {seed}")
                try:
                    generate_one(args, pipeline, cfg, prompt, seed, baseline_video, baseline_log, None)
                    elapsed = parse_elapsed(baseline_log)
                    baseline_time.write_text(f"elapsed_seconds={elapsed if elapsed is not None else ''}\n", encoding="utf-8")
                    run_ffprobe(args.ffprobe_bin, baseline_video, baseline_ffprobe)
                except Exception as exc:
                    write_failed(exp_root, f"baseline_prompt_{prompt_index}", {
                        "method": "baseline", "prompt_index": prompt_index, "seed": seed,
                        "status": "exception", "error": repr(exc), "log": baseline_log,
                    })
                    raise
        else:
            baseline_video, baseline_ffprobe, baseline_time, baseline_log = copy_baseline_artifacts(args, prompt_index, exp_root)

        for threshold in thresholds:
            label = threshold_label(threshold)
            method_id = f"seacache_{label}"
            output = exp_root / "seacache" / label / f"prompt_{prompt_index}.mp4"
            log_path = exp_root / "logs" / f"{method_id}_prompt_{prompt_index}.log"
            time_file = exp_root / "logs" / f"{method_id}_prompt_{prompt_index}.time"
            ffprobe_json = exp_root / "ffprobe" / f"{method_id}_prompt_{prompt_index}.json"
            psnr_json = exp_root / "psnr" / label / f"prompt_{prompt_index}.json"
            psnr_log = exp_root / "psnr" / label / f"prompt_{prompt_index}.log"
            save_command_record(
                exp_root / "commands" / f"{method_id}_prompt_{prompt_index}.sh",
                root_dir,
                sys.argv,
                {
                    "method": "seacache",
                    "threshold": threshold,
                    "prompt_index": prompt_index,
                    "seed": seed,
                    "output": output,
                    "block_cache": "none",
                    "cfg_cache": "none",
                },
            )
            if args.resume_existing and maybe_completed(output, time_file, ffprobe_json, psnr_json):
                print(f"Skipping existing SeaCache {threshold} prompt {prompt_index}")
                continue
            print(f"Running SeaCache {threshold} prompt {prompt_index} seed {seed}")
            try:
                if not (args.resume_existing and output.exists() and output.stat().st_size > 0 and time_file.exists() and time_file.stat().st_size > 0):
                    generate_one(args, pipeline, cfg, prompt, seed, output, log_path, make_seacache_config(args, threshold))
                    elapsed = parse_elapsed(log_path)
                    time_file.write_text(f"elapsed_seconds={elapsed if elapsed is not None else ''}\n", encoding="utf-8")
                    torch.cuda.empty_cache()
                if not (args.resume_existing and ffprobe_json.exists() and ffprobe_json.stat().st_size > 0):
                    run_ffprobe(args.ffprobe_bin, output, ffprobe_json)
                if not (args.resume_existing and psnr_json.exists() and psnr_json.stat().st_size > 0):
                    run_psnr(args.python_bin, tools_dir, baseline_video, output, psnr_json, psnr_log)
            except Exception as exc:
                write_failed(exp_root, f"{method_id}_prompt_{prompt_index}", {
                    "method": "seacache", "threshold": threshold, "threshold_label": label,
                    "prompt_index": prompt_index, "seed": seed, "status": "exception",
                    "error": repr(exc), "log": log_path,
                })
                raise

    summary_log = exp_root / "results" / "summary.log"
    cmd = [
        args.python_bin,
        str(root_dir / "experiments" / "seacache_50step_45f_480p" / "summarize_results.py"),
        "--experiment-root", str(exp_root),
        "--output", str(exp_root / "results" / "summary.csv"),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    summary_log.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.returncode != 0:
        write_failed(exp_root, "summary", {"method": "summary", "status": proc.returncode, "log": summary_log})
        raise SystemExit(proc.returncode)
    print(f"Completed experiment: {exp_root}")


if __name__ == "__main__":
    main()
