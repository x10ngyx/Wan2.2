#!/usr/bin/env python3
import argparse
import ast
import contextlib
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASELINE_ROOT = "/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625"
DEFAULT_FFPROBE = "/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe"
DEFAULT_THRESHOLDS = "0.05 0.10 0.20 0.40 1.00"


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


def value_label(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


def candidate_label(ts_threshold: str, block_threshold: str, cfg_threshold: str) -> str:
    return (
        f"sea_ts_{value_label(ts_threshold)}"
        f"__sea_bg_{value_label(block_threshold)}"
        f"__sea_cfg_{value_label(cfg_threshold)}"
    )


def parse_elapsed(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def read_time_file(path: Path):
    if not path.exists():
        return None
    match = re.search(r"elapsed_seconds=([0-9.]+)", path.read_text(encoding="utf-8", errors="replace"))
    return float(match.group(1)) if match else None


def maybe_completed(video: Path, time_file: Path, ffprobe_json: Path, psnr_json: Path):
    return all(path.exists() and path.stat().st_size > 0 for path in [video, time_file, ffprobe_json, psnr_json])


def write_failed(root: Path, name: str, fields):
    failed = root / "failed" / f"{name}.failed"
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
    source_video = source_root / "baseline" / "prompt_01.mp4"
    source_ffprobe = source_root / "ffprobe" / "baseline_prompt_01.json"
    if not source_video.exists() or source_video.stat().st_size == 0:
        raise FileNotFoundError(f"Missing baseline video: {source_video}")
    if not source_ffprobe.exists() or source_ffprobe.stat().st_size == 0:
        raise FileNotFoundError(f"Missing baseline ffprobe: {source_ffprobe}")
    baseline_video = exp_root / "baseline" / "prompt_01.mp4"
    baseline_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_video, baseline_video)
    shutil.copy2(source_ffprobe, exp_root / "ffprobe" / "baseline_prompt_01.json")
    (exp_root / "logs" / "baseline_prompt_01.time").write_text(
        f"elapsed_seconds={args.baseline_compute_seconds}\n", encoding="utf-8")
    return baseline_video


def run_ffprobe(ffprobe_bin: str, video: Path, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffprobe_bin, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration",
        "-of", "json", str(video),
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


def save_command_record(path: Path, argv, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"cd {REPO_ROOT}", " ".join(subprocess.list2cmdline([item]) for item in argv)]
    lines.extend(f"# {k}={v}" for k, v in extra.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_timestep_config(args, threshold: str):
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


def make_block_config(args, threshold: str):
    from wan.block_group_cache import BlockGroupCacheConfig

    return BlockGroupCacheConfig(
        enabled=True,
        group_size=args.block_group_size,
        threshold=float(threshold),
        metric="sea_full_rel_l1",
        decision="accumulated",
        start=0.0,
        end=1.0,
        max_reuse=args.block_group_max_reuse,
        eps=args.block_group_eps,
        ret_steps=args.block_group_ret_steps,
        cutoff_steps=args.block_group_cutoff_steps,
        sea_power_exp=args.block_group_sea_power_exp,
        sea_power_const=args.block_group_sea_power_const,
        sea_norm_mode=args.block_group_sea_norm_mode,
    )


def make_cfg_config(args, threshold: str):
    from wan.cfg_cache import SeaCFGCacheConfig

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


def generate_one(args, pipeline, cfg, prompt: str, output: Path, log_path: Path, ts_config, bg_config, cfg_cache_config):
    import torch
    from wan.configs import SIZE_CONFIGS
    from wan.utils.utils import save_video

    with run_log_context(log_path):
        logging.info(f"Input prompt: {prompt}")
        logging.info(f"Generating video to {output}")
        logging.info(f"timestep_cache_config={ts_config}")
        logging.info(f"block_group_cache_config={bg_config}")
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
            timestep_cache_config=ts_config,
            block_cache_config=None,
            block_group_cache_config=bg_config,
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


def parse_cache_summary(log_path: Path, prefix: str):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(rf"{re.escape(prefix)}: (.*)", text)
    if not matches:
        return {}
    try:
        return ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return {}


def sum_flat(summary, field):
    return sum(int(state.get(field, 0)) for state in summary.values() if isinstance(state, dict))


def sum_nested(summary, field):
    total = 0
    for branch_summary in summary.values():
        if not isinstance(branch_summary, dict):
            continue
        for group_summary in branch_summary.values():
            if isinstance(group_summary, dict):
                total += int(group_summary.get(field, 0))
    return total


def unique_flat_steps(summary, field):
    steps = set()
    for state in summary.values():
        if not isinstance(state, dict):
            continue
        for item in state.get(field, []):
            steps.add(item[0] if isinstance(item, (list, tuple)) and item else item)
    return len(steps)


def collect_json(summary, field):
    result = {}
    for key, state in summary.items():
        if isinstance(state, dict) and field in state:
            result[key] = state[field]
    return json.dumps(result, separators=(",", ":"))


def collect_nested_json(summary, field):
    result = {}
    for branch_key, branch_summary in summary.items():
        if not isinstance(branch_summary, dict):
            continue
        for group_key, group_summary in branch_summary.items():
            if isinstance(group_summary, dict) and field in group_summary:
                result[f"{branch_key}/group_{group_key}"] = group_summary[field]
    return json.dumps(result, separators=(",", ":"))


def build_candidates(args):
    candidates = []
    for ts_threshold, bg_threshold, cfg_threshold in itertools.product(
        args.timestep_thresholds.split(),
        args.block_thresholds.split(),
        args.cfg_thresholds.split(),
    ):
        label = candidate_label(ts_threshold, bg_threshold, cfg_threshold)
        candidates.append((ts_threshold, bg_threshold, cfg_threshold, label))
    if args.combo_limit > 0:
        candidates = candidates[:args.combo_limit]
    return candidates


def summarize(exp_root: Path, candidates):
    baseline_time = read_time_file(exp_root / "logs" / "baseline_prompt_01.time")
    rows = []
    for ts_threshold, bg_threshold, cfg_threshold, label in candidates:
        time_file = exp_root / "logs" / f"{label}_prompt_01.time"
        psnr_file = exp_root / "psnr" / f"{label}_prompt_01.json"
        ffprobe_file = exp_root / "ffprobe" / f"{label}_prompt_01.json"
        log_path = exp_root / "logs" / f"{label}_prompt_01.log"
        video_path = exp_root / "videos" / f"{label}_prompt_01.mp4"
        if not maybe_completed(video_path, time_file, ffprobe_file, psnr_file):
            continue
        elapsed = read_time_file(time_file)
        psnr = json.loads(psnr_file.read_text(encoding="utf-8"))
        meta = json.loads(ffprobe_file.read_text(encoding="utf-8"))["streams"][0]
        timestep_summary = parse_cache_summary(log_path, "Timestep cache summary")
        block_summary = parse_cache_summary(log_path, "Block-group cache summary")
        cfg_summary = parse_cache_summary(log_path, "CFG cache summary")
        rows.append({
            "prompt_id": "01",
            "candidate": label,
            "sample_id": "prompt_01",
            "seed": 42,
            "size": "832*480",
            "frame_num": 45,
            "sample_steps": 50,
            "sample_solver": "dpm++",
            "timestep_cache": "seacache",
            "timestep_threshold": ts_threshold,
            "block_cache": "block-group",
            "block_group_metric": "sea_full_rel_l1",
            "block_group_decision": "accumulated",
            "block_threshold": bg_threshold,
            "cfg_cache": "sea-threshold",
            "cfg_threshold": cfg_threshold,
            "baseline_elapsed_seconds": baseline_time,
            "candidate_elapsed_seconds": elapsed,
            "speedup": baseline_time / elapsed if baseline_time and elapsed else None,
            "mean_psnr": psnr.get("mean_psnr"),
            "min_psnr": psnr.get("min_psnr"),
            "max_psnr": psnr.get("max_psnr"),
            "psnr_frames": psnr.get("frames"),
            "ffprobe_width": meta.get("width"),
            "ffprobe_height": meta.get("height"),
            "ffprobe_frames": meta.get("nb_frames") or meta.get("nb_read_frames"),
            "ffprobe_fps": meta.get("r_frame_rate"),
            "ffprobe_duration": meta.get("duration"),
            "timestep_reuse_count": unique_flat_steps(timestep_summary, "skipping_path"),
            "timestep_recompute_count": unique_flat_steps(timestep_summary, "recompute_path"),
            "timestep_reuse_branch_call_count": sum_flat(timestep_summary, "reuse"),
            "timestep_recompute_branch_call_count": sum_flat(timestep_summary, "recompute"),
            "block_reuse_count": sum_nested(block_summary, "reuse"),
            "block_recompute_count": sum_nested(block_summary, "recompute"),
            "cfg_reuse_count": sum_flat(cfg_summary, "reuse"),
            "cfg_recompute_count": sum_flat(cfg_summary, "recompute"),
            "timestep_skipping_path": collect_json(timestep_summary, "skipping_path"),
            "timestep_recompute_path": collect_json(timestep_summary, "recompute_path"),
            "block_reuse_path": collect_nested_json(block_summary, "reuse_path"),
            "block_recompute_path": collect_nested_json(block_summary, "recompute_path"),
            "block_metric_path": collect_nested_json(block_summary, "metric_path"),
            "block_accumulated_metric_path": collect_nested_json(block_summary, "accumulated_metric_path"),
            "cfg_reuse_path": collect_json(cfg_summary, "reuse_path"),
            "cfg_recompute_path": collect_json(cfg_summary, "recompute_path"),
            "video_path": str(video_path),
            "log_path": str(log_path),
            "psnr_path": str(psnr_file),
            "ffprobe_path": str(ffprobe_file),
        })
    output = exp_root / "results" / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (exp_root / "results" / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def cpu_validate(args):
    candidates = build_candidates(args)
    missing = []
    baseline_root = Path(args.baseline_root)
    for rel in [Path("baseline") / "prompt_01.mp4", Path("ffprobe") / "baseline_prompt_01.json"]:
        path = baseline_root / rel
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
    print(json.dumps({
        "status": "ok" if not missing else "missing_baseline_artifacts",
        "candidate_count": len(candidates),
        "single_process_pipeline_load": True,
        "baseline_root": str(baseline_root),
        "thresholds": {
            "timestep": args.timestep_thresholds.split(),
            "block": args.block_thresholds.split(),
            "cfg": args.cfg_thresholds.split(),
        },
        "missing": missing,
    }, indent=2))
    if missing:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description="Prompt-01 three sea-style cache threshold-grid runner")
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
    parser.add_argument("--timestep_thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--block_thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--cfg_thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--combo_limit", type=int, default=0)
    parser.add_argument("--seacache_num_steps", type=int, default=None)
    parser.add_argument("--seacache_use_ret_steps", action="store_true")
    parser.add_argument("--seacache_power_exp", type=float, default=3.0)
    parser.add_argument("--seacache_power_const", type=float, default=1.0)
    parser.add_argument("--seacache_eps", type=float, default=1e-16)
    parser.add_argument("--seacache_norm_mode", default="mean", choices=["mean", "peak"])
    parser.add_argument("--block_group_size", type=int, default=5)
    parser.add_argument("--block_group_max_reuse", type=int, default=50)
    parser.add_argument("--block_group_eps", type=float, default=1e-6)
    parser.add_argument("--block_group_ret_steps", type=int, default=1)
    parser.add_argument("--block_group_cutoff_steps", type=int, default=1)
    parser.add_argument("--block_group_sea_power_exp", type=float, default=3.0)
    parser.add_argument("--block_group_sea_power_const", type=float, default=1.0)
    parser.add_argument("--block_group_sea_norm_mode", default="mean", choices=["mean", "peak"])
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
    from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS

    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_{stamp}"
    exp_root = Path(args.exp_root)
    candidates = build_candidates(args)

    cfg = WAN_CONFIGS[args.task]
    if args.task != "t2v-A14B":
        raise SystemExit("This runner currently supports t2v-A14B only")
    if args.size not in SUPPORTED_SIZES[args.task]:
        raise SystemExit(f"Unsupported size {args.size} for {args.task}")
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = tuple(cfg.sample_guide_scale)
    else:
        args.sample_guide_scale = tuple(args.sample_guide_scale)

    for subdir in ["baseline", "videos", "logs", "commands", "ffprobe", "psnr", "results", "failed"]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)
    subprocess.run(["nvidia-smi"], stdout=(exp_root / "gpu.txt").open("w", encoding="utf-8"), stderr=subprocess.STDOUT)
    baseline_video = copy_baseline_artifacts(args, exp_root)

    env = vars(args).copy()
    env.update({
        "experiment_root": str(exp_root),
        "baseline_video": str(baseline_video),
        "timestep_cache": "seacache",
        "block_cache": "block-group sea_full_rel_l1 accumulated",
        "cfg_cache": "sea-threshold",
        "candidate_count": len(candidates),
        "cache_order": "CFG outer; branch timestep cache; block-group cache only on timestep miss",
    })
    (exp_root / "experiment_config.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")

    prompt = read_prompts(Path(args.prompt_file))[0]
    print(json.dumps({k: str(v) for k, v in env.items()}, indent=2))
    print("Creating WanT2V pipeline once for prompt-01 three sea-style cache grid")
    with run_log_context(exp_root / "logs" / "pipeline_init.log"):
        pipeline = create_pipeline(args, cfg)

    for index, (ts_threshold, bg_threshold, cfg_threshold, label) in enumerate(candidates, start=1):
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
                "method": "three-sea-style-caches",
                "candidate_index": f"{index}/{len(candidates)}",
                "timestep_threshold": ts_threshold,
                "block_threshold": bg_threshold,
                "cfg_threshold": cfg_threshold,
                "prompt_index": "01",
                "seed": args.base_seed,
                "output": output,
            },
        )
        if args.resume_existing and maybe_completed(output, time_file, ffprobe_json, psnr_json):
            print(f"Skipping existing {label} ({index}/{len(candidates)})")
            continue
        print(f"Running {label} ({index}/{len(candidates)}) prompt 01 seed {args.base_seed}")
        try:
            if not (args.resume_existing and output.exists() and output.stat().st_size > 0 and time_file.exists() and time_file.stat().st_size > 0):
                generate_one(
                    args,
                    pipeline,
                    cfg,
                    prompt,
                    output,
                    log_path,
                    make_timestep_config(args, ts_threshold),
                    make_block_config(args, bg_threshold),
                    make_cfg_config(args, cfg_threshold),
                )
                elapsed = parse_elapsed(log_path)
                if elapsed is None:
                    raise RuntimeError(f"Missing inference_compute_elapsed_seconds in {log_path}")
                time_file.write_text(f"elapsed_seconds={elapsed}\n", encoding="utf-8")
                torch.cuda.empty_cache()
            if not (args.resume_existing and ffprobe_json.exists() and ffprobe_json.stat().st_size > 0):
                run_ffprobe(args.ffprobe_bin, output, ffprobe_json)
            if not (args.resume_existing and psnr_json.exists() and psnr_json.stat().st_size > 0):
                run_psnr(args.python_bin, baseline_video, output, psnr_json, psnr_log)
            rows = summarize(exp_root, candidates)
            print(f"Archived {label}; completed_rows={len(rows)}")
        except Exception as exc:
            write_failed(exp_root, f"{label}_prompt_01", {
                "method": "three-sea-style-caches",
                "candidate_index": f"{index}/{len(candidates)}",
                "timestep_threshold": ts_threshold,
                "block_threshold": bg_threshold,
                "cfg_threshold": cfg_threshold,
                "prompt_index": "01",
                "seed": args.base_seed,
                "status": "exception",
                "error": repr(exc),
                "log": log_path,
            })
            raise

    rows = summarize(exp_root, candidates)
    print(f"Completed experiment: {exp_root}; completed_rows={len(rows)}")


if __name__ == "__main__":
    main()
