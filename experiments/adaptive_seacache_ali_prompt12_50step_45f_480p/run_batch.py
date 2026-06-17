#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASELINE_ROOT = "/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427"
DEFAULT_FFPROBE = "/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe"
DEFAULT_GATE_MODEL = (
    "/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/"
    "temporal_mean/best_model.pt"
)


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


def read_prompts(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if prompts:
        return prompts
    text = path.read_text(encoding="utf-8").strip()
    data = ast.literal_eval("{" + text + "}")
    return [prompt.replace("\n", " ").strip() for prompt in data["prompts"]]


def target_label(value: str) -> str:
    return f"target_{value.replace('.', 'p').replace('-', '_')}"


def parse_elapsed(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def read_time_file(path: Path) -> float | None:
    if not path.exists():
        return None
    match = re.search(
        r"elapsed_seconds=([0-9.]+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    return float(match.group(1)) if match and match.group(1) else None


def load_psnr(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("mean_psnr")
    return float(value) if value is not None else None


def write_failed(root: Path, name: str, fields: dict[str, object]) -> None:
    failed = root / "failed" / f"{name}.txt"
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text(
        "\n".join(f"{key}={value}" for key, value in fields.items()) + "\n",
        encoding="utf-8",
    )


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


def run_ffprobe(ffprobe_bin: str, video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration",
        "-of",
        "json",
        str(video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        output.write_text(proc.stdout, encoding="utf-8")
        raise RuntimeError(f"ffprobe failed for {video}: {proc.stderr.strip()}")
    output.write_text(proc.stdout, encoding="utf-8")


def run_psnr(
    python_bin: str,
    tools_dir: Path,
    reference: Path,
    candidate: Path,
    output: Path,
    log_path: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(tools_dir / "compute_psnr.py"),
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--output",
        str(output),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"PSNR failed for {candidate}; see {log_path}")


def save_command_record(path: Path, root_dir: Path, argv: list[str], extra: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"cd {root_dir}", " ".join(subprocess.list2cmdline([item]) for item in argv)]
    lines.extend(f"# {key}={value}" for key, value in extra.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_completed(
    video: Path,
    time_file: Path,
    ffprobe_json: Path,
    psnr_json: Path | None = None,
    trace_json: Path | None = None,
) -> bool:
    required = [video, time_file, ffprobe_json]
    if psnr_json is not None:
        required.append(psnr_json)
    if trace_json is not None:
        required.append(trace_json)
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def copy_baseline_artifacts(args, prompt_index: str, exp_root: Path):
    source_root = Path(args.baseline_root)
    baseline_video = exp_root / "baseline" / f"prompt_{prompt_index}.mp4"
    baseline_ffprobe = exp_root / "ffprobe" / f"baseline_prompt_{prompt_index}.json"
    baseline_time = exp_root / "logs" / f"baseline_prompt_{prompt_index}.time"
    baseline_log = exp_root / "logs" / f"baseline_prompt_{prompt_index}.log"
    source_paths = [
        (source_root / "baseline" / f"prompt_{prompt_index}.mp4", baseline_video),
        (source_root / "ffprobe" / f"baseline_prompt_{prompt_index}.json", baseline_ffprobe),
        (source_root / "logs" / f"baseline_prompt_{prompt_index}.time", baseline_time),
        (source_root / "logs" / f"baseline_prompt_{prompt_index}.log", baseline_log),
    ]
    for source, dest in source_paths:
        if not source.exists() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Missing reusable baseline artifact: {source}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size == 0:
            shutil.copy2(source, dest)
    return baseline_video, baseline_ffprobe, baseline_time, baseline_log


def make_seacache_config(args):
    from wan.timestep_cache import SeaCacheTimestepCacheConfig

    return SeaCacheTimestepCacheConfig(
        enabled=True,
        threshold=args.seacache_threshold,
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


def extract_trace(summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, payload in sorted(summary.items()):
        branch = ""
        model_stage = ""
        try:
            parsed = ast.literal_eval(key)
            model_stage, branch = parsed[0], parsed[1]
        except Exception:
            pass
        for item in payload.get("adaptive_decision_trace", []):
            row = dict(item)
            row["cache_key"] = key
            row["model_stage"] = model_stage
            row["branch"] = branch
            rows.append(row)
    rows.sort(key=lambda row: (int(row["step_index"]), str(row["model_stage"]), str(row["branch"])))
    return rows


def write_trace(trace_rows: list[dict[str, object]], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(trace_rows, indent=2), encoding="utf-8")
    fields = [
        "step_index",
        "model_stage",
        "branch",
        "predicted_threshold",
        "rel_l1",
        "accumulated_rel_l1",
        "decision",
        "force_recompute",
        "cache_key",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trace_rows)


def trace_stats(trace_rows: list[dict[str, object]]) -> dict[str, object]:
    values = [float(row["predicted_threshold"]) for row in trace_rows]
    reuse = sum(1 for row in trace_rows if row.get("decision") == "reuse")
    recompute = sum(1 for row in trace_rows if row.get("decision") == "recompute")
    return {
        "trace_rows": len(trace_rows),
        "reuse_decisions": reuse,
        "recompute_decisions": recompute,
        "threshold_min": min(values) if values else None,
        "threshold_max": max(values) if values else None,
        "threshold_mean": sum(values) / len(values) if values else None,
    }


def generate_one(
    args,
    pipeline,
    cfg,
    prompt: str,
    seed: int,
    output: Path,
    log_path: Path,
    cache_config,
    adaptive_factory=None,
):
    import torch
    from wan.configs import SIZE_CONFIGS
    from wan.utils.utils import save_video

    with run_log_context(log_path):
        logging.info(f"Input prompt: {prompt}")
        logging.info(f"Generating video to {output}")
        logging.info(f"timestep_cache_config={cache_config}")
        logging.info("block_cache_config=None")
        logging.info("cfg_cache_config=None")
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
        summary = None
        if adaptive_factory is not None and adaptive_factory.last_instance is not None:
            summary = adaptive_factory.last_instance.summary()
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
    return summary


def cpu_validate(args) -> None:
    prompts = read_prompts(Path(args.prompt_file))
    selected = prompts[args.prompt_start:]
    if args.prompt_limit > 0:
        selected = selected[: args.prompt_limit]
    target_psnrs = args.target_psnrs.split()
    missing = []
    if not args.generate_baseline:
        baseline_root = Path(args.baseline_root)
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
        "target_psnrs": target_psnrs,
        "expected_candidate_runs": len(selected) * len(target_psnrs),
        "single_process_pipeline_load": True,
        "missing": missing,
    }
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive SeaCache ali prompt1/2 batch runner")
    parser.add_argument("--root_dir", default="/hy-tmp/work/Wan2.2")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt_file", default="/hy-tmp/work/Wan2.2/test_sets/ali_10/prompts.txt")
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
    parser.add_argument("--target_psnrs", default="20 25 30")
    parser.add_argument("--prompt_limit", type=int, default=2)
    parser.add_argument("--prompt_start", type=int, default=0)
    parser.add_argument("--offload_model", type=lambda value: value.lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--no_convert_model_dtype", dest="convert_model_dtype", action="store_false")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--generate_baseline", action="store_true")
    parser.add_argument("--cpu_validate", action="store_true")
    parser.add_argument("--ffprobe_bin", default=DEFAULT_FFPROBE)
    parser.add_argument("--seacache_threshold", type=float, default=0.2)
    parser.add_argument("--seacache_num_steps", type=int, default=None)
    parser.add_argument("--seacache_use_ret_steps", action="store_true", default=False)
    parser.add_argument("--seacache_power_exp", type=float, default=3.0)
    parser.add_argument("--seacache_power_const", type=float, default=1.0)
    parser.add_argument("--seacache_eps", type=float, default=1e-16)
    parser.add_argument("--seacache_norm_mode", default="mean", choices=["mean", "peak"])
    parser.add_argument("--adaptive_gate_model", default=DEFAULT_GATE_MODEL)
    parser.add_argument("--adaptive_feature_set", choices=("temporal_mean", "latent_pool"), default="temporal_mean")
    parser.add_argument("--adaptive_hidden_dim", type=int, default=16)
    parser.add_argument("--adaptive_feature_dim", type=int, default=128)
    parser.add_argument("--adaptive_grid_size", nargs=3, type=int, default=(2, 2, 2))
    parser.add_argument("--adaptive_psnr_min", type=float, default=10.0)
    parser.add_argument("--adaptive_psnr_max", type=float, default=50.0)
    parser.add_argument("--adaptive_min_threshold", type=float, default=0.0)
    parser.add_argument("--adaptive_max_threshold", type=float, default=1.0)
    args = parser.parse_args()

    if args.cpu_validate:
        cpu_validate(args)
        return

    import torch
    import wan.text2video as wan_text2video
    from adaptive_seacache_wan22.cache import AdaptiveSeaCacheGateConfig, build_adaptive_seacache_factory
    from adaptive_seacache_wan22.patch import patch_wan_model_forward_for_adaptive_seacache
    from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS

    root_dir = Path(args.root_dir)
    tools_dir = root_dir / "experiments" / "zeus_timestep_cache_50step_45f_480p"
    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_{stamp}"
    exp_root = Path(args.exp_root)
    target_psnrs = args.target_psnrs.split()

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
    prompts = prompts[args.prompt_start :]
    if args.prompt_limit > 0:
        prompts = prompts[: args.prompt_limit]
    if not prompts:
        raise SystemExit("No prompts selected")

    for subdir in [
        "baseline",
        "adaptive_seacache",
        "targets",
        "logs",
        "commands",
        "ffprobe",
        "psnr",
        "traces",
        "results",
        "failed",
    ]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)
    for target_psnr in target_psnrs:
        label = target_label(target_psnr)
        for subdir in ["adaptive_seacache", "psnr", "traces"]:
            (exp_root / subdir / label).mkdir(parents=True, exist_ok=True)
        (exp_root / "targets" / f"{label}.env").write_text(
            f"target_psnr={target_psnr}\n",
            encoding="utf-8",
        )

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
        "target_psnrs": args.target_psnrs,
        "prompt_start": args.prompt_start,
        "prompt_limit": args.prompt_limit,
        "resume_existing": args.resume_existing,
        "adaptive_gate_model": args.adaptive_gate_model,
        "adaptive_feature_set": args.adaptive_feature_set,
        "adaptive_hidden_dim": args.adaptive_hidden_dim,
        "adaptive_feature_dim": args.adaptive_feature_dim,
        "adaptive_grid_size": args.adaptive_grid_size,
        "adaptive_psnr_min": args.adaptive_psnr_min,
        "adaptive_psnr_max": args.adaptive_psnr_max,
        "adaptive_min_threshold": args.adaptive_min_threshold,
        "adaptive_max_threshold": args.adaptive_max_threshold,
        "seacache_threshold_config_default": args.seacache_threshold,
        "seacache_num_steps": args.seacache_num_steps,
        "seacache_use_ret_steps": args.seacache_use_ret_steps,
        "seacache_power_exp": args.seacache_power_exp,
        "seacache_power_const": args.seacache_power_const,
        "seacache_eps": args.seacache_eps,
        "seacache_norm_mode": args.seacache_norm_mode,
        "block_cache": "none",
        "cfg_cache": "none",
    }
    (exp_root / "experiment_config.json").write_text(
        json.dumps({k: str(v) for k, v in env.items()}, indent=2),
        encoding="utf-8",
    )
    (exp_root / "launch.env").write_text(
        "".join(f"{key}={value}\n" for key, value in env.items()),
        encoding="utf-8",
    )
    subprocess.run(
        ["nvidia-smi"],
        stdout=(exp_root / "gpu.txt").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    print(json.dumps({key: str(value) for key, value in env.items()}, indent=2))

    gate_config = AdaptiveSeaCacheGateConfig(
        model_path=Path(args.adaptive_gate_model),
        target_psnr=0.0,
        feature_set=args.adaptive_feature_set,
        hidden_dim=args.adaptive_hidden_dim,
        feature_dim=args.adaptive_feature_dim,
        grid_size=tuple(args.adaptive_grid_size),
        psnr_min=args.adaptive_psnr_min,
        psnr_max=args.adaptive_psnr_max,
        min_threshold=args.adaptive_min_threshold,
        max_threshold=args.adaptive_max_threshold,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    patch_wan_model_forward_for_adaptive_seacache()
    adaptive_factory = build_adaptive_seacache_factory(gate_config)
    wan_text2video.SeaCacheTimestepCache = adaptive_factory

    print("Creating WanT2V pipeline once for adaptive SeaCache batch run")
    pipeline_log = exp_root / "logs" / "pipeline_init.log"
    with run_log_context(pipeline_log):
        pipeline = create_pipeline(args, cfg)

    rows = []
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
            {
                "method": "baseline",
                "prompt_index": prompt_index,
                "seed": seed,
                "output": baseline_video,
            },
        )
        if args.generate_baseline:
            if args.resume_existing and maybe_completed(baseline_video, baseline_time, baseline_ffprobe):
                print(f"Skipping existing generated baseline prompt {prompt_index}")
            else:
                print(f"Running baseline prompt {prompt_index} seed {seed}")
                try:
                    summary = generate_one(
                        args,
                        pipeline,
                        cfg,
                        prompt,
                        seed,
                        baseline_video,
                        baseline_log,
                        None,
                    )
                    elapsed = parse_elapsed(baseline_log)
                    baseline_time.write_text(
                        f"elapsed_seconds={elapsed if elapsed is not None else ''}\n",
                        encoding="utf-8",
                    )
                    run_ffprobe(args.ffprobe_bin, baseline_video, baseline_ffprobe)
                except Exception as exc:
                    write_failed(
                        exp_root,
                        f"baseline_prompt_{prompt_index}",
                        {
                            "method": "baseline",
                            "prompt_index": prompt_index,
                            "seed": seed,
                            "status": "exception",
                            "error": repr(exc),
                            "log": baseline_log,
                        },
                    )
                    raise
        else:
            baseline_video, baseline_ffprobe, baseline_time, baseline_log = copy_baseline_artifacts(
                args,
                prompt_index,
                exp_root,
            )

        baseline_elapsed = read_time_file(baseline_time)
        for target_psnr in target_psnrs:
            target_value = float(target_psnr)
            gate_config.target_psnr = target_value
            label = target_label(target_psnr)
            method_id = f"adaptive_seacache_{label}"
            output = exp_root / "adaptive_seacache" / label / f"prompt_{prompt_index}.mp4"
            log_path = exp_root / "logs" / f"{method_id}_prompt_{prompt_index}.log"
            time_file = exp_root / "logs" / f"{method_id}_prompt_{prompt_index}.time"
            ffprobe_json = exp_root / "ffprobe" / f"{method_id}_prompt_{prompt_index}.json"
            psnr_json = exp_root / "psnr" / label / f"prompt_{prompt_index}.json"
            psnr_log = exp_root / "psnr" / label / f"prompt_{prompt_index}.log"
            trace_json = exp_root / "traces" / label / f"prompt_{prompt_index}.json"
            trace_csv = exp_root / "traces" / label / f"prompt_{prompt_index}.csv"
            save_command_record(
                exp_root / "commands" / f"{method_id}_prompt_{prompt_index}.sh",
                root_dir,
                sys.argv,
                {
                    "method": "adaptive_seacache",
                    "target_psnr": target_psnr,
                    "prompt_index": prompt_index,
                    "seed": seed,
                    "output": output,
                    "block_cache": "none",
                    "cfg_cache": "none",
                    "adaptive_gate_model": args.adaptive_gate_model,
                },
            )
            if args.resume_existing and maybe_completed(
                output,
                time_file,
                ffprobe_json,
                psnr_json,
                trace_json,
            ):
                print(f"Skipping existing adaptive SeaCache target {target_psnr} prompt {prompt_index}")
                continue
            print(f"Running adaptive SeaCache target {target_psnr} prompt {prompt_index} seed {seed}")
            try:
                if not (
                    args.resume_existing
                    and output.exists()
                    and output.stat().st_size > 0
                    and time_file.exists()
                    and time_file.stat().st_size > 0
                    and trace_json.exists()
                    and trace_json.stat().st_size > 0
                ):
                    summary = generate_one(
                        args,
                        pipeline,
                        cfg,
                        prompt,
                        seed,
                        output,
                        log_path,
                        make_seacache_config(args),
                        adaptive_factory=adaptive_factory,
                    )
                    elapsed = parse_elapsed(log_path)
                    time_file.write_text(
                        f"elapsed_seconds={elapsed if elapsed is not None else ''}\n",
                        encoding="utf-8",
                    )
                    trace_rows = extract_trace(summary or {})
                    write_trace(trace_rows, trace_json, trace_csv)
                    torch.cuda.empty_cache()
                if not (args.resume_existing and ffprobe_json.exists() and ffprobe_json.stat().st_size > 0):
                    run_ffprobe(args.ffprobe_bin, output, ffprobe_json)
                if not (args.resume_existing and psnr_json.exists() and psnr_json.stat().st_size > 0):
                    run_psnr(args.python_bin, tools_dir, baseline_video, output, psnr_json, psnr_log)

                trace_rows = json.loads(trace_json.read_text(encoding="utf-8"))
                stats = trace_stats(trace_rows)
                elapsed = read_time_file(time_file)
                psnr = load_psnr(psnr_json)
                rows.append(
                    {
                        "sample_id": f"ali_{int(prompt_index):03d}",
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "method": "adaptive_seacache",
                        "target_psnr": target_value,
                        "adaptive_feature_set": args.adaptive_feature_set,
                        "adaptive_hidden_dim": args.adaptive_hidden_dim,
                        "compute_elapsed_seconds": elapsed,
                        "baseline_compute_elapsed_seconds": baseline_elapsed,
                        "speedup": (
                            baseline_elapsed / elapsed
                            if baseline_elapsed is not None and elapsed and elapsed > 0
                            else None
                        ),
                        "mean_psnr": psnr,
                        "video_path": str(output),
                        "log_path": str(log_path),
                        "trace_json": str(trace_json),
                        "trace_csv": str(trace_csv),
                        **stats,
                    }
                )
            except Exception as exc:
                write_failed(
                    exp_root,
                    f"{method_id}_prompt_{prompt_index}",
                    {
                        "method": "adaptive_seacache",
                        "target_psnr": target_psnr,
                        "target_label": label,
                        "prompt_index": prompt_index,
                        "seed": seed,
                        "status": "exception",
                        "error": repr(exc),
                        "log": log_path,
                    },
                )
                raise

    summary_csv = exp_root / "results" / "summary.csv"
    fieldnames = [
        "sample_id",
        "prompt_index",
        "seed",
        "method",
        "target_psnr",
        "adaptive_feature_set",
        "adaptive_hidden_dim",
        "compute_elapsed_seconds",
        "baseline_compute_elapsed_seconds",
        "speedup",
        "mean_psnr",
        "trace_rows",
        "reuse_decisions",
        "recompute_decisions",
        "threshold_min",
        "threshold_max",
        "threshold_mean",
        "video_path",
        "log_path",
        "trace_json",
        "trace_csv",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (exp_root / "results" / "summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    print(f"Completed experiment: {exp_root}")


if __name__ == "__main__":
    main()
