#!/usr/bin/env python3
import argparse
import contextlib
import csv
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.taylorseer_wan22.cache import TaylorSeerConfig
from third_party.taylorseer_wan22.text2video import TaylorSeerWanT2V
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.text2video import WanT2V
from wan.utils.utils import save_video, str2bool


DEFAULT_PROMPTS = REPO_ROOT / "test_sets" / "vbench_every20" / "prompts.jsonl"
DEFAULT_CKPT = "/hy-tmp/models/Wan2.2-T2V-A14B"
DEFAULT_PYTHON = "/hy-tmp/miniconda3/envs/Wan2.2/bin/python"
DEFAULT_FFPROBE = "/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe"
PSNR_SCRIPT = REPO_ROOT / "experiments" / "zeus_timestep_cache_50step_45f_480p" / "compute_psnr.py"


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


@contextlib.contextmanager
def rank0_log_context(log_path: Path, enabled: bool):
    if not enabled:
        yield
        return
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


def init_logging(rank: int):
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def read_prompt_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                row = json.loads(line)
                rows.append({
                    "sample_id": row.get("sample_id") or f"prompt_{line_index:03d}",
                    "prompt_index": line_index,
                    "source_index_1based": row.get("source_index_1based", ""),
                    "prompt": row["text"],
                })
            else:
                rows.append({
                    "sample_id": f"prompt_{line_index:03d}",
                    "prompt_index": line_index,
                    "source_index_1based": "",
                    "prompt": line,
                })
    return rows


def select_rows(rows, prompt_start: int, prompt_limit: int):
    selected = rows[prompt_start:]
    if prompt_limit > 0:
        selected = selected[:prompt_limit]
    return selected


def run_ffprobe(ffprobe_bin: str, video: Path, output: Path):
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
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    output.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video}: {proc.stderr.strip()}")


def parse_elapsed(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def parse_cache_summary(log_path: Path):
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"TaylorSeer cache summary: (.*)", text)
    if not matches:
        return {}
    try:
        import ast

        return ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return {}


def cache_totals(summary):
    reuse = 0
    recompute = 0
    skipping = set()
    recompute_steps = set()
    for state in summary.values():
        if not isinstance(state, dict):
            continue
        reuse += int(state.get("reuse", 0))
        recompute += int(state.get("recompute", 0))
        skipping.update(state.get("skipping_path", []))
        recompute_steps.update(state.get("recompute_path", []))
    return {
        "taylorseer_reuse_branch_call_count": reuse,
        "taylorseer_recompute_branch_call_count": recompute,
        "taylorseer_reuse_step_count": len(skipping),
        "taylorseer_recompute_step_count": len(recompute_steps),
        "taylorseer_summary": json.dumps(summary, separators=(",", ":")),
    }


def artifact_complete(*paths: Path):
    return all(path.exists() and path.stat().st_size > 0 for path in [
        *paths,
    ])


def baseline_complete(video: Path, time_file: Path, ffprobe_json: Path):
    return artifact_complete(video, time_file, ffprobe_json)


def candidate_complete(video: Path, time_file: Path, ffprobe_json: Path, psnr_json: Path):
    return artifact_complete(video, time_file, ffprobe_json, psnr_json)


def run_psnr(args, baseline_video: Path, candidate_video: Path, psnr_json: Path, psnr_log: Path):
    psnr_json.parent.mkdir(parents=True, exist_ok=True)
    psnr_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        str(PSNR_SCRIPT),
        "--reference",
        str(baseline_video),
        "--candidate",
        str(candidate_video),
        "--output",
        str(psnr_json),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    psnr_log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"PSNR failed for {candidate_video} against {baseline_video}; see {psnr_log}")


def write_failed(root: Path, name: str, fields):
    path = root / "failed" / f"{name}.failed"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in fields.items()) + "\n",
        encoding="utf-8",
    )


def write_command(path: Path, command, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"cd {REPO_ROOT}", " ".join(subprocess.list2cmdline([x]) for x in command)]
    lines.extend(f"# {key}={value}" for key, value in extra.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(exp_root: Path, args, selected_rows):
    rows = []
    for row in selected_rows:
        sample_id = row["sample_id"]
        baseline_video = exp_root / "baseline" / f"{sample_id}.mp4"
        baseline_log = exp_root / "logs" / f"baseline_{sample_id}.log"
        baseline_time = exp_root / "logs" / f"baseline_{sample_id}.time"
        baseline_ffprobe_json = exp_root / "ffprobe" / f"baseline_{sample_id}.json"
        candidate_video = exp_root / "videos" / f"{sample_id}.mp4"
        candidate_log = exp_root / "logs" / f"{sample_id}.log"
        candidate_time = exp_root / "logs" / f"{sample_id}.time"
        candidate_ffprobe_json = exp_root / "ffprobe" / f"{sample_id}.json"
        psnr_json = exp_root / "psnr" / f"{sample_id}.json"
        psnr_log = exp_root / "psnr" / f"{sample_id}.log"
        if not (
                baseline_complete(baseline_video, baseline_time, baseline_ffprobe_json)
                and candidate_complete(candidate_video, candidate_time, candidate_ffprobe_json, psnr_json)):
            continue

        baseline_elapsed = parse_elapsed(baseline_log)
        if baseline_elapsed is None:
            match = re.search(
                r"elapsed_seconds=([0-9.]+)",
                baseline_time.read_text(encoding="utf-8", errors="replace"),
            )
            baseline_elapsed = float(match.group(1)) if match else None

        candidate_elapsed = parse_elapsed(candidate_log)
        if candidate_elapsed is None:
            match = re.search(
                r"elapsed_seconds=([0-9.]+)",
                candidate_time.read_text(encoding="utf-8", errors="replace"),
            )
            candidate_elapsed = float(match.group(1)) if match else None

        speedup = ""
        if baseline_elapsed and candidate_elapsed:
            speedup = baseline_elapsed / candidate_elapsed

        candidate_ffprobe = json.loads(candidate_ffprobe_json.read_text(encoding="utf-8"))
        candidate_meta = candidate_ffprobe["streams"][0] if candidate_ffprobe.get("streams") else {}
        baseline_ffprobe = json.loads(baseline_ffprobe_json.read_text(encoding="utf-8"))
        baseline_meta = baseline_ffprobe["streams"][0] if baseline_ffprobe.get("streams") else {}
        psnr = json.loads(psnr_json.read_text(encoding="utf-8"))
        summary = parse_cache_summary(candidate_log)
        cache_fields = cache_totals(summary)
        rows.append({
            "sample_id": sample_id,
            "prompt_index": row["prompt_index"],
            "source_index_1based": row.get("source_index_1based", ""),
            "prompt": row["prompt"],
            "task": args.task,
            "seed": args.base_seed,
            "size": args.size,
            "frame_num": args.frame_num,
            "sample_steps": args.sample_steps,
            "sample_solver": args.sample_solver,
            "sample_shift": args.sample_shift,
            "sample_guide_scale": json.dumps(args.sample_guide_scale),
            "method": "taylorseer",
            "taylorseer_fresh_threshold": args.taylorseer_fresh_threshold,
            "taylorseer_max_order": args.taylorseer_max_order,
            "taylorseer_first_enhance": args.taylorseer_first_enhance,
            "baseline_elapsed_seconds": baseline_elapsed,
            "compute_elapsed_seconds": candidate_elapsed,
            "speedup": speedup,
            "psnr": psnr.get("mean_psnr", ""),
            "psnr_min": psnr.get("min_psnr", ""),
            "psnr_max": psnr.get("max_psnr", ""),
            "psnr_frames": psnr.get("frames", ""),
            "psnr_decoded_frames_total": psnr.get("decoded_frames_total", ""),
            "psnr_excluded_perfect_frames": psnr.get("excluded_perfect_frames", ""),
            "ffprobe_width": candidate_meta.get("width", ""),
            "ffprobe_height": candidate_meta.get("height", ""),
            "ffprobe_frames": candidate_meta.get("nb_frames") or candidate_meta.get("nb_read_frames", ""),
            "ffprobe_fps": candidate_meta.get("r_frame_rate", ""),
            "ffprobe_duration": candidate_meta.get("duration", ""),
            "baseline_ffprobe_width": baseline_meta.get("width", ""),
            "baseline_ffprobe_height": baseline_meta.get("height", ""),
            "baseline_ffprobe_frames": baseline_meta.get("nb_frames") or baseline_meta.get("nb_read_frames", ""),
            "baseline_ffprobe_fps": baseline_meta.get("r_frame_rate", ""),
            "baseline_ffprobe_duration": baseline_meta.get("duration", ""),
            "baseline_video_path": str(baseline_video),
            "baseline_log_path": str(baseline_log),
            "baseline_ffprobe_path": str(baseline_ffprobe_json),
            "video_path": str(candidate_video),
            "log_path": str(candidate_log),
            "ffprobe_path": str(candidate_ffprobe_json),
            "psnr_path": str(psnr_json),
            "psnr_log_path": str(psnr_log),
            **cache_fields,
        })
    result_dir = exp_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "summary.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (result_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def create_baseline_pipeline(args, cfg, rank: int, device: int):
    return WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )


def create_taylorseer_pipeline(args, cfg, rank: int, device: int):
    return TaylorSeerWanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        taylorseer_config=TaylorSeerConfig(
            enabled=True,
            fresh_threshold=args.taylorseer_fresh_threshold,
            max_order=args.taylorseer_max_order,
            first_enhance=args.taylorseer_first_enhance,
        ),
    )


def generate_one(args, pipeline, cfg, row, output: Path, log_path: Path, rank: int, method: str):
    prompt = row["prompt"]
    with rank0_log_context(log_path, enabled=(rank == 0)):
        logging.info("sample_id=%s", row["sample_id"])
        logging.info("prompt=%s", prompt)
        logging.info("output=%s", output)
        logging.info("method=%s", method)
        if method == "taylorseer":
            logging.info("TaylorSeer fresh_threshold=%s max_order=%s first_enhance=%s",
                         args.taylorseer_fresh_threshold,
                         args.taylorseer_max_order,
                         args.taylorseer_first_enhance)
        start = time.perf_counter()
        generate_kwargs = {
            "size": SIZE_CONFIGS[args.size],
            "frame_num": args.frame_num,
            "shift": args.sample_shift,
            "sample_solver": args.sample_solver,
            "sampling_steps": args.sample_steps,
            "guide_scale": args.sample_guide_scale,
            "seed": args.base_seed,
            "offload_model": args.offload_model,
        }
        if method == "baseline":
            generate_kwargs.update({
                "timestep_cache_config": None,
                "block_cache_config": None,
                "block_group_cache_config": None,
                "cfg_cache_config": None,
            })
        video = pipeline.generate(prompt, **generate_kwargs)
        wall = time.perf_counter() - start
        logging.info("generation_wall_elapsed_seconds=%.3f", wall)
        if rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            logging.info("Saving generated video to %s", output)
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
        logging.info("Finished sample_id=%s method=%s", row["sample_id"], method)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch runner for TaylorSeer Wan2.2 on VBench prompts.")
    parser.add_argument("--python_bin", default=DEFAULT_PYTHON)
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    parser.add_argument("--prompt_jsonl", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--exp_root", default=None)
    parser.add_argument("--task", default="t2v-A14B")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_solver", default="dpm++", choices=["dpm++", "unipc"])
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, nargs=2, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--taylorseer_fresh_threshold", type=int, default=5)
    parser.add_argument("--taylorseer_max_order", type=int, default=1)
    parser.add_argument("--taylorseer_first_enhance", type=int, default=1)
    parser.add_argument("--prompt_start", type=int, default=0)
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--cpu_validate", action="store_true")
    parser.add_argument("--ffprobe_bin", default=DEFAULT_FFPROBE)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/hy-tmp/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/hy-tmp/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/hy-tmp/hf-cache/hub")

    all_rows = read_prompt_rows(Path(args.prompt_jsonl))
    selected_rows = select_rows(all_rows, args.prompt_start, args.prompt_limit)
    if args.cpu_validate:
        print(json.dumps({
            "status": "ok",
            "prompt_jsonl": args.prompt_jsonl,
            "total_prompts": len(all_rows),
            "selected_prompts": len(selected_rows),
            "first_sample_id": selected_rows[0]["sample_id"] if selected_rows else None,
            "last_sample_id": selected_rows[-1]["sample_id"] if selected_rows else None,
            "batch_runner": True,
            "baseline_then_taylorseer": True,
            "single_pipeline_load_per_process_per_phase": True,
            "psnr_script": str(PSNR_SCRIPT),
        }, indent=2))
        return

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    init_logging(rank)
    device = local_rank
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "FSDP requires torchrun multi-process execution."
        assert args.ulysses_size == 1, "ulysses_size > 1 requires torchrun."
    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "ulysses_size must equal WORLD_SIZE."
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.task != "t2v-A14B":
        raise SystemExit("This runner currently supports t2v-A14B only.")
    if args.size not in SUPPORTED_SIZES[args.task]:
        raise SystemExit(f"Unsupported size {args.size} for {args.task}.")
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, (
            f"`{cfg.num_heads=}` must be divisible by `{args.ulysses_size=}`.")
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = tuple(cfg.sample_guide_scale)
    else:
        args.sample_guide_scale = tuple(args.sample_guide_scale)
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_taylorseer_vbench_50step_45f_480p_{stamp}"
    exp_root = Path(args.exp_root)
    if rank == 0:
        for dirname in ["baseline", "videos", "logs", "commands", "ffprobe", "psnr", "results", "failed"]:
            (exp_root / dirname).mkdir(parents=True, exist_ok=True)
        env = vars(args).copy()
        env.update({
            "experiment_root": str(exp_root),
            "world_size": world_size,
            "rank0_only_artifacts": True,
            "method": "taylorseer",
            "prompt_source": str(Path(args.prompt_jsonl).resolve()),
            "selected_prompt_count": len(selected_rows),
            "baseline": "no-cache WanT2V baseline generated before TaylorSeer candidates",
            "psnr_script": str(PSNR_SCRIPT),
        })
        (exp_root / "experiment_config.json").write_text(
            json.dumps(env, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        launch_env = {
            key: os.environ.get(key, "")
            for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE"]
        }
        (exp_root / "launch.env").write_text(
            "\n".join(f"{key}={value}" for key, value in launch_env.items()) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["nvidia-smi"],
            stdout=(exp_root / "gpu.txt").open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
    if dist.is_initialized():
        dist.barrier()

    logging.info("Creating no-cache WanT2V baseline pipeline once for batch run.")
    with rank0_log_context(exp_root / "logs" / "pipeline_init_baseline.log", enabled=(rank == 0)):
        baseline_pipeline = create_baseline_pipeline(args, cfg, rank, device)

    for index, row in enumerate(selected_rows, start=1):
        sample_id = row["sample_id"]
        video = exp_root / "baseline" / f"{sample_id}.mp4"
        log_path = exp_root / "logs" / f"baseline_{sample_id}.log"
        time_file = exp_root / "logs" / f"baseline_{sample_id}.time"
        ffprobe_json = exp_root / "ffprobe" / f"baseline_{sample_id}.json"

        if rank == 0:
            write_command(
                exp_root / "commands" / f"baseline_{sample_id}.sh",
                sys.argv,
                {
                    "method": "baseline",
                    "sample_index": f"{index}/{len(selected_rows)}",
                    "sample_id": sample_id,
                    "prompt_index": row["prompt_index"],
                    "seed": args.base_seed,
                    "output": video,
                },
            )
        skip_current = False
        if args.resume_existing and rank == 0 and baseline_complete(video, time_file, ffprobe_json):
            logging.info("Skipping existing baseline %s (%s/%s)", sample_id, index, len(selected_rows))
            skip_current = True
        if dist.is_initialized():
            skip_list = [skip_current] if rank == 0 else [None]
            dist.broadcast_object_list(skip_list, src=0)
            skip_current = skip_list[0]
        if skip_current:
            if dist.is_initialized():
                dist.barrier()
            continue

        logging.info("Running baseline %s (%s/%s)", sample_id, index, len(selected_rows))
        try:
            generate_one(args, baseline_pipeline, cfg, row, video, log_path, rank, method="baseline")
            if rank == 0:
                elapsed = parse_elapsed(log_path)
                if elapsed is None:
                    raise RuntimeError(f"Missing inference_compute_elapsed_seconds in {log_path}")
                time_file.write_text(f"elapsed_seconds={elapsed}\n", encoding="utf-8")
                run_ffprobe(args.ffprobe_bin, video, ffprobe_json)
                logging.info("Archived baseline %s", sample_id)
        except Exception as exc:
            if rank == 0:
                write_failed(exp_root, f"baseline_{sample_id}", {
                    "sample_id": sample_id,
                    "method": "baseline",
                    "status": "exception",
                    "error": repr(exc),
                    "log": log_path,
                })
            raise
        finally:
            if dist.is_initialized():
                dist.barrier()

    del baseline_pipeline
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()

    logging.info("Creating TaylorSeerWanT2V pipeline once for batch run.")
    with rank0_log_context(exp_root / "logs" / "pipeline_init_taylorseer.log", enabled=(rank == 0)):
        pipeline = create_taylorseer_pipeline(args, cfg, rank, device)

    for index, row in enumerate(selected_rows, start=1):
        sample_id = row["sample_id"]
        baseline_video = exp_root / "baseline" / f"{sample_id}.mp4"
        video = exp_root / "videos" / f"{sample_id}.mp4"
        log_path = exp_root / "logs" / f"{sample_id}.log"
        time_file = exp_root / "logs" / f"{sample_id}.time"
        ffprobe_json = exp_root / "ffprobe" / f"{sample_id}.json"
        psnr_json = exp_root / "psnr" / f"{sample_id}.json"
        psnr_log = exp_root / "psnr" / f"{sample_id}.log"

        if rank == 0:
            write_command(
                exp_root / "commands" / f"{sample_id}.sh",
                sys.argv,
                {
                    "method": "taylorseer",
                    "sample_index": f"{index}/{len(selected_rows)}",
                    "sample_id": sample_id,
                    "prompt_index": row["prompt_index"],
                    "seed": args.base_seed,
                    "baseline": baseline_video,
                    "output": video,
                    "psnr": psnr_json,
                },
            )
        skip_current = False
        if args.resume_existing and rank == 0 and candidate_complete(video, time_file, ffprobe_json, psnr_json):
            logging.info("Skipping existing TaylorSeer %s (%s/%s)", sample_id, index, len(selected_rows))
            skip_current = True
        if dist.is_initialized():
            skip_list = [skip_current] if rank == 0 else [None]
            dist.broadcast_object_list(skip_list, src=0)
            skip_current = skip_list[0]
        if skip_current:
            if dist.is_initialized():
                dist.barrier()
            continue

        logging.info("Running TaylorSeer %s (%s/%s)", sample_id, index, len(selected_rows))
        try:
            generate_one(args, pipeline, cfg, row, video, log_path, rank, method="taylorseer")
            if rank == 0:
                elapsed = parse_elapsed(log_path)
                if elapsed is None:
                    raise RuntimeError(f"Missing inference_compute_elapsed_seconds in {log_path}")
                time_file.write_text(f"elapsed_seconds={elapsed}\n", encoding="utf-8")
                run_ffprobe(args.ffprobe_bin, video, ffprobe_json)
                run_psnr(args, baseline_video, video, psnr_json, psnr_log)
                rows = summarize(exp_root, args, selected_rows)
                logging.info("Archived TaylorSeer %s; completed_rows=%s", sample_id, len(rows))
        except Exception as exc:
            if rank == 0:
                write_failed(exp_root, sample_id, {
                    "sample_id": sample_id,
                    "method": "taylorseer",
                    "status": "exception",
                    "error": repr(exc),
                    "log": log_path,
                })
            raise
        finally:
            if dist.is_initialized():
                dist.barrier()

    if rank == 0:
        rows = summarize(exp_root, args, selected_rows)
        logging.info("Completed experiment: %s; completed_rows=%s", exp_root, len(rows))
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
