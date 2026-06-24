#!/usr/bin/env python3
import argparse
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

DEFAULT_PROMPTS = REPO_ROOT / "test_sets" / "Vbench10" / "prompts.jsonl"
DEFAULT_FFPROBE = "/hy-tmp/env/Wan2.2/bin/ffprobe"


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


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def load_vbench_records(path: Path):
    rows = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ["sample_id", "text"]:
            if field not in row:
                raise ValueError(f"Missing {field!r} in {path}:{line_no}")
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        row = dict(row)
        row["prompt"] = row["text"].replace("\n", " ").strip()
        rows.append(row)
    return rows


def select_records(records, start: int, limit: int):
    selected = records[start:]
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise SystemExit("No Vbench10 records selected")
    return selected


def parse_elapsed(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    if not matches:
        return None
    return float(matches[-1])


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
    env = os.environ.copy()
    env["PATH"] = f"/hy-tmp/env/Wan2.2/bin:{env.get('PATH', '')}"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
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


def write_manifest_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest_csv(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "source",
        "source_url",
        "source_index_1based",
        "source_set",
        "source_sample_id",
        "source_set_index_1based",
        "sampling_rule",
        "random_seed",
        "text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def make_zeus_config(args):
    from wan.timestep_cache import ZeusTimestepCacheConfig

    return ZeusTimestepCacheConfig(
        enabled=True,
        acc_range=(args.zeus_acc_start, args.zeus_acc_end),
        denominator=args.zeus_denominator,
        modular=tuple(args.zeus_modular),
        caching_mode=args.zeus_caching_mode,
        max_interval=args.zeus_max_interval,
        lagrange_term=args.zeus_lagrange_term,
        lagrange_int=args.zeus_lagrange_int,
        lagrange_step=args.zeus_lagrange_step,
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
    prompt_path = Path(args.prompt_path)
    if not prompt_path.exists():
        raise SystemExit(f"Missing prompt file: {prompt_path}")
    records = load_vbench_records(prompt_path)
    selected = select_records(records, args.prompt_start, args.prompt_limit)
    result = {
        "status": "ok",
        "prompt_path": str(prompt_path),
        "total_prompt_records": len(records),
        "prompt_start": args.prompt_start,
        "prompt_limit": args.prompt_limit,
        "selected_prompt_count": len(selected),
        "expected_baseline_runs": len(selected),
        "expected_candidate_runs": len(selected),
        "single_process_pipeline_load": True,
        "timestep_cache": "zeus",
        "block_cache": "none",
        "cfg_cache": "none",
        "sample_ids_head": [row["sample_id"] for row in selected[:5]],
        "sample_ids_tail": [row["sample_id"] for row in selected[-5:]],
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Single-process Vbench10 ZEUS Wan2.2 T2V runner")
    parser.add_argument("--root_dir", default=str(REPO_ROOT))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt_path", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--exp_root", default=None)
    parser.add_argument("--task", default="t2v-A14B")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_solver", default="unipc", choices=["dpm++", "unipc"])
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, nargs=2, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--prompt_start", type=int, default=0)
    parser.add_argument("--offload_model", type=parse_bool, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--no_convert_model_dtype", dest="convert_model_dtype", action="store_false")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--cpu_validate", action="store_true")
    parser.add_argument("--ffprobe_bin", default=DEFAULT_FFPROBE)
    parser.add_argument("--zeus_acc_start", type=int, default=8)
    parser.add_argument("--zeus_acc_end", type=int, default=47)
    parser.add_argument("--zeus_denominator", type=int, default=3)
    parser.add_argument("--zeus_modular", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--zeus_caching_mode", default="reuse_interp", choices=["reuse_interp", "interp_all", "reuse_all"])
    parser.add_argument("--zeus_max_interval", type=int, default=6)
    parser.add_argument("--zeus_lagrange_term", type=int, default=4)
    parser.add_argument("--zeus_lagrange_int", type=int, default=4)
    parser.add_argument("--zeus_lagrange_step", type=int, default=24)
    args = parser.parse_args()

    if args.cpu_validate:
        cpu_validate(args)
        return

    import torch
    from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; Vbench10 ZEUS runner requires GPU mode.")

    root_dir = Path(args.root_dir)
    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_zeus_unipc_vbench10_50step_45f_480p_{stamp}"
    exp_root = Path(args.exp_root)

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

    records = load_vbench_records(Path(args.prompt_path))
    selected = select_records(records, args.prompt_start, args.prompt_limit)

    for subdir in ["baseline", "zeus", "logs", "commands", "ffprobe", "psnr", "results", "failed", "manifests"]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)

    write_manifest_jsonl(exp_root / "manifests" / "selected_vbench10_records.jsonl", selected)
    write_manifest_csv(exp_root / "manifests" / "selected_vbench10_records.csv", selected)
    shutil.copy2(Path(args.prompt_path), exp_root / "manifests" / Path(args.prompt_path).name)

    env = {
        "experiment_root": exp_root,
        "root_dir": root_dir,
        "python_bin": args.python_bin,
        "ckpt_dir": args.ckpt_dir,
        "prompt_path": args.prompt_path,
        "ffprobe_bin": args.ffprobe_bin,
        "task": args.task,
        "size": args.size,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "sample_solver": args.sample_solver,
        "sample_shift": args.sample_shift,
        "sample_guide_scale": args.sample_guide_scale,
        "base_seed": args.base_seed,
        "prompt_start": args.prompt_start,
        "prompt_limit": args.prompt_limit,
        "selected_prompt_count": len(selected),
        "expected_candidate_runs": len(selected),
        "resume_existing": args.resume_existing,
        "timestep_cache": "zeus",
        "block_cache": "none",
        "cfg_cache": "none",
        "zeus_acc_start": args.zeus_acc_start,
        "zeus_acc_end": args.zeus_acc_end,
        "zeus_denominator": args.zeus_denominator,
        "zeus_modular": args.zeus_modular,
        "zeus_caching_mode": args.zeus_caching_mode,
        "zeus_max_interval": args.zeus_max_interval,
        "zeus_lagrange_term": args.zeus_lagrange_term,
        "zeus_lagrange_int": args.zeus_lagrange_int,
        "zeus_lagrange_step": args.zeus_lagrange_step,
    }
    (exp_root / "experiment_config.json").write_text(
        json.dumps({k: str(v) for k, v in env.items()}, indent=2),
        encoding="utf-8",
    )
    (exp_root / "launch.env").write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    if Path(nvidia_smi).exists():
        subprocess.run([nvidia_smi], stdout=(exp_root / "gpu.txt").open("w", encoding="utf-8"), stderr=subprocess.STDOUT)
    else:
        (exp_root / "gpu.txt").write_text("nvidia-smi not found\n", encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    print(json.dumps({k: str(v) for k, v in env.items()}, indent=2))
    print("Creating WanT2V pipeline once for Vbench10 ZEUS batch run")
    with run_log_context(exp_root / "logs" / "pipeline_init.log"):
        pipeline = create_pipeline(args, cfg)

    for row in selected:
        sample_id = row["sample_id"]
        prompt = row["prompt"]
        seed = args.base_seed
        baseline_video = exp_root / "baseline" / f"{sample_id}.mp4"
        baseline_log = exp_root / "logs" / f"baseline_{sample_id}.log"
        baseline_time = exp_root / "logs" / f"baseline_{sample_id}.time"
        baseline_ffprobe = exp_root / "ffprobe" / f"baseline_{sample_id}.json"
        save_command_record(
            exp_root / "commands" / f"baseline_{sample_id}.sh",
            root_dir,
            sys.argv,
            {
                "method": "baseline",
                "sample_id": sample_id,
                "seed": seed,
                "output": baseline_video,
                "prompt": prompt,
                "block_cache": "none",
                "cfg_cache": "none",
            },
        )
        if args.resume_existing and maybe_completed(baseline_video, baseline_time, baseline_ffprobe):
            print(f"Skipping existing baseline {sample_id}")
        else:
            print(f"Running baseline {sample_id} seed {seed}")
            try:
                generate_one(args, pipeline, cfg, prompt, seed, baseline_video, baseline_log, None)
                elapsed = parse_elapsed(baseline_log)
                baseline_time.write_text(f"elapsed_seconds={elapsed if elapsed is not None else ''}\n", encoding="utf-8")
                run_ffprobe(args.ffprobe_bin, baseline_video, baseline_ffprobe)
                torch.cuda.empty_cache()
            except Exception as exc:
                write_failed(exp_root, f"baseline_{sample_id}", {
                    "method": "baseline",
                    "sample_id": sample_id,
                    "seed": seed,
                    "status": "exception",
                    "error": repr(exc),
                    "log": baseline_log,
                })
                raise

        method_id = "zeus"
        output = exp_root / "zeus" / f"{sample_id}.mp4"
        log_path = exp_root / "logs" / f"{method_id}_{sample_id}.log"
        time_file = exp_root / "logs" / f"{method_id}_{sample_id}.time"
        ffprobe_json = exp_root / "ffprobe" / f"{method_id}_{sample_id}.json"
        psnr_json = exp_root / "psnr" / f"{sample_id}.json"
        psnr_log = exp_root / "psnr" / f"{sample_id}.log"
        save_command_record(
            exp_root / "commands" / f"{method_id}_{sample_id}.sh",
            root_dir,
            sys.argv,
            {
                "method": "zeus",
                "sample_id": sample_id,
                "seed": seed,
                "output": output,
                "prompt": prompt,
                "block_cache": "none",
                "cfg_cache": "none",
            },
        )
        if args.resume_existing and maybe_completed(output, time_file, ffprobe_json, psnr_json):
            print(f"Skipping existing ZEUS {sample_id}")
            continue
        print(f"Running ZEUS {sample_id} seed {seed}")
        try:
            if not (args.resume_existing and output.exists() and output.stat().st_size > 0 and time_file.exists() and time_file.stat().st_size > 0):
                generate_one(args, pipeline, cfg, prompt, seed, output, log_path, make_zeus_config(args))
                elapsed = parse_elapsed(log_path)
                time_file.write_text(f"elapsed_seconds={elapsed if elapsed is not None else ''}\n", encoding="utf-8")
                torch.cuda.empty_cache()
            if not (args.resume_existing and ffprobe_json.exists() and ffprobe_json.stat().st_size > 0):
                run_ffprobe(args.ffprobe_bin, output, ffprobe_json)
            if not (args.resume_existing and psnr_json.exists() and psnr_json.stat().st_size > 0):
                run_psnr(args.python_bin, baseline_video, output, psnr_json, psnr_log)
        except Exception as exc:
            write_failed(exp_root, f"{method_id}_{sample_id}", {
                "method": "zeus",
                "sample_id": sample_id,
                "seed": seed,
                "status": "exception",
                "error": repr(exc),
                "log": log_path,
            })
            raise

    summary_log = exp_root / "results" / "summary.log"
    cmd = [
        args.python_bin,
        str(root_dir / "experiments" / "zeus_vbench10_50step_45f_480p" / "summarize_results.py"),
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
