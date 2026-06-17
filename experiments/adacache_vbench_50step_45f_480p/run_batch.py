#!/usr/bin/env python3
import argparse
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

ADACACHE_ROOT = REPO_ROOT / "third_party" / "AdaCache"
if str(ADACACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(ADACACHE_ROOT))

DEFAULT_PROMPTS = REPO_ROOT / "test_sets" / "vbench_every20" / "prompts.jsonl"
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
        raise SystemExit("No VBench records selected")
    return selected


def parse_elapsed(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    if not matches:
        return None
    return float(matches[-1])


def parse_adacache_summary(log_path: Path):
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Wan2\.2 AdaCache summary: (\{.*\})", text)
    if not matches:
        return {}
    import ast
    return ast.literal_eval(matches[-1])


def sum_summary(summary, field: str) -> int:
    return sum(int(item.get(field, 0)) for item in summary.values())


def collect_summary(summary, field: str) -> str:
    parts = []
    for key, item in sorted(summary.items()):
        parts.append(f"{key}:{item.get(field, [])}")
    return " | ".join(parts)


def write_failed(root: Path, name: str, fields):
    failed = root / "failed" / f"{name}.txt"
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text(
        "\n".join(f"{k}={v}" for k, v in fields.items()) + "\n",
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
        "sampling_rule",
        "text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


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


def generate_one(
    args,
    pipeline,
    cfg,
    prompt: str,
    seed: int,
    output: Path,
    log_path: Path,
    adacache_runtime=None,
):
    import torch
    from wan.configs import SIZE_CONFIGS
    from wan.utils.utils import save_video

    with run_log_context(log_path):
        if adacache_runtime is not None:
            adacache_runtime.set_enabled(True)
        try:
            logging.info(f"Input prompt: {prompt}")
            logging.info(f"Generating video to {output}")
            logging.info(f"adacache_enabled={adacache_runtime is not None}")
            if adacache_runtime is not None:
                logging.info(f"adacache_config={adacache_runtime.config}")
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
                timestep_cache_config=None,
                block_cache_config=None,
                block_group_cache_config=None,
                cfg_cache_config=None,
            )
            wall = time.perf_counter() - start
            logging.info(f"generation_wall_elapsed_seconds={wall:.3f}")
            if adacache_runtime is not None:
                logging.info("Wan2.2 AdaCache summary: %s", adacache_runtime.summary())
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
        finally:
            if adacache_runtime is not None:
                adacache_runtime.set_enabled(False)


def summarize(exp_root: Path, selected):
    rows = []
    for row in selected:
        sample_id = row["sample_id"]
        baseline_video = exp_root / "baseline" / f"{sample_id}.mp4"
        candidate_video = exp_root / "adacache" / f"{sample_id}.mp4"
        if not baseline_video.exists() or not candidate_video.exists():
            continue
        baseline_elapsed = read_elapsed(exp_root / "logs" / f"baseline_{sample_id}.time")
        candidate_elapsed = read_elapsed(exp_root / "logs" / f"adacache_{sample_id}.time")
        psnr = json.loads((exp_root / "psnr" / f"{sample_id}.json").read_text(encoding="utf-8"))
        baseline_ffprobe = json.loads((exp_root / "ffprobe" / f"baseline_{sample_id}.json").read_text(encoding="utf-8"))
        candidate_ffprobe = json.loads((exp_root / "ffprobe" / f"adacache_{sample_id}.json").read_text(encoding="utf-8"))
        cache_summary = parse_adacache_summary(exp_root / "logs" / f"adacache_{sample_id}.log")
        rows.append({
            "sample_id": sample_id,
            "source": row.get("source", ""),
            "source_index_1based": row.get("source_index_1based", ""),
            "prompt": row.get("text", ""),
            "task": "t2v-A14B",
            "size": "832*480",
            "frame_num": 45,
            "sample_steps": 50,
            "sample_solver": "dpm++",
            "seed": 42,
            "method": "adacache",
            "adacache_cache_res": "",
            "adacache_cache_loc": "",
            "adacache_codebook": "",
            "adacache_apply_moreg": "",
            "baseline_elapsed_seconds": baseline_elapsed,
            "adacache_elapsed_seconds": candidate_elapsed,
            "speedup": baseline_elapsed / candidate_elapsed if candidate_elapsed else "",
            "mean_psnr": psnr["mean_psnr"],
            "min_psnr": psnr["min_psnr"],
            "max_psnr": psnr["max_psnr"],
            "psnr_frames": psnr["frames"],
            "decoded_frames_total": psnr.get("decoded_frames_total", ""),
            "excluded_perfect_frames": psnr.get("excluded_perfect_frames", ""),
            "adacache_reuse_count": sum_summary(cache_summary, "reuse"),
            "adacache_recompute_count": sum_summary(cache_summary, "recompute"),
            "adacache_reuse_path": collect_summary(cache_summary, "reuse_path"),
            "adacache_recompute_path": collect_summary(cache_summary, "recompute_path"),
            "adacache_diff_path": collect_summary(cache_summary, "diff_path"),
            "adacache_rate_path": collect_summary(cache_summary, "rate_path"),
            "baseline_video": str(baseline_video),
            "adacache_video": str(candidate_video),
            "baseline_log": str(exp_root / "logs" / f"baseline_{sample_id}.log"),
            "adacache_log": str(exp_root / "logs" / f"adacache_{sample_id}.log"),
            "baseline_ffprobe": json.dumps(baseline_ffprobe, sort_keys=True),
            "adacache_ffprobe": json.dumps(candidate_ffprobe, sort_keys=True),
        })
    if not rows:
        raise SystemExit("No completed AdaCache VBench pairs found")
    return rows


def read_elapsed(path: Path) -> float:
    text = path.read_text(encoding="utf-8").strip()
    match = re.search(r"elapsed_seconds=([0-9.]+)", text)
    if not match:
        raise ValueError(f"Missing elapsed_seconds in {path}")
    return float(match.group(1))


def write_results(exp_root: Path, rows, args):
    for row in rows:
        row["adacache_cache_res"] = args.adacache_res
        row["adacache_cache_loc"] = args.adacache_loc
        row["adacache_codebook"] = args.adacache_codebook
        row["adacache_apply_moreg"] = args.adacache_moreg
    summary_csv = exp_root / "results" / "summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_baseline = sum(float(row["baseline_elapsed_seconds"]) for row in rows)
    total_candidate = sum(float(row["adacache_elapsed_seconds"]) for row in rows)
    aggregate = {
        "num_pairs": len(rows),
        "total_baseline_elapsed_seconds": total_baseline,
        "total_adacache_elapsed_seconds": total_candidate,
        "overall_speedup": total_baseline / total_candidate if total_candidate else None,
        "mean_psnr": sum(float(row["mean_psnr"]) for row in rows) / len(rows),
        "min_psnr": min(float(row["min_psnr"]) for row in rows),
        "total_reuse_count": sum(int(row["adacache_reuse_count"]) for row in rows),
        "total_recompute_count": sum(int(row["adacache_recompute_count"]) for row in rows),
        "summary_csv": str(summary_csv),
    }
    (exp_root / "results" / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


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
        "method": "adacache",
        "adacache_res": args.adacache_res,
        "adacache_loc": args.adacache_loc,
        "adacache_codebook": args.adacache_codebook,
        "adacache_moreg": args.adacache_moreg,
        "sample_ids_head": [row["sample_id"] for row in selected[:5]],
        "sample_ids_tail": [row["sample_id"] for row in selected[-5:]],
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Single-process VBench AdaCache Wan2.2 T2V runner")
    parser.add_argument("--root_dir", default=str(REPO_ROOT))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt_path", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--exp_root", default=None)
    parser.add_argument("--task", default="t2v-A14B")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_solver", default="dpm++", choices=["dpm++", "unipc"])
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
    parser.add_argument("--adacache_res", default="t-attn", choices=["t-attn", "s-attn", "self-attn", "ca-mlp"])
    parser.add_argument("--adacache_loc", default="13")
    parser.add_argument("--adacache_codebook", default="0.03:12,0.05:10,0.07:8,0.09:6,0.11:4,1.0:3")
    parser.add_argument("--adacache_moreg", action="store_true", default=False)
    parser.add_argument("--adacache_moreg_strides", default="1")
    parser.add_argument("--adacache_moreg_steps", default="10,90")
    parser.add_argument("--adacache_moreg_hyp", default="0.385,8,1,2")
    parser.add_argument("--adacache_mograd_mul", type=float, default=10.0)
    parser.add_argument("--adacache_cache_dtype", default="input", choices=["input", "bf16", "fp16", "fp32"])
    args = parser.parse_args()

    if args.cpu_validate:
        cpu_validate(args)
        return

    import torch
    from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS
    from wan22_adacache import AdaCacheConfig, enable_wan22_adacache

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; AdaCache VBench runner requires GPU mode.")

    root_dir = Path(args.root_dir)
    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "moreg" if args.adacache_moreg else "default"
        args.exp_root = f"/hy-tmp/wan22_adacache_vbench_50step_45f_480p_{suffix}_{stamp}"
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

    for subdir in [
        "baseline",
        "adacache",
        "commands",
        "logs",
        "ffprobe",
        "psnr",
        "results",
        "failed",
        "manifests",
    ]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)

    write_manifest_jsonl(exp_root / "manifests" / "selected_vbench_records.jsonl", selected)
    write_manifest_csv(exp_root / "manifests" / "selected_vbench_records.csv", selected)
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
        "resume_existing": args.resume_existing,
        "timestep_cache": "none",
        "block_cache": "adacache",
        "cfg_cache": "none",
        "adacache_res": args.adacache_res,
        "adacache_loc": args.adacache_loc,
        "adacache_codebook": args.adacache_codebook,
        "adacache_moreg": args.adacache_moreg,
        "adacache_moreg_strides": args.adacache_moreg_strides,
        "adacache_moreg_steps": args.adacache_moreg_steps,
        "adacache_moreg_hyp": args.adacache_moreg_hyp,
        "adacache_mograd_mul": args.adacache_mograd_mul,
        "adacache_cache_dtype": args.adacache_cache_dtype,
    }
    (exp_root / "experiment_config.json").write_text(
        json.dumps({k: str(v) for k, v in env.items()}, indent=2),
        encoding="utf-8",
    )
    (exp_root / "launch.env").write_text(
        "".join(f"{k}={v}\n" for k, v in env.items()),
        encoding="utf-8",
    )
    subprocess.run(["nvidia-smi"], stdout=(exp_root / "gpu.txt").open("w", encoding="utf-8"), stderr=subprocess.STDOUT)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    print(json.dumps({k: str(v) for k, v in env.items()}, indent=2))
    print("Creating WanT2V pipeline once for VBench AdaCache batch run")
    with run_log_context(exp_root / "logs" / "pipeline_init.log"):
        pipeline = create_pipeline(args, cfg)

    adacache_config = AdaCacheConfig.from_strings(
        cache_res=args.adacache_res,
        cache_loc=args.adacache_loc,
        codebook=args.adacache_codebook,
        apply_moreg=args.adacache_moreg,
        moreg_strides=args.adacache_moreg_strides,
        moreg_steps=args.adacache_moreg_steps,
        moreg_hyp=args.adacache_moreg_hyp,
        mograd_mul=args.adacache_mograd_mul,
        cache_dtype=args.adacache_cache_dtype,
    )
    adacache_runtime = enable_wan22_adacache(adacache_config)
    adacache_runtime.set_enabled(False)

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

        output = exp_root / "adacache" / f"{sample_id}.mp4"
        log_path = exp_root / "logs" / f"adacache_{sample_id}.log"
        time_file = exp_root / "logs" / f"adacache_{sample_id}.time"
        ffprobe_json = exp_root / "ffprobe" / f"adacache_{sample_id}.json"
        psnr_json = exp_root / "psnr" / f"{sample_id}.json"
        psnr_log = exp_root / "psnr" / f"{sample_id}.log"
        save_command_record(
            exp_root / "commands" / f"adacache_{sample_id}.sh",
            root_dir,
            sys.argv,
            {
                "method": "adacache",
                "sample_id": sample_id,
                "seed": seed,
                "output": output,
                "prompt": prompt,
                "adacache_config": adacache_config,
            },
        )
        if args.resume_existing and maybe_completed(output, time_file, ffprobe_json, psnr_json):
            print(f"Skipping existing AdaCache {sample_id}")
            continue
        print(f"Running AdaCache {sample_id} seed {seed}")
        try:
            if not (args.resume_existing and output.exists() and output.stat().st_size > 0 and time_file.exists() and time_file.stat().st_size > 0):
                generate_one(args, pipeline, cfg, prompt, seed, output, log_path, adacache_runtime)
                elapsed = parse_elapsed(log_path)
                time_file.write_text(f"elapsed_seconds={elapsed if elapsed is not None else ''}\n", encoding="utf-8")
                torch.cuda.empty_cache()
            if not (args.resume_existing and ffprobe_json.exists() and ffprobe_json.stat().st_size > 0):
                run_ffprobe(args.ffprobe_bin, output, ffprobe_json)
            if not (args.resume_existing and psnr_json.exists() and psnr_json.stat().st_size > 0):
                run_psnr(args.python_bin, baseline_video, output, psnr_json, psnr_log)
        except Exception as exc:
            write_failed(exp_root, f"adacache_{sample_id}", {
                "method": "adacache",
                "sample_id": sample_id,
                "seed": seed,
                "status": "exception",
                "error": repr(exc),
                "log": log_path,
            })
            raise

    rows = summarize(exp_root, selected)
    write_results(exp_root, rows, args)
    print(f"Completed experiment: {exp_root}")


if __name__ == "__main__":
    main()
