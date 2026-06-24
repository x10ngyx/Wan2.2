#!/usr/bin/env python3
import argparse
import csv
import json
import os
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

from experiments.adacache_vbench_50step_45f_480p import run_batch as base


DEFAULT_PROMPTS = REPO_ROOT / "test_sets" / "Vbench10" / "prompts.jsonl"
DEFAULT_FFPROBE = os.environ.get("FFPROBE_BIN", "/hy-tmp/env/Wan2.2-sm120/bin/ffprobe")
DEFAULT_SLOW_HIGH = "wan22_50_slow"
DEFAULT_SLOW_LOW = "wan22_50_slow"
DEFAULT_FAST_HIGH = "wan22_50_fast"
DEFAULT_FAST_LOW = "wan22_50_fast"


def resolve_preset(name: str) -> str:
    return base.CODEBOOK_PRESETS[name]


def create_adacache_runtime(args, high_preset: str, low_preset: str):
    from wan22_adacache import AdaCacheConfig, enable_wan22_adacache

    config = AdaCacheConfig.from_strings(
        cache_res=args.adacache_res,
        cache_loc=args.adacache_loc,
        codebook=resolve_preset(high_preset),
        high_codebook=resolve_preset(high_preset),
        low_codebook=resolve_preset(low_preset),
        apply_moreg=args.adacache_moreg,
        moreg_strides=args.adacache_moreg_strides,
        moreg_steps=args.adacache_moreg_steps,
        moreg_hyp=args.adacache_moreg_hyp,
        mograd_mul=args.adacache_mograd_mul,
        cache_dtype=args.adacache_cache_dtype,
    )
    runtime = enable_wan22_adacache(config)
    runtime.set_enabled(False)
    return runtime, config


def method_paths(exp_root: Path, method: str, sample_id: str):
    return {
        "video": exp_root / method / f"{sample_id}.mp4",
        "log": exp_root / "logs" / f"{method}_{sample_id}.log",
        "time": exp_root / "logs" / f"{method}_{sample_id}.time",
        "ffprobe": exp_root / "ffprobe" / f"{method}_{sample_id}.json",
        "psnr": exp_root / "psnr" / f"{method}_{sample_id}.json",
        "psnr_log": exp_root / "psnr" / f"{method}_{sample_id}.log",
        "command": exp_root / "commands" / f"{method}_{sample_id}.sh",
    }


def generate_and_probe(args, pipeline, cfg, record, exp_root: Path, method: str, runtime, config):
    import torch

    sample_id = record["sample_id"]
    prompt = record["prompt"]
    paths = method_paths(exp_root, method, sample_id)
    base.save_command_record(
        paths["command"],
        Path(args.root_dir),
        sys.argv,
        {
            "method": method,
            "sample_id": sample_id,
            "seed": args.base_seed,
            "output": paths["video"],
            "prompt": prompt,
            "adacache_config": config,
        },
    )
    if args.resume_existing and base.maybe_completed(paths["video"], paths["time"], paths["ffprobe"]):
        print(f"Skipping existing {method} {sample_id}")
        return
    print(f"Running {method} {sample_id}")
    base.generate_one(
        args,
        pipeline,
        cfg,
        prompt,
        args.base_seed,
        paths["video"],
        paths["log"],
        runtime,
    )
    elapsed = base.parse_elapsed(paths["log"])
    paths["time"].write_text(f"elapsed_seconds={elapsed if elapsed is not None else ''}\n", encoding="utf-8")
    base.run_ffprobe(args.ffprobe_bin, paths["video"], paths["ffprobe"])
    torch.cuda.empty_cache()


def run_method_subprocess(args, record, exp_root: Path, method: str):
    sample_id = record["sample_id"]
    paths = method_paths(exp_root, method, sample_id)
    if args.resume_existing and base.maybe_completed(paths["video"], paths["time"], paths["ffprobe"]):
        print(f"Skipping existing {method} {sample_id}")
        return

    cmd = [
        args.python_bin,
        str(Path(args.root_dir) / "experiments" / "adacache_vbench10_slow_fast_50step_45f_480p" / "run_one_method.py"),
        "--python_bin", args.python_bin,
        "--ckpt_dir", args.ckpt_dir,
        "--prompt_path", args.prompt_path,
        "--exp_root", str(exp_root),
        "--shard_id", f"shard_{args.shard_id}_{method}_{sample_id}",
        "--task", args.task,
        "--size", args.size,
        "--frame_num", str(args.frame_num),
        "--sample_steps", str(args.sample_steps),
        "--sample_solver", args.sample_solver,
        "--sample_shift", str(args.sample_shift),
        "--sample_guide_scale", str(args.sample_guide_scale[0]), str(args.sample_guide_scale[1]),
        "--base_seed", str(args.base_seed),
        "--sample_id", sample_id,
        "--method", method,
        "--offload_model", str(args.offload_model),
        "--ffprobe_bin", args.ffprobe_bin,
        "--adacache_res", args.adacache_res,
        "--adacache_loc", args.adacache_loc,
        "--adacache_moreg_strides", args.adacache_moreg_strides,
        "--adacache_moreg_steps", args.adacache_moreg_steps,
        "--adacache_moreg_hyp", args.adacache_moreg_hyp,
        "--adacache_mograd_mul", str(args.adacache_mograd_mul),
        "--adacache_cache_dtype", args.adacache_cache_dtype,
        "--slow_high_codebook_preset", args.slow_high_codebook_preset,
        "--slow_low_codebook_preset", args.slow_low_codebook_preset,
        "--fast_high_codebook_preset", args.fast_high_codebook_preset,
        "--fast_low_codebook_preset", args.fast_low_codebook_preset,
    ]
    if args.convert_model_dtype:
        cmd.append("--convert_model_dtype")
    else:
        cmd.append("--no_convert_model_dtype")
    if args.resume_existing:
        cmd.append("--resume_existing")
    if args.adacache_moreg:
        cmd.append("--adacache_moreg")

    log_path = exp_root / "logs" / f"subprocess_{method}_{sample_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("FFPROBE_BIN", args.ffprobe_bin)
    ffmpeg_bin = os.environ.get("FFMPEG_BIN")
    if ffmpeg_bin:
        env.setdefault("FFMPEG_BIN", ffmpeg_bin)

    base.save_command_record(
        paths["command"],
        Path(args.root_dir),
        cmd,
        {
            "method": method,
            "sample_id": sample_id,
            "seed": args.base_seed,
            "output": paths["video"],
            "prompt": record["prompt"],
            "subprocess": True,
        },
    )
    print(f"Running {method} {sample_id} in subprocess")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=args.root_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        base.write_failed(exp_root, f"shard_{args.shard_id}_{method}_{sample_id}", {
            "sample_id": sample_id,
            "method": method,
            "status": "subprocess_failed",
            "returncode": proc.returncode,
            "log": str(log_path),
        })
        raise RuntimeError(f"{method} {sample_id} subprocess failed; see {log_path}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(exp_root: Path, records, args):
    rows = []
    for record in records:
        sample_id = record["sample_id"]
        baseline = method_paths(exp_root, "baseline", sample_id)
        baseline_elapsed = base.read_elapsed(baseline["time"])
        baseline_ffprobe = load_json(baseline["ffprobe"])
        for method in ["slow", "fast"]:
            paths = method_paths(exp_root, method, sample_id)
            if not paths["video"].exists() or not paths["psnr"].exists():
                continue
            elapsed = base.read_elapsed(paths["time"])
            psnr = load_json(paths["psnr"])
            ffprobe = load_json(paths["ffprobe"])
            cache_summary = base.parse_adacache_summary(paths["log"])
            high_preset = getattr(args, f"{method}_high_codebook_preset")
            low_preset = getattr(args, f"{method}_low_codebook_preset")
            rows.append({
                "sample_id": sample_id,
                "source": record.get("source", ""),
                "source_index_1based": record.get("source_index_1based", ""),
                "prompt": record.get("text", ""),
                "task": args.task,
                "size": args.size,
                "frame_num": args.frame_num,
                "sample_steps": args.sample_steps,
                "sample_solver": args.sample_solver,
                "seed": args.base_seed,
                "method": f"adacache_{method}",
                "adacache_cache_res": args.adacache_res,
                "adacache_cache_loc": args.adacache_loc,
                "adacache_high_codebook_preset": high_preset,
                "adacache_low_codebook_preset": low_preset,
                "adacache_high_codebook": resolve_preset(high_preset),
                "adacache_low_codebook": resolve_preset(low_preset),
                "adacache_apply_moreg": args.adacache_moreg,
                "baseline_elapsed_seconds": baseline_elapsed,
                "adacache_elapsed_seconds": elapsed,
                "speedup": baseline_elapsed / elapsed if elapsed else "",
                "mean_psnr": psnr["mean_psnr"],
                "min_psnr": psnr["min_psnr"],
                "max_psnr": psnr["max_psnr"],
                "psnr_frames": psnr["frames"],
                "decoded_frames_total": psnr.get("decoded_frames_total", ""),
                "excluded_perfect_frames": psnr.get("excluded_perfect_frames", ""),
                "adacache_reuse_count": base.sum_summary(cache_summary, "reuse"),
                "adacache_recompute_count": base.sum_summary(cache_summary, "recompute"),
                "adacache_reuse_path": base.collect_summary(cache_summary, "reuse_path"),
                "adacache_recompute_path": base.collect_summary(cache_summary, "recompute_path"),
                "adacache_diff_path": base.collect_summary(cache_summary, "diff_path"),
                "adacache_rate_path": base.collect_summary(cache_summary, "rate_path"),
                "baseline_video": str(baseline["video"]),
                "adacache_video": str(paths["video"]),
                "baseline_log": str(baseline["log"]),
                "adacache_log": str(paths["log"]),
                "baseline_ffprobe": json.dumps(baseline_ffprobe, sort_keys=True),
                "adacache_ffprobe": json.dumps(ffprobe, sort_keys=True),
            })
    return rows


def write_summary(exp_root: Path, rows, shard_id: str):
    if not rows:
        raise SystemExit("No completed AdaCache slow/fast rows found")
    result_dir = exp_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = result_dir / f"summary_shard_{shard_id}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {}
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            "num_rows": len(subset),
            "mean_speedup": sum(float(row["speedup"]) for row in subset) / len(subset),
            "mean_psnr": sum(float(row["mean_psnr"]) for row in subset) / len(subset),
            "min_psnr": min(float(row["min_psnr"]) for row in subset),
        }
    aggregate_path = result_dir / f"aggregate_shard_{shard_id}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_csv), "aggregate": aggregate}, indent=2))


def cpu_validate(args):
    records = base.load_vbench_records(Path(args.prompt_path))
    selected = base.select_records(records, args.prompt_start, args.prompt_limit)
    print(json.dumps({
        "status": "ok",
        "prompt_path": args.prompt_path,
        "selected_prompt_count": len(selected),
        "expected_baseline_runs": len(selected),
        "expected_slow_runs": len(selected),
        "expected_fast_runs": len(selected),
        "slow_high_codebook_preset": args.slow_high_codebook_preset,
        "slow_low_codebook_preset": args.slow_low_codebook_preset,
        "fast_high_codebook_preset": args.fast_high_codebook_preset,
        "fast_low_codebook_preset": args.fast_low_codebook_preset,
        "sample_ids": [row["sample_id"] for row in selected],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description="VBench10 AdaCache baseline/slow/fast runner")
    parser.add_argument("--root_dir", default=str(REPO_ROOT))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt_path", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--exp_root", default=None)
    parser.add_argument("--shard_id", default="0")
    parser.add_argument("--task", default="t2v-A14B")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_solver", default="dpm++", choices=["dpm++", "unipc"])
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, nargs=2, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--prompt_start", type=int, default=0)
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--offload_model", type=base.parse_bool, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--no_convert_model_dtype", dest="convert_model_dtype", action="store_false")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--method_subprocess", action="store_true")
    parser.add_argument("--cpu_validate", action="store_true")
    parser.add_argument("--ffprobe_bin", default=DEFAULT_FFPROBE)
    parser.add_argument("--adacache_res", default="t-attn", choices=["t-attn", "s-attn", "self-attn", "ca-mlp"])
    parser.add_argument("--adacache_loc", default="13")
    parser.add_argument("--adacache_moreg", action="store_true", default=False)
    parser.add_argument("--adacache_moreg_strides", default="1")
    parser.add_argument("--adacache_moreg_steps", default="10,90")
    parser.add_argument("--adacache_moreg_hyp", default="0.385,8,1,2")
    parser.add_argument("--adacache_mograd_mul", type=float, default=10.0)
    parser.add_argument("--adacache_cache_dtype", default="input", choices=["input", "bf16", "fp16", "fp32"])
    parser.add_argument("--slow_high_codebook_preset", choices=sorted(base.CODEBOOK_PRESETS), default=DEFAULT_SLOW_HIGH)
    parser.add_argument("--slow_low_codebook_preset", choices=sorted(base.CODEBOOK_PRESETS), default=DEFAULT_SLOW_LOW)
    parser.add_argument("--fast_high_codebook_preset", choices=sorted(base.CODEBOOK_PRESETS), default=DEFAULT_FAST_HIGH)
    parser.add_argument("--fast_low_codebook_preset", choices=sorted(base.CODEBOOK_PRESETS), default=DEFAULT_FAST_LOW)
    args = parser.parse_args()

    if args.cpu_validate:
        cpu_validate(args)
        return

    import torch
    from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    cfg = WAN_CONFIGS[args.task]
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

    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_adacache_vbench10_slow_fast_50step_45f_480p_{stamp}"
    exp_root = Path(args.exp_root)
    for subdir in ["baseline", "slow", "fast", "commands", "logs", "ffprobe", "psnr", "results", "failed", "manifests"]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)

    records = base.load_vbench_records(Path(args.prompt_path))
    selected = base.select_records(records, args.prompt_start, args.prompt_limit)
    base.write_manifest_jsonl(exp_root / "manifests" / f"selected_records_shard_{args.shard_id}.jsonl", selected)
    base.write_manifest_csv(exp_root / "manifests" / f"selected_records_shard_{args.shard_id}.csv", selected)
    shutil.copy2(Path(args.prompt_path), exp_root / "manifests" / Path(args.prompt_path).name)
    config = vars(args).copy()
    config["selected_prompt_count"] = len(selected)
    (exp_root / f"experiment_config_shard_{args.shard_id}.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    subprocess.run(["nvidia-smi"], stdout=(exp_root / f"gpu_shard_{args.shard_id}.txt").open("w", encoding="utf-8"), stderr=subprocess.STDOUT)

    print(json.dumps(config, indent=2, sort_keys=True))

    if args.method_subprocess:
        for record in selected:
            try:
                run_method_subprocess(args, record, exp_root, "baseline")
                for method in ["slow", "fast"]:
                    run_method_subprocess(args, record, exp_root, method)
                    paths = method_paths(exp_root, method, record["sample_id"])
                    baseline = method_paths(exp_root, "baseline", record["sample_id"])
                    if not (args.resume_existing and paths["psnr"].exists() and paths["psnr"].stat().st_size > 0):
                        base.run_psnr(args.python_bin, baseline["video"], paths["video"], paths["psnr"], paths["psnr_log"])
            except Exception as exc:
                base.write_failed(exp_root, f"shard_{args.shard_id}_{record['sample_id']}", {
                    "sample_id": record["sample_id"],
                    "status": "exception",
                    "error": repr(exc),
                })
                raise
    else:
        with base.run_log_context(exp_root / "logs" / f"pipeline_init_shard_{args.shard_id}.log"):
            pipeline = base.create_pipeline(args, cfg)

        for record in selected:
            try:
                generate_and_probe(args, pipeline, cfg, record, exp_root, "baseline", None, None)
                for method in ["slow", "fast"]:
                    high = getattr(args, f"{method}_high_codebook_preset")
                    low = getattr(args, f"{method}_low_codebook_preset")
                    runtime, adacache_config = create_adacache_runtime(args, high, low)
                    generate_and_probe(args, pipeline, cfg, record, exp_root, method, runtime, adacache_config)
                    paths = method_paths(exp_root, method, record["sample_id"])
                    baseline = method_paths(exp_root, "baseline", record["sample_id"])
                    if not (args.resume_existing and paths["psnr"].exists() and paths["psnr"].stat().st_size > 0):
                        base.run_psnr(args.python_bin, baseline["video"], paths["video"], paths["psnr"], paths["psnr_log"])
                    runtime.set_enabled(False)
                    torch.cuda.empty_cache()
            except Exception as exc:
                base.write_failed(exp_root, f"shard_{args.shard_id}_{record['sample_id']}", {
                    "sample_id": record["sample_id"],
                    "status": "exception",
                    "error": repr(exc),
                })
                raise

    rows = build_rows(exp_root, selected, args)
    write_summary(exp_root, rows, args.shard_id)
    print(f"Completed shard {args.shard_id}: {exp_root}")


if __name__ == "__main__":
    main()
