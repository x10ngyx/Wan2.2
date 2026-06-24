#!/usr/bin/env python3
"""Run one VBench10 prompt/method in a fresh process.

This is intentionally small and reuses the batch runner helpers so the
generation path stays identical to the formal experiment.
"""

import argparse
import json
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

from experiments.adacache_vbench10_slow_fast_50step_45f_480p import run_batch
from experiments.adacache_vbench_50step_45f_480p import run_batch as base


def main():
    parser = argparse.ArgumentParser(description="Run one AdaCache VBench10 method")
    parser.add_argument("--root_dir", default=str(REPO_ROOT))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt_path", default=str(run_batch.DEFAULT_PROMPTS))
    parser.add_argument("--exp_root", required=True)
    parser.add_argument("--shard_id", default="single")
    parser.add_argument("--task", default="t2v-A14B")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_solver", default="dpm++", choices=["dpm++", "unipc"])
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, nargs=2, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--method", required=True, choices=["baseline", "slow", "fast"])
    parser.add_argument("--offload_model", type=base.parse_bool, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--no_convert_model_dtype", dest="convert_model_dtype", action="store_false")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--ffprobe_bin", default=run_batch.DEFAULT_FFPROBE)
    parser.add_argument("--adacache_res", default="t-attn", choices=["t-attn", "s-attn", "self-attn", "ca-mlp"])
    parser.add_argument("--adacache_loc", default="13")
    parser.add_argument("--adacache_moreg", action="store_true", default=False)
    parser.add_argument("--adacache_moreg_strides", default="1")
    parser.add_argument("--adacache_moreg_steps", default="10,90")
    parser.add_argument("--adacache_moreg_hyp", default="0.385,8,1,2")
    parser.add_argument("--adacache_mograd_mul", type=float, default=10.0)
    parser.add_argument("--adacache_cache_dtype", default="input", choices=["input", "bf16", "fp16", "fp32"])
    parser.add_argument("--slow_high_codebook_preset", choices=sorted(base.CODEBOOK_PRESETS), default=run_batch.DEFAULT_SLOW_HIGH)
    parser.add_argument("--slow_low_codebook_preset", choices=sorted(base.CODEBOOK_PRESETS), default=run_batch.DEFAULT_SLOW_LOW)
    parser.add_argument("--fast_high_codebook_preset", choices=sorted(base.CODEBOOK_PRESETS), default=run_batch.DEFAULT_FAST_HIGH)
    parser.add_argument("--fast_low_codebook_preset", choices=sorted(base.CODEBOOK_PRESETS), default=run_batch.DEFAULT_FAST_LOW)
    args = parser.parse_args()

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

    exp_root = Path(args.exp_root)
    for subdir in ["baseline", "slow", "fast", "commands", "logs", "ffprobe", "psnr", "results", "failed", "manifests"]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)

    records = base.load_vbench_records(Path(args.prompt_path))
    matches = [record for record in records if record["sample_id"] == args.sample_id]
    if not matches:
        raise SystemExit(f"sample_id not found: {args.sample_id}")
    record = matches[0]
    base.write_manifest_jsonl(exp_root / "manifests" / f"selected_records_{args.shard_id}.jsonl", [record])
    base.write_manifest_csv(exp_root / "manifests" / f"selected_records_{args.shard_id}.csv", [record])
    shutil.copy2(Path(args.prompt_path), exp_root / "manifests" / Path(args.prompt_path).name)
    config = vars(args).copy()
    (exp_root / f"experiment_config_{args.shard_id}.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    subprocess.run(
        ["nvidia-smi"],
        stdout=(exp_root / f"gpu_{args.shard_id}.txt").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    print(json.dumps(config, indent=2, sort_keys=True))
    with base.run_log_context(exp_root / "logs" / f"pipeline_init_{args.shard_id}.log"):
        pipeline = base.create_pipeline(args, cfg)

    runtime = None
    adacache_config = None
    if args.method in {"slow", "fast"}:
        high = getattr(args, f"{args.method}_high_codebook_preset")
        low = getattr(args, f"{args.method}_low_codebook_preset")
        runtime, adacache_config = run_batch.create_adacache_runtime(args, high, low)

    try:
        run_batch.generate_and_probe(
            args, pipeline, cfg, record, exp_root, args.method, runtime, adacache_config)
    except Exception as exc:
        base.write_failed(exp_root, f"{args.shard_id}_{args.method}_{args.sample_id}", {
            "sample_id": args.sample_id,
            "method": args.method,
            "status": "exception",
            "error": repr(exc),
        })
        raise
    finally:
        if runtime is not None:
            runtime.set_enabled(False)
        torch.cuda.empty_cache()

    print(f"Completed {args.method} {args.sample_id}: {exp_root}")


if __name__ == "__main__":
    main()
