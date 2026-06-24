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

DEFAULT_PROMPTS = REPO_ROOT / "test_sets" / "Vbench10" / "prompts.jsonl"
DEFAULT_THRESHOLDS = "0.05 0.10 0.20 0.50"
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


def value_label(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


def candidate_label(ts_threshold: str, block_threshold: str, cfg_threshold: str) -> str:
    return (
        f"sea_ts_{value_label(ts_threshold)}"
        f"__sea_bg_{value_label(block_threshold)}"
        f"__sea_cfg_{value_label(cfg_threshold)}"
    )


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
        raise SystemExit("No VBench10 records selected")
    return selected


def parse_elapsed(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def read_time_file(path: Path):
    if not path.exists():
        return None
    match = re.search(r"elapsed_seconds=([0-9.]+)", path.read_text(encoding="utf-8", errors="replace"))
    return float(match.group(1)) if match else None


def maybe_completed(video: Path, time_file: Path, ffprobe_json: Path, psnr_json: Path | None = None):
    required = [video, time_file, ffprobe_json]
    if psnr_json is not None:
        required.append(psnr_json)
    return all(path.exists() and path.stat().st_size > 0 for path in required)


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


def prepend_binary_dirs(args) -> dict:
    env = os.environ.copy()
    dirs = []
    for binary in [args.ffprobe_bin, args.python_bin]:
        if binary:
            parent = str(Path(binary).expanduser().resolve().parent)
            if parent not in dirs:
                dirs.append(parent)
    for fallback in ["/hy-tmp/env/Wan2.2/bin", "/hy-tmp/miniconda3/envs/Wan2.2/bin"]:
        if Path(fallback).exists() and fallback not in dirs:
            dirs.append(fallback)
    env["PATH"] = ":".join(dirs + [env.get("PATH", "")])
    return env


def run_ffprobe(args, video: Path, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.ffprobe_bin, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration",
        "-of", "json", str(video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=prepend_binary_dirs(args))
    output.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video}: {proc.stderr.strip()}")


def run_psnr(args, reference: Path, candidate: Path, output: Path, log_path: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_bin,
        str(REPO_ROOT / "experiments" / "zeus_timestep_cache_50step_45f_480p" / "compute_psnr.py"),
        "--reference", str(reference),
        "--candidate", str(candidate),
        "--output", str(output),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=prepend_binary_dirs(args))
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"PSNR failed for {candidate}; see {log_path}")


def save_command_record(path: Path, root_dir: Path, argv, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"cd {root_dir}", " ".join(subprocess.list2cmdline([item]) for item in argv)]
    lines.extend(f"# {k}={v}" for k, v in extra.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest_csv(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "source", "source_url", "source_index_1based", "source_set",
        "source_sample_id", "source_set_index_1based", "sampling_rule", "random_seed", "text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


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


def generate_one(args, pipeline, cfg, prompt: str, seed: int, output: Path, log_path: Path,
                 ts_config, bg_config, cfg_cache_config):
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
            seed=seed,
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
    if not log_path.exists():
        return {}
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
        candidates.append((ts_threshold, bg_threshold, cfg_threshold, candidate_label(ts_threshold, bg_threshold, cfg_threshold)))
    if args.combo_limit > 0:
        candidates = candidates[:args.combo_limit]
    return candidates


def prompt_root(exp_root: Path, sample_id: str) -> Path:
    return exp_root / "prompts" / sample_id


def summarize_prompt(sample_root: Path, record, candidates):
    baseline_time = read_time_file(sample_root / "logs" / "baseline.time")
    rows = []
    for ts_threshold, bg_threshold, cfg_threshold, label in candidates:
        time_file = sample_root / "logs" / f"{label}.time"
        psnr_file = sample_root / "psnr" / f"{label}.json"
        ffprobe_file = sample_root / "ffprobe" / f"{label}.json"
        log_path = sample_root / "logs" / f"{label}.log"
        video_path = sample_root / "videos" / f"{label}.mp4"
        if not maybe_completed(video_path, time_file, ffprobe_file, psnr_file):
            continue
        elapsed = read_time_file(time_file)
        psnr = json.loads(psnr_file.read_text(encoding="utf-8"))
        meta = json.loads(ffprobe_file.read_text(encoding="utf-8"))["streams"][0]
        timestep_summary = parse_cache_summary(log_path, "Timestep cache summary")
        block_summary = parse_cache_summary(log_path, "Block-group cache summary")
        cfg_summary = parse_cache_summary(log_path, "CFG cache summary")
        rows.append({
            "sample_id": record["sample_id"],
            "source": record.get("source", ""),
            "source_index_1based": record.get("source_index_1based", ""),
            "source_set": record.get("source_set", ""),
            "source_sample_id": record.get("source_sample_id", ""),
            "prompt": record.get("text", ""),
            "candidate": label,
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
            "decoded_frames_total": psnr.get("decoded_frames_total", ""),
            "excluded_perfect_frames": psnr.get("excluded_perfect_frames", ""),
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
            "baseline_video_path": str(sample_root / "baseline" / "baseline.mp4"),
            "video_path": str(video_path),
            "log_path": str(log_path),
            "psnr_path": str(psnr_file),
            "ffprobe_path": str(ffprobe_file),
            "prompt_result_dir": str(sample_root),
        })
    output = sample_root / "results" / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (sample_root / "results" / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def summarize_experiment(exp_root: Path, selected, candidates):
    all_rows = []
    for record in selected:
        all_rows.extend(summarize_prompt(prompt_root(exp_root, record["sample_id"]), record, candidates))
    output = exp_root / "results" / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    (exp_root / "results" / "summary.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    aggregate = []
    for label in sorted({row["candidate"] for row in all_rows}):
        subset = [row for row in all_rows if row["candidate"] == label]
        total_baseline = sum(float(row["baseline_elapsed_seconds"]) for row in subset if row["baseline_elapsed_seconds"] is not None)
        total_candidate = sum(float(row["candidate_elapsed_seconds"]) for row in subset if row["candidate_elapsed_seconds"] is not None)
        finite_psnr = [float(row["mean_psnr"]) for row in subset if row["mean_psnr"] not in {None, ""}]
        aggregate.append({
            "candidate": label,
            "timestep_threshold": subset[0]["timestep_threshold"],
            "block_threshold": subset[0]["block_threshold"],
            "cfg_threshold": subset[0]["cfg_threshold"],
            "num_prompts": len(subset),
            "total_baseline_elapsed_seconds": total_baseline,
            "total_candidate_elapsed_seconds": total_candidate,
            "overall_speedup": total_baseline / total_candidate if total_candidate else None,
            "mean_psnr": sum(finite_psnr) / len(finite_psnr) if finite_psnr else None,
            "min_psnr": min(finite_psnr) if finite_psnr else None,
            "total_timestep_reuse_count": sum(int(row["timestep_reuse_count"]) for row in subset),
            "total_block_reuse_count": sum(int(row["block_reuse_count"]) for row in subset),
            "total_cfg_reuse_count": sum(int(row["cfg_reuse_count"]) for row in subset),
        })
    if aggregate:
        with (exp_root / "results" / "aggregate_by_candidate.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate)
    (exp_root / "results" / "aggregate_by_candidate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return all_rows


def cpu_validate(args):
    prompt_path = Path(args.prompt_path)
    if not prompt_path.exists():
        raise SystemExit(f"Missing prompt file: {prompt_path}")
    records = load_vbench_records(prompt_path)
    selected = select_records(records, args.prompt_start, args.prompt_limit)
    candidates = build_candidates(args)
    invalid = []
    for value in args.timestep_thresholds.split() + args.block_thresholds.split() + args.cfg_thresholds.split():
        try:
            if float(value) < 0:
                invalid.append(value)
        except ValueError:
            invalid.append(value)
    ffprobe_exists = Path(args.ffprobe_bin).exists()
    print(json.dumps({
        "status": "ok" if not invalid and ffprobe_exists else "invalid",
        "prompt_path": str(prompt_path),
        "total_prompt_records": len(records),
        "prompt_start": args.prompt_start,
        "prompt_limit": args.prompt_limit,
        "selected_prompt_count": len(selected),
        "candidate_count_per_prompt": len(candidates),
        "expected_baseline_runs": len(selected),
        "expected_candidate_runs": len(selected) * len(candidates),
        "timestep_thresholds": args.timestep_thresholds.split(),
        "block_thresholds": args.block_thresholds.split(),
        "cfg_thresholds": args.cfg_thresholds.split(),
        "ffprobe_bin": args.ffprobe_bin,
        "ffprobe_exists": ffprobe_exists,
        "invalid_thresholds": invalid,
        "sample_ids": [row["sample_id"] for row in selected],
        "per_prompt_result_dirs": True,
        "cache_order": "CFG outer; branch timestep; block-group on timestep miss",
    }, indent=2))
    if invalid or not ffprobe_exists:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description="VBench10 three sea-style cache threshold-grid runner")
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
    parser.add_argument("--timestep_thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--block_thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--cfg_thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--combo_limit", type=int, default=0)
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--prompt_start", type=int, default=0)
    parser.add_argument("--offload_model", type=parse_bool, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--no_convert_model_dtype", dest="convert_model_dtype", action="store_false")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--cpu_validate", action="store_true")
    parser.add_argument("--ffprobe_bin", default=DEFAULT_FFPROBE)
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

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; VBench10 three-cache runner requires GPU mode.")

    root_dir = Path(args.root_dir)
    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_three_cache_sea_vbench10_50step_45f_480p_{stamp}"
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

    records = load_vbench_records(Path(args.prompt_path))
    selected = select_records(records, args.prompt_start, args.prompt_limit)

    for subdir in ["prompts", "logs", "results", "failed", "manifests"]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)
    write_manifest_jsonl(exp_root / "manifests" / "selected_vbench10_records.jsonl", selected)
    write_manifest_csv(exp_root / "manifests" / "selected_vbench10_records.csv", selected)
    shutil.copy2(Path(args.prompt_path), exp_root / "manifests" / Path(args.prompt_path).name)

    env = vars(args).copy()
    env.update({
        "experiment_root": str(exp_root),
        "selected_prompt_count": len(selected),
        "candidate_count_per_prompt": len(candidates),
        "expected_candidate_runs": len(selected) * len(candidates),
        "timestep_cache": "seacache",
        "block_cache": "block-group sea_full_rel_l1 accumulated",
        "cfg_cache": "sea-threshold",
        "cache_order": "CFG outer; branch timestep cache; block-group cache only on timestep miss",
        "per_prompt_result_dirs": True,
    })
    (exp_root / "experiment_config.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    (exp_root / "launch.env").write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
    subprocess.run(["nvidia-smi"], stdout=(exp_root / "gpu.txt").open("w", encoding="utf-8"), stderr=subprocess.STDOUT)

    print(json.dumps({k: str(v) for k, v in env.items()}, indent=2))
    print("Creating WanT2V pipeline once for VBench10 three sea-style cache grid")
    with run_log_context(exp_root / "logs" / "pipeline_init.log"):
        pipeline = create_pipeline(args, cfg)

    for row in selected:
        sample_id = row["sample_id"]
        sample_root = prompt_root(exp_root, sample_id)
        for subdir in ["baseline", "videos", "logs", "commands", "ffprobe", "psnr", "results", "failed"]:
            (sample_root / subdir).mkdir(parents=True, exist_ok=True)
        (sample_root / "prompt.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

        prompt = row["prompt"]
        seed = args.base_seed
        baseline_video = sample_root / "baseline" / "baseline.mp4"
        baseline_log = sample_root / "logs" / "baseline.log"
        baseline_time = sample_root / "logs" / "baseline.time"
        baseline_ffprobe = sample_root / "ffprobe" / "baseline.json"
        save_command_record(
            sample_root / "commands" / "baseline.sh",
            root_dir,
            sys.argv,
            {"method": "baseline", "sample_id": sample_id, "seed": seed, "output": baseline_video, "prompt": prompt},
        )
        if args.resume_existing and maybe_completed(baseline_video, baseline_time, baseline_ffprobe):
            print(f"Skipping existing baseline {sample_id}")
        else:
            print(f"Running baseline {sample_id} seed {seed}")
            try:
                generate_one(args, pipeline, cfg, prompt, seed, baseline_video, baseline_log, None, None, None)
                elapsed = parse_elapsed(baseline_log)
                if elapsed is None:
                    raise RuntimeError(f"Missing inference_compute_elapsed_seconds in {baseline_log}")
                baseline_time.write_text(f"elapsed_seconds={elapsed}\n", encoding="utf-8")
                run_ffprobe(args, baseline_video, baseline_ffprobe)
                torch.cuda.empty_cache()
            except Exception as exc:
                write_failed(sample_root, "baseline", {
                    "method": "baseline", "sample_id": sample_id, "seed": seed,
                    "status": "exception", "error": repr(exc), "log": baseline_log,
                })
                write_failed(exp_root, f"baseline_{sample_id}", {
                    "method": "baseline", "sample_id": sample_id, "status": "exception",
                    "error": repr(exc), "log": baseline_log,
                })
                raise

        for index, (ts_threshold, bg_threshold, cfg_threshold, label) in enumerate(candidates, start=1):
            output = sample_root / "videos" / f"{label}.mp4"
            log_path = sample_root / "logs" / f"{label}.log"
            time_file = sample_root / "logs" / f"{label}.time"
            ffprobe_json = sample_root / "ffprobe" / f"{label}.json"
            psnr_json = sample_root / "psnr" / f"{label}.json"
            psnr_log = sample_root / "psnr" / f"{label}.log"
            save_command_record(
                sample_root / "commands" / f"{label}.sh",
                root_dir,
                sys.argv,
                {
                    "method": "three-sea-style-caches",
                    "candidate_index": f"{index}/{len(candidates)}",
                    "sample_id": sample_id,
                    "timestep_threshold": ts_threshold,
                    "block_threshold": bg_threshold,
                    "cfg_threshold": cfg_threshold,
                    "seed": seed,
                    "output": output,
                    "prompt": prompt,
                },
            )
            if args.resume_existing and maybe_completed(output, time_file, ffprobe_json, psnr_json):
                print(f"Skipping existing {label} {sample_id} ({index}/{len(candidates)})")
                continue
            print(f"Running {label} {sample_id} ({index}/{len(candidates)}) seed {seed}")
            try:
                if not (args.resume_existing and output.exists() and output.stat().st_size > 0 and time_file.exists() and time_file.stat().st_size > 0):
                    generate_one(
                        args,
                        pipeline,
                        cfg,
                        prompt,
                        seed,
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
                    run_ffprobe(args, output, ffprobe_json)
                if not (args.resume_existing and psnr_json.exists() and psnr_json.stat().st_size > 0):
                    run_psnr(args, baseline_video, output, psnr_json, psnr_log)
                rows = summarize_prompt(sample_root, row, candidates)
                print(f"Archived {label} {sample_id}; prompt_completed_rows={len(rows)}")
            except Exception as exc:
                write_failed(sample_root, label, {
                    "method": "three-sea-style-caches",
                    "candidate_index": f"{index}/{len(candidates)}",
                    "sample_id": sample_id,
                    "timestep_threshold": ts_threshold,
                    "block_threshold": bg_threshold,
                    "cfg_threshold": cfg_threshold,
                    "seed": seed,
                    "status": "exception",
                    "error": repr(exc),
                    "log": log_path,
                })
                write_failed(exp_root, f"{label}_{sample_id}", {
                    "method": "three-sea-style-caches",
                    "sample_id": sample_id,
                    "status": "exception",
                    "error": repr(exc),
                    "log": log_path,
                })
                raise
        summarize_experiment(exp_root, selected, candidates)

    rows = summarize_experiment(exp_root, selected, candidates)
    print(f"Completed experiment: {exp_root}; completed_rows={len(rows)}")


if __name__ == "__main__":
    main()
