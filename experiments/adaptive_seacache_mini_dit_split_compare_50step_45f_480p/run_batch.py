#!/usr/bin/env python3
from __future__ import annotations

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

DEFAULT_FFPROBE = "/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe"
DEFAULT_PYTHON = "/hy-tmp/miniconda3/envs/Wan2.2/bin/python"
DEFAULT_VBENCH_PROMPTS = REPO_ROOT / "test_sets" / "Vbench10" / "prompts.jsonl"
DEFAULT_OPENVID_PROMPTS = REPO_ROOT / "test_sets" / "openvid_100" / "prompts.jsonl"
DEFAULT_VBENCH_BASELINE_SUMMARY = (
    "/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030/results/summary.csv"
)
DEFAULT_OPENVID_BASELINE_ROOTS = (
    "/hy-tmp/openvid_100_seacache_trace_data "
    "/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814"
)
DEFAULT_SAMPLE_SPLIT_MODEL = (
    "/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_"
    "20260629_214906/best_model_checkpoint.pt"
)
DEFAULT_ROW_SPLIT_MODEL = (
    "/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_"
    "20260629_232659/best_model_checkpoint.pt"
)
DEFAULT_SAMPLE_SPLIT_JSON = (
    "/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_"
    "20260629_214906/split.json"
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


def target_label(value: str) -> str:
    return f"target_{value.replace('.', 'p').replace('-', '_')}"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["prompt"] = str(row["text"]).replace("\n", " ").strip()
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def load_vbench_records(args) -> list[dict[str, object]]:
    rows = load_jsonl(Path(args.vbench_prompt_jsonl))
    selected = rows[args.vbench_prompt_start:]
    if args.prompt_count > 0:
        selected = selected[:args.prompt_count]
    if len(selected) != args.prompt_count:
        raise ValueError(
            f"Expected {args.prompt_count} VBench prompts, got {len(selected)}"
        )
    for row in selected:
        row["dataset"] = "vbench10"
        row["source_id"] = row["sample_id"]
    return selected


def find_openvid_baseline(source_id: str, baseline_roots: list[Path]) -> Path | None:
    for root in baseline_roots:
        matches = sorted(root.glob(f"**/baseline/{source_id}.mp4"))
        if matches:
            return matches[0]
    return None


def load_openvid_train_records(args) -> list[dict[str, object]]:
    split = json.loads(Path(args.sample_split_json).read_text(encoding="utf-8"))
    train_ids = set(split["train_sample_ids"])
    baseline_roots = [Path(item) for item in args.openvid_baseline_roots.split()]
    selected = []
    for row in load_jsonl(Path(args.openvid_prompt_jsonl)):
        source_id = str(row["source_id"])
        baseline = find_openvid_baseline(source_id, baseline_roots)
        if source_id in train_ids and baseline is not None:
            row["dataset"] = "openvid100_train"
            row["baseline_video"] = str(baseline)
            selected.append(row)
        if len(selected) >= args.prompt_count:
            break
    if len(selected) != args.prompt_count:
        raise ValueError(
            f"Expected {args.prompt_count} OpenVid train prompts with reusable baselines, "
            f"got {len(selected)}"
        )
    return selected


def load_vbench_baselines(summary_csv: Path) -> dict[str, dict[str, object]]:
    baselines: dict[str, dict[str, object]] = {}
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("sample_solver") != "dpm++":
                continue
            sample_id = str(row["sample_id"])
            if sample_id not in baselines:
                baselines[sample_id] = {
                    "baseline_video": row["baseline_video"],
                    "baseline_elapsed_seconds": row["baseline_elapsed_seconds"],
                    "baseline_log": row.get("baseline_log", ""),
                    "baseline_ffprobe": row.get("baseline_ffprobe", ""),
                }
    return baselines


def parse_elapsed_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"inference_compute_elapsed_seconds=([0-9.]+)", text)
    return float(matches[-1]) if matches else None


def parse_elapsed_from_time(time_path: Path) -> float | None:
    if not time_path.exists():
        return None
    match = re.search(
        r"elapsed_seconds=([0-9.]+)",
        time_path.read_text(encoding="utf-8", errors="replace"),
    )
    return float(match.group(1)) if match else None


def resolve_openvid_baseline_artifacts(baseline_video: Path) -> dict[str, object]:
    root = baseline_video.parent.parent
    stem = baseline_video.stem
    log = root / "logs" / f"baseline_{stem}.log"
    time_file = root / "logs" / f"baseline_{stem}.time"
    ffprobe = root / "ffprobe" / f"baseline_{stem}.json"
    missing = [
        path for path in [baseline_video, log, time_file, ffprobe]
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            "Incomplete OpenVid baseline artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    return {
        "baseline_video": str(baseline_video),
        "baseline_elapsed_seconds": parse_elapsed_from_time(time_file),
        "baseline_log": str(log),
        "baseline_ffprobe": str(ffprobe),
    }


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
    output.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video}: {proc.stderr.strip()}")


def run_psnr(
    python_bin: str,
    reference: Path,
    candidate: Path,
    output: Path,
    log_path: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(REPO_ROOT / "experiments" / "zeus_timestep_cache_50step_45f_480p" / "compute_psnr.py"),
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--output",
        str(output),
    ]
    env = os.environ.copy()
    env["PATH"] = f"/hy-tmp/miniconda3/envs/Wan2.2/bin:{env.get('PATH', '')}"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"PSNR failed for {candidate}; see {log_path}")


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


def build_factory(args, model_path: Path, target_psnr: float, target_speedup: float):
    from adaptive_seacache_wan22.cache import (
        AdaptiveSeaCacheGateConfig,
        build_adaptive_seacache_factory,
    )

    device = "cuda"
    config = AdaptiveSeaCacheGateConfig(
        model_path=model_path,
        target_psnr=target_psnr,
        target_speedup=target_speedup,
        model_type="auto",
        min_threshold=args.adaptive_min_threshold,
        max_threshold=args.adaptive_max_threshold,
        device=device,
        measure_predictor_timing=args.measure_predictor_timing,
    )
    return build_adaptive_seacache_factory(config)


def extract_trace(summary: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, payload in sorted(summary.items()):
        model_stage = ""
        branch = ""
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
        "predictor_elapsed_seconds",
        "cache_key",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trace_rows)


def trace_stats(trace_rows: list[dict[str, object]]) -> dict[str, object]:
    values = [float(row["predicted_threshold"]) for row in trace_rows]
    elapsed = [
        float(row["predictor_elapsed_seconds"])
        for row in trace_rows
        if row.get("predictor_elapsed_seconds") not in {None, ""}
    ]
    return {
        "trace_rows": len(trace_rows),
        "reuse_decisions": sum(1 for row in trace_rows if row.get("decision") == "reuse"),
        "recompute_decisions": sum(1 for row in trace_rows if row.get("decision") == "recompute"),
        "threshold_min": min(values) if values else None,
        "threshold_mean": sum(values) / len(values) if values else None,
        "threshold_max": max(values) if values else None,
        "predictor_call_count": len(elapsed),
        "predictor_elapsed_total_seconds": sum(elapsed) if elapsed else None,
        "predictor_elapsed_mean_seconds": (sum(elapsed) / len(elapsed)) if elapsed else None,
        "predictor_elapsed_max_seconds": max(elapsed) if elapsed else None,
    }


def release_factory(factory) -> None:
    if factory is None:
        return
    if hasattr(factory, "clear_last_instance"):
        factory.clear_last_instance()
    elif getattr(factory, "last_instance", None) is not None:
        instance = factory.last_instance
        if hasattr(instance, "clear_runtime_state"):
            instance.clear_runtime_state()
        factory.last_instance = None


def generate_one(args, pipeline, cfg, prompt: str, seed: int, output: Path, log_path: Path, factory):
    import torch
    from wan.configs import SIZE_CONFIGS
    from wan.utils.utils import save_video

    with run_log_context(log_path):
        logging.info(f"Input prompt: {prompt}")
        logging.info(f"Generating video to {output}")
        logging.info(f"timestep_cache_config={make_seacache_config(args)}")
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
            timestep_cache_config=make_seacache_config(args),
            block_cache_config=None,
            block_group_cache_config=None,
            cfg_cache_config=None,
        )
        wall = time.perf_counter() - start
        logging.info(f"generation_wall_elapsed_seconds={wall:.3f}")
        summary = factory.last_instance.summary() if factory.last_instance is not None else {}
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


def save_command_record(path: Path, argv: list[str], extra: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"cd {REPO_ROOT}",
        " ".join(subprocess.list2cmdline([item]) for item in argv),
    ]
    lines.extend(f"# {key}={value}" for key, value in extra.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_completed(video: Path, time_file: Path, ffprobe_json: Path, psnr_json: Path, trace_json: Path) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in [
        video,
        time_file,
        ffprobe_json,
        psnr_json,
        trace_json,
    ])


def prepare_records(args) -> list[dict[str, object]]:
    vbench_baselines = load_vbench_baselines(Path(args.vbench_baseline_summary))
    records = []
    for row in load_vbench_records(args):
        sample_id = str(row["sample_id"])
        if sample_id not in vbench_baselines:
            raise FileNotFoundError(f"Missing reusable VBench dpm++ baseline for {sample_id}")
        row.update(vbench_baselines[sample_id])
        records.append(row)
    for row in load_openvid_train_records(args):
        row.update(resolve_openvid_baseline_artifacts(Path(str(row["baseline_video"]))))
        records.append(row)
    for row in records:
        baseline_video = Path(str(row["baseline_video"]))
        if not baseline_video.exists() or baseline_video.stat().st_size == 0:
            raise FileNotFoundError(f"Missing baseline video: {baseline_video}")
    return records


def cpu_validate(args) -> None:
    records = prepare_records(args)
    model_specs = model_specs_from_args(args)
    targets = args.target_psnrs.split()
    target_speedups_by_psnr = target_speedups_from_args(args, targets)
    target_pairs = [
        (target, target_speedup)
        for target in targets
        for target_speedup in target_speedups_by_psnr[target]
    ]
    payload = {
        "status": "ok",
        "records": [
            {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "source_id": row.get("source_id"),
                "baseline_video": row["baseline_video"],
                "baseline_elapsed_seconds": row.get("baseline_elapsed_seconds"),
            }
            for row in records
        ],
        "models": model_specs,
        "target_psnrs": targets,
        "target_speedups_by_psnr": target_speedups_by_psnr,
        "expected_candidates": len(records) * len(model_specs) * len(target_pairs),
        "generate_baseline": False,
        "single_process_pipeline_load": True,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def model_specs_from_args(args) -> list[dict[str, object]]:
    available = [
        {
            "model_split": "sample_split",
            "checkpoint": args.sample_split_model,
        },
        {
            "model_split": "row_split",
            "checkpoint": args.row_split_model,
        },
    ]
    requested = set(args.model_splits.split())
    unknown = requested - {str(item["model_split"]) for item in available}
    if unknown:
        raise ValueError(f"Unknown model_splits: {sorted(unknown)}")
    return [item for item in available if str(item["model_split"]) in requested]


def target_speedups_from_args(args, targets: list[str]) -> dict[str, list[float]]:
    if args.target_speedups_by_psnr:
        result: dict[str, list[float]] = {}
        for spec in args.target_speedups_by_psnr.split():
            if ":" not in spec:
                raise ValueError(
                    f"Invalid --target_speedups_by_psnr entry {spec!r}; expected PSNR:speedup[,speedup]."
                )
            target, values = spec.split(":", 1)
            speeds = [float(item) for item in values.split(",") if item]
            if not speeds:
                raise ValueError(f"No speedup values provided for target PSNR {target!r}.")
            result[target] = speeds
        missing = [target for target in targets if target not in result]
        if missing:
            raise ValueError(f"Missing target speedups for PSNR targets: {missing}")
        return {target: result[target] for target in targets}
    return {target: [float(args.target_speedup)] for target in targets}


def write_summary(exp_root: Path, rows: list[dict[str, object]]) -> None:
    output = exp_root / "results" / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "sample_id",
        "source_id",
        "prompt",
        "model_split",
        "checkpoint",
        "target_psnr",
        "target_speedup",
        "compute_elapsed_seconds",
        "generation_wall_elapsed_seconds",
        "baseline_compute_elapsed_seconds",
        "speedup",
        "mean_psnr",
        "trace_rows",
        "reuse_decisions",
        "recompute_decisions",
        "threshold_min",
        "threshold_mean",
        "threshold_max",
        "predictor_call_count",
        "predictor_elapsed_total_seconds",
        "predictor_elapsed_mean_seconds",
        "predictor_elapsed_max_seconds",
        "video_path",
        "baseline_video_path",
        "log_path",
        "trace_json",
        "trace_csv",
        "ffprobe_path",
        "psnr_json",
        "psnr_log",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    aggregate_path = exp_root / "results" / "aggregate_by_dataset_model_target.csv"
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((
            str(row["dataset"]),
            str(row["model_split"]),
            str(row["target_psnr"]),
            str(row["target_speedup"]),
        ), []).append(row)
    aggregate_rows = []
    for (dataset, split, target, target_speedup), group in sorted(groups.items()):
        completed = [
            row for row in group
            if row.get("compute_elapsed_seconds") not in {None, ""}
            and row.get("mean_psnr") not in {None, ""}
        ]
        if not completed:
            continue
        total_baseline = sum(float(row["baseline_compute_elapsed_seconds"]) for row in completed)
        total_compute = sum(float(row["compute_elapsed_seconds"]) for row in completed)
        aggregate_rows.append({
            "dataset": dataset,
            "model_split": split,
            "target_psnr": target,
            "target_speedup": target_speedup,
            "num_completed": len(completed),
            "overall_speedup": total_baseline / total_compute,
            "mean_psnr": sum(float(row["mean_psnr"]) for row in completed) / len(completed),
            "mean_target_error": sum(float(row["mean_psnr"]) - float(target) for row in completed) / len(completed),
            "mean_reuse_decisions": sum(float(row["reuse_decisions"]) for row in completed) / len(completed),
            "mean_threshold": sum(float(row["threshold_mean"]) for row in completed) / len(completed),
        })
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "model_split",
            "target_psnr",
            "target_speedup",
            "num_completed",
            "overall_speedup",
            "mean_psnr",
            "mean_target_error",
            "mean_reuse_decisions",
            "mean_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sample-split vs row-split MiniDiT adaptive SeaCache online."
    )
    parser.add_argument("--root_dir", default=str(REPO_ROOT))
    parser.add_argument("--python_bin", default=DEFAULT_PYTHON)
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--exp_root", default=None)
    parser.add_argument("--task", default="t2v-A14B")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_solver", default="dpm++", choices=["dpm++"])
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, nargs=2, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--target_psnrs", default="22 28")
    parser.add_argument("--target_speedup", type=float, default=2.0)
    parser.add_argument(
        "--target_speedups_by_psnr",
        default="",
        help="Optional mapping such as '22:2.2,2.5,2.8 28:1.4,1.7,2.0'.",
    )
    parser.add_argument(
        "--model_splits",
        default="sample_split row_split",
        help="Whitespace-separated subset of sample_split and row_split.",
    )
    parser.add_argument("--prompt_count", type=int, default=3)
    parser.add_argument("--vbench_prompt_jsonl", default=str(DEFAULT_VBENCH_PROMPTS))
    parser.add_argument("--vbench_prompt_start", type=int, default=0)
    parser.add_argument("--openvid_prompt_jsonl", default=str(DEFAULT_OPENVID_PROMPTS))
    parser.add_argument("--sample_split_json", default=DEFAULT_SAMPLE_SPLIT_JSON)
    parser.add_argument("--vbench_baseline_summary", default=DEFAULT_VBENCH_BASELINE_SUMMARY)
    parser.add_argument("--openvid_baseline_roots", default=DEFAULT_OPENVID_BASELINE_ROOTS)
    parser.add_argument("--sample_split_model", default=DEFAULT_SAMPLE_SPLIT_MODEL)
    parser.add_argument("--row_split_model", default=DEFAULT_ROW_SPLIT_MODEL)
    parser.add_argument("--offload_model", type=lambda value: value.lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--no_convert_model_dtype", dest="convert_model_dtype", action="store_false")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--cpu_validate", action="store_true")
    parser.add_argument("--allow_custom_candidate_count", action="store_true")
    parser.add_argument("--ffprobe_bin", default=DEFAULT_FFPROBE)
    parser.add_argument("--seacache_threshold", type=float, default=0.2)
    parser.add_argument("--seacache_num_steps", type=int, default=None)
    parser.add_argument("--seacache_use_ret_steps", action="store_true", default=False)
    parser.add_argument("--seacache_power_exp", type=float, default=3.0)
    parser.add_argument("--seacache_power_const", type=float, default=1.0)
    parser.add_argument("--seacache_eps", type=float, default=1e-16)
    parser.add_argument("--seacache_norm_mode", default="mean", choices=["mean", "peak"])
    parser.add_argument("--adaptive_min_threshold", type=float, default=0.0)
    parser.add_argument("--adaptive_max_threshold", type=float, default=1.0)
    parser.add_argument("--measure_predictor_timing", action="store_true")
    args = parser.parse_args()

    if args.cpu_validate:
        cpu_validate(args)
        return

    import torch
    import wan.text2video as wan_text2video
    from adaptive_seacache_wan22.patch import patch_wan_model_forward_for_adaptive_seacache
    from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this runner requires GPU mode.")

    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_{stamp}"
    exp_root = Path(args.exp_root)

    cfg = WAN_CONFIGS[args.task]
    if args.task != "t2v-A14B":
        raise SystemExit("This runner currently supports t2v-A14B only.")
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

    records = prepare_records(args)
    model_specs = model_specs_from_args(args)
    targets = args.target_psnrs.split()
    target_speedups_by_psnr = target_speedups_from_args(args, targets)
    target_pairs = [
        (target, target_speedup)
        for target in targets
        for target_speedup in target_speedups_by_psnr[target]
    ]
    expected = len(records) * len(model_specs) * len(target_pairs)
    allowed_expected = {24, 36}
    if expected not in allowed_expected and not args.allow_custom_candidate_count:
        raise SystemExit(
            f"Expected one of {sorted(allowed_expected)} candidates for this pilot, got {expected}. "
            "Pass --allow_custom_candidate_count for custom sweeps."
        )

    for subdir in [
        "adaptive_seacache",
        "commands",
        "failed",
        "ffprobe",
        "logs",
        "manifests",
        "psnr",
        "results",
        "traces",
    ]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)
    (exp_root / "manifests" / "selected_records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    config_payload = {
        "exp_root": str(exp_root),
        "task": args.task,
        "size": args.size,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "sample_solver": args.sample_solver,
        "sample_shift": args.sample_shift,
        "sample_guide_scale": args.sample_guide_scale,
        "base_seed": args.base_seed,
        "target_psnrs": targets,
        "target_speedup": args.target_speedup,
        "target_speedups_by_psnr": target_speedups_by_psnr,
        "model_specs": model_specs,
        "records": [
            {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "source_id": row.get("source_id"),
                "baseline_video": row["baseline_video"],
                "baseline_elapsed_seconds": row.get("baseline_elapsed_seconds"),
            }
            for row in records
        ],
        "expected_candidates": expected,
        "generate_baseline": False,
        "single_process_pipeline_load": True,
    }
    (exp_root / "experiment_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (exp_root / "launch.env").write_text(
        "".join(f"{key}={value}\n" for key, value in config_payload.items()),
        encoding="utf-8",
    )
    nvidia_smi = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"
    if Path(nvidia_smi).exists():
        with (exp_root / "gpu.txt").open("w", encoding="utf-8") as handle:
            subprocess.run([nvidia_smi], stdout=handle, stderr=subprocess.STDOUT)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    print(json.dumps(config_payload, indent=2, ensure_ascii=False))
    print("Creating WanT2V pipeline once for adaptive MiniDiT split comparison")
    with run_log_context(exp_root / "logs" / "pipeline_init.log"):
        pipeline = create_pipeline(args, cfg)

    patch_wan_model_forward_for_adaptive_seacache()
    original_seacache_factory = wan_text2video.SeaCacheTimestepCache
    summary_rows: list[dict[str, object]] = []

    try:
        for row in records:
            dataset = str(row["dataset"])
            sample_id = str(row["sample_id"])
            source_id = str(row.get("source_id", sample_id))
            prompt = str(row["prompt"])
            baseline_video = Path(str(row["baseline_video"]))
            baseline_elapsed = float(row["baseline_elapsed_seconds"])
            for model in model_specs:
                model_split = str(model["model_split"])
                checkpoint = Path(str(model["checkpoint"]))
                for target, target_speedup in target_pairs:
                    target_value = float(target)
                    label = target_label(target)
                    speedup_label = f"speedup_{str(target_speedup).replace('.', 'p').replace('-', '_')}"
                    method_id = f"{dataset}_{sample_id}_{model_split}_{label}_{speedup_label}"
                    output = (
                        exp_root
                        / "adaptive_seacache"
                        / dataset
                        / model_split
                        / label
                        / speedup_label
                        / f"{sample_id}.mp4"
                    )
                    log_path = exp_root / "logs" / f"{method_id}.log"
                    time_file = exp_root / "logs" / f"{method_id}.time"
                    ffprobe_json = exp_root / "ffprobe" / f"{method_id}.json"
                    psnr_json = exp_root / "psnr" / dataset / model_split / label / speedup_label / f"{sample_id}.json"
                    psnr_log = exp_root / "psnr" / dataset / model_split / label / speedup_label / f"{sample_id}.log"
                    trace_json = exp_root / "traces" / dataset / model_split / label / speedup_label / f"{sample_id}.json"
                    trace_csv = exp_root / "traces" / dataset / model_split / label / speedup_label / f"{sample_id}.csv"
                    save_command_record(
                        exp_root / "commands" / f"{method_id}.sh",
                        sys.argv,
                        {
                            "dataset": dataset,
                            "sample_id": sample_id,
                            "source_id": source_id,
                            "model_split": model_split,
                            "checkpoint": checkpoint,
                            "target_psnr": target,
                            "target_speedup": target_speedup,
                            "baseline_video": baseline_video,
                            "output": output,
                            "prompt": prompt,
                        },
                    )

                    if args.resume_existing and maybe_completed(output, time_file, ffprobe_json, psnr_json, trace_json):
                        print(f"Skipping existing {method_id}")
                    else:
                        print(f"Running {method_id}")
                        factory = None
                        try:
                            factory = build_factory(args, checkpoint, target_value, target_speedup)
                            wan_text2video.SeaCacheTimestepCache = factory
                            summary = generate_one(
                                args,
                                pipeline,
                                cfg,
                                prompt,
                                args.base_seed,
                                output,
                                log_path,
                                factory,
                            )
                            elapsed = parse_elapsed_from_log(log_path)
                            time_file.write_text(
                                f"elapsed_seconds={elapsed if elapsed is not None else ''}\n",
                                encoding="utf-8",
                            )
                            trace_rows = extract_trace(summary)
                            write_trace(trace_rows, trace_json, trace_csv)
                            run_ffprobe(args.ffprobe_bin, output, ffprobe_json)
                            run_psnr(args.python_bin, baseline_video, output, psnr_json, psnr_log)
                        except Exception as exc:
                            write_failed(
                                exp_root,
                                method_id,
                                {
                                    "dataset": dataset,
                                    "sample_id": sample_id,
                                    "source_id": source_id,
                                    "model_split": model_split,
                                    "target_psnr": target,
                                    "target_speedup": target_speedup,
                                    "checkpoint": checkpoint,
                                    "status": "exception",
                                    "error": repr(exc),
                                    "log": log_path,
                                },
                            )
                            raise
                        finally:
                            release_factory(factory)
                            wan_text2video.SeaCacheTimestepCache = original_seacache_factory
                            del factory
                            torch.cuda.empty_cache()

                    elapsed = parse_elapsed_from_time(time_file)
                    trace_rows = json.loads(trace_json.read_text(encoding="utf-8")) if trace_json.exists() else []
                    stats = trace_stats(trace_rows)
                    mean_psnr = load_psnr(psnr_json)
                    row_out = {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "prompt": prompt,
                        "model_split": model_split,
                        "checkpoint": str(checkpoint),
                        "target_psnr": target,
                        "target_speedup": target_speedup,
                        "compute_elapsed_seconds": elapsed,
                        "generation_wall_elapsed_seconds": None,
                        "baseline_compute_elapsed_seconds": baseline_elapsed,
                        "speedup": (baseline_elapsed / elapsed) if elapsed else None,
                        "mean_psnr": mean_psnr,
                        "video_path": str(output),
                        "baseline_video_path": str(baseline_video),
                        "log_path": str(log_path),
                        "trace_json": str(trace_json),
                        "trace_csv": str(trace_csv),
                        "ffprobe_path": str(ffprobe_json),
                        "psnr_json": str(psnr_json),
                        "psnr_log": str(psnr_log),
                        **stats,
                    }
                    summary_rows = [
                        existing for existing in summary_rows
                        if not (
                            existing["dataset"] == dataset
                            and existing["sample_id"] == sample_id
                            and existing["model_split"] == model_split
                            and existing["target_psnr"] == target
                            and existing["target_speedup"] == target_speedup
                        )
                    ]
                    summary_rows.append(row_out)
                    write_summary(exp_root, summary_rows)
    finally:
        wan_text2video.SeaCacheTimestepCache = original_seacache_factory
        torch.cuda.empty_cache()

    write_summary(exp_root, summary_rows)
    print(f"Completed experiment: {exp_root}")


if __name__ == "__main__":
    main()
