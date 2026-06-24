#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import json
import logging
import random
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
DEFAULT_BASELINE_ROOT = (
    "/hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data"
)
DEFAULT_SPLIT_JSON = (
    "/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/"
    "temporal_mean/split.json"
)
DEFAULT_PROMPT_JSONL = "/hy-tmp/work/Wan2.2/test_sets/openvid_100/prompts.jsonl"
DEFAULT_THRESHOLDS = "0.1 0.2 0.4 0.6"


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


def load_openvid_records(prompt_jsonl: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with prompt_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["text"] = str(row["text"]).replace("\n", " ").strip()
            records[str(row["source_id"])] = row
    return records


def select_train_records(args) -> list[dict[str, object]]:
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    train_ids = list(split["train_sample_ids"])
    test_ids = list(split["val_sample_ids"])
    if args.selected_source_ids:
        selected_ids = args.selected_source_ids.split()
        split_by_id = {
            **{source_id: "train" for source_id in train_ids},
            **{source_id: "test" for source_id in test_ids},
        }
    else:
        if args.train_prompt_count < 0 or args.test_prompt_count < 0:
            raise ValueError("--train_prompt_count and --test_prompt_count must be non-negative.")
        rng = random.Random(args.random_seed)
        selected_train = rng.sample(train_ids, args.train_prompt_count)
        selected_test = rng.sample(test_ids, args.test_prompt_count)
        selected_ids = selected_train + selected_test
        split_by_id = {
            **{source_id: "train" for source_id in selected_train},
            **{source_id: "test" for source_id in selected_test},
        }
    known_ids = set(train_ids) | set(test_ids)
    outside_split = [source_id for source_id in selected_ids if source_id not in known_ids]
    if outside_split:
        raise ValueError(f"Selected source IDs are not in train/test split: {outside_split}")
    by_source = load_openvid_records(Path(args.prompt_jsonl))
    missing = [source_id for source_id in selected_ids if source_id not in by_source]
    if missing:
        raise ValueError(f"Selected source IDs are missing from prompt JSONL: {missing}")
    records = [dict(by_source[source_id]) for source_id in selected_ids]
    for order, row in enumerate(records, start=1):
        row["selection_order"] = order
        row["predictor_split"] = split_by_id[str(row["source_id"])]
    return records


def threshold_label(value: str) -> str:
    return f"th_{value.replace('.', 'p').replace('-', '_')}"


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


def load_psnr_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_cache_summary(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"SeaCache timestep cache summary: (\{.*\})", text)
    if not matches:
        matches = re.findall(r"Timestep cache summary: (\{.*\})", text)
    if not matches:
        return {}
    return ast.literal_eval(matches[-1])


def sum_cache(summary: dict[str, dict[str, object]], field: str) -> int:
    return sum(int(item.get(field, 0)) for item in summary.values())


def count_unique_steps(summary: dict[str, dict[str, object]], field: str) -> int:
    steps = set()
    for item in summary.values():
        steps.update(int(step) for step in item.get(field, []))
    return len(steps)


def collect_paths(summary: dict[str, dict[str, object]], field: str) -> str:
    parts = []
    for key, item in sorted(summary.items()):
        parts.append(f"{key}:{item.get(field, [])}")
    return " | ".join(parts)


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
) -> bool:
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


def resolve_openvid_baseline_artifacts(args, source_id: str):
    baseline_root = Path(args.baseline_root).resolve()
    matches = sorted(baseline_root.glob(f"shards/*/baseline/{source_id}.mp4"))
    if not matches:
        matches = sorted(baseline_root.glob(f"**/baseline/{source_id}.mp4"))
    if not matches:
        raise FileNotFoundError(f"Missing reusable OpenVid baseline video for {source_id} under {baseline_root}")
    baseline_video = matches[0]
    shard_root = baseline_video.parent.parent
    baseline_log = shard_root / "logs" / f"baseline_{source_id}.log"
    baseline_time = shard_root / "logs" / f"baseline_{source_id}.time"
    baseline_ffprobe = shard_root / "ffprobe" / f"baseline_{source_id}.json"
    missing = [
        path
        for path in [baseline_video, baseline_log, baseline_time, baseline_ffprobe]
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete reusable baseline artifacts for {source_id}: "
            + ", ".join(str(path) for path in missing)
        )
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


def generate_one(
    args,
    pipeline,
    cfg,
    prompt: str,
    seed: int,
    output: Path,
    log_path: Path,
    cache_config,
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


def cpu_validate(args) -> None:
    selected = select_train_records(args)
    thresholds = args.thresholds.split()
    threshold_errors = []
    for value in thresholds:
        try:
            parsed = float(value)
            if parsed < 0:
                threshold_errors.append(value)
        except ValueError:
            threshold_errors.append(value)
    result = {
        "status": "ok" if not threshold_errors else "invalid_thresholds",
        "split_json": args.split_json,
        "prompt_jsonl": args.prompt_jsonl,
        "random_seed": args.random_seed,
        "prompt_count": len(selected),
        "train_prompt_count": sum(1 for row in selected if row["predictor_split"] == "train"),
        "test_prompt_count": sum(1 for row in selected if row["predictor_split"] == "test"),
        "selected_source_ids": [row["source_id"] for row in selected],
        "selected_sample_ids": [row["sample_id"] for row in selected],
        "selected_predictor_splits": [row["predictor_split"] for row in selected],
        "thresholds": thresholds,
        "invalid_thresholds": threshold_errors,
        "expected_candidate_runs": len(selected) * len(thresholds),
        "generate_baseline": False,
        "single_process_pipeline_load": True,
        "timestep_cache": "seacache",
        "block_cache": "none",
        "cfg_cache": "none",
    }
    missing = []
    for row in selected:
        try:
            resolve_openvid_baseline_artifacts(args, str(row["source_id"]))
        except FileNotFoundError as exc:
            missing.append(str(exc))
    result["missing_baseline_artifacts"] = missing
    print(json.dumps(result, indent=2))
    if threshold_errors or missing:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-threshold SeaCache train15/test5 OpenVid runner")
    parser.add_argument("--root_dir", default="/hy-tmp/work/Wan2.2")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--ckpt_dir", default="/hy-tmp/models/Wan2.2-T2V-A14B")
    parser.add_argument("--prompt_jsonl", default=DEFAULT_PROMPT_JSONL)
    parser.add_argument("--split_json", default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--random_seed", type=int, default=20260619)
    parser.add_argument("--selected_source_ids", default="")
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
    parser.add_argument("--prompt_limit", type=int, default=20)
    parser.add_argument("--train_prompt_count", type=int, default=15)
    parser.add_argument("--test_prompt_count", type=int, default=5)
    parser.add_argument("--prompt_start", type=int, default=0)
    parser.add_argument("--offload_model", type=lambda value: value.lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--convert_model_dtype", action="store_true", default=True)
    parser.add_argument("--no_convert_model_dtype", dest="convert_model_dtype", action="store_false")
    parser.add_argument("--resume_existing", action="store_true")
    parser.add_argument("--generate_baseline", action="store_true", default=False)
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
    if args.generate_baseline:
        raise SystemExit(
            "This runner reuses existing OpenVid baselines; do not pass --generate_baseline."
        )

    import torch
    from wan.configs import SUPPORTED_SIZES, WAN_CONFIGS

    root_dir = Path(args.root_dir)
    tools_dir = root_dir / "experiments" / "zeus_timestep_cache_50step_45f_480p"
    if args.exp_root is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.exp_root = f"/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_{stamp}"
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

    records = select_train_records(args)
    if not records:
        raise SystemExit("No prompts selected")

    for subdir in [
        "baseline",
        "seacache",
        "thresholds",
        "logs",
        "commands",
        "ffprobe",
        "psnr",
        "results",
        "failed",
    ]:
        (exp_root / subdir).mkdir(parents=True, exist_ok=True)
    (exp_root / "selected_prompts.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    for threshold in thresholds:
        label = threshold_label(threshold)
        for subdir in ["seacache", "psnr"]:
            (exp_root / subdir / label).mkdir(parents=True, exist_ok=True)
        (exp_root / "thresholds" / f"{label}.env").write_text(
            f"threshold={threshold}\n",
            encoding="utf-8",
        )

    env = {
        "experiment_root": exp_root,
        "root_dir": root_dir,
        "python_bin": args.python_bin,
        "ckpt_dir": args.ckpt_dir,
        "prompt_jsonl": args.prompt_jsonl,
        "split_json": args.split_json,
        "random_seed": args.random_seed,
        "selected_source_ids": " ".join(str(row["source_id"]) for row in records),
        "selected_predictor_splits": " ".join(str(row["predictor_split"]) for row in records),
        "train_prompt_count": sum(1 for row in records if row["predictor_split"] == "train"),
        "test_prompt_count": sum(1 for row in records if row["predictor_split"] == "test"),
        "baseline_root": args.baseline_root,
        "generate_baseline": False,
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

    print("Creating WanT2V pipeline once for fixed-threshold SeaCache batch run")
    pipeline_log = exp_root / "logs" / "pipeline_init.log"
    with run_log_context(pipeline_log):
        pipeline = create_pipeline(args, cfg)

    rows = []
    for record in records:
        prompt = str(record["text"])
        source_id = str(record["source_id"])
        sample_id = str(record["sample_id"])
        seed = args.base_seed
        baseline_video, baseline_ffprobe, baseline_time, baseline_log = resolve_openvid_baseline_artifacts(
            args,
            source_id,
        )
        save_command_record(
            exp_root / "commands" / f"baseline_{source_id}.sh",
            root_dir,
            sys.argv,
            {
                "method": "baseline",
                "sample_id": sample_id,
                "source_id": source_id,
                "seed": seed,
                "output": baseline_video,
                "source_log": baseline_log,
                "source_time": baseline_time,
                "source_ffprobe": baseline_ffprobe,
            },
        )
        print(f"Reusing baseline {source_id}: {baseline_video}")

        baseline_elapsed = read_time_file(baseline_time)
        for threshold in thresholds:
            label = threshold_label(threshold)
            method_id = f"seacache_{label}"
            output = exp_root / "seacache" / label / f"{source_id}.mp4"
            log_path = exp_root / "logs" / f"{method_id}_{source_id}.log"
            time_file = exp_root / "logs" / f"{method_id}_{source_id}.time"
            ffprobe_json = exp_root / "ffprobe" / f"{method_id}_{source_id}.json"
            psnr_json = exp_root / "psnr" / label / f"{source_id}.json"
            psnr_log = exp_root / "psnr" / label / f"{source_id}.log"
            save_command_record(
                exp_root / "commands" / f"{method_id}_{source_id}.sh",
                root_dir,
                sys.argv,
                {
                    "method": "seacache",
                    "threshold": threshold,
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "seed": seed,
                    "output": output,
                    "block_cache": "none",
                    "cfg_cache": "none",
                },
            )
            completed = args.resume_existing and maybe_completed(
                output,
                time_file,
                ffprobe_json,
                psnr_json,
            )
            if completed:
                print(f"Skipping existing SeaCache threshold {threshold} {source_id}")
            else:
                print(f"Running SeaCache threshold {threshold} {source_id} seed {seed}")
            try:
                if not (
                    args.resume_existing
                    and output.exists()
                    and output.stat().st_size > 0
                    and time_file.exists()
                    and time_file.stat().st_size > 0
                ):
                    generate_one(
                        args,
                        pipeline,
                        cfg,
                        prompt,
                        seed,
                        output,
                        log_path,
                        make_seacache_config(args, threshold),
                    )
                    elapsed = parse_elapsed(log_path)
                    time_file.write_text(
                        f"elapsed_seconds={elapsed if elapsed is not None else ''}\n",
                        encoding="utf-8",
                    )
                    torch.cuda.empty_cache()
                if not (args.resume_existing and ffprobe_json.exists() and ffprobe_json.stat().st_size > 0):
                    run_ffprobe(args.ffprobe_bin, output, ffprobe_json)
                if not (args.resume_existing and psnr_json.exists() and psnr_json.stat().st_size > 0):
                    run_psnr(args.python_bin, tools_dir, baseline_video, output, psnr_json, psnr_log)

                elapsed = read_time_file(time_file)
                psnr_payload = load_psnr_payload(psnr_json)
                cache_summary = read_cache_summary(log_path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "selection_order": record["selection_order"],
                        "predictor_split": record["predictor_split"],
                        "source_video": record.get("source_video", ""),
                        "content_group": record.get("content_group", ""),
                        "portrait_group": record.get("portrait_group", ""),
                        "motion_group": record.get("motion_group", ""),
                        "prompt": prompt,
                        "seed": seed,
                        "method": "seacache",
                        "threshold_label": label,
                        "threshold": float(threshold),
                        "compute_elapsed_seconds": elapsed,
                        "baseline_compute_elapsed_seconds": baseline_elapsed,
                        "speedup": (
                            baseline_elapsed / elapsed
                            if baseline_elapsed is not None and elapsed and elapsed > 0
                            else None
                        ),
                        "mean_psnr": psnr_payload.get("mean_psnr"),
                        "min_psnr": psnr_payload.get("min_psnr"),
                        "max_psnr": psnr_payload.get("max_psnr"),
                        "psnr_frames": psnr_payload.get("frames"),
                        "decoded_frames_total": psnr_payload.get("decoded_frames_total", ""),
                        "excluded_perfect_frames": psnr_payload.get("excluded_perfect_frames", ""),
                        "seacache_reuse_count": count_unique_steps(cache_summary, "skipping_path"),
                        "seacache_recompute_count": count_unique_steps(cache_summary, "recompute_path"),
                        "seacache_reuse_branch_call_count": sum_cache(cache_summary, "reuse"),
                        "seacache_recompute_branch_call_count": sum_cache(cache_summary, "recompute"),
                        "seacache_skipping_path": collect_paths(cache_summary, "skipping_path"),
                        "seacache_recompute_path": collect_paths(cache_summary, "recompute_path"),
                        "seacache_rel_l1_path": collect_paths(cache_summary, "rel_l1_path"),
                        "seacache_accumulated_rel_l1_path": collect_paths(
                            cache_summary,
                            "accumulated_rel_l1_path",
                        ),
                        "video_path": str(output),
                        "log_path": str(log_path),
                    }
                )
            except Exception as exc:
                torch.cuda.empty_cache()
                write_failed(
                    exp_root,
                    f"{method_id}_{source_id}",
                    {
                        "method": "seacache",
                        "threshold": threshold,
                        "threshold_label": label,
                        "sample_id": sample_id,
                        "source_id": source_id,
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
        "source_id",
        "selection_order",
        "predictor_split",
        "source_video",
        "content_group",
        "portrait_group",
        "motion_group",
        "prompt",
        "seed",
        "method",
        "threshold_label",
        "threshold",
        "compute_elapsed_seconds",
        "baseline_compute_elapsed_seconds",
        "speedup",
        "mean_psnr",
        "min_psnr",
        "max_psnr",
        "psnr_frames",
        "decoded_frames_total",
        "excluded_perfect_frames",
        "seacache_reuse_count",
        "seacache_recompute_count",
        "seacache_reuse_branch_call_count",
        "seacache_recompute_branch_call_count",
        "seacache_skipping_path",
        "seacache_recompute_path",
        "seacache_rel_l1_path",
        "seacache_accumulated_rel_l1_path",
        "video_path",
        "log_path",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (exp_root / "results" / "summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    aggregate_rows = []
    for label in sorted({str(row["threshold_label"]) for row in rows}):
        subset = [row for row in rows if row["threshold_label"] == label]
        total_baseline = sum(
            float(row["baseline_compute_elapsed_seconds"] or 0.0) for row in subset
        )
        total_candidate = sum(float(row["compute_elapsed_seconds"] or 0.0) for row in subset)
        psnr_values = [
            float(row["mean_psnr"])
            for row in subset
            if row["mean_psnr"] is not None
        ]
        min_psnr_values = [
            float(row["min_psnr"])
            for row in subset
            if row["min_psnr"] is not None
        ]
        aggregate_rows.append(
            {
                "threshold_label": label,
                "threshold": subset[0]["threshold"],
                "num_pairs": len(subset),
                "total_baseline_compute_elapsed_seconds": total_baseline,
                "total_seacache_compute_elapsed_seconds": total_candidate,
                "overall_speedup": (
                    total_baseline / total_candidate if total_candidate else None
                ),
                "mean_psnr": (
                    sum(psnr_values) / len(psnr_values) if psnr_values else None
                ),
                "min_psnr": min(min_psnr_values) if min_psnr_values else None,
                "total_reuse_count": sum(int(row["seacache_reuse_count"]) for row in subset),
                "total_recompute_count": sum(
                    int(row["seacache_recompute_count"]) for row in subset
                ),
                "total_reuse_branch_call_count": sum(
                    int(row["seacache_reuse_branch_call_count"]) for row in subset
                ),
                "total_recompute_branch_call_count": sum(
                    int(row["seacache_recompute_branch_call_count"]) for row in subset
                ),
            }
        )
    aggregate_csv = exp_root / "results" / "aggregate_by_threshold.csv"
    if aggregate_rows:
        with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)
        (exp_root / "results" / "aggregate_by_threshold.json").write_text(
            json.dumps(aggregate_rows, indent=2),
            encoding="utf-8",
        )
    print(f"Completed experiment: {exp_root}")


if __name__ == "__main__":
    main()
