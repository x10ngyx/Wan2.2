#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import math
import re
from pathlib import Path


def read_elapsed(path: Path) -> float:
    text = path.read_text(encoding="utf-8").strip()
    match = re.search(r"elapsed_seconds=([0-9.]+)", text)
    if not match:
        raise ValueError(f"Missing elapsed_seconds in {path}")
    return float(match.group(1))


def read_threshold(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    match = re.search(r"threshold=([0-9.eE+-]+)", text)
    if not match:
        raise ValueError(f"Missing threshold in {path}")
    return match.group(1)


def read_cache_summary(path: Path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Timestep cache summary: (\{.*\})", text)
    if not matches:
        matches = re.findall(r"SeaCache timestep cache summary: (\{.*\})", text)
    if not matches:
        return {}
    return ast.literal_eval(matches[-1])


def sum_cache(summary, field: str) -> int:
    return sum(int(item.get(field, 0)) for item in summary.values())


def count_unique_steps(summary, field: str) -> int:
    steps = set()
    for item in summary.values():
        steps.update(int(step) for step in item.get(field, []))
    return len(steps)


def collect_paths(summary, field: str) -> str:
    parts = []
    for key, item in sorted(summary.items()):
        parts.append(f"{key}:{item.get(field, [])}")
    return " | ".join(parts)


def finite_values(values):
    result = []
    for value in values:
        value = float(value)
        if math.isfinite(value):
            result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.experiment_root)
    rows = []

    threshold_dirs = sorted((root / "seacache").glob("th_*"))
    for threshold_dir in threshold_dirs:
        threshold_label = threshold_dir.name
        threshold = read_threshold(root / "thresholds" / f"{threshold_label}.env")
        method_id = f"seacache_{threshold_label}"

        for candidate_video in sorted(threshold_dir.glob("prompt_*.mp4")):
            prompt_id = candidate_video.stem.replace("prompt_", "")
            baseline_video = root / "baseline" / f"prompt_{prompt_id}.mp4"
            if not baseline_video.exists():
                continue

            baseline_elapsed = read_elapsed(root / "logs" / f"baseline_prompt_{prompt_id}.time")
            candidate_elapsed = read_elapsed(root / "logs" / f"{method_id}_prompt_{prompt_id}.time")
            psnr = json.loads((root / "psnr" / threshold_label / f"prompt_{prompt_id}.json").read_text(encoding="utf-8"))
            baseline_ffprobe = json.loads((root / "ffprobe" / f"baseline_prompt_{prompt_id}.json").read_text(encoding="utf-8"))
            candidate_ffprobe = json.loads((root / "ffprobe" / f"{method_id}_prompt_{prompt_id}.json").read_text(encoding="utf-8"))
            cache_summary = read_cache_summary(root / "logs" / f"{method_id}_prompt_{prompt_id}.log")

            rows.append({
                "prompt_id": prompt_id,
                "threshold_label": threshold_label,
                "threshold": threshold,
                "baseline_elapsed_seconds": baseline_elapsed,
                "seacache_elapsed_seconds": candidate_elapsed,
                "speedup": baseline_elapsed / candidate_elapsed if candidate_elapsed else "",
                "mean_psnr": psnr["mean_psnr"],
                "min_psnr": psnr["min_psnr"],
                "max_psnr": psnr["max_psnr"],
                "psnr_frames": psnr["frames"],
                "seacache_reuse_count": count_unique_steps(cache_summary, "skipping_path"),
                "seacache_recompute_count": count_unique_steps(cache_summary, "recompute_path"),
                "seacache_reuse_branch_call_count": sum_cache(cache_summary, "reuse"),
                "seacache_recompute_branch_call_count": sum_cache(cache_summary, "recompute"),
                "seacache_skipping_path": collect_paths(cache_summary, "skipping_path"),
                "seacache_recompute_path": collect_paths(cache_summary, "recompute_path"),
                "seacache_rel_l1_path": collect_paths(cache_summary, "rel_l1_path"),
                "seacache_accumulated_rel_l1_path": collect_paths(cache_summary, "accumulated_rel_l1_path"),
                "baseline_video": str(baseline_video),
                "seacache_video": str(candidate_video),
                "baseline_log": str(root / "logs" / f"baseline_prompt_{prompt_id}.log"),
                "seacache_log": str(root / "logs" / f"{method_id}_prompt_{prompt_id}.log"),
                "baseline_ffprobe": json.dumps(baseline_ffprobe, sort_keys=True),
                "seacache_ffprobe": json.dumps(candidate_ffprobe, sort_keys=True),
            })

    if not rows:
        raise SystemExit("No completed SeaCache prompt pairs found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for threshold_label in sorted({row["threshold_label"] for row in rows}):
        subset = [row for row in rows if row["threshold_label"] == threshold_label]
        total_baseline = sum(row["baseline_elapsed_seconds"] for row in subset)
        total_candidate = sum(row["seacache_elapsed_seconds"] for row in subset)
        mean_psnr_values = finite_values(row["mean_psnr"] for row in subset)
        min_psnr_values = finite_values(row["min_psnr"] for row in subset)
        aggregate_rows.append({
            "threshold_label": threshold_label,
            "threshold": subset[0]["threshold"],
            "num_pairs": len(subset),
            "total_baseline_elapsed_seconds": total_baseline,
            "total_seacache_elapsed_seconds": total_candidate,
            "overall_speedup": total_baseline / total_candidate if total_candidate else None,
            "mean_psnr": sum(mean_psnr_values) / len(mean_psnr_values) if mean_psnr_values else float("inf"),
            "min_psnr": min(min_psnr_values) if min_psnr_values else float("inf"),
            "total_reuse_count": sum(int(row["seacache_reuse_count"]) for row in subset),
            "total_recompute_count": sum(int(row["seacache_recompute_count"]) for row in subset),
            "total_reuse_branch_call_count": sum(int(row["seacache_reuse_branch_call_count"]) for row in subset),
            "total_recompute_branch_call_count": sum(int(row["seacache_recompute_branch_call_count"]) for row in subset),
        })

    aggregate_csv = output.parent / "aggregate_by_threshold.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    aggregate_json = {
        "summary_csv": str(output),
        "aggregate_csv": str(aggregate_csv),
        "thresholds": aggregate_rows,
    }
    (output.parent / "aggregate_by_threshold.json").write_text(
        json.dumps(aggregate_json, indent=2),
        encoding="utf-8")
    print(json.dumps(aggregate_json, indent=2))


if __name__ == "__main__":
    main()
