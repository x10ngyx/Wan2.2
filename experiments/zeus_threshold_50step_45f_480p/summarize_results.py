#!/usr/bin/env python3
import argparse
import ast
import csv
import json
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
    match = re.search(r"ZEUS timestep cache summary: (\{.*\})", text)
    if not match:
        return {}
    return ast.literal_eval(match.group(1))


def sum_cache(summary, field: str) -> int:
    total = 0
    for item in summary.values():
        total += int(item.get(field, 0))
    return total


def collect_paths(summary, field: str) -> str:
    parts = []
    for key, item in sorted(summary.items()):
        parts.append(f"{key}:{item.get(field, [])}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.experiment_root)
    rows = []

    threshold_dirs = sorted((root / "zeus_threshold").glob("th_*"))
    for threshold_dir in threshold_dirs:
        threshold_label = threshold_dir.name
        threshold = read_threshold(root / "thresholds" / f"{threshold_label}.env")
        method_id = f"zeus_threshold_{threshold_label}"

        for candidate_video in sorted(threshold_dir.glob("prompt_*.mp4")):
            prompt_id = candidate_video.stem.replace("prompt_", "")
            baseline_video = root / "baseline" / f"prompt_{prompt_id}.mp4"
            if not baseline_video.exists():
                continue

            baseline_elapsed = read_elapsed(root / "logs" / f"baseline_prompt_{prompt_id}.time")
            threshold_elapsed = read_elapsed(root / "logs" / f"{method_id}_prompt_{prompt_id}.time")
            psnr = json.loads((root / "psnr" / threshold_label / f"prompt_{prompt_id}.json").read_text(encoding="utf-8"))
            baseline_ffprobe = json.loads((root / "ffprobe" / f"baseline_prompt_{prompt_id}.json").read_text(encoding="utf-8"))
            threshold_ffprobe = json.loads((root / "ffprobe" / f"{method_id}_prompt_{prompt_id}.json").read_text(encoding="utf-8"))
            cache_summary = read_cache_summary(root / "logs" / f"{method_id}_prompt_{prompt_id}.log")

            rows.append({
                "prompt_id": prompt_id,
                "threshold_label": threshold_label,
                "threshold": threshold,
                "baseline_elapsed_seconds": baseline_elapsed,
                "zeus_threshold_elapsed_seconds": threshold_elapsed,
                "speedup": baseline_elapsed / threshold_elapsed if threshold_elapsed else "",
                "mean_psnr": psnr["mean_psnr"],
                "min_psnr": psnr["min_psnr"],
                "max_psnr": psnr["max_psnr"],
                "psnr_frames": psnr["frames"],
                "zeus_threshold_reuse_count": sum_cache(cache_summary, "reuse"),
                "zeus_threshold_recompute_count": sum_cache(cache_summary, "recompute"),
                "zeus_threshold_skipping_path": collect_paths(cache_summary, "skipping_path"),
                "zeus_threshold_recompute_path": collect_paths(cache_summary, "recompute_path"),
                "zeus_threshold_rel_l1_path": collect_paths(cache_summary, "rel_l1_path"),
                "baseline_video": str(baseline_video),
                "zeus_threshold_video": str(candidate_video),
                "baseline_log": str(root / "logs" / f"baseline_prompt_{prompt_id}.log"),
                "zeus_threshold_log": str(root / "logs" / f"{method_id}_prompt_{prompt_id}.log"),
                "baseline_ffprobe": json.dumps(baseline_ffprobe, sort_keys=True),
                "zeus_threshold_ffprobe": json.dumps(threshold_ffprobe, sort_keys=True),
            })

    if not rows:
        raise SystemExit("No completed threshold prompt pairs found")

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
        total_threshold = sum(row["zeus_threshold_elapsed_seconds"] for row in subset)
        aggregate_rows.append({
            "threshold_label": threshold_label,
            "threshold": subset[0]["threshold"],
            "num_pairs": len(subset),
            "total_baseline_elapsed_seconds": total_baseline,
            "total_zeus_threshold_elapsed_seconds": total_threshold,
            "overall_speedup": total_baseline / total_threshold if total_threshold else None,
            "mean_psnr": sum(float(row["mean_psnr"]) for row in subset) / len(subset),
            "min_psnr": min(float(row["min_psnr"]) for row in subset),
            "total_reuse_count": sum(int(row["zeus_threshold_reuse_count"]) for row in subset),
            "total_recompute_count": sum(int(row["zeus_threshold_recompute_count"]) for row in subset),
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
