#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import re
from pathlib import Path


METHODS = [
    ("bwcache", "BWCache block cache summary", "bwcache"),
    ("block_group", "Block-group cache summary", "block_group"),
]


def read_elapsed(path: Path) -> float:
    text = path.read_text(encoding="utf-8").strip()
    match = re.search(r"elapsed_seconds=([0-9.]+)", text)
    if not match:
        raise ValueError(f"Missing elapsed_seconds in {path}")
    return float(match.group(1))


def read_env(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def read_cache_summary(path: Path, marker: str):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(re.escape(marker) + r": (\{.*\})", text)
    if not match:
        return {}
    return ast.literal_eval(match.group(1))


def sum_bwcache(summary, field: str) -> int:
    return sum(int(item.get(field, 0)) for item in summary.values())


def collect_bwcache(summary, field: str) -> str:
    parts = []
    for key, item in sorted(summary.items()):
        parts.append(f"{key}:{item.get(field, [])}")
    return " | ".join(parts)


def sum_block_group(summary, field: str) -> int:
    total = 0
    for branch_summary in summary.values():
        for group_summary in branch_summary.values():
            total += int(group_summary.get(field, 0))
    return total


def collect_block_group(summary, field: str) -> str:
    parts = []
    for branch_key, branch_summary in sorted(summary.items()):
        for group_key, group_summary in sorted(branch_summary.items(), key=lambda kv: int(kv[0])):
            parts.append(f"{branch_key}/group_{group_key}:{group_summary.get(field, [])}")
    return " | ".join(parts)


def cache_counts(method: str, summary: dict) -> dict:
    if method == "bwcache":
        return {
            "cache_reuse_count": sum_bwcache(summary, "reuse"),
            "cache_recompute_count": sum_bwcache(summary, "recompute"),
            "cache_reuse_path": collect_bwcache(summary, "reuse_path"),
            "cache_recompute_path": collect_bwcache(summary, "recompute_path"),
            "cache_metric_path": collect_bwcache(summary, "acu_l1_path"),
        }
    return {
        "cache_reuse_count": sum_block_group(summary, "reuse"),
        "cache_recompute_count": sum_block_group(summary, "recompute"),
        "cache_reuse_path": collect_block_group(summary, "reuse_path"),
        "cache_recompute_path": collect_block_group(summary, "recompute_path"),
        "cache_metric_path": collect_block_group(summary, "metric_path"),
    }


def threshold_dirs(root: Path, method: str):
    parent = root / ("block_group" if method == "block_group" else "bwcache")
    return sorted(parent.glob("th_*"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.experiment_root)
    rows = []

    for method, marker, directory_name in METHODS:
        for threshold_dir in threshold_dirs(root, method):
            threshold_label = threshold_dir.name
            method_id = f"{directory_name}_{threshold_label}"
            env = read_env(root / "thresholds" / f"{method_id}.env")
            threshold = env["threshold"]

            for candidate_video in sorted(threshold_dir.glob("prompt_*.mp4")):
                prompt_id = candidate_video.stem.replace("prompt_", "")
                baseline_video = root / "baseline" / f"prompt_{prompt_id}.mp4"
                if not baseline_video.exists():
                    continue

                baseline_elapsed = read_elapsed(root / "logs" / f"baseline_prompt_{prompt_id}.time")
                method_elapsed = read_elapsed(root / "logs" / f"{method_id}_prompt_{prompt_id}.time")
                psnr = json.loads((root / "psnr" / method_id / f"prompt_{prompt_id}.json").read_text(encoding="utf-8"))
                baseline_ffprobe = json.loads((root / "ffprobe" / f"baseline_prompt_{prompt_id}.json").read_text(encoding="utf-8"))
                method_ffprobe = json.loads((root / "ffprobe" / f"{method_id}_prompt_{prompt_id}.json").read_text(encoding="utf-8"))
                cache_summary = read_cache_summary(root / "logs" / f"{method_id}_prompt_{prompt_id}.log", marker)
                counts = cache_counts(method, cache_summary)

                rows.append({
                    "prompt_id": prompt_id,
                    "method": method,
                    "method_id": method_id,
                    "threshold_label": threshold_label,
                    "threshold": threshold,
                    "baseline_elapsed_seconds": baseline_elapsed,
                    "method_elapsed_seconds": method_elapsed,
                    "speedup": baseline_elapsed / method_elapsed if method_elapsed else "",
                    "mean_psnr": psnr["mean_psnr"],
                    "min_psnr": psnr["min_psnr"],
                    "max_psnr": psnr["max_psnr"],
                    "psnr_frames": psnr["frames"],
                    **counts,
                    "baseline_video": str(baseline_video),
                    "method_video": str(candidate_video),
                    "baseline_log": str(root / "logs" / f"baseline_prompt_{prompt_id}.log"),
                    "method_log": str(root / "logs" / f"{method_id}_prompt_{prompt_id}.log"),
                    "baseline_ffprobe": json.dumps(baseline_ffprobe, sort_keys=True),
                    "method_ffprobe": json.dumps(method_ffprobe, sort_keys=True),
                })

    if not rows:
        raise SystemExit("No completed block-cache prompt pairs found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for method_id in sorted({row["method_id"] for row in rows}):
        subset = [row for row in rows if row["method_id"] == method_id]
        total_baseline = sum(row["baseline_elapsed_seconds"] for row in subset)
        total_method = sum(row["method_elapsed_seconds"] for row in subset)
        aggregate_rows.append({
            "method": subset[0]["method"],
            "method_id": method_id,
            "threshold_label": subset[0]["threshold_label"],
            "threshold": subset[0]["threshold"],
            "num_pairs": len(subset),
            "total_baseline_elapsed_seconds": total_baseline,
            "total_method_elapsed_seconds": total_method,
            "overall_speedup": total_baseline / total_method if total_method else None,
            "mean_psnr": sum(float(row["mean_psnr"]) for row in subset) / len(subset),
            "min_psnr": min(float(row["min_psnr"]) for row in subset),
            "total_cache_reuse_count": sum(int(row["cache_reuse_count"]) for row in subset),
            "total_cache_recompute_count": sum(int(row["cache_recompute_count"]) for row in subset),
        })

    aggregate_csv = output.parent / "aggregate_by_method_threshold.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    aggregate_json = {
        "summary_csv": str(output),
        "aggregate_csv": str(aggregate_csv),
        "methods": aggregate_rows,
    }
    (output.parent / "aggregate_by_method_threshold.json").write_text(
        json.dumps(aggregate_json, indent=2),
        encoding="utf-8")
    print(json.dumps(aggregate_json, indent=2))


if __name__ == "__main__":
    main()
