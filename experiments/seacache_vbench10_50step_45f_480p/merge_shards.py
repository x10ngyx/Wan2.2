#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def finite_values(values):
    result = []
    for value in values:
        value = float(value)
        if math.isfinite(value):
            result.append(value)
    return result


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        raise SystemExit(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    parent = Path(args.parent_root)
    output_dir = Path(args.output_dir) if args.output_dir else parent / "merged"
    shard_summaries = sorted(parent.glob("shard_*/results/summary.csv"))
    if not shard_summaries:
        raise SystemExit(f"No shard summary.csv files found under {parent}")

    rows = []
    for summary in shard_summaries:
        shard_name = summary.parents[1].name
        shard_rows = read_csv(summary)
        for row in shard_rows:
            row["shard"] = shard_name
        rows.extend(shard_rows)

    rows.sort(key=lambda row: (row["threshold_label"], row["sample_id"]))
    write_csv(output_dir / "summary.csv", rows)

    aggregate_rows = []
    for threshold_label in sorted({row["threshold_label"] for row in rows}):
        subset = [row for row in rows if row["threshold_label"] == threshold_label]
        total_baseline = sum(float(row["baseline_elapsed_seconds"]) for row in subset)
        total_candidate = sum(float(row["seacache_elapsed_seconds"]) for row in subset)
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

    write_csv(output_dir / "aggregate_by_threshold.csv", aggregate_rows)
    payload = {
        "parent_root": str(parent),
        "shard_summaries": [str(path) for path in shard_summaries],
        "summary_csv": str(output_dir / "summary.csv"),
        "aggregate_csv": str(output_dir / "aggregate_by_threshold.csv"),
        "thresholds": aggregate_rows,
    }
    (output_dir / "aggregate_by_threshold.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
