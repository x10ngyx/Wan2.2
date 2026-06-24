#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def finite_values(values):
    result = []
    for value in values:
        if value in {None, ""}:
            continue
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
    shard_summaries = sorted(parent.glob("*/results/summary.csv"))
    if not shard_summaries:
        raise SystemExit(f"No shard summary.csv files found under {parent}")

    rows = []
    for summary in shard_summaries:
        shard_name = summary.parents[1].name
        for row in read_csv(summary):
            row["shard"] = shard_name
            rows.append(row)

    rows.sort(key=lambda row: (row["candidate"], row["sample_id"]))
    write_csv(output_dir / "summary.csv", rows)

    aggregate_rows = []
    for candidate in sorted({row["candidate"] for row in rows}):
        subset = [row for row in rows if row["candidate"] == candidate]
        total_baseline = sum(float(row["baseline_elapsed_seconds"]) for row in subset if row["baseline_elapsed_seconds"])
        total_candidate = sum(float(row["candidate_elapsed_seconds"]) for row in subset if row["candidate_elapsed_seconds"])
        mean_psnr_values = finite_values(row["mean_psnr"] for row in subset)
        min_psnr_values = finite_values(row["min_psnr"] for row in subset)
        aggregate_rows.append({
            "candidate": candidate,
            "timestep_threshold": subset[0]["timestep_threshold"],
            "block_threshold": subset[0]["block_threshold"],
            "cfg_threshold": subset[0]["cfg_threshold"],
            "num_prompts": len(subset),
            "total_baseline_elapsed_seconds": total_baseline,
            "total_candidate_elapsed_seconds": total_candidate,
            "overall_speedup": total_baseline / total_candidate if total_candidate else None,
            "mean_psnr": sum(mean_psnr_values) / len(mean_psnr_values) if mean_psnr_values else None,
            "min_psnr": min(min_psnr_values) if min_psnr_values else None,
            "total_timestep_reuse_count": sum(int(row["timestep_reuse_count"]) for row in subset),
            "total_timestep_recompute_count": sum(int(row["timestep_recompute_count"]) for row in subset),
            "total_timestep_reuse_branch_call_count": sum(int(row["timestep_reuse_branch_call_count"]) for row in subset),
            "total_timestep_recompute_branch_call_count": sum(int(row["timestep_recompute_branch_call_count"]) for row in subset),
            "total_block_reuse_count": sum(int(row["block_reuse_count"]) for row in subset),
            "total_block_recompute_count": sum(int(row["block_recompute_count"]) for row in subset),
            "total_cfg_reuse_count": sum(int(row["cfg_reuse_count"]) for row in subset),
            "total_cfg_recompute_count": sum(int(row["cfg_recompute_count"]) for row in subset),
        })

    write_csv(output_dir / "aggregate_by_candidate.csv", aggregate_rows)
    payload = {
        "parent_root": str(parent),
        "shard_summaries": [str(path) for path in shard_summaries],
        "summary_csv": str(output_dir / "summary.csv"),
        "aggregate_csv": str(output_dir / "aggregate_by_candidate.csv"),
        "candidates": aggregate_rows,
    }
    (output_dir / "aggregate_by_candidate.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
