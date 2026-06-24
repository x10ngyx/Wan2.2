#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_FIXED_ROOT = Path(
    "/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513"
)
DEFAULT_ADAPTIVE_ROOT = Path(
    "/hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521"
)
DEFAULT_OUT_DIR = Path("/hy-tmp/work/Wan2.2/reports/assets/seacache_adaptive_train15_test5")

FIXED_COLOR = "#1f77b4"
ADAPTIVE_COLOR = "#d62728"
POINT_ALPHA = 0.32


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing {key} in row: {row}")
    return float(value)


def aggregate_fixed(summary_rows: list[dict[str, str]]) -> list[dict[str, float | str | int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["threshold"]].append(row)
    result = []
    for threshold in sorted(grouped, key=float):
        rows = grouped[threshold]
        total_base = sum(as_float(row, "baseline_compute_elapsed_seconds") for row in rows)
        total_candidate = sum(as_float(row, "compute_elapsed_seconds") for row in rows)
        psnr_values = [as_float(row, "mean_psnr") for row in rows]
        min_values = [as_float(row, "min_psnr") for row in rows]
        result.append(
            {
                "label": f"th={float(threshold):g}",
                "short_label": f"{float(threshold):g}",
                "x": total_base / total_candidate,
                "y": sum(psnr_values) / len(psnr_values),
                "min_psnr": min(min_values),
                "n": len(rows),
                "sort_key": float(threshold),
            }
        )
    return result


def aggregate_adaptive(summary_rows: list[dict[str, str]]) -> list[dict[str, float | str | int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["target_psnr"]].append(row)
    result = []
    for target in sorted(grouped, key=float):
        rows = grouped[target]
        total_base = sum(as_float(row, "baseline_compute_elapsed_seconds") for row in rows)
        total_candidate = sum(as_float(row, "compute_elapsed_seconds") for row in rows)
        psnr_values = [as_float(row, "mean_psnr") for row in rows]
        threshold_values = [as_float(row, "threshold_mean") for row in rows]
        result.append(
            {
                "label": f"target={float(target):g}",
                "short_label": f"T{float(target):g}",
                "x": total_base / total_candidate,
                "y": sum(psnr_values) / len(psnr_values),
                "threshold_mean": sum(threshold_values) / len(threshold_values),
                "n": len(rows),
                "sort_key": float(target),
            }
        )
    return result


def write_aggregate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_rows_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fixed_result_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result = []
    for row in rows:
        result.append(
            {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "selection_order": int(row["selection_order"]),
                "split": row["predictor_split"],
                "method": "fixed_seacache",
                "threshold": as_float(row, "threshold"),
                "baseline_s": as_float(row, "baseline_compute_elapsed_seconds"),
                "candidate_s": as_float(row, "compute_elapsed_seconds"),
                "speedup": as_float(row, "speedup"),
                "mean_psnr": as_float(row, "mean_psnr"),
                "min_psnr": as_float(row, "min_psnr"),
                "max_psnr": as_float(row, "max_psnr"),
                "reuse": int(row["seacache_reuse_count"]),
                "recompute": int(row["seacache_recompute_count"]),
                "reuse_branch_calls": int(row["seacache_reuse_branch_call_count"]),
                "recompute_branch_calls": int(row["seacache_recompute_branch_call_count"]),
                "video_path": row["video_path"],
                "log_path": row["log_path"],
            }
        )
    return result


def adaptive_result_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result = []
    for row in rows:
        result.append(
            {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "selection_order": int(row["selection_order"]),
                "split": row["predictor_split"],
                "method": "adaptive_seacache",
                "target_psnr": as_float(row, "target_psnr"),
                "threshold_min": as_float(row, "threshold_min"),
                "threshold_max": as_float(row, "threshold_max"),
                "threshold_mean": as_float(row, "threshold_mean"),
                "baseline_s": as_float(row, "baseline_compute_elapsed_seconds"),
                "candidate_s": as_float(row, "compute_elapsed_seconds"),
                "speedup": as_float(row, "speedup"),
                "mean_psnr": as_float(row, "mean_psnr"),
                "reuse": int(row["reuse_decisions"]),
                "recompute": int(row["recompute_decisions"]),
                "trace_rows": int(row["trace_rows"]),
                "video_path": row["video_path"],
                "log_path": row["log_path"],
                "trace_json": row["trace_json"],
            }
        )
    return result


def base_axes(title: str, source_note: str, figsize: tuple[float, float] = (10.2, 6.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Overall speedup vs no-cache baseline (20-prompt total compute time)", fontsize=11)
    ax.set_ylabel("Mean FFmpeg PSNR vs no-cache baseline (dB)", fontsize=11)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.text(0.012, 0.018, source_note, fontsize=8.2, color="#555555")
    return fig, ax


def annotate_points(ax, aggregates: list[dict[str, object]], xytext=(6, 5)) -> None:
    for row in aggregates:
        ax.annotate(
            str(row["short_label"]),
            (float(row["x"]), float(row["y"])),
            textcoords="offset points",
            xytext=xytext,
            fontsize=9,
            color="#222222",
        )


def set_limits(ax, xs: list[float], ys: list[float]) -> None:
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max((xmax - xmin) * 0.12, 0.12)
    ypad = max((ymax - ymin) * 0.12, 1.2)
    ax.set_xlim(max(0.8, xmin - xpad), xmax + xpad)
    ax.set_ylim(max(10.0, ymin - ypad), ymax + ypad)


def save_all(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_fixed(summary_rows, aggregates, out_dir: Path, fixed_summary: Path) -> None:
    source = (
        "Source: fixed SeaCache OpenVid20 summary.csv. "
        "Points: 80/80 candidate rows, aggregates: 4 fixed thresholds."
    )
    fig, ax = base_axes("OpenVid20 Fixed SeaCache: PSNR-Speedup Pareto Scatter", source)
    x = [as_float(row, "speedup") for row in summary_rows]
    y = [as_float(row, "mean_psnr") for row in summary_rows]
    ax.scatter(x, y, s=26, color=FIXED_COLOR, alpha=POINT_ALPHA, label="per-prompt candidates")
    ax.plot(
        [float(row["x"]) for row in aggregates],
        [float(row["y"]) for row in aggregates],
        color=FIXED_COLOR,
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="20-prompt aggregate by threshold",
    )
    annotate_points(ax, aggregates)
    set_limits(ax, x + [float(row["x"]) for row in aggregates], y + [float(row["y"]) for row in aggregates])
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    save_all(fig, out_dir / "openvid20_fixed_seacache_pareto_scatter")


def plot_adaptive(summary_rows, aggregates, out_dir: Path, adaptive_summary: Path) -> None:
    source = (
        "Source: adaptive SeaCache OpenVid20 summary.csv. "
        "Points: 60/60 candidate rows, aggregates: 3 target PSNRs."
    )
    fig, ax = base_axes("OpenVid20 Adaptive SeaCache: PSNR-Speedup Pareto Scatter", source)
    x = [as_float(row, "speedup") for row in summary_rows]
    y = [as_float(row, "mean_psnr") for row in summary_rows]
    ax.scatter(x, y, s=30, color=ADAPTIVE_COLOR, alpha=POINT_ALPHA, marker="s", label="per-prompt candidates")
    ax.plot(
        [float(row["x"]) for row in aggregates],
        [float(row["y"]) for row in aggregates],
        color=ADAPTIVE_COLOR,
        marker="s",
        linewidth=2.2,
        markersize=7,
        label="20-prompt aggregate by target",
    )
    annotate_points(ax, aggregates)
    set_limits(ax, x + [float(row["x"]) for row in aggregates], y + [float(row["y"]) for row in aggregates])
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    save_all(fig, out_dir / "openvid20_adaptive_seacache_pareto_scatter")


def plot_overlay(
    fixed_rows,
    fixed_agg,
    adaptive_rows,
    adaptive_agg,
    out_dir: Path,
    fixed_summary_path: Path,
    adaptive_summary_path: Path,
) -> None:
    source = (
        "Sources: fixed and adaptive SeaCache OpenVid20 summary.csv. "
        "Aggregate points use total baseline compute / total candidate compute speedup."
    )
    fig, ax = base_axes(
        "OpenVid20 Fixed SeaCache vs Adaptive SeaCache: PSNR-Speedup Pareto Comparison",
        source,
        figsize=(10.2, 6.7),
    )
    fixed_x = [as_float(row, "speedup") for row in fixed_rows]
    fixed_y = [as_float(row, "mean_psnr") for row in fixed_rows]
    adaptive_x = [as_float(row, "speedup") for row in adaptive_rows]
    adaptive_y = [as_float(row, "mean_psnr") for row in adaptive_rows]
    ax.scatter(fixed_x, fixed_y, s=22, color=FIXED_COLOR, alpha=0.20, label="fixed per-prompt")
    ax.scatter(adaptive_x, adaptive_y, s=28, color=ADAPTIVE_COLOR, alpha=0.24, marker="s", label="adaptive per-prompt")
    ax.plot(
        [float(row["x"]) for row in fixed_agg],
        [float(row["y"]) for row in fixed_agg],
        color=FIXED_COLOR,
        marker="o",
        linewidth=2.4,
        markersize=7,
        label="fixed threshold aggregate",
    )
    ax.plot(
        [float(row["x"]) for row in adaptive_agg],
        [float(row["y"]) for row in adaptive_agg],
        color=ADAPTIVE_COLOR,
        marker="s",
        linewidth=2.4,
        markersize=7,
        label="adaptive target aggregate",
    )
    annotate_points(ax, fixed_agg, xytext=(6, 5))
    annotate_points(ax, adaptive_agg, xytext=(6, -12))
    all_x = (
        fixed_x
        + adaptive_x
        + [float(row["x"]) for row in fixed_agg]
        + [float(row["x"]) for row in adaptive_agg]
    )
    all_y = (
        fixed_y
        + adaptive_y
        + [float(row["y"]) for row in fixed_agg]
        + [float(row["y"]) for row in adaptive_agg]
    )
    set_limits(ax, all_x, all_y)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    save_all(fig, out_dir / "openvid20_fixed_vs_adaptive_seacache_pareto_overlay")


def plot_train_overlay(fixed_rows, fixed_agg, adaptive_rows, adaptive_agg, out_dir: Path) -> None:
    source = (
        "Sources: train split only from fixed and adaptive SeaCache OpenVid20 summary.csv. "
        "Aggregate points use total baseline compute / total candidate compute speedup."
    )
    fig, ax = base_axes(
        "OpenVid20 Train Split: Fixed SeaCache vs Adaptive SeaCache Pareto Comparison",
        source,
        figsize=(10.2, 6.7),
    )
    ax.set_xlabel("Overall speedup vs no-cache baseline (15-train-prompt total compute time)", fontsize=11)
    fixed_x = [as_float(row, "speedup") for row in fixed_rows]
    fixed_y = [as_float(row, "mean_psnr") for row in fixed_rows]
    adaptive_x = [as_float(row, "speedup") for row in adaptive_rows]
    adaptive_y = [as_float(row, "mean_psnr") for row in adaptive_rows]
    ax.scatter(fixed_x, fixed_y, s=24, color=FIXED_COLOR, alpha=0.24, label="fixed train candidates")
    ax.scatter(
        adaptive_x,
        adaptive_y,
        s=30,
        color=ADAPTIVE_COLOR,
        alpha=0.28,
        marker="s",
        label="adaptive train candidates",
    )
    ax.plot(
        [float(row["x"]) for row in fixed_agg],
        [float(row["y"]) for row in fixed_agg],
        color=FIXED_COLOR,
        marker="o",
        linewidth=2.4,
        markersize=7,
        label="fixed train aggregate",
    )
    ax.plot(
        [float(row["x"]) for row in adaptive_agg],
        [float(row["y"]) for row in adaptive_agg],
        color=ADAPTIVE_COLOR,
        marker="s",
        linewidth=2.4,
        markersize=7,
        label="adaptive train aggregate",
    )
    annotate_points(ax, fixed_agg, xytext=(6, 5))
    annotate_points(ax, adaptive_agg, xytext=(6, -12))
    all_x = (
        fixed_x
        + adaptive_x
        + [float(row["x"]) for row in fixed_agg]
        + [float(row["x"]) for row in adaptive_agg]
    )
    all_y = (
        fixed_y
        + adaptive_y
        + [float(row["y"]) for row in fixed_agg]
        + [float(row["y"]) for row in adaptive_agg]
    )
    set_limits(ax, all_x, all_y)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    save_all(fig, out_dir / "openvid20_train_fixed_vs_adaptive_seacache_pareto_overlay")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OpenVid20 fixed/adaptive SeaCache Pareto charts.")
    parser.add_argument("--fixed-root", type=Path, default=DEFAULT_FIXED_ROOT)
    parser.add_argument("--adaptive-root", type=Path, default=DEFAULT_ADAPTIVE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    fixed_summary_path = args.fixed_root / "results" / "summary.csv"
    adaptive_summary_path = args.adaptive_root / "results" / "summary.csv"
    fixed_rows = read_csv(fixed_summary_path)
    adaptive_rows = read_csv(adaptive_summary_path)
    fixed_agg = aggregate_fixed(fixed_rows)
    adaptive_agg = aggregate_adaptive(adaptive_rows)
    fixed_train_rows = [row for row in fixed_rows if row["predictor_split"] == "train"]
    adaptive_train_rows = [row for row in adaptive_rows if row["predictor_split"] == "train"]
    fixed_train_agg = aggregate_fixed(fixed_train_rows)
    adaptive_train_agg = aggregate_adaptive(adaptive_train_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_aggregate_csv(args.out_dir / "openvid20_fixed_seacache_plot_aggregate.csv", fixed_agg)
    write_aggregate_csv(args.out_dir / "openvid20_adaptive_seacache_plot_aggregate.csv", adaptive_agg)
    write_aggregate_csv(args.out_dir / "openvid20_train_fixed_seacache_aggregate.csv", fixed_train_agg)
    write_aggregate_csv(args.out_dir / "openvid20_train_adaptive_seacache_aggregate.csv", adaptive_train_agg)
    write_aggregate_csv(
        args.out_dir / "openvid20_train_fixed_vs_adaptive_seacache_aggregate.csv",
        [
            {"method": "fixed_seacache", **row}
            for row in fixed_train_agg
        ]
        + [
            {"method": "adaptive_seacache", **row}
            for row in adaptive_train_agg
        ],
    )
    write_rows_csv(
        args.out_dir / "openvid20_train_fixed_seacache_results.csv",
        fixed_result_rows(fixed_train_rows),
        [
            "sample_id",
            "source_id",
            "selection_order",
            "split",
            "method",
            "threshold",
            "baseline_s",
            "candidate_s",
            "speedup",
            "mean_psnr",
            "min_psnr",
            "max_psnr",
            "reuse",
            "recompute",
            "reuse_branch_calls",
            "recompute_branch_calls",
            "video_path",
            "log_path",
        ],
    )
    write_rows_csv(
        args.out_dir / "openvid20_train_adaptive_seacache_results.csv",
        adaptive_result_rows(adaptive_train_rows),
        [
            "sample_id",
            "source_id",
            "selection_order",
            "split",
            "method",
            "target_psnr",
            "threshold_min",
            "threshold_max",
            "threshold_mean",
            "baseline_s",
            "candidate_s",
            "speedup",
            "mean_psnr",
            "reuse",
            "recompute",
            "trace_rows",
            "video_path",
            "log_path",
            "trace_json",
        ],
    )
    (args.out_dir / "plot_inputs.json").write_text(
        json.dumps(
            {
                "fixed_summary": str(fixed_summary_path),
                "adaptive_summary": str(adaptive_summary_path),
                "fixed_rows": len(fixed_rows),
                "adaptive_rows": len(adaptive_rows),
                "fixed_train_rows": len(fixed_train_rows),
                "adaptive_train_rows": len(adaptive_train_rows),
                "fixed_aggregates": fixed_agg,
                "adaptive_aggregates": adaptive_agg,
                "fixed_train_aggregates": fixed_train_agg,
                "adaptive_train_aggregates": adaptive_train_agg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_fixed(fixed_rows, fixed_agg, args.out_dir, fixed_summary_path)
    plot_adaptive(adaptive_rows, adaptive_agg, args.out_dir, adaptive_summary_path)
    plot_overlay(
        fixed_rows,
        fixed_agg,
        adaptive_rows,
        adaptive_agg,
        args.out_dir,
        fixed_summary_path,
        adaptive_summary_path,
    )
    plot_train_overlay(
        fixed_train_rows,
        fixed_train_agg,
        adaptive_train_rows,
        adaptive_train_agg,
        args.out_dir,
    )
    print(f"wrote charts to {args.out_dir}")


if __name__ == "__main__":
    main()
