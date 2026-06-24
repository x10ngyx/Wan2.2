from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FEATURES = (
    "latent_pool",
    "temporal_mean",
    "temporal_var",
    "frame_diff_mean",
    "frame_diff_var",
)

GRIDS = ("2x2x2", "2x4x4", "3x4x4", "4x4x4")


@dataclass(frozen=True)
class RunSpec:
    group: str
    grid: str
    feature: str
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot adaptive threshold predictor training curves."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("reports/assets/adaptive_training_curves"),
    )
    return parser.parse_args()


def load_metrics(spec: RunSpec) -> dict:
    with spec.path.open("r") as handle:
        metrics = json.load(handle)
    epochs = metrics["epochs"]
    best = min(epochs, key=lambda item: item["val_loss"])
    return {
        "group": spec.group,
        "grid": spec.grid,
        "feature": spec.feature,
        "label": spec.label,
        "path": str(spec.path),
        "parameters": metrics.get("parameters"),
        "num_examples": metrics.get("num_examples"),
        "train_examples": metrics.get("train_examples"),
        "val_examples": metrics.get("val_examples"),
        "train_samples": metrics.get("train_samples"),
        "val_samples": metrics.get("val_samples"),
        "batch_size": metrics.get("batch_size"),
        "lr": metrics.get("lr"),
        "weight_decay": metrics.get("weight_decay"),
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_train_loss": best["train_loss"],
        "best_val_loss": best["val_loss"],
        "best_val_mae": best["val_mae"],
        "last_train_loss": epochs[-1]["train_loss"],
        "last_val_loss": epochs[-1]["val_loss"],
        "last_val_mae": epochs[-1]["val_mae"],
    }


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    base = Path("/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409")
    for feature in FEATURES:
        specs.append(
            RunSpec(
                group="grid_ablation_3epoch",
                grid="2x2x2",
                feature=feature,
                label=f"2x2x2 {feature}",
                path=base / feature / "metrics.json",
            )
        )

    grid_root = Path("/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314")
    for grid in ("2x4x4", "3x4x4", "4x4x4"):
        for feature in FEATURES:
            specs.append(
                RunSpec(
                    group="grid_ablation_3epoch",
                    grid=grid,
                    feature=feature,
                    label=f"{grid} {feature}",
                    path=grid_root
                    / f"feature_ablation_grid_{grid}"
                    / feature
                    / "metrics.json",
                )
            )

    long_root = Path("/hy-tmp/wan22_adaptive_threshold_feature_ablation_long_20260616")
    for feature in ("latent_pool", "temporal_mean"):
        specs.append(
            RunSpec(
                group="long_30epoch_hdim64",
                grid="2x2x2",
                feature=feature,
                label=f"30epoch hdim64 {feature}",
                path=long_root / feature / "metrics.json",
            )
        )

    for hdim in ("8", "16"):
        hdim_root = Path(f"/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim{hdim}_20260616")
        for feature in ("latent_pool", "temporal_mean"):
            specs.append(
                RunSpec(
                    group=f"long_30epoch_hdim{hdim}",
                    grid="2x2x2",
                    feature=feature,
                    label=f"30epoch hdim{hdim} {feature}",
                    path=hdim_root / feature / "metrics.json",
                )
            )

    control_root = Path("/hy-tmp/wan22_adaptive_threshold_controls_20260616")
    for feature in ("condition_only", "noise_feature"):
        specs.append(
            RunSpec(
                group="control_3epoch",
                grid="none",
                feature=feature,
                label=feature,
                path=control_root / feature / "metrics.json",
            )
        )
    return specs


def save_summary(rows: list[dict], out_dir: Path) -> None:
    summary_rows = []
    for row in rows:
        summary_rows.append(
            {
                key: row[key]
                for key in (
                    "group",
                    "grid",
                    "feature",
                    "label",
                    "parameters",
                    "num_examples",
                    "train_examples",
                    "val_examples",
                    "train_samples",
                    "val_samples",
                    "batch_size",
                    "lr",
                    "weight_decay",
                    "best_epoch",
                    "best_train_loss",
                    "best_val_loss",
                    "best_val_mae",
                    "last_train_loss",
                    "last_val_loss",
                    "last_val_mae",
                    "path",
                )
            }
        )
    with (out_dir / "training_curve_summary.json").open("w") as handle:
        json.dump(summary_rows, handle, indent=2)
    with (out_dir / "training_curve_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_feature_curves(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    rows = [
        row
        for row in rows
        if row["group"] == "grid_ablation_3epoch" and row["grid"] == "2x2x2"
    ]
    for row in rows:
        epochs = [item["epoch"] for item in row["epochs"]]
        axes[0].plot(
            epochs,
            [item["train_loss"] for item in row["epochs"]],
            marker="o",
            linewidth=2,
            label=row["feature"],
        )
        axes[1].plot(
            epochs,
            [item["val_loss"] for item in row["epochs"]],
            marker="o",
            linewidth=2,
            label=row["feature"],
        )
    axes[0].set_title("Train loss, grid 2x2x2")
    axes[1].set_title("Validation loss, grid 2x2x2")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("SmoothL1 loss")
        axis.grid(True, alpha=0.28)
        axis.set_xticks([1, 2, 3])
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Feature ablation convergence curves")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_curves_2x2x2.svg", bbox_inches="tight")
    fig.savefig(out_dir / "feature_curves_2x2x2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_grid_curves(rows: list[dict], out_dir: Path) -> None:
    grid_rows = [
        row for row in rows if row["group"] == "grid_ablation_3epoch"
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axes = axes.flatten()
    for axis, feature in zip(axes, FEATURES):
        for grid in GRIDS:
            matches = [
                row
                for row in grid_rows
                if row["feature"] == feature and row["grid"] == grid
            ]
            if not matches:
                continue
            row = matches[0]
            epochs = [item["epoch"] for item in row["epochs"]]
            axis.plot(
                epochs,
                [item["val_loss"] for item in row["epochs"]],
                marker="o",
                linewidth=2,
                label=grid,
            )
        axis.set_title(feature)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation loss")
        axis.set_xticks([1, 2, 3])
        axis.grid(True, alpha=0.28)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.94, 0.08))
    fig.suptitle("Pooling-grid comparison by feature")
    fig.tight_layout()
    fig.savefig(out_dir / "grid_val_curves_by_feature.svg", bbox_inches="tight")
    fig.savefig(out_dir / "grid_val_curves_by_feature.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rows: list[dict], out_dir: Path) -> None:
    grid_rows = [
        row for row in rows if row["group"] == "grid_ablation_3epoch"
    ]
    matrix = np.full((len(FEATURES), len(GRIDS)), np.nan)
    params = np.full((len(FEATURES), len(GRIDS)), np.nan)
    for i, feature in enumerate(FEATURES):
        for j, grid in enumerate(GRIDS):
            matches = [
                row
                for row in grid_rows
                if row["feature"] == feature and row["grid"] == grid
            ]
            if matches:
                matrix[i, j] = matches[0]["best_val_loss"]
                params[i, j] = matches[0]["parameters"]
    fig, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(matrix, cmap="viridis_r", aspect="auto")
    axis.set_xticks(range(len(GRIDS)), labels=GRIDS)
    axis.set_yticks(range(len(FEATURES)), labels=FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(GRIDS)):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.5f}\n{int(params[i, j] / 1000)}K",
                ha="center",
                va="center",
                color="white" if matrix[i, j] < 0.0138 else "black",
                fontsize=8,
            )
    axis.set_title("Best validation loss by feature and pooling grid")
    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("Best validation SmoothL1 loss")
    fig.tight_layout()
    fig.savefig(out_dir / "best_val_loss_heatmap.svg", bbox_inches="tight")
    fig.savefig(out_dir / "best_val_loss_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_long_runs(rows: list[dict], out_dir: Path) -> None:
    long_rows = [
        row
        for row in rows
        if row["group"]
        in ("long_30epoch_hdim64", "long_30epoch_hdim16", "long_30epoch_hdim8")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for row in long_rows:
        epochs = [item["epoch"] for item in row["epochs"]]
        label = row["label"].replace("30epoch ", "")
        axes[0].plot(
            epochs,
            [item["train_loss"] for item in row["epochs"]],
            linewidth=1.8,
            label=label,
        )
        axes[1].plot(
            epochs,
            [item["val_loss"] for item in row["epochs"]],
            linewidth=1.8,
            label=label,
        )
    axes[0].set_title("Train loss")
    axes[1].set_title("Validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("SmoothL1 loss")
        axis.grid(True, alpha=0.28)
    axes[1].legend(loc="best", fontsize=7)
    fig.suptitle("30-epoch runs: convergence and overfitting check")
    fig.tight_layout()
    fig.savefig(out_dir / "long_run_curves.svg", bbox_inches="tight")
    fig.savefig(out_dir / "long_run_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = [spec for spec in build_specs() if spec.path.exists()]
    if not specs:
        raise RuntimeError("No metrics files found")
    rows = [load_metrics(spec) for spec in specs]
    save_summary(rows, args.out_dir)
    plot_feature_curves(rows, args.out_dir)
    plot_grid_curves(rows, args.out_dir)
    plot_heatmap(rows, args.out_dir)
    plot_long_runs(rows, args.out_dir)
    print(f"saved plots and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
