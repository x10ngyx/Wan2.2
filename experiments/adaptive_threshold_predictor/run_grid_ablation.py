from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_grid(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("grid must have three dims, e.g. 2x4x4")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def grid_label(grid: tuple[int, int, int]) -> str:
    return "x".join(str(dim) for dim in grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build feature caches and run feature ablations for grid sizes."
    )
    parser.add_argument("--python", default="/hy-tmp/miniconda3/envs/Wan2.2/bin/python")
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument(
        "--grids",
        nargs="+",
        type=parse_grid,
        default=[(2, 4, 4), (3, 4, 4), (4, 4, 4)],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--cache_batch_size", type=int, default=8)
    parser.add_argument("--cache_num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for grid in args.grids:
        label = grid_label(grid)
        cache_dir = args.out_root / f"feature_cache_grid_{label}"
        run_dir = args.out_root / f"feature_ablation_grid_{label}"

        build_cmd = [
            args.python,
            "-m",
            "experiments.adaptive_threshold_predictor.build_feature_cache",
            "--out_dir",
            str(cache_dir),
            "--dataset_mode",
            "candidate_inverse",
            "--dtype",
            args.dtype,
            "--batch_size",
            str(args.cache_batch_size),
            "--num_workers",
            str(args.cache_num_workers),
            "--device",
            args.device,
            "--grid_size",
            str(grid[0]),
            str(grid[1]),
            str(grid[2]),
        ]
        print("building cache:", " ".join(build_cmd), flush=True)
        subprocess.run(build_cmd, check=True)

        train_cmd = [
            args.python,
            "-m",
            "experiments.adaptive_threshold_predictor.run_feature_ablation",
            "--cache_dir",
            str(cache_dir),
            "--dataset_mode",
            "candidate_inverse",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.train_batch_size),
            "--num_workers",
            "0",
            "--device",
            args.device,
            "--save_val_predictions",
            "--out_root",
            str(run_dir),
        ]
        print("running ablation:", " ".join(train_cmd), flush=True)
        subprocess.run(train_cmd, check=True)

        best_path = run_dir / "feature_ablation_best_summary.csv"
        if not best_path.exists():
            # Build best summary for older run_feature_ablation outputs.
            rows = []
            for metrics_path in sorted(run_dir.glob("*/metrics.json")):
                metrics = json.load(metrics_path.open())
                best = min(metrics["epochs"], key=lambda item: item["val_loss"])
                last = metrics["epochs"][-1]
                rows.append(
                    {
                        "feature_set": metrics["feature_set"],
                        "parameters": metrics["parameters"],
                        "best_epoch": best["epoch"],
                        "best_train_loss": best["train_loss"],
                        "best_val_loss": best["val_loss"],
                        "best_val_mae": best["val_mae"],
                        "last_epoch": last["epoch"],
                        "last_train_loss": last["train_loss"],
                        "last_val_loss": last["val_loss"],
                        "last_val_mae": last["val_mae"],
                        "metrics_path": str(metrics_path),
                    }
                )
            rows = sorted(rows, key=lambda item: item["best_val_loss"])
            with best_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        with best_path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["grid_size"] = label
                row["cache_dir"] = str(cache_dir)
                row["run_dir"] = str(run_dir)
                all_rows.append(row)

    all_rows = sorted(
        all_rows,
        key=lambda item: (float(item["best_val_loss"]), item["grid_size"]),
    )
    summary_csv = args.out_root / "grid_feature_ablation_best_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    summary_json = args.out_root / "grid_feature_ablation_best_summary.json"
    with summary_json.open("w") as handle:
        json.dump(all_rows, handle, indent=2)

    print("grid_size,feature_set,best_epoch,best_val_loss,best_val_mae,parameters")
    for row in all_rows:
        print(
            f"{row['grid_size']},{row['feature_set']},{row['best_epoch']},"
            f"{float(row['best_val_loss']):.8f},{float(row['best_val_mae']):.8f},"
            f"{row['parameters']}"
        )
    print(f"saved summary: {summary_csv}")
    print(f"saved summary json: {summary_json}")


if __name__ == "__main__":
    main()
