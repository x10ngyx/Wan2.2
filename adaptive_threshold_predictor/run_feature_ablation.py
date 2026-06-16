from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from adaptive_threshold_predictor.data import DATASET_MODES
from adaptive_threshold_predictor.models import FEATURE_SETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-architecture feature ablations for threshold prediction."
    )
    parser.add_argument("--python", default="/hy-tmp/miniconda3/envs/Wan2.2/bin/python")
    parser.add_argument("--out_root", type=Path, default=Path("/hy-tmp/wan22_adaptive_threshold_feature_ablation"))
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--dataset_mode", choices=DATASET_MODES, default="candidate_inverse")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--psnr_min", type=float, default=10.0)
    parser.add_argument("--psnr_max", type=float, default=50.0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_val_predictions", action="store_true")
    parser.add_argument(
        "--feature_sets",
        nargs="+",
        choices=FEATURE_SETS,
        default=list(FEATURE_SETS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for feature_set in args.feature_sets:
        out_dir = args.out_root / feature_set
        cmd = [
            args.python,
            "-m",
            "adaptive_threshold_predictor.train_gate",
            "--dataset_mode",
            args.dataset_mode,
            "--feature_set",
            feature_set,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--hidden_dim",
            str(args.hidden_dim),
            "--psnr_min",
            str(args.psnr_min),
            "--psnr_max",
            str(args.psnr_max),
            "--out_dir",
            str(out_dir),
            "--num_workers",
            str(args.num_workers),
        ]
        if args.cache_dir is not None:
            cmd.extend(["--cache_dir", str(args.cache_dir)])
        if args.max_examples is not None:
            cmd.extend(["--max_examples", str(args.max_examples)])
        if args.device is not None:
            cmd.extend(["--device", args.device])
        if args.save_val_predictions:
            cmd.append("--save_val_predictions")

        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        metrics_path = out_dir / "metrics.json"
        with metrics_path.open("r") as handle:
            metrics = json.load(handle)
        last = metrics["epochs"][-1]
        summary.append(
            {
                "feature_set": feature_set,
                "parameters": metrics["parameters"],
                "train_loss": last["train_loss"],
                "val_loss": last["val_loss"],
                "val_mae": last["val_mae"],
                "metrics_path": str(metrics_path),
            }
        )

    summary_path = args.out_root / "feature_ablation_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    csv_path = args.out_root / "feature_ablation_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_set",
                "parameters",
                "train_loss",
                "val_loss",
                "val_mae",
                "metrics_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("feature_set,parameters,train_loss,val_loss,val_mae,metrics_path")
    for row in sorted(summary, key=lambda item: item["val_loss"]):
        print(
            f"{row['feature_set']},{row['parameters']},"
            f"{row['train_loss']:.8f},{row['val_loss']:.8f},"
            f"{row['val_mae']:.8f},{row['metrics_path']}"
        )
    print(f"saved summary: {summary_path}")
    print(f"saved summary csv: {csv_path}")


if __name__ == "__main__":
    main()
