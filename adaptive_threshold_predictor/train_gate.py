from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from adaptive_threshold_predictor.data import (
    CachedFeatureThresholdDataset,
    DATASET_MODES,
    DEFAULT_DATA_ROOT,
    TraceStepThresholdDataset,
    collate_cached_features,
    collate_trace_steps,
    split_indices_by_sample_id,
)
from adaptive_threshold_predictor.models import (
    CachedFeatureAdaCacheGate,
    ConditionOnlyAdaCacheGate,
    FEATURE_SETS,
    ImprovedAdaCacheGate,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the lightweight timestep threshold predictor."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("/hy-tmp/wan22_adaptive_threshold_predictor_debug"))
    parser.add_argument("--dataset_mode", choices=DATASET_MODES, default="candidate_inverse")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--feature_set", choices=FEATURE_SETS, default="latent_pool")
    parser.add_argument(
        "--control_mode",
        choices=("feature", "condition_only", "noise_feature"),
        default="feature",
        help=(
            "feature: use the selected latent-derived feature; "
            "condition_only: use only timestep and PSNR; "
            "noise_feature: keep the feature trunk but replace features with random noise"
        ),
    )
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument("--psnr_min", type=float, default=10.0)
    parser.add_argument("--psnr_max", type=float, default=50.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--save_val_predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    if args.control_mode == "noise_feature" and args.cache_dir is None:
        raise ValueError("--control_mode noise_feature currently requires --cache_dir")

    if args.cache_dir is not None:
        dataset = CachedFeatureThresholdDataset(
            cache_dir=args.cache_dir,
            feature_set=args.feature_set,
            max_examples=args.max_examples,
        )
        collate_fn = collate_cached_features
        input_mode = "cached_feature"
    else:
        dataset = TraceStepThresholdDataset(
            data_root=args.data_root,
            dataset_mode=args.dataset_mode,
            max_examples=args.max_examples,
        )
        collate_fn = collate_trace_steps
        input_mode = "raw_latent"
    if not dataset:
        raise RuntimeError("No training examples found")

    train_indices, val_indices = split_indices_by_sample_id(
        dataset,
        train_fraction=args.train_fraction,
        seed=args.split_seed,
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_sample_ids = {dataset.examples[index].sample_id for index in train_indices}
    val_sample_ids = {dataset.examples[index].sample_id for index in val_indices}
    split_payload = {
        "split": "group_by_sample_id",
        "split_seed": args.split_seed,
        "train_fraction": args.train_fraction,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_sample_ids": sorted(train_sample_ids),
        "val_sample_ids": sorted(val_sample_ids),
    }
    with (args.out_dir / "split.json").open("w") as handle:
        json.dump(split_payload, handle, indent=2)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    first = dataset[0]
    if args.control_mode == "condition_only":
        model = ConditionOnlyAdaCacheGate(
            hidden_dim=args.hidden_dim,
            psnr_min=args.psnr_min,
            psnr_max=args.psnr_max,
        ).to(args.device)
    elif input_mode == "cached_feature":
        feature_dim = int(first["feature"].numel())  # type: ignore[index,union-attr]
        model = CachedFeatureAdaCacheGate(
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            psnr_min=args.psnr_min,
            psnr_max=args.psnr_max,
        ).to(args.device)
    else:
        latent_channels = int(first["latent"].shape[0])  # type: ignore[index,union-attr]
        model = ImprovedAdaCacheGate(
            latent_channels=latent_channels,
            hidden_dim=args.hidden_dim,
            feature_set=args.feature_set,
            psnr_min=args.psnr_min,
            psnr_max=args.psnr_max,
        ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()

    metrics = {
        "num_examples": len(dataset),
        "train_examples": len(train_indices),
        "val_examples": len(val_indices),
        "train_samples": len(train_sample_ids),
        "val_samples": len(val_sample_ids),
        "split": "group_by_sample_id",
        "split_seed": args.split_seed,
        "dataset_mode": args.dataset_mode,
        "input_mode": input_mode,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "parameters": count_parameters(model),
        "feature_set": args.feature_set,
        "control_mode": args.control_mode,
        "noise_seed": args.noise_seed,
        "psnr_min": args.psnr_min,
        "psnr_max": args.psnr_max,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": [],
    }

    best_val_loss = float("inf")
    noise_generator = torch.Generator(device=args.device)
    noise_generator.manual_seed(args.noise_seed)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            timestep = batch["timestep"].to(args.device)
            target_psnr = batch["target_psnr"].to(args.device)
            label = batch["threshold"].to(args.device)

            if args.control_mode == "condition_only":
                batch_size = label.shape[0]
                pred = model(timestep, target_psnr, batch=batch_size, device=label.device)
            else:
                model_input = batch[
                    "feature" if input_mode == "cached_feature" else "latent"
                ].to(args.device)
                if args.control_mode == "noise_feature":
                    model_input = torch.randn(
                        model_input.shape,
                        generator=noise_generator,
                        device=model_input.device,
                        dtype=model_input.dtype,
                    )
                pred = model(model_input, timestep, target_psnr)
            loss = loss_fn(pred, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.detach()) * label.shape[0]
            train_count += label.shape[0]

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                timestep = batch["timestep"].to(args.device)
                target_psnr = batch["target_psnr"].to(args.device)
                label = batch["threshold"].to(args.device)
                if args.control_mode == "condition_only":
                    pred = model(
                        timestep,
                        target_psnr,
                        batch=label.shape[0],
                        device=label.device,
                    )
                else:
                    model_input = batch[
                        "feature" if input_mode == "cached_feature" else "latent"
                    ].to(args.device)
                    if args.control_mode == "noise_feature":
                        model_input = torch.randn(
                            model_input.shape,
                            generator=noise_generator,
                            device=model_input.device,
                            dtype=model_input.dtype,
                        )
                    pred = model(model_input, timestep, target_psnr)
                val_loss += float(loss_fn(pred, label)) * label.shape[0]
                val_mae += float(torch.abs(pred - label).mean()) * label.shape[0]
                val_count += label.shape[0]

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_count, 1),
            "val_loss": val_loss / max(val_count, 1),
            "val_mae": val_mae / max(val_count, 1),
        }
        metrics["epochs"].append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True))
        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")

    torch.save(model.state_dict(), args.out_dir / "final_model.pt")
    with (args.out_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    if args.save_val_predictions:
        predictions_path = args.out_dir / "val_predictions.csv"
        model.eval()
        with predictions_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "pred_threshold",
                    "label_threshold",
                    "abs_error",
                    "timestep",
                    "target_psnr",
                ],
            )
            writer.writeheader()
            with torch.no_grad():
                for batch in val_loader:
                    timestep = batch["timestep"].to(args.device)
                    target_psnr = batch["target_psnr"].to(args.device)
                    label = batch["threshold"].to(args.device)
                    if args.control_mode == "condition_only":
                        pred = model(
                            timestep,
                            target_psnr,
                            batch=label.shape[0],
                            device=label.device,
                        )
                    else:
                        model_input = batch[
                            "feature" if input_mode == "cached_feature" else "latent"
                        ].to(args.device)
                        if args.control_mode == "noise_feature":
                            model_input = torch.randn(
                                model_input.shape,
                                generator=noise_generator,
                                device=model_input.device,
                                dtype=model_input.dtype,
                            )
                        pred = model(model_input, timestep, target_psnr)
                    for sample_id, pred_i, label_i, step_i, psnr_i in zip(
                        batch["sample_id"],
                        pred.cpu().flatten().tolist(),
                        label.cpu().flatten().tolist(),
                        timestep.cpu().flatten().tolist(),
                        target_psnr.cpu().flatten().tolist(),
                    ):
                        writer.writerow(
                            {
                                "sample_id": sample_id,
                                "pred_threshold": pred_i,
                                "label_threshold": label_i,
                                "abs_error": abs(pred_i - label_i),
                                "timestep": step_i,
                                "target_psnr": psnr_i,
                            }
                        )
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
