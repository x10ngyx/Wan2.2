from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from adaptive_threshold_predictor.data import (
    CachedFeatureThresholdDataset,
    DATASET_MODES,
    DEFAULT_DATA_ROOT,
    GridFeatureThresholdDataset,
    PackedRawLatentThresholdDataset,
    TraceStepThresholdDataset,
    collate_cached_features,
    collate_grid_features,
    collate_trace_steps,
    split_indices_by_row,
    split_indices_by_sample_id,
)
from adaptive_threshold_predictor.models import (
    CachedFeatureAdaCacheGate,
    CachedGatedFeatureAdaCacheGate,
    ConditionOnlyAdaCacheGate,
    DEFAULT_GATED_FEATURE_SETS,
    FEATURE_SETS,
    GatedMultiFeatureAdaCacheGate,
    GridMLPThresholdPredictor,
    ImprovedAdaCacheGate,
    MiniDiTCLSAdaptiveThresholdPredictor,
    count_parameters,
)


MODEL_TYPES = ("mlp", "grid_mlp", "mini_dit_cls")
MINI_DIT_INPUT_SHAPE = (16, 12, 60, 104)


def selected_feature_sets(args: argparse.Namespace) -> tuple[str, ...]:
    if args.feature_sets is not None:
        return tuple(args.feature_sets)
    return (args.feature_set,)


def use_gated_features(args: argparse.Namespace) -> bool:
    return len(selected_feature_sets(args)) > 1


def feature_embedding_dim(args: argparse.Namespace) -> int:
    return args.feature_embedding_dim or args.hidden_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train adaptive timestep threshold predictors."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--grid_cache_dir", type=Path, default=None)
    parser.add_argument("--packed_latent_cache_dir", type=Path, default=None)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/hy-tmp/wan22_adaptive_threshold_predictor_debug"),
    )
    parser.add_argument("--dataset_mode", choices=DATASET_MODES, default="candidate_inverse")
    parser.add_argument("--model_type", choices=MODEL_TYPES, default="mlp")

    # Legacy MLP arguments.
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--feature_embedding_dim", type=int, default=None)
    parser.add_argument("--feature_set", choices=FEATURE_SETS, default="latent_pool")
    parser.add_argument(
        "--feature_sets",
        nargs="+",
        choices=FEATURE_SETS,
        default=None,
        help=(
            "Optional multi-feature MLP input. When set, features are encoded "
            "independently and fused with a condition-dependent softmax gate; "
            "--feature_set is ignored for model input."
        ),
    )
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

    # Grid MLP / MiniDiT arguments.
    parser.add_argument("--grid_mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--grid_mlp_depth", type=int, default=3)
    parser.add_argument("--dit_dim", type=int, default=96)
    parser.add_argument("--dit_layers", type=int, default=2)
    parser.add_argument("--dit_heads", type=int, default=4)
    parser.add_argument("--dit_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dit_dropout", type=float, default=0.05)
    parser.add_argument("--dit_gate_init", type=float, default=0.0)
    parser.add_argument("--dit_patch_size", nargs=3, type=int, default=(3, 12, 8))
    parser.add_argument("--min_threshold", type=float, default=0.10)
    parser.add_argument("--max_threshold", type=float, default=0.80)

    # Shared training arguments.
    parser.add_argument("--psnr_min", type=float, default=10.0)
    parser.add_argument("--psnr_max", type=float, default=50.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--smooth_l1_beta", type=float, default=0.02)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_mode", choices=("sample", "row"), default="sample")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--preload_packed_latents", action="store_true")
    parser.add_argument("--save_val_predictions", action="store_true")
    parser.add_argument("--save_epoch_val_predictions", action="store_true")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> tuple[Any, Any, str]:
    if args.model_type == "mini_dit_cls":
        if args.packed_latent_cache_dir is not None:
            dataset = PackedRawLatentThresholdDataset(
                cache_dir=args.packed_latent_cache_dir,
                max_examples=args.max_examples,
                preload=args.preload_packed_latents,
            )
            return dataset, collate_trace_steps, "packed_raw_latent"
        dataset = TraceStepThresholdDataset(
            data_root=args.data_root,
            dataset_mode=args.dataset_mode,
            max_examples=args.max_examples,
        )
        return dataset, collate_trace_steps, "raw_latent"

    if args.model_type == "grid_mlp":
        if args.grid_cache_dir is None:
            raise ValueError(f"--grid_cache_dir is required for model_type={args.model_type}")
        dataset = GridFeatureThresholdDataset(
            cache_dir=args.grid_cache_dir,
            max_examples=args.max_examples,
        )
        return dataset, collate_grid_features, "grid_feature"

    if args.control_mode == "noise_feature" and args.cache_dir is None:
        raise ValueError("--control_mode noise_feature currently requires --cache_dir")
    if args.cache_dir is not None:
        dataset = CachedFeatureThresholdDataset(
            cache_dir=args.cache_dir,
            feature_set=selected_feature_sets(args),
            max_examples=args.max_examples,
        )
        return dataset, collate_cached_features, "cached_feature"

    dataset = TraceStepThresholdDataset(
        data_root=args.data_root,
        dataset_mode=args.dataset_mode,
        max_examples=args.max_examples,
    )
    return dataset, collate_trace_steps, "raw_latent"


def build_model(args: argparse.Namespace, dataset: Any, input_mode: str) -> nn.Module:
    first = dataset[0]
    if args.model_type == "grid_mlp":
        return GridMLPThresholdPredictor(
            grid_shape=dataset.grid_shape,
            hidden_dim=args.grid_mlp_hidden_dim,
            depth=args.grid_mlp_depth,
            psnr_min=args.psnr_min,
            psnr_max=args.psnr_max,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            dropout=args.dit_dropout,
        ).to(args.device)
    if args.model_type == "mini_dit_cls":
        return MiniDiTCLSAdaptiveThresholdPredictor(
            input_shape=tuple(int(value) for value in first["latent"].shape),  # type: ignore[index,union-attr]
            patch_size=tuple(args.dit_patch_size),
            dim=args.dit_dim,
            num_layers=args.dit_layers,
            num_heads=args.dit_heads,
            mlp_ratio=args.dit_mlp_ratio,
            psnr_min=args.psnr_min,
            psnr_max=args.psnr_max,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            dropout=args.dit_dropout,
            gate_init=args.dit_gate_init,
        ).to(args.device)

    if args.control_mode == "condition_only":
        return ConditionOnlyAdaCacheGate(
            hidden_dim=args.hidden_dim,
            psnr_min=args.psnr_min,
            psnr_max=args.psnr_max,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
        ).to(args.device)
    if input_mode == "cached_feature":
        if use_gated_features(args):
            return CachedGatedFeatureAdaCacheGate(
                feature_dims=dataset.feature_dims,
                hidden_dim=args.hidden_dim,
                feature_embedding_dim=feature_embedding_dim(args),
                psnr_min=args.psnr_min,
                psnr_max=args.psnr_max,
                min_threshold=args.min_threshold,
                max_threshold=args.max_threshold,
                dropout=args.dit_dropout,
            ).to(args.device)
        feature_dim = int(first["feature"].numel())  # type: ignore[index,union-attr]
        return CachedFeatureAdaCacheGate(
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            psnr_min=args.psnr_min,
            psnr_max=args.psnr_max,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
        ).to(args.device)

    latent_channels = int(first["latent"].shape[0])  # type: ignore[index,union-attr]
    if use_gated_features(args):
        return GatedMultiFeatureAdaCacheGate(
            latent_channels=latent_channels,
            hidden_dim=args.hidden_dim,
            feature_embedding_dim=feature_embedding_dim(args),
            feature_sets=selected_feature_sets(args),
            psnr_min=args.psnr_min,
            psnr_max=args.psnr_max,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            dropout=args.dit_dropout,
        ).to(args.device)
    return ImprovedAdaCacheGate(
        latent_channels=latent_channels,
        hidden_dim=args.hidden_dim,
        feature_set=args.feature_set,
        psnr_min=args.psnr_min,
        psnr_max=args.psnr_max,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
    ).to(args.device)


def build_feature_extractor_config(
    args: argparse.Namespace,
    dataset: Any,
    model: nn.Module,
) -> dict[str, Any]:
    if args.model_type == "mini_dit_cls":
        input_shape = getattr(model, "input_shape", MINI_DIT_INPUT_SHAPE)
        patch_size = getattr(model, "patch_size", tuple(args.dit_patch_size))
        grid_shape = getattr(model, "grid_shape", None)
        token_grid_shape = tuple(grid_shape[1:]) if grid_shape is not None else None
        return {
            "type": "learned_conv3d_patch_embedding",
            "input_layout": "B,C,T,H,W",
            "input_shape": list(input_shape),
            "patch_size": list(patch_size),
            "token_grid_shape": list(token_grid_shape) if token_grid_shape is not None else None,
            "token_count": int(torch.tensor(token_grid_shape).prod().item())
            if token_grid_shape is not None
            else None,
            "in_channels": int(input_shape[0]),
            "out_channels": args.dit_dim,
            "normalization": "none",
            "runtime_dtype": "float32",
            "packed_latent_cache_dir": str(args.packed_latent_cache_dir)
            if args.packed_latent_cache_dir is not None
            else None,
        }
    if args.model_type == "grid_mlp":
        return {
            "type": "avg_pool3d_grid_cache",
            "input_layout": "B,C,T,H,W",
            "grid_shape": list(dataset.grid_shape),
            "grid_cache_dir": str(args.grid_cache_dir),
            "normalization": "none",
            "runtime_dtype": "float32",
        }
    if args.cache_dir is not None:
        return {
            "type": "pooled_feature_cache",
            "feature_set": args.feature_set,
            "feature_sets": list(selected_feature_sets(args)),
            "feature_fusion": "gated" if use_gated_features(args) else "single",
            "feature_embedding_dim": feature_embedding_dim(args),
            "cache_dir": str(args.cache_dir),
            "normalization": "none",
            "runtime_dtype": "float32",
        }
    return {
        "type": "raw_latent_internal_pool",
        "feature_set": args.feature_set,
        "feature_sets": list(selected_feature_sets(args)),
        "feature_fusion": "gated" if use_gated_features(args) else "single",
        "feature_embedding_dim": feature_embedding_dim(args),
        "normalization": "none",
        "runtime_dtype": "float32",
    }


def cosine_with_warmup_lambda(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def predict_batch(
    *,
    model: nn.Module,
    batch: dict[str, Any],
    args: argparse.Namespace,
    input_mode: str,
    noise_generator: torch.Generator,
) -> torch.Tensor:
    timestep = batch["timestep"].to(args.device)
    target_psnr = batch["target_psnr"].to(args.device)
    label = batch["threshold"].to(args.device)

    if args.model_type == "mini_dit_cls":
        model_input = batch["latent"].to(args.device)
        return model(model_input, timestep, target_psnr)

    if args.model_type == "grid_mlp":
        model_input = batch["grid_feature"].to(args.device)
        return model(model_input, timestep, target_psnr)

    if args.control_mode == "condition_only":
        return model(timestep, target_psnr, batch=label.shape[0], device=label.device)

    if use_gated_features(args) and input_mode == "cached_feature":
        features = {
            name: feature.to(args.device)
            for name, feature in batch["features"].items()
        }
        if args.control_mode == "noise_feature":
            features = {
                name: torch.randn(
                    feature.shape,
                    generator=noise_generator,
                    device=feature.device,
                    dtype=feature.dtype,
                )
                for name, feature in features.items()
            }
        return model(features, timestep, target_psnr)

    model_input = batch["feature" if input_mode == "cached_feature" else "latent"].to(args.device)
    if args.control_mode == "noise_feature":
        model_input = torch.randn(
            model_input.shape,
            generator=noise_generator,
            device=model_input.device,
            dtype=model_input.dtype,
        )
    return model(model_input, timestep, target_psnr)


def bucket_name(kind: str, value: float) -> str:
    if kind == "step":
        step = int(round(value * 49))
        if step <= 9:
            return "step_00_09"
        if step <= 39:
            return "step_10_39"
        return "step_40_49"
    if kind == "threshold":
        return f"threshold_{value:.2f}"
    if kind == "target_psnr":
        return f"target_{round(value):02d}"
    raise ValueError(f"Unknown bucket kind: {kind}")


def summarize_predictions(
    *,
    loss_sum: float,
    count: int,
    preds: list[float],
    labels: list[float],
    timesteps: list[float],
    target_psnrs: list[float],
) -> dict[str, Any]:
    errors = [abs(pred - label) for pred, label in zip(preds, labels)]
    signed = [pred - label for pred, label in zip(preds, labels)]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def grouped(kind: str, keys: list[float]) -> dict[str, dict[str, float]]:
        groups: dict[str, list[float]] = {}
        for key, error in zip(keys, errors):
            groups.setdefault(bucket_name(kind, key), []).append(error)
        return {
            name: {
                "count": len(values),
                "mae": sum(values) / len(values),
            }
            for name, values in sorted(groups.items())
        }

    return {
        "loss": loss_sum / max(count, 1),
        "mae": mean(errors),
        "bias": mean(signed),
        "pred_min": min(preds) if preds else None,
        "pred_max": max(preds) if preds else None,
        "pred_mean": mean(preds),
        "pred_std": float(torch.tensor(preds).std(unbiased=False)) if preds else None,
        "label_min": min(labels) if labels else None,
        "label_max": max(labels) if labels else None,
        "label_mean": mean(labels),
        "by_step_range": grouped("step", timesteps),
        "by_threshold": grouped("threshold", labels),
        "by_target_psnr": grouped("target_psnr", target_psnrs),
    }


def run_eval(
    *,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    args: argparse.Namespace,
    input_mode: str,
    noise_generator: torch.Generator,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    count = 0
    preds: list[float] = []
    labels: list[float] = []
    timesteps: list[float] = []
    target_psnrs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            label = batch["threshold"].to(args.device)
            pred = predict_batch(
                model=model,
                batch=batch,
                args=args,
                input_mode=input_mode,
                noise_generator=noise_generator,
            )
            loss_sum += float(loss_fn(pred, label)) * label.shape[0]
            count += label.shape[0]
            preds.extend(pred.detach().cpu().flatten().tolist())
            labels.extend(label.detach().cpu().flatten().tolist())
            timesteps.extend(batch["timestep"].flatten().tolist())
            target_psnrs.extend(batch["target_psnr"].flatten().tolist())
    return summarize_predictions(
        loss_sum=loss_sum,
        count=count,
        preds=preds,
        labels=labels,
        timesteps=timesteps,
        target_psnrs=target_psnrs,
    )


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    feature_extractor_config: dict[str, Any],
) -> None:
    torch.save(model.state_dict(), path)
    payload = {
        "epoch": epoch,
        "model_type": args.model_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "feature_extractor": feature_extractor_config,
        "metrics": metrics,
    }
    torch.save(payload, path.with_name(path.stem + "_checkpoint.pt"))


def write_epoch_csv(path: Path, epochs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "lr",
        "elapsed_seconds",
        "train_loss",
        "train_mae",
        "train_bias",
        "val_loss",
        "val_mae",
        "val_bias",
        "val_pred_min",
        "val_pred_max",
        "val_pred_mean",
        "val_pred_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in epochs:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_predictions_csv(
    *,
    path: Path,
    model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    input_mode: str,
    noise_generator: torch.Generator,
) -> None:
    model.eval()
    gate_feature_names = (
        list(getattr(model, "feature_sets", []))
        if hasattr(model, "fusion")
        else []
    )
    gate_fieldnames = [f"gate_{name}" for name in gate_feature_names]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "pred_threshold",
                "label_threshold",
                "abs_error",
                "signed_error",
                "timestep",
                "step_index_estimate",
                "target_psnr",
            ]
            + gate_fieldnames,
        )
        writer.writeheader()
        with torch.no_grad():
            for batch in loader:
                timestep = batch["timestep"].to(args.device)
                target_psnr = batch["target_psnr"].to(args.device)
                label = batch["threshold"].to(args.device)
                pred = predict_batch(
                    model=model,
                    batch=batch,
                    args=args,
                    input_mode=input_mode,
                    noise_generator=noise_generator,
                )
                gate_weights = getattr(getattr(model, "fusion", None), "last_gate_weights", None)
                gate_rows = (
                    gate_weights.cpu().tolist()
                    if gate_weights is not None and gate_feature_names
                    else [None] * len(batch["sample_id"])
                )
                for sample_id, pred_i, label_i, step_i, psnr_i, gate_i in zip(
                    batch["sample_id"],
                    pred.cpu().flatten().tolist(),
                    label.cpu().flatten().tolist(),
                    timestep.cpu().flatten().tolist(),
                    target_psnr.cpu().flatten().tolist(),
                    gate_rows,
                ):
                    row = {
                        "sample_id": sample_id,
                        "pred_threshold": pred_i,
                        "label_threshold": label_i,
                        "abs_error": abs(pred_i - label_i),
                        "signed_error": pred_i - label_i,
                        "timestep": step_i,
                        "step_index_estimate": int(round(step_i * 49)),
                        "target_psnr": psnr_i,
                    }
                    if gate_i is not None:
                        row.update({
                            f"gate_{name}": value
                            for name, value in zip(gate_feature_names, gate_i)
                        })
                    writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.selected_feature_sets = list(selected_feature_sets(args))
    args.resolved_feature_embedding_dim = feature_embedding_dim(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    dataset, collate_fn, input_mode = load_dataset(args)
    if not dataset:
        raise RuntimeError("No training examples found")

    if args.split_mode == "row":
        train_indices, val_indices = split_indices_by_row(
            dataset,
            train_fraction=args.train_fraction,
            seed=args.split_seed,
        )
    else:
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
        "split": args.split_mode,
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
        pin_memory=str(args.device).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=str(args.device).startswith("cuda"),
    )

    model = build_model(args, dataset, input_mode)
    feature_extractor_config = build_feature_extractor_config(args, dataset, model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup_lambda(
            step,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=args.min_lr / args.lr if args.lr > 0 else 0.0,
        ),
    )
    loss_fn = nn.SmoothL1Loss(beta=args.smooth_l1_beta)

    metrics: dict[str, Any] = {
        "num_examples": len(dataset),
        "train_examples": len(train_indices),
        "val_examples": len(val_indices),
        "train_samples": len(train_sample_ids),
        "val_samples": len(val_sample_ids),
        "validation_empty": len(val_indices) == 0,
        "split": args.split_mode,
        "split_seed": args.split_seed,
        "dataset_mode": args.dataset_mode,
        "input_mode": input_mode,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "grid_cache_dir": str(args.grid_cache_dir) if args.grid_cache_dir is not None else None,
        "packed_latent_cache_dir": str(args.packed_latent_cache_dir)
        if args.packed_latent_cache_dir is not None
        else None,
        "preload_packed_latents": args.preload_packed_latents,
        "grid_shape": list(dataset.grid_shape) if hasattr(dataset, "grid_shape") else None,
        "feature_extractor": feature_extractor_config,
        "model_type": args.model_type,
        "parameters": count_parameters(model),
        "feature_set": args.feature_set,
        "feature_sets": list(selected_feature_sets(args)),
        "feature_fusion": "gated" if use_gated_features(args) else "single",
        "feature_embedding_dim": feature_embedding_dim(args),
        "control_mode": args.control_mode,
        "noise_seed": args.noise_seed,
        "psnr_min": args.psnr_min,
        "psnr_max": args.psnr_max,
        "min_threshold": args.min_threshold,
        "max_threshold": args.max_threshold,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "smooth_l1_beta": args.smooth_l1_beta,
        "grad_clip": args.grad_clip,
        "early_stop_patience": args.early_stop_patience,
        "epochs": [],
    }
    with (args.out_dir / "model_summary.json").open("w") as handle:
        json.dump(
            {
                "model": repr(model),
                "parameters": metrics["parameters"],
                "input_mode": input_mode,
            },
            handle,
            indent=2,
        )

    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0
    global_step = 0
    noise_generator = torch.Generator(device=args.device)
    noise_generator.manual_seed(args.noise_seed)
    epoch_jsonl = args.out_dir / "epoch_metrics.jsonl"
    if epoch_jsonl.exists():
        epoch_jsonl.unlink()

    for epoch in range(args.epochs):
        t0 = perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_preds: list[float] = []
        train_labels: list[float] = []
        train_timesteps: list[float] = []
        train_target_psnrs: list[float] = []

        for batch in train_loader:
            label = batch["threshold"].to(args.device)
            pred = predict_batch(
                model=model,
                batch=batch,
                args=args,
                input_mode=input_mode,
                noise_generator=noise_generator,
            )
            loss = loss_fn(pred, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss_sum += float(loss.detach()) * label.shape[0]
            train_count += label.shape[0]
            train_preds.extend(pred.detach().cpu().flatten().tolist())
            train_labels.extend(label.detach().cpu().flatten().tolist())
            train_timesteps.extend(batch["timestep"].flatten().tolist())
            train_target_psnrs.extend(batch["target_psnr"].flatten().tolist())

        train_summary = summarize_predictions(
            loss_sum=train_loss_sum,
            count=train_count,
            preds=train_preds,
            labels=train_labels,
            timesteps=train_timesteps,
            target_psnrs=train_target_psnrs,
        )
        val_summary = run_eval(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            args=args,
            input_mode=input_mode,
            noise_generator=noise_generator,
        )
        epoch_metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": round(perf_counter() - t0, 3),
            "train_loss": train_summary["loss"],
            "train_mae": train_summary["mae"],
            "train_bias": train_summary["bias"],
            "train_pred_min": train_summary["pred_min"],
            "train_pred_max": train_summary["pred_max"],
            "train_pred_mean": train_summary["pred_mean"],
            "train_pred_std": train_summary["pred_std"],
            "train_by_step_range": train_summary["by_step_range"],
            "train_by_threshold": train_summary["by_threshold"],
            "train_by_target_psnr": train_summary["by_target_psnr"],
            "val_loss": val_summary["loss"],
            "val_mae": val_summary["mae"],
            "val_bias": val_summary["bias"],
            "val_pred_min": val_summary["pred_min"],
            "val_pred_max": val_summary["pred_max"],
            "val_pred_mean": val_summary["pred_mean"],
            "val_pred_std": val_summary["pred_std"],
            "val_by_step_range": val_summary["by_step_range"],
            "val_by_threshold": val_summary["by_threshold"],
            "val_by_target_psnr": val_summary["by_target_psnr"],
        }
        metrics["epochs"].append(epoch_metrics)
        with epoch_jsonl.open("a") as handle:
            handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
        write_epoch_csv(args.out_dir / "epoch_metrics.csv", metrics["epochs"])
        with (args.out_dir / "metrics.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

        if args.save_epoch_val_predictions:
            save_predictions_csv(
                path=args.out_dir / f"val_predictions_epoch_{epoch + 1:03d}.csv",
                model=model,
                loader=val_loader,
                args=args,
                input_mode=input_mode,
                noise_generator=noise_generator,
            )

        metric_for_best = epoch_metrics["val_mae"]
        if metric_for_best is None:
            metric_for_best = epoch_metrics["train_mae"]
        metric_for_best = float(metric_for_best)
        if metric_for_best < best_val_mae:
            best_val_mae = metric_for_best
            best_epoch = epoch + 1
            bad_epochs = 0
            save_checkpoint(
                path=args.out_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                args=args,
                metrics=epoch_metrics,
                feature_extractor_config=feature_extractor_config,
            )
        else:
            bad_epochs += 1
            if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
                metrics["early_stopped"] = True
                metrics["early_stop_epoch"] = epoch + 1
                break

    metrics["best_epoch"] = best_epoch
    metrics["best_val_mae"] = best_val_mae
    save_checkpoint(
        path=args.out_dir / "final_model.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=len(metrics["epochs"]),
        args=args,
        metrics=metrics["epochs"][-1] if metrics["epochs"] else {},
        feature_extractor_config=feature_extractor_config,
    )
    with (args.out_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    if args.save_val_predictions:
        save_predictions_csv(
            path=args.out_dir / "val_predictions.csv",
            model=model,
            loader=val_loader,
            args=args,
            input_mode=input_mode,
            noise_generator=noise_generator,
        )
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
