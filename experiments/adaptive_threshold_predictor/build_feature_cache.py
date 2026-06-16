from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.adaptive_threshold_predictor.data import (
    DATASET_MODES,
    DEFAULT_DATA_ROOT,
    TraceStepThresholdDataset,
    collate_trace_steps,
)
from experiments.adaptive_threshold_predictor.models import FEATURE_SETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute pooled latent features for adaptive threshold training."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset_mode", choices=DATASET_MODES, default="candidate_inverse")
    parser.add_argument("--grid_size", nargs=3, type=int, default=(2, 2, 2))
    parser.add_argument("--feature_sets", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def frame_diff(latent: torch.Tensor) -> torch.Tensor:
    if latent.shape[2] <= 1:
        return torch.zeros_like(latent)
    diff = torch.abs(latent[:, :, 1:] - latent[:, :, :-1])
    pad = torch.zeros_like(latent[:, :, :1])
    return torch.cat([diff, pad], dim=2)


def extract_feature(
    latent: torch.Tensor,
    feature_set: str,
    pool: nn.AdaptiveAvgPool3d,
) -> torch.Tensor:
    batch, channels, _, _, _ = latent.shape
    temporal_bins, height_bins, width_bins = pool.output_size
    if feature_set == "latent_pool":
        return pool(latent).flatten(start_dim=1)
    elif feature_set == "temporal_mean":
        spatial = torch.nn.functional.adaptive_avg_pool2d(
            latent.mean(dim=2), (height_bins, width_bins)
        )
        return (
            spatial.unsqueeze(2)
            .expand(batch, channels, temporal_bins, height_bins, width_bins)
            .flatten(start_dim=1)
        )
    elif feature_set == "temporal_var":
        spatial = torch.nn.functional.adaptive_avg_pool2d(
            latent.var(dim=2, unbiased=False), (height_bins, width_bins)
        )
        return (
            spatial.unsqueeze(2)
            .expand(batch, channels, temporal_bins, height_bins, width_bins)
            .flatten(start_dim=1)
        )
    elif feature_set == "frame_diff_mean":
        spatial = torch.nn.functional.adaptive_avg_pool2d(
            frame_diff(latent).mean(dim=2), (height_bins, width_bins)
        )
        return (
            spatial.unsqueeze(2)
            .expand(batch, channels, temporal_bins, height_bins, width_bins)
            .flatten(start_dim=1)
        )
    elif feature_set == "frame_diff_var":
        spatial = torch.nn.functional.adaptive_avg_pool2d(
            frame_diff(latent).var(dim=2, unbiased=False), (height_bins, width_bins)
        )
        return (
            spatial.unsqueeze(2)
            .expand(batch, channels, temporal_bins, height_bins, width_bins)
            .flatten(start_dim=1)
        )
    else:
        raise ValueError(f"Unhandled feature_set: {feature_set}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "cache_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    dataset = TraceStepThresholdDataset(
        data_root=args.data_root,
        dataset_mode=args.dataset_mode,
        max_examples=args.max_examples,
    )
    if not dataset:
        raise RuntimeError(f"No examples found under {args.data_root}")

    device = torch.device(args.device)
    pool = nn.AdaptiveAvgPool3d(tuple(args.grid_size)).to(device)
    feature_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    feature_buffers = {feature_set: [] for feature_set in args.feature_sets}
    sample_ids: list[str] = []
    timestep_batches: list[torch.Tensor] = []
    target_psnr_batches: list[torch.Tensor] = []
    threshold_batches: list[torch.Tensor] = []
    step_index_values: list[int] = []
    source_index_values: list[int] = []
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_trace_steps,
        pin_memory=(device.type == "cuda"),
    )

    t0 = perf_counter()
    processed = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            latent = batch["latent"].to(device, non_blocking=True).float()
            for feature_set in args.feature_sets:
                feature_buffers[feature_set].append(
                    extract_feature(latent, feature_set, pool)
                    .to("cpu", dtype=feature_dtype)
                )
            sample_ids.extend(batch["sample_id"])
            timestep_batches.append(batch["timestep"].flatten().cpu())
            target_psnr_batches.append(batch["target_psnr"].flatten().cpu())
            threshold_batches.append(batch["threshold"].flatten().cpu())
            start = batch_index * args.batch_size
            stop = start + len(batch["sample_id"])
            for index in range(start, stop):
                example = dataset.examples[index]
                step_index_values.append(int(example.step_index))
                source_index_values.append(index)

            processed += len(batch["sample_id"])
            if processed % 1000 == 0 or processed == len(dataset):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = perf_counter() - t0
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "total": len(dataset),
                            "elapsed_seconds": round(elapsed, 2),
                            "examples_per_second": round(processed / elapsed, 2),
                        }
                    ),
                    flush=True,
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    for feature_set, values in feature_buffers.items():
        torch.save(torch.cat(values, dim=0), args.out_dir / f"features_{feature_set}.pt")

    metadata = {
        "sample_id": sample_ids,
        "timestep": torch.cat(timestep_batches).to(torch.float32),
        "target_psnr": torch.cat(target_psnr_batches).to(torch.float32),
        "threshold": torch.cat(threshold_batches).to(torch.float32),
        "step_index": torch.tensor(step_index_values, dtype=torch.long),
        "source_index": torch.tensor(source_index_values, dtype=torch.long),
    }
    torch.save(metadata, args.out_dir / "metadata.pt")

    manifest = {
        "dataset_mode": args.dataset_mode,
        "data_root": str(args.data_root),
        "num_examples": len(dataset),
        "feature_sets": args.feature_sets,
        "grid_size": list(args.grid_size),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "feature_dim": int(next(iter(feature_buffers.values()))[0].shape[1]),
        "elapsed_seconds": round(perf_counter() - t0, 3),
    }
    with (args.out_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
