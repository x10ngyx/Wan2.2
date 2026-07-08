from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from adaptive_threshold_predictor.data import (
    DEFAULT_DATA_ROOT,
    TraceStepThresholdDataset,
    collate_trace_steps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute fixed 3D grid latent features for MiniDiT training."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--patch_size", nargs=3, type=int, default=(3, 12, 8))
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def extract_grid_feature(latent: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    _, _, frames, height, width = latent.shape
    if (
        frames % patch_size[0] != 0
        or height % patch_size[1] != 0
        or width % patch_size[2] != 0
    ):
        raise ValueError(
            "Latent shape must be divisible by patch_size for fixed grid cache: "
            f"latent={(frames, height, width)}, patch_size={patch_size}"
        )
    return F.avg_pool3d(latent, kernel_size=patch_size, stride=patch_size)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patch_size = tuple(args.patch_size)
    with (args.out_dir / "cache_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    dataset = TraceStepThresholdDataset(
        data_root=args.data_root,
        max_examples=args.max_examples,
    )
    if not dataset:
        raise RuntimeError(f"No examples found under {args.data_root}")

    device = torch.device(args.device)
    feature_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    grid_batches: list[torch.Tensor] = []
    sample_ids: list[str] = []
    timestep_batches: list[torch.Tensor] = []
    target_psnr_batches: list[torch.Tensor] = []
    target_speedup_batches: list[torch.Tensor] = []
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
    grid_shape: tuple[int, int, int, int] | None = None
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            latent = batch["latent"].to(device, non_blocking=True).float()
            grid = extract_grid_feature(latent, patch_size)
            if grid_shape is None:
                grid_shape = tuple(int(value) for value in grid.shape[1:])
            grid_batches.append(grid.to("cpu", dtype=feature_dtype))
            sample_ids.extend(batch["sample_id"])
            timestep_batches.append(batch["timestep"].flatten().cpu())
            target_psnr_batches.append(batch["target_psnr"].flatten().cpu())
            target_speedup_batches.append(batch["target_speedup"].flatten().cpu())
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
    torch.save(torch.cat(grid_batches, dim=0), args.out_dir / "grid_features.pt")

    metadata = {
        "sample_id": sample_ids,
        "timestep": torch.cat(timestep_batches).to(torch.float32),
        "target_psnr": torch.cat(target_psnr_batches).to(torch.float32),
        "target_speedup": torch.cat(target_speedup_batches).to(torch.float32),
        "threshold": torch.cat(threshold_batches).to(torch.float32),
        "step_index": torch.tensor(step_index_values, dtype=torch.long),
        "source_index": torch.tensor(source_index_values, dtype=torch.long),
    }
    torch.save(metadata, args.out_dir / "metadata.pt")

    manifest = {
        "data_root": str(args.data_root),
        "num_examples": len(dataset),
        "patch_size": list(patch_size),
        "grid_shape": list(grid_shape or ()),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "elapsed_seconds": round(perf_counter() - t0, 3),
    }
    with (args.out_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
