from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from adaptive_threshold_predictor.data import (
    DEFAULT_DATA_ROOT,
    TraceStepThresholdDataset,
    collate_trace_steps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack raw step latents into fp16 shards for faster training."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--shard_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume_existing", action="store_true")
    return parser.parse_args()


def tensor_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def save_shard(
    *,
    out_dir: Path,
    shard_index: int,
    latents: list[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[str, int]:
    shard_name = f"latents_shard_{shard_index:05d}.pt"
    shard_path = out_dir / shard_name
    stacked = torch.cat(latents, dim=0).to(dtype=dtype)
    torch.save(stacked, shard_path)
    return shard_name, int(stacked.shape[0])


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0:
        raise ValueError("--shard_size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = args.out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.out_dir / "cache_config.json"
    with config_path.open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    dataset = TraceStepThresholdDataset(
        data_root=args.data_root,
        max_examples=args.max_examples,
    )
    if not dataset:
        raise RuntimeError(f"No examples found under {args.data_root}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_trace_steps,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    dtype = tensor_dtype(args.dtype)
    t0 = perf_counter()
    processed = 0
    shard_index = 0
    shard_latents: list[torch.Tensor] = []
    shard_count = 0
    shard_records: list[dict[str, object]] = []
    sample_ids: list[str] = []
    timestep_batches: list[torch.Tensor] = []
    target_psnr_batches: list[torch.Tensor] = []
    target_speedup_batches: list[torch.Tensor] = []
    threshold_batches: list[torch.Tensor] = []
    step_index_values: list[int] = []
    source_index_values: list[int] = []
    shard_name_values: list[str] = []
    shard_offset_values: list[int] = []

    for batch_index, batch in enumerate(loader):
        latents = batch["latent"]
        batch_size = int(latents.shape[0])
        start_index = processed
        batch_offsets = list(range(batch_size))
        cursor = 0
        while cursor < batch_size:
            take = min(args.shard_size - shard_count, batch_size - cursor)
            shard_latents.append(latents[cursor:cursor + take].contiguous())
            for local_offset in range(take):
                global_index = start_index + cursor + local_offset
                example = dataset.examples[global_index]
                sample_ids.append(batch["sample_id"][cursor + local_offset])
                step_index_values.append(int(example.step_index))
                source_index_values.append(global_index)
                shard_name_values.append(f"latents_shard_{shard_index:05d}.pt")
                shard_offset_values.append(shard_count + local_offset)
            timestep_batches.append(batch["timestep"][cursor:cursor + take].flatten().cpu())
            target_psnr_batches.append(
                batch["target_psnr"][cursor:cursor + take].flatten().cpu()
            )
            target_speedup_batches.append(
                batch["target_speedup"][cursor:cursor + take].flatten().cpu()
            )
            threshold_batches.append(batch["threshold"][cursor:cursor + take].flatten().cpu())
            shard_count += take
            cursor += take

            if shard_count == args.shard_size:
                shard_name, saved_count = save_shard(
                    out_dir=shards_dir,
                    shard_index=shard_index,
                    latents=shard_latents,
                    dtype=dtype,
                )
                shard_records.append(
                    {
                        "shard_index": shard_index,
                        "path": f"shards/{shard_name}",
                        "count": saved_count,
                    }
                )
                shard_index += 1
                shard_latents = []
                shard_count = 0

        processed += batch_size
        if processed % 1024 == 0 or processed == len(dataset):
            elapsed = perf_counter() - t0
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "total": len(dataset),
                        "elapsed_seconds": round(elapsed, 2),
                        "examples_per_second": round(processed / elapsed, 2),
                        "shards": len(shard_records),
                    }
                ),
                flush=True,
            )

    if shard_count > 0:
        shard_name, saved_count = save_shard(
            out_dir=shards_dir,
            shard_index=shard_index,
            latents=shard_latents,
            dtype=dtype,
        )
        shard_records.append(
            {
                "shard_index": shard_index,
                "path": f"shards/{shard_name}",
                "count": saved_count,
            }
        )

    metadata = {
        "sample_id": sample_ids,
        "timestep": torch.cat(timestep_batches).to(torch.float32),
        "target_psnr": torch.cat(target_psnr_batches).to(torch.float32),
        "target_speedup": torch.cat(target_speedup_batches).to(torch.float32),
        "threshold": torch.cat(threshold_batches).to(torch.float32),
        "step_index": torch.tensor(step_index_values, dtype=torch.long),
        "source_index": torch.tensor(source_index_values, dtype=torch.long),
        "shard_name": shard_name_values,
        "shard_offset": torch.tensor(shard_offset_values, dtype=torch.long),
    }
    torch.save(metadata, args.out_dir / "metadata.pt")

    latent_shape = list(torch.load(shards_dir / shard_records[0]["path"].split("/")[-1],
                                   map_location="cpu",
                                   weights_only=True).shape[1:])
    manifest = {
        "data_root": str(args.data_root),
        "num_examples": len(dataset),
        "dtype": args.dtype,
        "latent_shape": latent_shape,
        "shard_size": args.shard_size,
        "num_shards": len(shard_records),
        "shards": shard_records,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "elapsed_seconds": round(perf_counter() - t0, 3),
    }
    with (args.out_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
