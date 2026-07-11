from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from data import load_or_compute_latent_mse, load_summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cached tensors needed by offline IQL training."
    )
    parser.add_argument(
        "--feature_cache",
        type=Path,
        default=Path("/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409"),
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/hy-tmp/openvid_100_seacache_trace_data"),
    )
    parser.add_argument(
        "--latent_mse_cache",
        type=Path,
        default=None,
        help="Defaults to <feature_cache>/latent_mse_to_baseline.pt.",
    )
    parser.add_argument(
        "--raw_latent_cache",
        type=Path,
        default=Path("/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805"),
        help="Packed candidate latent cache used for fast latent-MSE preparation.",
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite an existing latent MSE cache.",
    )
    return parser.parse_args()


class PackedLatentReader:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.shards_dir = cache_dir / "shards"
        self.metadata = torch.load(
            cache_dir / "metadata.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.shard_names = self.metadata["shard_name"]
        self.shard_offsets = self.metadata["shard_offset"].long()
        self._loaded_name: str | None = None
        self._loaded_tensor: torch.Tensor | None = None

    def latent(self, index: int) -> torch.Tensor:
        shard_name = self.shard_names[index]
        if shard_name != self._loaded_name:
            self._loaded_tensor = torch.load(
                self.shards_dir / shard_name,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            self._loaded_name = shard_name
        assert self._loaded_tensor is not None
        return self._loaded_tensor[int(self.shard_offsets[index].item())]


def load_baseline_next_latents(step_root: Path, num_steps: int) -> list[torch.Tensor]:
    latents = []
    for step in range(num_steps - 1):
        payload = torch.load(
            step_root / f"step_{step + 1:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        latents.append(payload["latent"].float())
    done = torch.load(
        step_root / "trace_done.pt",
        map_location="cpu",
        weights_only=True,
    )
    latents.append(done["final_latent"].float())
    return latents


def compute_latent_mse_fast(
    *,
    data_root: Path,
    rows: list[dict[str, str]],
    feature_metadata: dict[str, object],
    raw_latent_cache: Path,
    cache_path: Path,
    max_examples: int | None,
    progress_every: int,
    save_every: int,
) -> torch.Tensor:
    source_index = feature_metadata["source_index"].long()  # type: ignore[union-attr]
    step_indices = feature_metadata["step_index"].long()  # type: ignore[union-attr]
    expected_count = int(source_index.numel())
    if max_examples is not None:
        expected_count = min(expected_count, max_examples)
        source_index = source_index[:expected_count]
        step_indices = step_indices[:expected_count]
    num_steps = int(step_indices.max().item()) + 1

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True).float()
        if int(cached.numel()) >= expected_count:
            return cached[:expected_count]
        values = torch.empty(expected_count, dtype=torch.float32)
        values[: int(cached.numel())] = cached
        start_index = int(cached.numel())
    else:
        values = torch.empty(expected_count, dtype=torch.float32)
        start_index = 0

    reader = PackedLatentReader(raw_latent_cache)
    current_sample_id: str | None = None
    baseline_next_latents: list[torch.Tensor] | None = None
    chunk_steps = 10
    start_candidate = start_index // num_steps
    candidate_total = (expected_count + num_steps - 1) // num_steps

    for candidate_index in range(start_candidate, candidate_total):
        row = rows[candidate_index]
        sample_id = row["sample_id"]
        if sample_id != current_sample_id:
            baseline_next_latents = load_baseline_next_latents(
                data_root / row["baseline_step_inputs"],
                num_steps,
            )
            current_sample_id = sample_id
        assert baseline_next_latents is not None
        candidate_start = candidate_index * num_steps
        candidate_stop = min(candidate_start + num_steps, expected_count)
        step_start = 0
        if candidate_index == start_candidate:
            step_start = start_index - candidate_start
        for chunk_start in range(step_start, candidate_stop - candidate_start, chunk_steps):
            chunk_stop = min(chunk_start + chunk_steps, candidate_stop - candidate_start)
            candidate_latents = []
            baseline_latents = []
            for step in range(chunk_start, chunk_stop):
                index = candidate_start + step
                actual_step = int(step_indices[index].item())
                if actual_step + 1 < num_steps:
                    candidate_latents.append(
                        reader.latent(int(source_index[index].item()) + 1)
                    )
                else:
                    done = torch.load(
                        data_root / row["seacache_step_inputs"] / "trace_done.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    candidate_latents.append(done["final_latent"])
                baseline_latents.append(baseline_next_latents[actual_step])

            candidate_batch = torch.stack(candidate_latents).float()
            baseline_batch = torch.stack(baseline_latents).float()
            mse = (candidate_batch - baseline_batch).pow(2).flatten(1).mean(dim=1)
            values[candidate_start + chunk_start:candidate_start + chunk_stop] = mse.cpu()

            completed = candidate_start + chunk_stop
            if save_every > 0 and completed % save_every == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(values[:completed], cache_path)
            if progress_every > 0 and (
                completed % progress_every == 0 or completed == expected_count
            ):
                print(
                    json.dumps({
                        "latent_mse_completed": completed,
                        "latent_mse_total": expected_count,
                        "mode": "packed_raw_latent",
                    }),
                    flush=True,
                )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(values, cache_path)
    return values


def main() -> None:
    args = parse_args()
    latent_mse_cache = (
        args.latent_mse_cache
        if args.latent_mse_cache is not None
        else args.feature_cache / "latent_mse_to_baseline.pt"
    )
    if args.force and latent_mse_cache.exists():
        latent_mse_cache.unlink()

    metadata_path = args.feature_cache / "metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature-cache metadata: {metadata_path}")
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    step_indices = metadata["step_index"].long()
    source_index = metadata["source_index"].long()
    if args.max_examples is not None:
        step_indices = step_indices[:args.max_examples]
        source_index = source_index[:args.max_examples]
    num_steps = int(step_indices.max().item()) + 1
    rows = load_summary_rows(args.data_root)

    start = perf_counter()
    if args.raw_latent_cache.exists():
        latent_mse = compute_latent_mse_fast(
            data_root=args.data_root,
            rows=rows,
            feature_metadata=metadata,
            raw_latent_cache=args.raw_latent_cache,
            cache_path=latent_mse_cache,
            max_examples=args.max_examples,
            progress_every=args.progress_every,
            save_every=args.save_every,
        )
        mode = "packed_raw_latent"
    else:
        latent_mse = load_or_compute_latent_mse(
            data_root=args.data_root,
            rows=rows,
            source_index=source_index,
            step_indices=step_indices,
            num_steps=num_steps,
            cache_path=latent_mse_cache,
            progress_every=args.progress_every,
            save_every=args.save_every,
        )
        mode = "step_inputs"
    elapsed = perf_counter() - start
    summary = {
        "feature_cache": str(args.feature_cache),
        "data_root": str(args.data_root),
        "latent_mse_cache": str(latent_mse_cache),
        "num_examples": int(latent_mse.numel()),
        "num_steps": num_steps,
        "mode": mode,
        "raw_latent_cache": str(args.raw_latent_cache) if args.raw_latent_cache.exists() else None,
        "elapsed_seconds": round(elapsed, 3),
        "latent_mse_min": float(latent_mse.min().item()),
        "latent_mse_mean": float(latent_mse.mean().item()),
        "latent_mse_max": float(latent_mse.max().item()),
    }
    summary_path = latent_mse_cache.with_suffix(".json")
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
