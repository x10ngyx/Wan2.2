from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.adaptive_threshold_predictor.data import DEFAULT_DATA_ROOT
from experiments.adaptive_threshold_predictor.models import (
    ImprovedAdaCacheGate,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect traced Wan latent inputs.")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sample_id", default="openvidhd_part1_000")
    parser.add_argument("--step_index", type=int, default=0)
    parser.add_argument("--target_psnr", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step_path = (
        args.data_root
        / "data"
        / "baseline"
        / "step_inputs"
        / args.sample_id
        / f"step_{args.step_index:03d}.pt"
    )
    payload = torch.load(step_path, map_location="cpu")
    latent = payload["latent"]
    print(f"step_path: {step_path}")
    print(f"latent shape: {tuple(latent.shape)} dtype={latent.dtype}")
    print(f"timestep: {payload['timestep']}")
    print(f"model_stage: {payload.get('model_stage')}")
    print(f"scheduler_sigma: {payload.get('scheduler_sigma')}")

    model = ImprovedAdaCacheGate(latent_channels=latent.shape[0], hidden_dim=64)
    with torch.no_grad():
        pred = model(
            latent,
            torch.tensor([args.step_index / max(payload["num_steps"] - 1, 1)], dtype=torch.float32),
            torch.tensor([args.target_psnr], dtype=torch.float32),
        )
    print(f"model parameters: {count_parameters(model)}")
    print(f"predicted threshold shape: {tuple(pred.shape)}")
    print(f"predicted threshold range [0,1]: {pred.flatten().tolist()}")


if __name__ == "__main__":
    main()
