from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

import generate as wan_generate
import wan.text2video as wan_text2video

from adaptive_seacache_wan22.cache import (
    AdaptiveSeaCacheGateConfig,
    build_adaptive_seacache_factory,
)
from adaptive_seacache_wan22.patch import patch_wan_model_forward_for_adaptive_seacache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--adaptive_gate_model",
        type=Path,
        default=None,
        help="Path to a trained adaptive predictor best_model.pt.",
    )
    parser.add_argument(
        "--target_psnr",
        type=float,
        default=None,
        help="Target PSNR conditioning value passed to the adaptive threshold predictor.",
    )
    parser.add_argument(
        "--adaptive_feature_set",
        choices=("temporal_mean", "latent_pool"),
        default="temporal_mean",
        help="Online latent feature extraction used by the adaptive predictor.",
    )
    parser.add_argument(
        "--adaptive_hidden_dim",
        type=int,
        default=16,
        help="Hidden dimension used by the trained adaptive predictor.",
    )
    parser.add_argument(
        "--adaptive_feature_dim",
        type=int,
        default=128,
        help="Flattened pooled feature dimension used by the trained predictor.",
    )
    parser.add_argument(
        "--adaptive_grid_size",
        nargs=3,
        type=int,
        default=(2, 2, 2),
        help="Pooled feature grid size used by the trained predictor.",
    )
    parser.add_argument(
        "--adaptive_psnr_min",
        type=float,
        default=10.0,
        help="Minimum PSNR normalization value used during predictor training.",
    )
    parser.add_argument(
        "--adaptive_psnr_max",
        type=float,
        default=50.0,
        help="Maximum PSNR normalization value used during predictor training.",
    )
    parser.add_argument(
        "--adaptive_min_threshold",
        type=float,
        default=0.0,
        help="Clamp lower bound for predicted SeaCache threshold.",
    )
    parser.add_argument(
        "--adaptive_max_threshold",
        type=float,
        default=1.0,
        help="Clamp upper bound for predicted SeaCache threshold.",
    )
    adaptive_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = wan_generate._parse_args()
    finally:
        sys.argv = original_argv

    for key, value in vars(adaptive_args).items():
        setattr(args, key, value)
    if args.adaptive_gate_model is None:
        raise ValueError("--adaptive_gate_model is required.")
    if args.target_psnr is None:
        raise ValueError("--target_psnr is required.")
    return args


def main() -> None:
    args = parse_args()
    if args.task != "t2v-A14B":
        raise NotImplementedError("Adaptive SeaCache prototype currently targets t2v-A14B.")
    if args.timestep_cache != "seacache":
        raise ValueError("Use --timestep_cache seacache for adaptive SeaCache inference.")
    if args.block_cache != "none" or args.cfg_cache != "none":
        raise ValueError("This prototype is timestep-only; use --block_cache none --cfg_cache none.")
    if args.ulysses_size > 1:
        raise NotImplementedError("Adaptive SeaCache prototype has only been prepared for single-process T2V.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate_config = AdaptiveSeaCacheGateConfig(
        model_path=args.adaptive_gate_model,
        target_psnr=args.target_psnr,
        feature_set=args.adaptive_feature_set,
        hidden_dim=args.adaptive_hidden_dim,
        feature_dim=args.adaptive_feature_dim,
        grid_size=tuple(args.adaptive_grid_size),
        psnr_min=args.adaptive_psnr_min,
        psnr_max=args.adaptive_psnr_max,
        min_threshold=args.adaptive_min_threshold,
        max_threshold=args.adaptive_max_threshold,
        device=device,
    )
    patch_wan_model_forward_for_adaptive_seacache()
    wan_text2video.SeaCacheTimestepCache = build_adaptive_seacache_factory(gate_config)
    wan_generate.generate(args)


if __name__ == "__main__":
    main()
