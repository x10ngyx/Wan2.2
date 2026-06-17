#!/usr/bin/env python3
"""Run Wan2.2 T2V with the AdaCache runtime adapter.

This wrapper keeps Wan2.2 source files unchanged. It consumes AdaCache-specific
arguments, enables the runtime monkey patch, removes unsupported arguments from
`sys.argv`, and then delegates to Wan2.2's normal `generate.py`.
"""

from __future__ import annotations

import argparse
import os
import sys


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}.")


def _consume_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--block_cache", choices=["adacache"])
    parser.add_argument("--adacache_res", default="t-attn")
    parser.add_argument("--adacache_cache_res", dest="adacache_res")
    parser.add_argument("--adacache_loc", default="13")
    parser.add_argument("--adacache_cache_loc", dest="adacache_loc")
    parser.add_argument(
        "--adacache_codebook",
        default="0.03:12,0.05:10,0.07:8,0.09:6,0.11:4,1.0:3",
    )
    parser.add_argument("--adacache_moreg", action="store_true")
    parser.add_argument("--adacache_apply_moreg", type=_str2bool, default=None)
    parser.add_argument("--adacache_moreg_strides", default="1")
    parser.add_argument("--adacache_moreg_steps", default="10,90")
    parser.add_argument("--adacache_moreg_hyp", default="0.385,8,1,2")
    parser.add_argument("--adacache_mograd_mul", type=float, default=10.0)
    parser.add_argument(
        "--adacache_cache_dtype",
        default="input",
        choices=["input", "bf16", "fp16", "fp32"],
    )
    args, remaining = parser.parse_known_args(argv)
    enabled = args.block_cache == "adacache"
    return enabled, args, remaining


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    adapter_root = os.path.abspath(os.path.dirname(__file__))
    if adapter_root not in sys.path:
        sys.path.insert(0, adapter_root)

    enabled, adacache_args, remaining = _consume_wrapper_args(sys.argv[1:])
    if enabled:
        from wan22_adacache import AdaCacheConfig, enable_wan22_adacache

        apply_moreg = adacache_args.adacache_moreg
        if adacache_args.adacache_apply_moreg is not None:
            apply_moreg = adacache_args.adacache_apply_moreg
        config = AdaCacheConfig.from_strings(
            cache_res=adacache_args.adacache_res,
            cache_loc=adacache_args.adacache_loc,
            codebook=adacache_args.adacache_codebook,
            apply_moreg=apply_moreg,
            moreg_strides=adacache_args.adacache_moreg_strides,
            moreg_steps=adacache_args.adacache_moreg_steps,
            moreg_hyp=adacache_args.adacache_moreg_hyp,
            mograd_mul=adacache_args.adacache_mograd_mul,
            cache_dtype=adacache_args.adacache_cache_dtype,
        )
        runtime = enable_wan22_adacache(config)
    else:
        runtime = None

    sys.argv = [os.path.join(repo_root, "generate.py"), *remaining]
    import generate

    args = generate._parse_args()
    generate.generate(args)
    if runtime is not None:
        import logging

        logging.info("Wan2.2 AdaCache summary: %s", runtime.summary())


if __name__ == "__main__":
    main()
