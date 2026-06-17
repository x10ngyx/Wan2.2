import argparse
import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist

from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool

try:
    from .cache import TaylorSeerConfig
    from .text2video import TaylorSeerWanT2V
except ImportError:
    from third_party.taylorseer_wan22.cache import TaylorSeerConfig
    from third_party.taylorseer_wan22.text2video import TaylorSeerWanT2V


EXAMPLE_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone TaylorSeer Wan2.2 T2V runner.")
    parser.add_argument("--task", type=str, default="t2v-A14B")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--frame_num", type=int, default=45)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=EXAMPLE_PROMPT)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--sample_solver", type=str, default="dpm++",
                        choices=["unipc", "dpm++"])
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument("--taylorseer_fresh_threshold", type=int, default=5)
    parser.add_argument("--taylorseer_max_order", type=int, default=1)
    parser.add_argument("--taylorseer_first_enhance", type=int, default=1)
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    assert args.ckpt_dir is not None, "Please specify --ckpt_dir."
    assert args.task == "t2v-A14B", "This standalone runner currently targets t2v-A14B."
    assert args.task in WAN_CONFIGS, f"Unsupported task: {args.task}"
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupported size: {args.size}"
    cfg = WAN_CONFIGS[args.task]
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)


def init_logging():
    rank = int(os.getenv("RANK", 0))
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])


def main():
    args = parse_args()
    init_logging()

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    device = local_rank
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed mode."
        assert args.ulysses_size == 1, "ulysses_size > 1 requires torchrun."
    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, (
            "ulysses_size must equal WORLD_SIZE for this Wan2.2 TaylorSeer runner.")
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, (
            f"`{cfg.num_heads=}` must be divisible by `{args.ulysses_size=}`.")
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info("Creating standalone TaylorSeer WanT2V pipeline.")
    pipeline = TaylorSeerWanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        taylorseer_config=TaylorSeerConfig(
            enabled=True,
            fresh_threshold=args.taylorseer_fresh_threshold,
            max_order=args.taylorseer_max_order,
            first_enhance=args.taylorseer_first_enhance,
        ),
    )

    logging.info("Generating video with standalone TaylorSeer.")
    start = time.perf_counter()
    video = pipeline.generate(
        args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )
    elapsed = time.perf_counter() - start
    logging.info(f"generation_wall_elapsed_seconds={elapsed:.3f}")

    if args.save_file is None:
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_file = f"taylorseer_t2v_{formatted_time}.mp4"
    if rank == 0:
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        logging.info(f"Saved video to {args.save_file}")
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
