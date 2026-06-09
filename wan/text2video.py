# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .block_cache import BWBlockCache, BWBlockCacheConfig
from .block_group_cache import BlockGroupCache, BlockGroupCacheConfig
from .cfg_cache import CFGCache, CFGCacheConfig
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .timestep_cache import (
    ZeusThresholdTimestepCache,
    ZeusThresholdTimestepCacheConfig,
    ZeusTimestepCache,
)
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 timestep_cache_config=None,
                 block_cache_config=None,
                 block_group_cache_config=None,
                 cfg_cache_config=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM
            timestep_cache_config (`ZeusTimestepCacheConfig`, *optional*, defaults to None):
                ZEUS-style timestep cache configuration.
            block_cache_config (`BWBlockCacheConfig`, *optional*, defaults to None):
                BWCache-style block cache configuration. It is evaluated only
                when the timestep cache does not reuse the whole model output.
            block_group_cache_config (`BlockGroupCacheConfig`, *optional*, defaults to None):
                Threshold-based grouped block cache configuration.
            cfg_cache_config (`CFGCacheConfig`, *optional*, defaults to None):
                CFG delta cache configuration. It is evaluated as the outer
                cond/uncond branch-selection logic.

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        compute_elapsed = 0.0
        weight_transfer_elapsed = 0.0

        def sync_cuda():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

        if not self.t5_cpu:
            transfer_start = time.perf_counter()
            self.text_encoder.model.to(self.device)
            sync_cuda()
            weight_transfer_elapsed += time.perf_counter() - transfer_start

            compute_start = time.perf_counter()
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            sync_cuda()
            compute_elapsed += time.perf_counter() - compute_start

            if offload_model:
                transfer_start = time.perf_counter()
                self.text_encoder.model.cpu()
                sync_cuda()
                weight_transfer_elapsed += time.perf_counter() - transfer_start
        else:
            compute_start = time.perf_counter()
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            sync_cuda()
            compute_elapsed += time.perf_counter() - compute_start

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            timestep_cache = None
            timestep_cache_is_threshold = isinstance(
                timestep_cache_config, ZeusThresholdTimestepCacheConfig)
            if timestep_cache_config is not None and timestep_cache_config.enabled:
                if timestep_cache_is_threshold:
                    timestep_cache = ZeusThresholdTimestepCache(timestep_cache_config)
                else:
                    timestep_cache = ZeusTimestepCache(timestep_cache_config)

            block_cache = None
            if block_cache_config is not None and block_cache_config.enabled:
                block_cache = BWBlockCache(block_cache_config)

            block_group_cache = None
            if block_group_cache_config is not None and block_group_cache_config.enabled:
                block_group_cache = BlockGroupCache(block_group_cache_config)

            cfg_cache = None
            if cfg_cache_config is not None and cfg_cache_config.enabled:
                cfg_cache = CFGCache(cfg_cache_config)

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            def model_forward(
                model,
                latent_input,
                timestep_tensor,
                model_stage,
                branch,
                kwargs,
                force_recompute=False,
            ):
                return model(
                    latent_input,
                    t=timestep_tensor,
                    block_cache=block_cache,
                    block_cache_key=(model_stage, branch),
                    block_cache_step_index=step_index,
                    block_cache_num_steps=len(timesteps),
                    block_cache_force_recompute=force_recompute,
                    block_group_cache=block_group_cache,
                    block_group_cache_key=(model_stage, branch),
                    block_group_cache_step_index=step_index,
                    block_group_cache_num_steps=len(timesteps),
                    block_group_cache_force_recompute=force_recompute,
                    **kwargs)[0]

            previous_block_cache_stage = None
            for step_index, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                transfer_start = time.perf_counter()
                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sync_cuda()
                weight_transfer_elapsed += time.perf_counter() - transfer_start

                compute_start = time.perf_counter()
                model_stage = 'high' if t.item() >= boundary else 'low'
                if (previous_block_cache_stage is not None and
                        previous_block_cache_stage != model_stage):
                    if block_cache is not None:
                        block_cache.clear_stage(previous_block_cache_stage)
                    if block_group_cache is not None:
                        block_group_cache.clear_stage(previous_block_cache_stage)
                    logging.info(
                        f"Cleared block cache state for completed {previous_block_cache_stage} stage.")
                previous_block_cache_stage = model_stage
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                def branch_forward(branch, kwargs, force_recompute=False):
                    cache_key = (model_stage, branch)
                    if timestep_cache is None:
                        return model_forward(
                            model,
                            latent_model_input,
                            timestep,
                            model_stage,
                            branch,
                            kwargs,
                            force_recompute=force_recompute)
                    if timestep_cache_is_threshold:
                        return timestep_cache.call(
                            cache_key,
                            step_index,
                            lambda: model_forward(
                                model,
                                latent_model_input,
                                timestep,
                                model_stage,
                                branch,
                                kwargs,
                                force_recompute=force_recompute),
                            latent=latent_model_input[0],
                            force_recompute=force_recompute)
                    return timestep_cache.call(
                        cache_key,
                        step_index,
                        lambda: model_forward(
                            model,
                            latent_model_input,
                            timestep,
                            model_stage,
                            branch,
                            kwargs,
                            force_recompute=force_recompute),
                        force_recompute=force_recompute)

                noise_pred_cond = branch_forward('cond', arg_c)
                cfg_key = model_stage
                if cfg_cache is not None and cfg_cache.should_reuse(
                        cfg_key,
                        step_index,
                        len(timesteps),
                        noise_pred_cond):
                    noise_pred = cfg_cache.reuse(
                        cfg_key,
                        step_index,
                        noise_pred_cond,
                        sample_guide_scale)
                else:
                    force_uncond_recompute = (
                        cfg_cache is not None and
                        cfg_cache.config.force_uncond_recompute_on_miss)
                    noise_pred_uncond = branch_forward(
                        'uncond',
                        arg_null,
                        force_recompute=force_uncond_recompute)
                    if cfg_cache is not None:
                        noise_pred = cfg_cache.recompute(
                            cfg_key,
                            step_index,
                            noise_pred_cond,
                            noise_pred_uncond,
                            sample_guide_scale)
                    else:
                        noise_pred = noise_pred_uncond + sample_guide_scale * (
                            noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
                sync_cuda()
                compute_elapsed += time.perf_counter() - compute_start

            if timestep_cache is not None and self.rank == 0:
                logging.info(f"ZEUS timestep cache summary: {timestep_cache.summary()}")
            if block_cache is not None and self.rank == 0:
                logging.info(f"BWCache block cache summary: {block_cache.summary()}")
            if block_group_cache is not None and self.rank == 0:
                logging.info(f"Block-group cache summary: {block_group_cache.summary()}")
            if cfg_cache is not None and self.rank == 0:
                logging.info(f"CFG cache summary: {cfg_cache.summary()}")

            x0 = latents
            if offload_model:
                transfer_start = time.perf_counter()
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
                sync_cuda()
                weight_transfer_elapsed += time.perf_counter() - transfer_start
            if self.rank == 0:
                compute_start = time.perf_counter()
                videos = self.vae.decode(x0)
                sync_cuda()
                compute_elapsed += time.perf_counter() - compute_start
                logging.info(f"inference_compute_elapsed_seconds={compute_elapsed:.3f}")
                logging.info(f"inference_weight_transfer_elapsed_seconds={weight_transfer_elapsed:.3f}")

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
