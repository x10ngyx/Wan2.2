import gc
import logging
import math
import random
import sys
import time
from contextlib import contextmanager

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from wan.text2video import WanT2V
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from .cache import TaylorSeerCache, TaylorSeerConfig
from .patch import install_taylorseer, set_taylorseer_context


class TaylorSeerWanT2V(WanT2V):
    """Standalone Wan2.2 T2V runner with official-style TaylorSeer blocks."""

    def __init__(self, *args, taylorseer_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.taylorseer_cache = TaylorSeerCache(
            taylorseer_config or TaylorSeerConfig())
        install_taylorseer(self.high_noise_model, self.taylorseer_cache, "high")
        install_taylorseer(self.low_noise_model, self.taylorseer_cache, "low")

    def generate(
        self,
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
    ):
        self.taylorseer_cache.reset()
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.patch_size[1] * self.patch_size[2]) *
            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.low_noise_model, 'no_sync', noop_no_sync)

        def sync_cuda():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

        compute_elapsed = 0.0
        weight_transfer_elapsed = 0.0
        boundary = self.boundary
        if not isinstance(guide_scale, (tuple, list)):
            guide_scale = (guide_scale, guide_scale)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
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

            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            for step_index, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = torch.stack([t])
                model_stage = 'high' if t.item() >= boundary else 'low'

                transfer_start = time.perf_counter()
                model = self._prepare_model_for_timestep(t, boundary, offload_model)
                sync_cuda()
                weight_transfer_elapsed += time.perf_counter() - transfer_start

                compute_start = time.perf_counter()
                sample_guide_scale = (
                    guide_scale[1] if t.item() >= boundary else guide_scale[0])

                def branch_forward(stream, kwargs):
                    step_type = self.taylorseer_cache.begin_branch(
                        model_stage,
                        stream,
                        step_index,
                        len(timesteps),
                    )
                    set_taylorseer_context(model, stream, step_type)
                    return model(latent_model_input, t=timestep, **kwargs)[0]

                noise_pred_cond = branch_forward('cond_stream', arg_c)
                noise_pred_uncond = branch_forward('uncond_stream', arg_null)
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

            if self.rank == 0:
                logging.info(
                    f"TaylorSeer cache summary: {self.taylorseer_cache.summary()}")

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
                logging.info(
                    f"inference_compute_elapsed_seconds={compute_elapsed:.3f}")
                logging.info(
                    "inference_weight_transfer_elapsed_seconds="
                    f"{weight_transfer_elapsed:.3f}")

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
