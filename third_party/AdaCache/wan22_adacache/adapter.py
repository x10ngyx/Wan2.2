"""Runtime AdaCache adapter for Wan2.2 without editing Wan2.2 source files.

This module mirrors the official AdaCache implementation style:

* every transformer block owns cached residuals;
* the selected residual type only decides the adaptive schedule metric;
* the schedule maps residual change to the next cache interval via a codebook;
* first, penultimate, and final denoising steps are forced to recompute;
* cache state is keyed by the explicit Wan2.2 `(model_stage, branch)` key;
* switching low/high model stage clears the previous stage, so the next stage
  starts cold.

Wan2.2 does not expose Open-Sora's separate spatial/temporal blocks. For Wan2.2
both official `t-attn` and `s-attn` options map to the block self-attention
residual. The official `ca-mlp` option maps to cross-attention plus FFN residual.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Hashable, List, Optional, Tuple

import torch


_PATCHED = False
_ORIGINAL_MODEL_FORWARD = None
_ORIGINAL_BLOCK_FORWARD = None
_GLOBAL_CACHE = None
_CURRENT_CONTEXT = None


def _parse_codebook(value: str) -> Dict[float, int]:
    result: Dict[float, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, rate = item.partition(":")
        if not sep:
            raise ValueError(
                f"Invalid AdaCache codebook item {item!r}; expected threshold:rate.")
        threshold = float(key)
        interval = int(rate)
        if threshold <= 0:
            raise ValueError("AdaCache codebook thresholds must be positive.")
        if interval <= 0:
            raise ValueError("AdaCache codebook intervals must be positive.")
        result[threshold] = interval
    if not result:
        raise ValueError("AdaCache codebook cannot be empty.")
    return dict(sorted(result.items()))


@dataclasses.dataclass
class AdaCacheConfig:
    enabled: bool = True
    cache_res: str = "t-attn"
    cache_loc: Tuple[int, ...] = (13,)
    codebook: Dict[float, int] = dataclasses.field(
        default_factory=lambda: {
            0.03: 12,
            0.05: 10,
            0.07: 8,
            0.09: 6,
            0.11: 4,
            1.00: 3,
        })
    apply_moreg: bool = False
    moreg_strides: Tuple[int, ...] = (1,)
    moreg_steps: Tuple[int, int] = (10, 90)
    moreg_hyp: Tuple[float, float, float, float] = (0.385, 8.0, 1.0, 2.0)
    mograd_mul: float = 10.0
    cache_dtype: str = "input"
    norm_ord: int = 1

    @classmethod
    def from_strings(
        cls,
        cache_res: str = "t-attn",
        cache_loc: str = "13",
        codebook: str = "0.03:12,0.05:10,0.07:8,0.09:6,0.11:4,1.0:3",
        apply_moreg: bool = False,
        moreg_strides: str = "1",
        moreg_steps: str = "10,90",
        moreg_hyp: str = "0.385,8,1,2",
        mograd_mul: float = 10.0,
        cache_dtype: str = "input",
    ) -> "AdaCacheConfig":
        return cls(
            cache_res=cache_res,
            cache_loc=tuple(int(x) for x in cache_loc.split(",") if x.strip()),
            codebook=_parse_codebook(codebook),
            apply_moreg=apply_moreg,
            moreg_strides=tuple(
                int(x) for x in moreg_strides.split(",") if x.strip()),
            moreg_steps=tuple(
                int(x) for x in moreg_steps.split(",") if x.strip()),  # type: ignore[arg-type]
            moreg_hyp=tuple(
                float(x) for x in moreg_hyp.split(",") if x.strip()),  # type: ignore[arg-type]
            mograd_mul=mograd_mul,
            cache_dtype=cache_dtype,
        )

    def __post_init__(self):
        if self.cache_res not in {"t-attn", "s-attn", "self-attn", "ca-mlp"}:
            raise ValueError(
                "AdaCache cache_res must be t-attn, s-attn, self-attn, or ca-mlp.")
        if not self.cache_loc:
            raise ValueError("AdaCache cache_loc must contain at least one block index.")
        if any(i < 0 for i in self.cache_loc):
            raise ValueError("AdaCache cache_loc entries must be non-negative.")
        if not self.codebook:
            raise ValueError("AdaCache codebook cannot be empty.")
        if not self.moreg_strides:
            raise ValueError("AdaCache moreg_strides cannot be empty.")
        if len(self.moreg_steps) != 2:
            raise ValueError("AdaCache moreg_steps must contain two integers.")
        if len(self.moreg_hyp) != 4:
            raise ValueError("AdaCache moreg_hyp must contain four numbers.")
        if self.cache_dtype not in {"input", "bf16", "fp16", "fp32"}:
            raise ValueError("AdaCache cache_dtype must be input, bf16, fp16, or fp32.")


@dataclasses.dataclass
class _BlockState:
    self_attn_cache: Optional[torch.Tensor] = None
    ca_mlp_cache: Optional[torch.Tensor] = None


@dataclasses.dataclass
class _KeyState:
    num_blocks: int
    block_states: List[_BlockState]
    cache_rate: int = 1
    new_cache_rate: int = 1
    prev_compute_step: int = 0
    prev_moreg: float = 1.0
    avg_diff: Optional[torch.Tensor] = None
    count: int = 0
    reuse_count: int = 0
    recompute_count: int = 0
    reuse_path: List[int] = dataclasses.field(default_factory=list)
    recompute_path: List[int] = dataclasses.field(default_factory=list)
    diff_path: List[Tuple[int, float]] = dataclasses.field(default_factory=list)
    rate_path: List[Tuple[int, int]] = dataclasses.field(default_factory=list)


class _AdaCacheRuntime:
    def __init__(self, config: AdaCacheConfig):
        self.config = config
        self.enabled = True
        self.states: Dict[Hashable, _KeyState] = {}
        self.archived_summaries: Dict[str, Dict[str, object]] = {}
        self.previous_stage: Optional[str] = None

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if not enabled:
            self.states.clear()
            self.archived_summaries.clear()
            self.previous_stage = None

    def begin_forward(
        self,
        model,
        key: Hashable,
        step_index: int,
        num_steps: int,
        force_recompute: bool,
    ) -> Dict[str, object]:
        if not isinstance(key, tuple) or len(key) < 2:
            raise ValueError(
                "Wan2.2 AdaCache requires explicit key `(model_stage, branch)`.")
        model_stage = key[0]
        if self.previous_stage is not None and self.previous_stage != model_stage:
            self.clear_stage(self.previous_stage)
        self.previous_stage = model_stage

        num_blocks = len(model.blocks)
        state = self.states.get(key)
        cold_start = state is None or state.num_blocks != num_blocks
        if cold_start:
            state = _KeyState(
                num_blocks=num_blocks,
                block_states=[_BlockState() for _ in range(num_blocks)],
            )
            self.states[key] = state

        for index, block in enumerate(model.blocks):
            setattr(block, "_wan22_adacache_block_index", index)

        state.avg_diff = None
        state.count = 0
        fwd_id = step_index + 1
        force_steps = {1, max(1, num_steps - 1), num_steps}
        next_step = state.prev_compute_step + state.cache_rate
        active = cold_start or force_recompute or fwd_id in force_steps or fwd_id == next_step
        return {
            "key": key,
            "state": state,
            "fwd_id": fwd_id,
            "num_steps": num_steps,
            "active": active,
        }

    def finish_forward(self, context: Dict[str, object]):
        state = context["state"]
        assert isinstance(state, _KeyState)
        state.cache_rate = state.new_cache_rate

    def clear_stage(self, model_stage: str):
        for key in list(self.states):
            if isinstance(key, tuple) and key and key[0] == model_stage:
                self.archived_summaries[str(key)] = self._summary_for_state(
                    self.states[key])
                del self.states[key]

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = dict(self.archived_summaries)
        for key, state in self.states.items():
            result[str(key)] = self._summary_for_state(state)
        return result

    def _summary_for_state(self, state: _KeyState) -> Dict[str, object]:
        return {
            "reuse": state.reuse_count,
            "recompute": state.recompute_count,
            "reuse_path": list(state.reuse_path),
            "recompute_path": list(state.recompute_path),
            "diff_path": list(state.diff_path),
            "rate_path": list(state.rate_path),
            "cache_res": self.config.cache_res,
            "cache_loc": list(self.config.cache_loc),
            "codebook": dict(self.config.codebook),
            "apply_moreg": self.config.apply_moreg,
        }


def _cache_dtype(config: AdaCacheConfig, input_dtype: torch.dtype) -> torch.dtype:
    if config.cache_dtype == "input":
        return input_dtype
    if config.cache_dtype == "bf16":
        return torch.bfloat16
    if config.cache_dtype == "fp16":
        return torch.float16
    return torch.float32


def _store_residual(residual: torch.Tensor, config: AdaCacheConfig) -> torch.Tensor:
    return residual.detach().to(dtype=_cache_dtype(config, residual.dtype)).clone()


def _compute_moreg(
    residual: torch.Tensor,
    spatial_dim: int,
    state: _KeyState,
    config: AdaCacheConfig,
    fwd_id: int,
    prev_rate: int,
) -> float:
    if not config.apply_moreg:
        return 1.0
    if not (config.moreg_steps[0] <= fwd_id <= config.moreg_steps[1]):
        moreg = 1.0
    else:
        moreg_tensor = residual.new_tensor(0.0, dtype=torch.float32)
        for stride in config.moreg_strides:
            offset = stride * spatial_dim
            if offset >= residual.shape[1]:
                continue
            left = residual[:, offset:, :].float()
            right = residual[:, :-offset, :].float()
            denom = left.norm(p=config.norm_ord) + right.norm(p=config.norm_ord)
            if denom.item() == 0:
                continue
            moreg_tensor = moreg_tensor + (left - right).norm(
                p=config.norm_ord) / denom
        moreg_tensor = moreg_tensor / max(1, len(config.moreg_strides))
        moreg = float(
            (((1.0 / config.moreg_hyp[0] * moreg_tensor)**
              config.moreg_hyp[1]) / config.moreg_hyp[2]).detach().cpu())

    mograd = config.mograd_mul * (moreg - state.prev_moreg) / max(1, prev_rate)
    state.prev_moreg = moreg
    return moreg + abs(mograd)


def _update_schedule(
    previous: torch.Tensor,
    current: torch.Tensor,
    state: _KeyState,
    config: AdaCacheConfig,
    fwd_id: int,
    spatial_dim: int,
):
    prev_rate = max(1, state.cache_rate)
    diff = (
        (previous.float() - current.float()).norm(p=config.norm_ord) /
        previous.float().norm(p=config.norm_ord).clamp_min(1e-12))
    diff = diff / prev_rate

    if state.avg_diff is None:
        avg = diff
    else:
        avg = (diff + state.count * state.avg_diff) / (state.count + 1)
    state.avg_diff = avg.detach()
    state.count += 1

    moreg = _compute_moreg(current, spatial_dim, state, config, fwd_id, prev_rate)
    metric = avg * moreg
    metric_float = float(metric.detach().cpu())

    thresholds = list(config.codebook.keys())
    rates = list(config.codebook.values())
    new_rate = rates[-1]
    for threshold, rate in zip(thresholds, rates):
        if metric_float < threshold:
            new_rate = rate
            break
    state.new_cache_rate = int(new_rate)
    state.diff_path.append((fwd_id - 1, metric_float))
    state.rate_path.append((fwd_id - 1, int(new_rate)))


def _patched_block_forward(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
    bwcache_state=None,
    bwcache_block_index=None,
    bwcache_cal_list_updated=True,
    bwcache_is_first_step=True,
    bwcache_metric="pooled_rel_l1",
):
    if _CURRENT_CONTEXT is None or bwcache_state is not None:
        return _ORIGINAL_BLOCK_FORWARD(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            bwcache_state=bwcache_state,
            bwcache_block_index=bwcache_block_index,
            bwcache_cal_list_updated=bwcache_cal_list_updated,
            bwcache_is_first_step=bwcache_is_first_step,
            bwcache_metric=bwcache_metric,
        )

    runtime = _GLOBAL_CACHE
    assert runtime is not None
    config = runtime.config
    state = _CURRENT_CONTEXT["state"]
    assert isinstance(state, _KeyState)
    block_index = getattr(self, "_wan22_adacache_block_index")
    block_state = state.block_states[block_index]
    fwd_id = int(_CURRENT_CONTEXT["fwd_id"])
    active = bool(_CURRENT_CONTEXT["active"])

    assert e.dtype == torch.float32
    with torch.amp.autocast("cuda", dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    assert e[0].dtype == torch.float32

    if active or block_state.self_attn_cache is None:
        x_m = self._modulated_norm1(x, e)
        y = self.self_attn(x_m, seq_lens, grid_sizes, freqs)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            self_attn_res = y * e[2].squeeze(2)
            x = x + self_attn_res

        if (block_state.self_attn_cache is not None and
                block_index in config.cache_loc and
                config.cache_res in {"t-attn", "s-attn", "self-attn"}):
            spatial_dim = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
            _update_schedule(
                block_state.self_attn_cache,
                self_attn_res,
                state,
                config,
                fwd_id,
                spatial_dim,
            )
        block_state.self_attn_cache = _store_residual(self_attn_res, config)
    else:
        x = x + block_state.self_attn_cache.to(device=x.device, dtype=x.dtype)

    if active or block_state.ca_mlp_cache is None:
        x_before = x
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(
            self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
            ca_mlp_res = x - x_before

        if (block_state.ca_mlp_cache is not None and
                block_index in config.cache_loc and
                config.cache_res == "ca-mlp"):
            spatial_dim = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
            _update_schedule(
                block_state.ca_mlp_cache,
                ca_mlp_res,
                state,
                config,
                fwd_id,
                spatial_dim,
            )
        block_state.ca_mlp_cache = _store_residual(ca_mlp_res, config)
    else:
        x = x + block_state.ca_mlp_cache.to(device=x.device, dtype=x.dtype)

    if block_index == state.num_blocks - 1:
        if active:
            state.recompute_count += 1
            state.recompute_path.append(fwd_id - 1)
            state.prev_compute_step = fwd_id
        else:
            state.reuse_count += 1
            state.reuse_path.append(fwd_id - 1)

    return x


def _patched_model_forward(self, *args, **kwargs):
    global _CURRENT_CONTEXT
    if (_GLOBAL_CACHE is None or not _GLOBAL_CACHE.enabled or
            not _GLOBAL_CACHE.config.enabled):
        return _ORIGINAL_MODEL_FORWARD(self, *args, **kwargs)

    key = kwargs.get("block_cache_key")
    step_index = kwargs.get("block_cache_step_index")
    num_steps = kwargs.get("block_cache_num_steps")
    if key is None or step_index is None or num_steps is None:
        return _ORIGINAL_MODEL_FORWARD(self, *args, **kwargs)

    context = _GLOBAL_CACHE.begin_forward(
        self,
        key=key,
        step_index=int(step_index),
        num_steps=int(num_steps),
        force_recompute=bool(kwargs.get("block_cache_force_recompute", False)),
    )
    old_context = _CURRENT_CONTEXT
    _CURRENT_CONTEXT = context
    try:
        result = _ORIGINAL_MODEL_FORWARD(self, *args, **kwargs)
    finally:
        _GLOBAL_CACHE.finish_forward(context)
        _CURRENT_CONTEXT = old_context
    return result


def enable_wan22_adacache(config: AdaCacheConfig):
    """Enable the Wan2.2 AdaCache monkey patch for the current Python process."""
    global _PATCHED, _ORIGINAL_MODEL_FORWARD, _ORIGINAL_BLOCK_FORWARD, _GLOBAL_CACHE

    from wan.modules.model import WanAttentionBlock, WanModel

    _GLOBAL_CACHE = _AdaCacheRuntime(config)
    if not _PATCHED:
        _ORIGINAL_MODEL_FORWARD = WanModel.forward
        _ORIGINAL_BLOCK_FORWARD = WanAttentionBlock.forward
        WanModel.forward = _patched_model_forward
        WanAttentionBlock.forward = _patched_block_forward
        _PATCHED = True
    return _GLOBAL_CACHE
