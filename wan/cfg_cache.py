import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Tuple

import torch


@dataclass
class CFGCacheConfig:
    enabled: bool = False
    start: float = 0.1
    end: float = 0.9
    threshold: float = 0.05
    max_reuse: int = 3
    eps: float = 1e-6
    metric: str = "timestep_modulated_input_rel_l1"
    force_uncond_recompute_on_miss: bool = False

    def __post_init__(self):
        if not (0 <= self.start <= 1):
            raise ValueError("CFG cache start must be in [0, 1].")
        if not (0 <= self.end <= 1):
            raise ValueError("CFG cache end must be in [0, 1].")
        if self.start > self.end:
            raise ValueError("CFG cache start must be <= end.")
        if self.threshold < 0:
            raise ValueError("CFG cache threshold must be non-negative.")
        if self.max_reuse <= 0:
            raise ValueError("CFG cache max_reuse must be positive.")
        if self.eps <= 0:
            raise ValueError("CFG cache eps must be positive.")
        if self.metric not in {
                "cond_output_rel_l1",
                "timestep_modulated_input_rel_l1",
                "timestep_modulated_latent_rel_l1",
        }:
            raise ValueError(
                "CFG cache metric must be cond_output_rel_l1 or "
                "timestep_modulated_input_rel_l1.")


@dataclass
class CFGCacheState:
    cached_delta: Optional[torch.Tensor] = None
    last_full_cond: Optional[torch.Tensor] = None
    last_metric_feature: Optional[torch.Tensor] = None
    reuse_streak: int = 0
    reuse_count: int = 0
    recompute_count: int = 0
    diff_path: List[Tuple[int, float]] = field(default_factory=list)
    reuse_path: List[int] = field(default_factory=list)
    recompute_path: List[int] = field(default_factory=list)
    accumulated_diff_path: List[Tuple[int, float]] = field(default_factory=list)
    accumulated_diff: float = 0.0

    def has_history(self) -> bool:
        return (
            self.cached_delta is not None and
            self.last_full_cond is not None and
            self.last_metric_feature is not None)

    def record_reuse(self, step_index: int):
        self.reuse_streak += 1
        self.reuse_count += 1
        self.reuse_path.append(step_index)

    def record_recompute(
        self,
        step_index: int,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        metric_feature: torch.Tensor,
    ):
        self.cached_delta = self._enhance(cond - uncond).detach().clone()
        self.last_full_cond = cond.detach().clone()
        self.last_metric_feature = metric_feature.detach().clone()
        self.reuse_streak = 0
        self.recompute_count += 1
        self.recompute_path.append(step_index)

    def _enhance(self, delta: torch.Tensor) -> torch.Tensor:
        return delta


class CFGCache:
    """CFG delta cache with independent states per caller-provided key."""

    def __init__(self, config: CFGCacheConfig):
        self.config = config
        self.states: Dict[Hashable, CFGCacheState] = {}

    def state(self, key: Hashable) -> CFGCacheState:
        return self.states.setdefault(key, CFGCacheState())

    def should_reuse(
        self,
        key: Hashable,
        step_index: int,
        num_steps: int,
        cond: Optional[torch.Tensor] = None,
        metric_feature: Optional[torch.Tensor] = None,
    ) -> bool:
        state = self.state(key)
        if not self.config.enabled or not state.has_history():
            return False
        if num_steps <= 0:
            return False

        progress = step_index / num_steps
        if not (self.config.start <= progress <= self.config.end):
            return False
        if state.reuse_streak >= self.config.max_reuse:
            return False

        metric_feature = self.metric_feature(
            cond,
            metric_feature=metric_feature)
        cfg_diff = self._relative_l1(metric_feature, state.last_metric_feature)
        state.diff_path.append((step_index, cfg_diff))
        return cfg_diff < self.config.threshold

    def reuse(
        self,
        key: Hashable,
        step_index: int,
        cond: torch.Tensor,
        guide_scale: float,
    ) -> torch.Tensor:
        state = self.state(key)
        state.record_reuse(step_index)
        return cond + (guide_scale - 1.0) * state.cached_delta

    def recompute(
        self,
        key: Hashable,
        step_index: int,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guide_scale: float,
        metric_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = self.state(key)
        metric_feature = self.metric_feature(
            cond,
            metric_feature=metric_feature)
        state.record_recompute(step_index, cond, uncond, metric_feature)
        return cond + (guide_scale - 1.0) * (cond - uncond)

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = {}
        for key, state in self.states.items():
            result[str(key)] = {
                "reuse": state.reuse_count,
                "recompute": state.recompute_count,
                "reuse_path": list(state.reuse_path),
                "recompute_path": list(state.recompute_path),
                "diff_path": list(state.diff_path),
                "threshold": self.config.threshold,
                "max_reuse": self.config.max_reuse,
                "start": self.config.start,
                "end": self.config.end,
                "metric": self.config.metric,
            }
        return result

    def metric_feature(
        self,
        cond: Optional[torch.Tensor] = None,
        metric_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.metric == "cond_output_rel_l1":
            if cond is None:
                raise ValueError("cond_output_rel_l1 requires cond output.")
            return cond.detach()

        if metric_feature is None:
            raise ValueError(
                "timestep_modulated_input_rel_l1 requires a model-provided "
                "metric_feature.")
        return metric_feature.detach()

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        diff = (current - previous).abs().sum()
        denom = previous.abs().sum().clamp_min(self.config.eps)
        return (diff / denom).detach().float().item()


@dataclass
class SeaCFGCacheConfig(CFGCacheConfig):
    metric: str = "sea_filtered_input_accum_rel_l1"
    power_exp: float = 3.0
    power_const: float = 1.0
    norm_mode: str = "mean"
    ret_steps: int = 1
    cutoff_steps: int = 1

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("Sea CFG cache threshold must be non-negative.")
        if self.max_reuse <= 0:
            raise ValueError("Sea CFG cache max_reuse must be positive.")
        if self.eps <= 0:
            raise ValueError("Sea CFG cache eps must be positive.")
        if self.power_exp <= 0:
            raise ValueError("Sea CFG cache power_exp must be positive.")
        if self.power_const <= 0:
            raise ValueError("Sea CFG cache power_const must be positive.")
        if self.norm_mode not in {"mean", "peak"}:
            raise ValueError("Sea CFG cache norm_mode must be mean or peak.")
        if self.ret_steps < 0:
            raise ValueError("Sea CFG cache ret_steps must be non-negative.")
        if self.cutoff_steps < 0:
            raise ValueError("Sea CFG cache cutoff_steps must be non-negative.")
        if self.metric != "sea_filtered_input_accum_rel_l1":
            raise ValueError(
                "Sea CFG cache metric must be sea_filtered_input_accum_rel_l1.")


class SeaCFGCache(CFGCache):
    """CFG delta cache using SeaCache-style filtered accumulated distance."""

    def __init__(self, config: SeaCFGCacheConfig):
        super().__init__(config)
        self.config: SeaCFGCacheConfig = config

    def should_reuse(
        self,
        key: Hashable,
        step_index: int,
        num_steps: int,
        cond: Optional[torch.Tensor] = None,
        metric_feature: Optional[torch.Tensor] = None,
        grid_size: Optional[torch.Tensor] = None,
        scheduler_sigmas: Optional[torch.Tensor] = None,
    ) -> bool:
        state = self.state(key)
        if not self.config.enabled or num_steps <= 0:
            return False

        filtered_feature = self.metric_feature(
            metric_feature=metric_feature,
            grid_size=grid_size,
            step_index=step_index,
            num_steps=num_steps,
            scheduler_sigmas=scheduler_sigmas,
        )

        if self._must_recompute(state, step_index, num_steps):
            state.last_metric_feature = filtered_feature.detach().clone()
            state.accumulated_diff = 0.0
            return False

        cfg_diff = self._relative_l1(
            filtered_feature,
            state.last_metric_feature)
        state.accumulated_diff += cfg_diff
        state.diff_path.append((step_index, cfg_diff))
        state.accumulated_diff_path.append(
            (step_index, state.accumulated_diff))
        state.last_metric_feature = filtered_feature.detach().clone()

        if state.accumulated_diff < self.config.threshold:
            return True

        state.accumulated_diff = 0.0
        return False

    def recompute(
        self,
        key: Hashable,
        step_index: int,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        guide_scale: float,
        metric_feature: Optional[torch.Tensor] = None,
        grid_size: Optional[torch.Tensor] = None,
        scheduler_sigmas: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        state = self.state(key)
        if metric_feature is not None:
            if num_steps is None:
                raise ValueError("Sea CFG recompute requires num_steps.")
            filtered_feature = self.metric_feature(
                metric_feature=metric_feature,
                grid_size=grid_size,
                step_index=step_index,
                num_steps=num_steps,
                scheduler_sigmas=scheduler_sigmas,
            )
        else:
            filtered_feature = state.last_metric_feature
            if filtered_feature is None:
                raise ValueError("Sea CFG recompute requires metric_feature.")
        state.record_recompute(step_index, cond, uncond, filtered_feature)
        state.accumulated_diff = 0.0
        return cond + (guide_scale - 1.0) * (cond - uncond)

    def metric_feature(
        self,
        cond: Optional[torch.Tensor] = None,
        metric_feature: Optional[torch.Tensor] = None,
        grid_size: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        num_steps: Optional[int] = None,
        scheduler_sigmas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if metric_feature is None or grid_size is None:
            raise ValueError(
                "Sea CFG cache requires model-provided feature and grid_size.")
        if step_index is None or num_steps is None:
            raise ValueError("Sea CFG cache requires step_index and num_steps.")
        return self._filter_feature(
            metric_feature,
            grid_size,
            step_index,
            num_steps,
            scheduler_sigmas,
        )

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = super().summary()
        for key, state in self.states.items():
            result[str(key)].update({
                "accumulated_diff_path": list(state.accumulated_diff_path),
                "ret_steps": self.config.ret_steps,
                "cutoff_steps": self.config.cutoff_steps,
                "power_exp": self.config.power_exp,
                "power_const": self.config.power_const,
                "norm_mode": self.config.norm_mode,
            })
        return result

    def _must_recompute(
        self,
        state: CFGCacheState,
        step_index: int,
        num_steps: int,
    ) -> bool:
        cutoff_start = max(0, num_steps - self.config.cutoff_steps)
        if step_index < self.config.ret_steps or step_index >= cutoff_start:
            return True
        if not state.has_history():
            return True
        return state.last_metric_feature is None

    def _filter_feature(
        self,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
        step_index: int,
        num_steps: int,
        scheduler_sigmas: Optional[torch.Tensor],
    ) -> torch.Tensor:
        feature_5d = self._reshape_to_grid(feature, grid_size)
        a, b = self._ab_from_flow_scheduler(step_index, num_steps,
                                            scheduler_sigmas)
        filtered = self._apply_sea_from_ab(feature_5d, a, b, dims=(-2, -3, -4))
        return filtered.reshape(filtered.shape[0], -1,
                                filtered.shape[-1]).detach()

    def _reshape_to_grid(
        self,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
    ) -> torch.Tensor:
        f, h, w = [int(v) for v in grid_size.detach().cpu().tolist()]
        return feature.reshape(feature.shape[0], f, h, w, feature.shape[-1])

    def _ab_from_flow_scheduler(
        self,
        step_index: int,
        num_steps: int,
        scheduler_sigmas: Optional[torch.Tensor],
    ) -> Tuple[float, float]:
        if scheduler_sigmas is not None:
            sigma = float(scheduler_sigmas[step_index].detach().cpu().item())
        else:
            sigma = 1.0 - (step_index + 1) / float(num_steps)
        sigma = max(1e-6, min(1.0 - 1e-6, sigma))
        return 1.0 - sigma, sigma

    def _apply_sea_from_ab(
        self,
        x: torch.Tensor,
        a: float,
        b: float,
        dims: Tuple[int, ...],
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.contiguous().to(torch.float32)
        spectrum = torch.fft.fftn(x32, dim=dims)
        gain = None
        for axis in dims:
            axis_len = x32.shape[axis]
            freq = torch.fft.fftfreq(
                axis_len,
                device=x32.device,
                dtype=torch.float32,
            )
            rad = torch.abs(freq)
            signal_power = self.config.power_const / (
                (rad ** self.config.power_exp) + self.config.eps)
            axis_gain = (a * signal_power) / (
                (a * a * signal_power) + (b * b) + self.config.eps)
            axis_shape = [1] * x32.ndim
            axis_shape[axis] = axis_gain.shape[0]
            axis_gain = axis_gain.reshape(axis_shape)
            gain = axis_gain if gain is None else gain * axis_gain

        if self.config.norm_mode == "peak":
            maxv = torch.amax(gain)
            if torch.isfinite(maxv) and maxv > 0:
                gain = gain / maxv
        else:
            meanv = torch.mean(gain)
            if torch.isfinite(meanv) and meanv > 0:
                gain = gain / meanv

        filtered = torch.fft.ifftn(spectrum * gain, dim=dims).real
        return filtered.to(orig_dtype)

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        diff = (current.float() - previous.float()).abs().mean()
        denom = previous.float().abs().mean().clamp_min(self.config.eps)
        value = diff / denom
        if not torch.isfinite(value):
            return math.inf
        return value.detach().float().item()
