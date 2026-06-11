import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import torch


@dataclass
class ZeusTimestepCacheConfig:
    enabled: bool = False
    acc_range: Tuple[int, int] = (8, 47)
    denominator: int = 3
    modular: Tuple[int, ...] = (0, 1)
    caching_mode: str = "reuse_interp"
    max_interval: int = 6
    lagrange_term: int = 4
    lagrange_int: int = 4
    lagrange_step: int = 24

    def __post_init__(self):
        if self.denominator <= 0:
            raise ValueError("ZEUS denominator must be positive.")
        if any(m < 0 or m >= self.denominator for m in self.modular):
            raise ValueError("ZEUS modular entries must be in [0, denominator).")
        if self.caching_mode not in {"reuse_interp", "interp_all", "reuse_all"}:
            raise ValueError("ZEUS caching_mode must be reuse_interp, interp_all, or reuse_all.")
        if self.max_interval <= 0:
            raise ValueError("ZEUS max_interval must be positive.")
        if self.lagrange_term:
            if self.lagrange_int <= 0 or self.lagrange_step <= 0:
                raise ValueError("ZEUS lagrange_int and lagrange_step must be positive when lagrange_term is enabled.")
            if self.lagrange_step % self.lagrange_int != 0:
                raise ValueError("ZEUS lagrange_step must be divisible by lagrange_int.")


@dataclass
class ZeusTimestepCacheState:
    prev_f: List[torch.Tensor] = field(default_factory=list)
    prev_interp: Optional[torch.Tensor] = None
    cons_skip: int = 0
    reuse_count: int = 0
    recompute_count: int = 0
    skipping_path: List[int] = field(default_factory=list)
    recompute_path: List[int] = field(default_factory=list)

    def has_history(self) -> bool:
        return len(self.prev_f) >= 3 and self.prev_f[0] is not None

    def record_recompute(self, step_index: int, output: torch.Tensor) -> torch.Tensor:
        self.cons_skip = 0
        self.recompute_count += 1
        self.recompute_path.append(step_index)
        self.prev_f.append(output.clone())
        if len(self.prev_f) > 3:
            self.prev_f.pop(0)
        return output

    def record_reuse(self, step_index: int) -> torch.Tensor:
        self.cons_skip += 1
        self.reuse_count += 1
        self.skipping_path.append(step_index)
        return self._estimate()

    def _estimate(self) -> torch.Tensor:
        prev, last = self.prev_f[-2], self.prev_f[-1]
        interp = last + (last - prev)
        return interp


class ZeusTimestepCache:
    """ZEUS-style timestep cache with independent states per caller-provided key."""

    def __init__(self, config: ZeusTimestepCacheConfig):
        self.config = config
        self.states: Dict[Hashable, ZeusTimestepCacheState] = {}

    def call(
        self,
        key: Hashable,
        step_index: int,
        compute_fn: Callable[[], torch.Tensor],
        force_recompute: bool = False,
    ) -> torch.Tensor:
        state = self.states.setdefault(key, ZeusTimestepCacheState())

        if not force_recompute and self._should_skip(state, step_index):
            return self._reuse(state, step_index)

        output = compute_fn()
        return state.record_recompute(step_index, output)

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = {}
        for key, state in self.states.items():
            result[str(key)] = {
                "reuse": state.reuse_count,
                "recompute": state.recompute_count,
                "skipping_path": list(state.skipping_path),
                "recompute_path": list(state.recompute_path),
            }
        return result

    def _should_skip(self, state: ZeusTimestepCacheState, step_index: int) -> bool:
        if not self.config.enabled or not state.has_history():
            return False
        if state.cons_skip >= self.config.max_interval:
            return False

        start, end = self.config.acc_range
        if not (start <= step_index < end):
            return False

        if self.config.lagrange_term:
            if step_index < self.config.lagrange_step:
                return step_index % self.config.denominator in self.config.modular
            return step_index % self.config.lagrange_int != 0

        return step_index % self.config.denominator in self.config.modular

    def _reuse(self, state: ZeusTimestepCacheState, step_index: int) -> torch.Tensor:
        next_skip = state.cons_skip + 1
        interp = state.record_reuse(step_index)

        if self.config.caching_mode == "interp_all":
            return interp

        if self.config.caching_mode == "reuse_all":
            return state.prev_f[-1].clone()

        if next_skip == 1:
            state.prev_interp = interp.clone()
            return interp
        if next_skip % 2 == 1 and state.prev_interp is not None:
            return state.prev_interp.clone()
        return state.prev_f[-1].clone()


@dataclass
class ZeusThresholdTimestepCacheConfig:
    enabled: bool = False
    acc_range: Tuple[int, int] = (8, 47)
    caching_mode: str = "reuse_interp"
    max_interval: int = 6
    threshold: float = 0.1
    metric: str = "rel_l1"
    eps: float = 1e-6

    def __post_init__(self):
        if self.caching_mode not in {"reuse_interp", "interp_all", "reuse_all", "timestep_aware_interp"}:
            raise ValueError("ZEUS threshold caching_mode must be reuse_interp, interp_all, reuse_all, or timestep_aware_interp.")
        if self.max_interval <= 0:
            raise ValueError("ZEUS threshold max_interval must be positive.")
        if self.threshold < 0:
            raise ValueError("ZEUS threshold must be non-negative.")
        if self.metric != "rel_l1":
            raise ValueError("ZEUS threshold metric currently only supports rel_l1.")
        if self.eps <= 0:
            raise ValueError("ZEUS threshold eps must be positive.")


@dataclass
class ZeusThresholdTimestepCacheState(ZeusTimestepCacheState):
    prev_input: Optional[torch.Tensor] = None
    rel_l1_path: List[Tuple[int, float]] = field(default_factory=list)
    recompute_steps: List[int] = field(default_factory=list)

    def record_recompute(
        self,
        step_index: int,
        output: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        result = super().record_recompute(step_index, output)
        self.prev_input = latent.detach().clone()
        self.recompute_steps.append(step_index)
        if len(self.recompute_steps) > 3:
            self.recompute_steps.pop(0)
        return result


class ZeusThresholdTimestepCache(ZeusTimestepCache):
    """ZEUS-style timestep cache with latent-threshold skip decisions."""

    def __init__(self, config: ZeusThresholdTimestepCacheConfig):
        super().__init__(config)
        self.states: Dict[Hashable, ZeusThresholdTimestepCacheState] = {}

    def call(
        self,
        key: Hashable,
        step_index: int,
        compute_fn: Callable[[], torch.Tensor],
        latent: torch.Tensor,
        force_recompute: bool = False,
    ) -> torch.Tensor:
        state = self.states.setdefault(key, ZeusThresholdTimestepCacheState())

        if not force_recompute and self._should_skip_threshold(state, step_index, latent):
            return self._reuse(state, step_index)

        output = compute_fn()
        return state.record_recompute(step_index, output, latent)

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = super().summary()
        for key, state in self.states.items():
            result[str(key)].update({
                "rel_l1_path": list(state.rel_l1_path),
                "recompute_steps": list(state.recompute_steps),
                "threshold": self.config.threshold,
            })
        return result

    def _reuse(self, state: ZeusThresholdTimestepCacheState, step_index: int) -> torch.Tensor:
        if self.config.caching_mode != "timestep_aware_interp":
            return super()._reuse(state, step_index)

        state.cons_skip += 1
        state.reuse_count += 1
        state.skipping_path.append(step_index)

        if len(state.prev_f) < 2 or len(state.recompute_steps) < 2:
            return state.prev_f[-1].clone()

        prev, last = state.prev_f[-2], state.prev_f[-1]
        prev_step, last_step = state.recompute_steps[-2], state.recompute_steps[-1]
        step_delta = last_step - prev_step
        if step_delta <= 0:
            return last.clone()

        scale = (step_index - last_step) / step_delta
        return last + (last - prev) * scale

    def _should_skip_threshold(
        self,
        state: ZeusThresholdTimestepCacheState,
        step_index: int,
        latent: torch.Tensor,
    ) -> bool:
        if not self.config.enabled or not state.has_history():
            return False
        if state.cons_skip >= self.config.max_interval:
            return False

        start, end = self.config.acc_range
        if not (start <= step_index < end):
            return False

        if state.prev_input is None:
            return False

        rel_l1 = self._relative_l1(latent, state.prev_input)
        state.rel_l1_path.append((step_index, rel_l1))
        return rel_l1 < self.config.threshold

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        diff = (current - previous).abs().mean()
        denom = previous.abs().mean().clamp_min(self.config.eps)
        return (diff / denom).detach().float().item()


@dataclass
class SeaCacheTimestepCacheConfig:
    enabled: bool = False
    threshold: float = 0.2
    num_steps: Optional[int] = None
    use_ret_steps: bool = False
    power_exp: float = 3.0
    power_const: float = 1.0
    eps: float = 1e-16
    norm_mode: str = "mean"

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("SeaCache threshold must be non-negative.")
        if self.num_steps is not None and self.num_steps <= 0:
            raise ValueError("SeaCache num_steps must be positive when set.")
        if self.power_exp <= 0:
            raise ValueError("SeaCache power_exp must be positive.")
        if self.power_const <= 0:
            raise ValueError("SeaCache power_const must be positive.")
        if self.eps <= 0:
            raise ValueError("SeaCache eps must be positive.")
        if self.norm_mode not in {"mean", "peak"}:
            raise ValueError("SeaCache norm_mode must be mean or peak.")


@dataclass
class SeaCacheTimestepCacheState:
    previous_feature: Optional[torch.Tensor] = None
    previous_residual: Optional[torch.Tensor] = None
    accumulated_rel_l1_distance: float = 0.0
    reuse_count: int = 0
    recompute_count: int = 0
    skipping_path: List[int] = field(default_factory=list)
    recompute_path: List[int] = field(default_factory=list)
    rel_l1_path: List[Tuple[int, float]] = field(default_factory=list)
    accumulated_rel_l1_path: List[Tuple[int, float]] = field(default_factory=list)


class SeaCacheTimestepCache:
    """SeaCache block-residual timestep cache with independent caller keys.

    SeaCache reuses the previous transformer-block residual, not the final
    denoiser output. The caller still runs the model head and unpatchify after a
    cache hit, matching the original Wan2.1 SeaCache implementation.
    """

    def __init__(self, config: SeaCacheTimestepCacheConfig):
        self.config = config
        self.states: Dict[Hashable, SeaCacheTimestepCacheState] = {}

    def should_reuse_blocks(
        self,
        key: Hashable,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
        scheduler_sigmas: Optional[torch.Tensor] = None,
        force_recompute: bool = False,
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        state = self.states.setdefault(key, SeaCacheTimestepCacheState())
        filtered_feature = self._filter_feature(
            feature,
            grid_size,
            step_index,
            num_steps,
            scheduler_sigmas,
        )
        should_recompute = force_recompute or self._should_recompute(
            state,
            step_index,
            num_steps,
            filtered_feature,
        )
        state.previous_feature = filtered_feature.detach().clone()
        if should_recompute:
            return False, None

        state.reuse_count += 1
        state.skipping_path.append(step_index)
        return True, state.previous_residual.clone()

    def record_recompute_blocks(
        self,
        key: Hashable,
        step_index: int,
        residual: torch.Tensor,
    ) -> None:
        state = self.states.setdefault(key, SeaCacheTimestepCacheState())
        self._record_recompute(state, step_index, residual)

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = {}
        for key, state in self.states.items():
            result[str(key)] = {
                "reuse": state.reuse_count,
                "recompute": state.recompute_count,
                "skipping_path": list(state.skipping_path),
                "recompute_path": list(state.recompute_path),
                "rel_l1_path": list(state.rel_l1_path),
                "accumulated_rel_l1_path": list(state.accumulated_rel_l1_path),
                "threshold": self.config.threshold,
            }
        return result

    def _should_recompute(
        self,
        state: SeaCacheTimestepCacheState,
        step_index: int,
        num_steps: int,
        filtered_feature: torch.Tensor,
    ) -> bool:
        ret_steps, cutoff_steps = self._ret_cutoff_steps(num_steps)
        if (step_index < ret_steps or step_index >= cutoff_steps or
                state.previous_feature is None or state.previous_residual is None):
            state.accumulated_rel_l1_distance = 0.0
            return True

        rel_l1 = self._relative_l1(filtered_feature, state.previous_feature)
        state.accumulated_rel_l1_distance += rel_l1
        state.rel_l1_path.append((step_index, rel_l1))
        state.accumulated_rel_l1_path.append(
            (step_index, state.accumulated_rel_l1_distance))
        if state.accumulated_rel_l1_distance < self.config.threshold:
            return False

        state.accumulated_rel_l1_distance = 0.0
        return True

    def _record_recompute(
        self,
        state: SeaCacheTimestepCacheState,
        step_index: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        state.previous_residual = output.detach().clone()
        state.recompute_count += 1
        state.recompute_path.append(step_index)
        return output

    def _ret_cutoff_steps(self, num_steps: int) -> Tuple[int, int]:
        configured_steps = self.config.num_steps or num_steps
        if self.config.use_ret_steps:
            return min(5, configured_steps), configured_steps
        return min(1, configured_steps), max(0, configured_steps - 1)

    def _filter_feature(
        self,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
        step_index: int,
        num_steps: int,
        scheduler_sigmas: Optional[torch.Tensor],
    ) -> torch.Tensor:
        feature_5d = self._reshape_to_grid(feature, grid_size)
        a, b = self._ab_from_flow_scheduler(step_index, num_steps, scheduler_sigmas)
        filtered = self._apply_sea_from_ab(
            feature_5d,
            a,
            b,
            dims=(-2, -3, -4),
        )
        return filtered.reshape(filtered.shape[0], -1, filtered.shape[-1]).detach()

    def _reshape_to_grid(self, feature: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
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
