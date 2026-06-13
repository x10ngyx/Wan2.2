import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Tuple

import torch


@dataclass
class BlockGroupCacheConfig:
    enabled: bool = False
    group_size: int = 5
    threshold: float = 0.05
    metric: str = "pooled_rel_l1"
    decision: str = "instant"
    start: float = 0.0
    end: float = 1.0
    max_reuse: int = 3
    eps: float = 1e-6
    ret_steps: int = 1
    cutoff_steps: int = 1
    sea_power_exp: float = 3.0
    sea_power_const: float = 1.0
    sea_norm_mode: str = "mean"

    def __post_init__(self):
        if self.group_size <= 0:
            raise ValueError("Block-group cache group_size must be positive.")
        if self.threshold < 0:
            raise ValueError("Block-group cache threshold must be non-negative.")
        if self.metric not in {"pooled_rel_l1", "full_rel_l1", "sea_full_rel_l1"}:
            raise ValueError("Block-group cache metric must be pooled_rel_l1, full_rel_l1, or sea_full_rel_l1.")
        if self.decision not in {"instant", "accumulated"}:
            raise ValueError("Block-group cache decision must be instant or accumulated.")
        if not (0 <= self.start <= 1):
            raise ValueError("Block-group cache start must be in [0, 1].")
        if not (0 <= self.end <= 1):
            raise ValueError("Block-group cache end must be in [0, 1].")
        if self.start > self.end:
            raise ValueError("Block-group cache start must be <= end.")
        if self.max_reuse <= 0:
            raise ValueError("Block-group cache max_reuse must be positive.")
        if self.eps <= 0:
            raise ValueError("Block-group cache eps must be positive.")
        if self.ret_steps < 0:
            raise ValueError("Block-group cache ret_steps must be non-negative.")
        if self.cutoff_steps < 0:
            raise ValueError("Block-group cache cutoff_steps must be non-negative.")
        if self.sea_power_exp <= 0:
            raise ValueError("Block-group cache sea_power_exp must be positive.")
        if self.sea_power_const <= 0:
            raise ValueError("Block-group cache sea_power_const must be positive.")
        if self.sea_norm_mode not in {"mean", "peak"}:
            raise ValueError("Block-group cache sea_norm_mode must be mean or peak.")


@dataclass
class BlockGroupState:
    last_feature: Optional[torch.Tensor] = None
    pending_feature: Optional[torch.Tensor] = None
    cached_residual: Optional[torch.Tensor] = None
    accumulated_rel_l1_distance: float = 0.0
    reuse_streak: int = 0
    reuse_count: int = 0
    recompute_count: int = 0
    metric_path: List[Tuple[int, float]] = field(default_factory=list)
    accumulated_metric_path: List[Tuple[int, float]] = field(default_factory=list)
    reuse_path: List[int] = field(default_factory=list)
    recompute_path: List[int] = field(default_factory=list)

    def has_history(self) -> bool:
        return self.last_feature is not None and self.cached_residual is not None

    def record_reuse(self, step_index: int):
        self.reuse_streak += 1
        self.reuse_count += 1
        self.reuse_path.append(step_index)

    def record_recompute(
        self,
        step_index: int,
        feature: torch.Tensor,
        residual: torch.Tensor,
    ):
        feature_to_store = self.pending_feature
        if feature_to_store is None:
            feature_to_store = feature
        self.last_feature = feature_to_store.detach().clone()
        self.pending_feature = None
        self.cached_residual = residual.detach().clone()
        self.accumulated_rel_l1_distance = 0.0
        self.reuse_streak = 0
        self.recompute_count += 1
        self.recompute_path.append(step_index)


class BlockGroupCache:
    """Threshold-based cache that reuses residuals for groups of transformer blocks."""

    def __init__(self, config: BlockGroupCacheConfig):
        self.config = config
        self.states: Dict[Hashable, List[BlockGroupState]] = {}
        self.archived_summaries: Dict[str, Dict[str, Dict[str, object]]] = {}

    def state(self, key: Hashable, num_groups: int) -> List[BlockGroupState]:
        states = self.states.get(key)
        if states is None or len(states) != num_groups:
            states = [BlockGroupState() for _ in range(num_groups)]
            self.states[key] = states
        return states

    def clear_stage(self, model_stage: str):
        for key in list(self.states):
            if isinstance(key, tuple) and key and key[0] == model_stage:
                self.archived_summaries[str(key)] = self._group_states_summary(
                    self.states[key])
                del self.states[key]

    def should_reuse(
        self,
        group_state: BlockGroupState,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: Optional[torch.Tensor] = None,
        scheduler_sigmas: Optional[torch.Tensor] = None,
    ) -> bool:
        if not self.config.enabled:
            return False
        if num_steps <= 0:
            return False

        metric_feature = self._metric_feature(
            feature,
            grid_size,
            step_index,
            num_steps,
            scheduler_sigmas,
        )
        group_state.pending_feature = metric_feature
        if not group_state.has_history():
            return False

        progress = step_index / num_steps
        if not (self.config.start <= progress <= self.config.end):
            return False
        if group_state.reuse_streak >= self.config.max_reuse:
            return False

        if self.config.decision == "accumulated":
            if self._must_recompute_accumulated(group_state, step_index, num_steps):
                return False
            rel_l1 = self._relative_l1(metric_feature, group_state.last_feature)
            group_state.accumulated_rel_l1_distance += rel_l1
            group_state.metric_path.append((step_index, rel_l1))
            group_state.accumulated_metric_path.append(
                (step_index, group_state.accumulated_rel_l1_distance))
            group_state.last_feature = metric_feature.detach().clone()
            group_state.pending_feature = None
            if group_state.accumulated_rel_l1_distance < self.config.threshold:
                return True
            group_state.accumulated_rel_l1_distance = 0.0
            group_state.pending_feature = metric_feature
            return False

        rel_l1 = self._relative_l1(metric_feature, group_state.last_feature)
        group_state.metric_path.append((step_index, rel_l1))
        return rel_l1 < self.config.threshold

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = dict(self.archived_summaries)
        for key, group_states in self.states.items():
            result[str(key)] = self._group_states_summary(group_states)
        return result

    def _group_states_summary(
        self,
        group_states: List[BlockGroupState],
    ) -> Dict[str, Dict[str, object]]:
        result = {}
        for group_index, state in enumerate(group_states):
            result[str(group_index)] = {
                "reuse": state.reuse_count,
                "recompute": state.recompute_count,
                "reuse_path": list(state.reuse_path),
                "recompute_path": list(state.recompute_path),
                "metric_path": list(state.metric_path),
                "accumulated_metric_path": list(state.accumulated_metric_path),
                "threshold": self.config.threshold,
                "max_reuse": self.config.max_reuse,
                "group_size": self.config.group_size,
                "metric": self.config.metric,
                "decision": self.config.decision,
                "ret_steps": self.config.ret_steps,
                "cutoff_steps": self.config.cutoff_steps,
                "sea_power_exp": self.config.sea_power_exp,
                "sea_power_const": self.config.sea_power_const,
                "sea_norm_mode": self.config.sea_norm_mode,
            }
        return result

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        diff = (current.float() - previous.float()).abs().mean()
        denom = previous.float().abs().mean().clamp_min(self.config.eps)
        value = diff / denom
        if not torch.isfinite(value):
            return math.inf
        return value.detach().float().item()

    def _metric_feature(
        self,
        feature: torch.Tensor,
        grid_size: Optional[torch.Tensor],
        step_index: int,
        num_steps: int,
        scheduler_sigmas: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.config.metric != "sea_full_rel_l1":
            return feature.detach()
        if grid_size is None:
            raise ValueError("sea_full_rel_l1 block-group metric requires grid_size.")
        filtered = self._filter_feature(
            feature,
            grid_size,
            step_index,
            num_steps,
            scheduler_sigmas,
        )
        return filtered.detach()

    def _must_recompute_accumulated(
        self,
        group_state: BlockGroupState,
        step_index: int,
        num_steps: int,
    ) -> bool:
        cutoff_start = max(0, num_steps - self.config.cutoff_steps)
        if step_index < self.config.ret_steps or step_index >= cutoff_start:
            group_state.accumulated_rel_l1_distance = 0.0
            return True
        return group_state.last_feature is None or group_state.cached_residual is None

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
        return filtered.reshape(filtered.shape[0], -1, filtered.shape[-1])

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
            signal_power = self.config.sea_power_const / (
                (rad ** self.config.sea_power_exp) + self.config.eps)
            axis_gain = (a * signal_power) / (
                (a * a * signal_power) + (b * b) + self.config.eps)
            axis_shape = [1] * x32.ndim
            axis_shape[axis] = axis_gain.shape[0]
            axis_gain = axis_gain.reshape(axis_shape)
            gain = axis_gain if gain is None else gain * axis_gain

        if self.config.sea_norm_mode == "peak":
            maxv = torch.amax(gain)
            if torch.isfinite(maxv) and maxv > 0:
                gain = gain / maxv
        else:
            meanv = torch.mean(gain)
            if torch.isfinite(meanv) and meanv > 0:
                gain = gain / meanv

        filtered = torch.fft.ifftn(spectrum * gain, dim=dims).real
        return filtered.to(orig_dtype)
