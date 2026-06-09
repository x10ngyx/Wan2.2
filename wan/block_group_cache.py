from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Tuple

import torch


@dataclass
class BlockGroupCacheConfig:
    enabled: bool = False
    group_size: int = 5
    threshold: float = 0.05
    metric: str = "pooled_rel_l1"
    start: float = 0.0
    end: float = 1.0
    max_reuse: int = 3
    eps: float = 1e-6

    def __post_init__(self):
        if self.group_size <= 0:
            raise ValueError("Block-group cache group_size must be positive.")
        if self.threshold < 0:
            raise ValueError("Block-group cache threshold must be non-negative.")
        if self.metric not in {"pooled_rel_l1", "full_rel_l1"}:
            raise ValueError("Block-group cache metric must be pooled_rel_l1 or full_rel_l1.")
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


@dataclass
class BlockGroupState:
    last_feature: Optional[torch.Tensor] = None
    cached_residual: Optional[torch.Tensor] = None
    reuse_streak: int = 0
    reuse_count: int = 0
    recompute_count: int = 0
    metric_path: List[Tuple[int, float]] = field(default_factory=list)
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
        self.last_feature = feature.detach().clone()
        self.cached_residual = residual.detach().clone()
        self.reuse_streak = 0
        self.recompute_count += 1
        self.recompute_path.append(step_index)


class BlockGroupCache:
    """Threshold-based cache that reuses residuals for groups of transformer blocks."""

    def __init__(self, config: BlockGroupCacheConfig):
        self.config = config
        self.states: Dict[Hashable, List[BlockGroupState]] = {}

    def state(self, key: Hashable, num_groups: int) -> List[BlockGroupState]:
        states = self.states.get(key)
        if states is None or len(states) != num_groups:
            states = [BlockGroupState() for _ in range(num_groups)]
            self.states[key] = states
        return states

    def clear_stage(self, model_stage: str):
        for key in list(self.states):
            if isinstance(key, tuple) and key and key[0] == model_stage:
                del self.states[key]

    def should_reuse(
        self,
        group_state: BlockGroupState,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
    ) -> bool:
        if not self.config.enabled or not group_state.has_history():
            return False
        if num_steps <= 0:
            return False

        progress = step_index / num_steps
        if not (self.config.start <= progress <= self.config.end):
            return False
        if group_state.reuse_streak >= self.config.max_reuse:
            return False

        rel_l1 = self._relative_l1(feature, group_state.last_feature)
        group_state.metric_path.append((step_index, rel_l1))
        return rel_l1 < self.config.threshold

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = {}
        for key, group_states in self.states.items():
            result[str(key)] = {}
            for group_index, state in enumerate(group_states):
                result[str(key)][str(group_index)] = {
                    "reuse": state.reuse_count,
                    "recompute": state.recompute_count,
                    "reuse_path": list(state.reuse_path),
                    "recompute_path": list(state.recompute_path),
                    "metric_path": list(state.metric_path),
                    "threshold": self.config.threshold,
                    "max_reuse": self.config.max_reuse,
                    "group_size": self.config.group_size,
                    "metric": self.config.metric,
                }
        return result

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        diff = (current.float() - previous.float()).abs().mean()
        denom = previous.float().abs().mean().clamp_min(self.config.eps)
        return (diff / denom).detach().float().item()
