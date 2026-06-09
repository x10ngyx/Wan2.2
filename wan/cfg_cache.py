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


@dataclass
class CFGCacheState:
    cached_delta: Optional[torch.Tensor] = None
    last_full_cond: Optional[torch.Tensor] = None
    reuse_streak: int = 0
    reuse_count: int = 0
    recompute_count: int = 0
    diff_path: List[Tuple[int, float]] = field(default_factory=list)
    reuse_path: List[int] = field(default_factory=list)
    recompute_path: List[int] = field(default_factory=list)

    def has_history(self) -> bool:
        return self.cached_delta is not None and self.last_full_cond is not None

    def record_reuse(self, step_index: int):
        self.reuse_streak += 1
        self.reuse_count += 1
        self.reuse_path.append(step_index)

    def record_recompute(
        self,
        step_index: int,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ):
        self.cached_delta = self._enhance(cond - uncond).detach().clone()
        self.last_full_cond = cond.detach().clone()
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
        cond: torch.Tensor,
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

        cfg_diff = self._relative_l1(cond, state.last_full_cond)
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
    ) -> torch.Tensor:
        state = self.state(key)
        state.record_recompute(step_index, cond, uncond)
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
            }
        return result

    def _relative_l1(self, current: torch.Tensor, previous: torch.Tensor) -> float:
        diff = (current - previous).abs().sum()
        denom = previous.abs().sum().clamp_min(self.config.eps)
        return (diff / denom).detach().float().item()
