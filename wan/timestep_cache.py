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

    def call(self, key: Hashable, step_index: int, compute_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        state = self.states.setdefault(key, ZeusTimestepCacheState())

        if self._should_skip(state, step_index):
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
        if self.caching_mode not in {"reuse_interp", "interp_all", "reuse_all"}:
            raise ValueError("ZEUS threshold caching_mode must be reuse_interp, interp_all, or reuse_all.")
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

    def record_recompute(
        self,
        step_index: int,
        output: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        result = super().record_recompute(step_index, output)
        self.prev_input = latent.detach().clone()
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
    ) -> torch.Tensor:
        state = self.states.setdefault(key, ZeusThresholdTimestepCacheState())

        if self._should_skip_threshold(state, step_index, latent):
            return self._reuse(state, step_index)

        output = compute_fn()
        return state.record_recompute(step_index, output, latent)

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = super().summary()
        for key, state in self.states.items():
            result[str(key)].update({
                "rel_l1_path": list(state.rel_l1_path),
                "threshold": self.config.threshold,
            })
        return result

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
