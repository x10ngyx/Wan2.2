from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional

import torch


@dataclass
class BWBlockCacheConfig:
    enabled: bool = False
    thresh: float = 0.15
    reuse_interval: int = 3
    last_step: float = 0.5

    def __post_init__(self):
        if self.thresh < 0:
            raise ValueError("BWCache thresh must be non-negative.")
        if self.reuse_interval <= 0:
            raise ValueError("BWCache reuse_interval must be positive.")
        if self.last_step < 0:
            raise ValueError("BWCache last_step must be non-negative.")


@dataclass
class BWBlockCacheState:
    cal_list: Optional[torch.Tensor] = None
    cal_list_updated: bool = False
    previous_residual: Optional[torch.Tensor] = None
    last_x_m: List[Optional[torch.Tensor]] = field(default_factory=list)
    reuse_count: int = 0
    recompute_count: int = 0
    reuse_path: List[int] = field(default_factory=list)
    recompute_path: List[int] = field(default_factory=list)
    update_step: Optional[int] = None
    acu_l1_path: List[tuple] = field(default_factory=list)
    cal_list_path: List[int] = field(default_factory=list)

    def ensure(self, num_steps: int, num_blocks: int, device: torch.device):
        if self.cal_list is None or self.cal_list.numel() != num_steps:
            self.cal_list = torch.ones(num_steps, dtype=torch.long, device=device)
            self.cal_list_updated = False
            self.previous_residual = None
            self.last_x_m = [None for _ in range(num_blocks)]
            self.reuse_count = 0
            self.recompute_count = 0
            self.reuse_path = []
            self.recompute_path = []
            self.update_step = None
            self.acu_l1_path = []
            self.cal_list_path = [1 for _ in range(num_steps)]
        elif self.cal_list.device != device:
            self.cal_list = self.cal_list.to(device)


class BWBlockCache:
    """BWCache-style block cache with independent states per caller key."""

    def __init__(self, config: BWBlockCacheConfig):
        self.config = config
        self.states: Dict[Hashable, BWBlockCacheState] = {}

    def state(self, key: Hashable, num_steps: int, num_blocks: int, device: torch.device) -> BWBlockCacheState:
        state = self.states.setdefault(key, BWBlockCacheState())
        state.ensure(num_steps, num_blocks, device)
        return state

    def if_reuse_cache(self, acu_l1: float, depth: int) -> bool:
        return acu_l1 / depth < self.config.thresh

    def update_cal_list(
        self,
        state: BWBlockCacheState,
        step_index: int,
        num_steps: int,
    ) -> bool:
        if state.cal_list_updated or not self.if_reuse_cache(
                state.acu_l1_path[-1][1], len(state.last_x_m)):
            return False

        pattern = [0] * self.config.reuse_interval + [1]
        new_cal_list = torch.ones(
            num_steps, dtype=torch.long, device=state.cal_list.device)
        for i in range(step_index + 1, num_steps):
            new_cal_list[i] = pattern[(i - (step_index + 1)) %
                                      (self.config.reuse_interval + 1)]

        tail_start = step_index * self.config.last_step if self.config.last_step < 1 else self.config.last_step
        new_cal_list[-int(tail_start):] = 1

        state.cal_list.copy_(new_cal_list)
        state.cal_list_updated = True
        state.update_step = step_index
        state.cal_list_path = [int(v) for v in new_cal_list.detach().cpu().tolist()]
        return True

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = {}
        for key, state in self.states.items():
            result[str(key)] = {
                "reuse": state.reuse_count,
                "recompute": state.recompute_count,
                "reuse_path": list(state.reuse_path),
                "recompute_path": list(state.recompute_path),
                "update_step": state.update_step,
                "acu_l1_path": list(state.acu_l1_path),
                "cal_list": list(state.cal_list_path),
                "thresh": self.config.thresh,
                "reuse_interval": self.config.reuse_interval,
                "last_step": self.config.last_step,
            }
        return result
