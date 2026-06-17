import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List

import torch


@dataclass
class TaylorSeerConfig:
    enabled: bool = True
    fresh_threshold: int = 5
    max_order: int = 1
    first_enhance: int = 1

    def __post_init__(self):
        if self.fresh_threshold <= 0:
            raise ValueError("fresh_threshold must be positive.")
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative.")
        if self.first_enhance < 0:
            raise ValueError("first_enhance must be non-negative.")


@dataclass
class TaylorSeerState:
    cache_counter: int = 0
    cal_threshold: int = 5
    current_type: str = "full"
    current_step: int = 0
    activated_steps: List[int] = field(default_factory=lambda: [0])
    cache: Dict[str, Dict[int, Dict[str, Dict[int, torch.Tensor]]]] = field(
        default_factory=dict)
    reuse_count: int = 0
    recompute_count: int = 0
    skipping_path: List[int] = field(default_factory=list)
    recompute_path: List[int] = field(default_factory=list)


class TaylorSeerCache:
    """Official Wan TaylorSeer-style per-stage cache.

    The public TaylorSeer-Wan2.1 code keeps one scheduler state per model and
    separate module caches for cond/uncond streams. Wan2.2 has separate high
    and low denoisers, so this wrapper stores one state per caller key.
    """

    def __init__(self, config: TaylorSeerConfig):
        self.config = config
        self.states: Dict[Hashable, TaylorSeerState] = {}

    def reset(self) -> None:
        self.states.clear()

    def begin_branch(
        self,
        key: Hashable,
        stream: str,
        step_index: int,
        num_steps: int,
        force_full: bool = False,
    ) -> str:
        state = self.states.setdefault(key, TaylorSeerState())
        state.current_step = step_index
        if stream == "cond_stream" or force_full:
            self._cal_type(state, step_index, num_steps, force_full)

        if state.current_type == "full":
            state.recompute_count += 1
            state.recompute_path.append(step_index)
        elif state.current_type == "Taylor":
            state.reuse_count += 1
            state.skipping_path.append(step_index)
        else:
            raise ValueError(f"Unsupported TaylorSeer step type: {state.current_type}")
        return state.current_type

    def record_module(
        self,
        key: Hashable,
        stream: str,
        layer: int,
        module: str,
        feature: torch.Tensor,
    ) -> None:
        state = self.states.setdefault(key, TaylorSeerState())
        module_cache = self._module_cache(state, stream, layer, module)
        distance = state.activated_steps[-1] - state.activated_steps[-2]
        updated = {0: feature.detach().clone()}
        for order in range(self.config.max_order):
            previous = module_cache.get(order)
            if previous is None or state.current_step <= self.config.first_enhance - 2:
                break
            updated[order + 1] = (updated[order] - previous) / distance
        state.cache[stream][layer][module] = updated

    def predict_module(
        self,
        key: Hashable,
        stream: str,
        layer: int,
        module: str,
    ) -> torch.Tensor:
        state = self.states.setdefault(key, TaylorSeerState())
        module_cache = self._module_cache(state, stream, layer, module)
        distance = state.current_step - state.activated_steps[-1]
        if not module_cache:
            raise ValueError(
                f"TaylorSeer cache missing for key={key}, stream={stream}, "
                f"layer={layer}, module={module}.")
        output = 0
        for order in range(len(module_cache)):
            output = output + (
                module_cache[order] * (distance ** order) /
                math.factorial(order))
        return output

    def summary(self) -> Dict[str, Dict[str, object]]:
        result = {}
        for key, state in self.states.items():
            result[str(key)] = {
                "reuse": state.reuse_count,
                "recompute": state.recompute_count,
                "skipping_path": list(state.skipping_path),
                "recompute_path": list(state.recompute_path),
                "activated_steps": list(state.activated_steps),
                "fresh_threshold": self.config.fresh_threshold,
                "max_order": self.config.max_order,
                "first_enhance": self.config.first_enhance,
            }
        return result

    def _cal_type(
        self,
        state: TaylorSeerState,
        step_index: int,
        num_steps: int,
        force_full: bool,
    ) -> None:
        if force_full or not self.config.enabled:
            state.current_type = "full"
            state.cache_counter = 0
            if state.activated_steps[-1] != step_index:
                state.activated_steps.append(step_index)
            state.cal_threshold = self._force_scheduler(step_index, num_steps)
            return

        first_step = step_index < self.config.first_enhance
        fresh_interval = self.config.fresh_threshold if first_step else state.cal_threshold
        if first_step or state.cache_counter == fresh_interval - 1:
            state.current_type = "full"
            state.cache_counter = 0
            if state.activated_steps[-1] != step_index:
                state.activated_steps.append(step_index)
            state.cal_threshold = self._force_scheduler(step_index, num_steps)
            return

        state.cache_counter += 1
        state.current_type = "Taylor"

    def _force_scheduler(self, step_index: int, num_steps: int) -> int:
        del step_index, num_steps
        return self.config.fresh_threshold

    def _module_cache(
        self,
        state: TaylorSeerState,
        stream: str,
        layer: int,
        module: str,
    ) -> Dict[int, torch.Tensor]:
        stream_cache = state.cache.setdefault(stream, {})
        layer_cache = stream_cache.setdefault(layer, {})
        return layer_cache.setdefault(module, {})
