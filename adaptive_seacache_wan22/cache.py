from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable, Optional

import torch
from torch import nn

from adaptive_threshold_predictor.models import CachedFeatureAdaCacheGate
from wan.timestep_cache import SeaCacheTimestepCache, SeaCacheTimestepCacheConfig


@dataclass
class AdaptiveSeaCacheGateConfig:
    model_path: Path
    target_psnr: float
    feature_set: str = "temporal_mean"
    hidden_dim: int = 16
    feature_dim: int = 128
    grid_size: tuple[int, int, int] = (2, 2, 2)
    psnr_min: float = 10.0
    psnr_max: float = 50.0
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    device: str = "cuda"
    measure_predictor_timing: bool = False


class OnlineCachedFeatureGate(nn.Module):
    """Predict thresholds from live Wan latents using training-time feature extraction.

    The adaptive predictor was trained with precomputed pooled features, so this
    wrapper reproduces ``build_feature_cache.extract_feature`` online and feeds
    the result into ``CachedFeatureAdaCacheGate``.
    """

    def __init__(self, config: AdaptiveSeaCacheGateConfig) -> None:
        super().__init__()
        self.config = config
        self.pool = nn.AdaptiveAvgPool3d(config.grid_size)
        self.model = CachedFeatureAdaCacheGate(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            psnr_min=config.psnr_min,
            psnr_max=config.psnr_max,
        )

    @classmethod
    def load(cls, config: AdaptiveSeaCacheGateConfig) -> "OnlineCachedFeatureGate":
        gate = cls(config)
        state = torch.load(config.model_path, map_location="cpu")
        gate.model.load_state_dict(state)
        gate.eval().requires_grad_(False)
        gate.to(torch.device(config.device))
        return gate

    @torch.no_grad()
    def predict(
        self,
        latent: torch.Tensor,
        step_index: int,
        num_steps: int,
    ) -> float:
        latent = self._normalize_latent_rank(latent).to(
            device=next(self.parameters()).device,
            dtype=torch.float32,
        )
        feature = self._extract_feature(latent)
        step_fraction = float(step_index) / float(max(num_steps - 1, 1))
        pred = self.model(feature, step_fraction, self.config.target_psnr)
        threshold = float(pred.flatten()[0].detach().cpu().item())
        return max(
            self.config.min_threshold,
            min(self.config.max_threshold, threshold),
        )

    def _extract_feature(self, latent: torch.Tensor) -> torch.Tensor:
        feature_set = self.config.feature_set
        batch, channels, _, _, _ = latent.shape
        temporal_bins, height_bins, width_bins = self.pool.output_size
        if feature_set == "latent_pool":
            return self.pool(latent).flatten(start_dim=1)
        if feature_set == "temporal_mean":
            spatial = torch.nn.functional.adaptive_avg_pool2d(
                latent.mean(dim=2),
                (height_bins, width_bins),
            )
            return (
                spatial.unsqueeze(2)
                .expand(batch, channels, temporal_bins, height_bins, width_bins)
                .flatten(start_dim=1)
            )
        raise ValueError(
            f"Adaptive SeaCache inference currently supports latent_pool and "
            f"temporal_mean, got {feature_set!r}."
        )

    @staticmethod
    def _normalize_latent_rank(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            return latent.unsqueeze(0)
        if latent.ndim == 5:
            return latent
        raise ValueError(
            f"Expected latent shape [C,T,H,W] or [B,C,T,H,W], got {tuple(latent.shape)}"
        )


@dataclass
class AdaptiveSeaCachePrediction:
    step_index: int
    threshold: float


@dataclass
class AdaptiveSeaCacheKeyState:
    threshold_path: list[tuple[int, float]] = field(default_factory=list)
    decision_trace: list[dict[str, object]] = field(default_factory=list)
    predictor_elapsed_path: list[tuple[int, float]] = field(default_factory=list)


class AdaptiveSeaCacheTimestepCache(SeaCacheTimestepCache):
    """SeaCache timestep cache with a per-step neural threshold predictor."""

    def __init__(
        self,
        config: SeaCacheTimestepCacheConfig,
        gate: OnlineCachedFeatureGate,
    ) -> None:
        super().__init__(config)
        self.gate = gate
        self._key_states: dict[Hashable, AdaptiveSeaCacheKeyState] = {}
        self._current_latents: dict[Hashable, torch.Tensor] = {}

    def set_current_latent(
        self,
        key: Hashable,
        latent: torch.Tensor,
    ) -> None:
        self._current_latents[key] = latent.detach()

    def should_reuse_blocks(
        self,
        key: Hashable,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
        scheduler_sigmas: Optional[torch.Tensor] = None,
        force_recompute: bool = False,
    ):
        if key not in self._current_latents:
            raise RuntimeError(
                "AdaptiveSeaCacheTimestepCache did not receive the current latent. "
                "Call patch_wan_model_forward_for_adaptive_seacache() before inference."
            )
        latent = self._current_latents[key]
        predictor_elapsed = None
        if self.gate.config.measure_predictor_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            predictor_start = time.perf_counter()
            threshold = self.gate.predict(latent, step_index, num_steps)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            predictor_elapsed = time.perf_counter() - predictor_start
        else:
            threshold = self.gate.predict(latent, step_index, num_steps)
        previous_threshold = self.config.threshold
        self.config.threshold = threshold
        key_state = self._key_states.setdefault(key, AdaptiveSeaCacheKeyState())
        key_state.threshold_path.append((step_index, threshold))
        if predictor_elapsed is not None:
            key_state.predictor_elapsed_path.append((step_index, predictor_elapsed))
        try:
            should_reuse, cached_residual = super().should_reuse_blocks(
                key,
                step_index,
                num_steps,
                feature,
                grid_size,
                scheduler_sigmas=scheduler_sigmas,
                force_recompute=force_recompute,
            )
            sea_state = self.states[key]
            accumulated_rel_l1 = sea_state.accumulated_rel_l1_distance
            rel_l1 = None
            if sea_state.rel_l1_path and sea_state.rel_l1_path[-1][0] == step_index:
                rel_l1 = sea_state.rel_l1_path[-1][1]
            key_state.decision_trace.append(
                {
                    "step_index": step_index,
                    "predicted_threshold": threshold,
                    "rel_l1": rel_l1,
                    "accumulated_rel_l1": accumulated_rel_l1,
                    "decision": "reuse" if should_reuse else "recompute",
                    "force_recompute": bool(force_recompute),
                    "predictor_elapsed_seconds": predictor_elapsed,
                }
            )
            return should_reuse, cached_residual
        finally:
            self.config.threshold = previous_threshold

    def summary(self):
        result = super().summary()
        for key, state in self._key_states.items():
            result.setdefault(str(key), {})
            result[str(key)]["adaptive_threshold_path"] = list(state.threshold_path)
            if state.threshold_path:
                values = [value for _, value in state.threshold_path]
                result[str(key)]["adaptive_threshold_min"] = min(values)
                result[str(key)]["adaptive_threshold_max"] = max(values)
                result[str(key)]["adaptive_threshold_mean"] = sum(values) / len(values)
            if state.predictor_elapsed_path:
                elapsed_values = [value for _, value in state.predictor_elapsed_path]
                result[str(key)]["adaptive_predictor_elapsed_path"] = list(
                    state.predictor_elapsed_path)
                result[str(key)]["adaptive_predictor_elapsed_total_seconds"] = sum(
                    elapsed_values)
                result[str(key)]["adaptive_predictor_elapsed_mean_seconds"] = (
                    sum(elapsed_values) / len(elapsed_values))
                result[str(key)]["adaptive_predictor_elapsed_max_seconds"] = max(
                    elapsed_values)
                result[str(key)]["adaptive_predictor_call_count"] = len(elapsed_values)
            result[str(key)]["adaptive_decision_trace"] = list(state.decision_trace)
        return result


class ReplaySeaCacheTimestepCache(SeaCacheTimestepCache):
    """SeaCache timestep cache that replays a saved adaptive threshold trace."""

    def __init__(
        self,
        config: SeaCacheTimestepCacheConfig,
        threshold_trace: dict[tuple[str, str, int], float],
    ) -> None:
        super().__init__(config)
        self.threshold_trace = threshold_trace
        self.decision_trace: dict[Hashable, list[dict[str, object]]] = {}
        self.threshold_path: dict[Hashable, list[tuple[int, float]]] = {}

    def should_reuse_blocks(
        self,
        key: Hashable,
        step_index: int,
        num_steps: int,
        feature: torch.Tensor,
        grid_size: torch.Tensor,
        scheduler_sigmas: Optional[torch.Tensor] = None,
        force_recompute: bool = False,
    ):
        try:
            model_stage, branch = key
        except Exception as exc:
            raise ValueError(f"Replay SeaCache expected key=(stage, branch), got {key!r}") from exc
        lookup_key = (str(model_stage), str(branch), int(step_index))
        if lookup_key not in self.threshold_trace:
            raise KeyError(f"Missing replay threshold for {lookup_key}")
        threshold = self.threshold_trace[lookup_key]
        previous_threshold = self.config.threshold
        self.config.threshold = threshold
        self.threshold_path.setdefault(key, []).append((step_index, threshold))
        try:
            should_reuse, cached_residual = super().should_reuse_blocks(
                key,
                step_index,
                num_steps,
                feature,
                grid_size,
                scheduler_sigmas=scheduler_sigmas,
                force_recompute=force_recompute,
            )
            sea_state = self.states[key]
            accumulated_rel_l1 = sea_state.accumulated_rel_l1_distance
            rel_l1 = None
            if sea_state.rel_l1_path and sea_state.rel_l1_path[-1][0] == step_index:
                rel_l1 = sea_state.rel_l1_path[-1][1]
            self.decision_trace.setdefault(key, []).append(
                {
                    "step_index": step_index,
                    "predicted_threshold": threshold,
                    "rel_l1": rel_l1,
                    "accumulated_rel_l1": accumulated_rel_l1,
                    "decision": "reuse" if should_reuse else "recompute",
                    "force_recompute": bool(force_recompute),
                    "predictor_elapsed_seconds": None,
                    "replay_threshold": True,
                }
            )
            return should_reuse, cached_residual
        finally:
            self.config.threshold = previous_threshold

    def summary(self):
        result = super().summary()
        for key, rows in self.decision_trace.items():
            result.setdefault(str(key), {})
            result[str(key)]["adaptive_decision_trace"] = list(rows)
            result[str(key)]["adaptive_threshold_path"] = list(
                self.threshold_path.get(key, []))
            if self.threshold_path.get(key):
                values = [value for _, value in self.threshold_path[key]]
                result[str(key)]["adaptive_threshold_min"] = min(values)
                result[str(key)]["adaptive_threshold_max"] = max(values)
                result[str(key)]["adaptive_threshold_mean"] = sum(values) / len(values)
        return result

def build_adaptive_seacache_factory(
    gate_config: AdaptiveSeaCacheGateConfig,
):
    gate = OnlineCachedFeatureGate.load(gate_config)
    logging.info(
        "Loaded adaptive SeaCache gate: model=%s target_psnr=%.3f feature_set=%s "
        "hidden_dim=%d",
        gate_config.model_path,
        gate_config.target_psnr,
        gate_config.feature_set,
        gate_config.hidden_dim,
    )

    class AdaptiveSeaCacheFactory:
        def __init__(self) -> None:
            self.instances: list[AdaptiveSeaCacheTimestepCache] = []
            self.last_instance: AdaptiveSeaCacheTimestepCache | None = None

        def __call__(
            self,
            config: SeaCacheTimestepCacheConfig,
        ) -> AdaptiveSeaCacheTimestepCache:
            cache = AdaptiveSeaCacheTimestepCache(config, gate)
            self.instances.append(cache)
            self.last_instance = cache
            return cache

    return AdaptiveSeaCacheFactory()


def build_replay_seacache_factory(
    threshold_trace: dict[tuple[str, str, int], float],
):
    class ReplaySeaCacheFactory:
        def __init__(self) -> None:
            self.instances: list[ReplaySeaCacheTimestepCache] = []
            self.last_instance: ReplaySeaCacheTimestepCache | None = None

        def __call__(
            self,
            config: SeaCacheTimestepCacheConfig,
        ) -> ReplaySeaCacheTimestepCache:
            cache = ReplaySeaCacheTimestepCache(config, threshold_trace)
            self.instances.append(cache)
            self.last_instance = cache
            return cache

    return ReplaySeaCacheFactory()
