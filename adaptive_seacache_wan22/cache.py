from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Hashable, Optional

import torch
from torch import nn

from adaptive_threshold_predictor.models import (
    CachedFeatureAdaCacheGate,
    CachedGatedFeatureAdaCacheGate,
    MiniDiTCLSAdaptiveThresholdPredictor,
    normalize_feature_sets,
)
from wan.timestep_cache import SeaCacheTimestepCache, SeaCacheTimestepCacheConfig


@dataclass
class AdaptiveSeaCacheGateConfig:
    model_path: Path
    target_psnr: float
    model_type: str = "auto"
    feature_set: str = "temporal_mean"
    feature_sets: tuple[str, ...] = (
        "latent_pool",
        "temporal_mean",
        "temporal_var",
        "frame_diff_mean",
        "frame_diff_var",
    )
    hidden_dim: int = 16
    feature_embedding_dim: int | None = None
    feature_dim: int = 128
    grid_size: tuple[int, int, int] = (2, 2, 2)
    dit_input_shape: tuple[int, int, int, int] | None = None
    dit_patch_size: tuple[int, int, int] | None = None
    dit_dim: int = 96
    dit_layers: int = 2
    dit_heads: int = 4
    dit_mlp_ratio: float = 2.0
    dit_dropout: float = 0.05
    dit_gate_init: float = 0.0
    psnr_min: float = 10.0
    psnr_max: float = 50.0
    min_threshold: float = 0.0
    max_threshold: float = 1.0
    device: str = "cuda"
    measure_predictor_timing: bool = False


class OnlineAdaptiveThresholdGate(nn.Module):
    """Predict thresholds from live Wan latents.

    The legacy MLP path reproduces cached feature extraction online. The MiniDiT
    path feeds the raw latent directly into the learned Conv3d patch embedder.
    """

    def __init__(self, config: AdaptiveSeaCacheGateConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.pool = nn.AdaptiveAvgPool3d(config.grid_size)
        if self.model_type == "mlp":
            self.model = CachedFeatureAdaCacheGate(
                feature_dim=config.feature_dim,
                hidden_dim=config.hidden_dim,
                psnr_min=config.psnr_min,
                psnr_max=config.psnr_max,
                min_threshold=config.min_threshold,
                max_threshold=config.max_threshold,
            )
        elif self.model_type == "mlp_gated":
            feature_sets = normalize_feature_sets(config.feature_sets)
            self.model = CachedGatedFeatureAdaCacheGate(
                feature_dims={name: config.feature_dim for name in feature_sets},
                hidden_dim=config.hidden_dim,
                feature_embedding_dim=config.feature_embedding_dim,
                psnr_min=config.psnr_min,
                psnr_max=config.psnr_max,
                min_threshold=config.min_threshold,
                max_threshold=config.max_threshold,
            )
        elif self.model_type == "mini_dit_cls":
            if config.dit_input_shape is None:
                raise ValueError("dit_input_shape is required for mini_dit_cls.")
            if config.dit_patch_size is None:
                raise ValueError("dit_patch_size is required for mini_dit_cls.")
            self.model = MiniDiTCLSAdaptiveThresholdPredictor(
                input_shape=config.dit_input_shape,
                patch_size=config.dit_patch_size,
                dim=config.dit_dim,
                num_layers=config.dit_layers,
                num_heads=config.dit_heads,
                mlp_ratio=config.dit_mlp_ratio,
                psnr_min=config.psnr_min,
                psnr_max=config.psnr_max,
                min_threshold=config.min_threshold,
                max_threshold=config.max_threshold,
                dropout=config.dit_dropout,
                gate_init=config.dit_gate_init,
            )
        else:
            raise ValueError(
                f"Unsupported adaptive gate model_type {self.model_type!r}; "
                "expected 'mlp', 'mlp_gated', or 'mini_dit_cls'."
            )

    @classmethod
    def load(cls, config: AdaptiveSeaCacheGateConfig) -> "OnlineAdaptiveThresholdGate":
        config, state = _load_adaptive_gate_state(config)
        gate = cls(config)
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
        step_fraction = float(step_index) / float(max(num_steps - 1, 1))
        if self.model_type == "mini_dit_cls":
            pred = self.model(latent, step_fraction, self.config.target_psnr)
        elif self.model_type == "mlp_gated":
            features = {
                name: self._extract_feature(latent, name)
                for name in self.model.feature_sets
            }
            pred = self.model(features, step_fraction, self.config.target_psnr)
        else:
            feature = self._extract_feature(latent, self.config.feature_set)
            pred = self.model(feature, step_fraction, self.config.target_psnr)
        threshold = float(pred.flatten()[0].detach().cpu().item())
        return max(
            self.config.min_threshold,
            min(self.config.max_threshold, threshold),
        )

    def _extract_feature(self, latent: torch.Tensor, feature_set: str) -> torch.Tensor:
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
        if feature_set == "temporal_var":
            spatial = torch.nn.functional.adaptive_avg_pool2d(
                latent.var(dim=2, unbiased=False),
                (height_bins, width_bins),
            )
            return (
                spatial.unsqueeze(2)
                .expand(batch, channels, temporal_bins, height_bins, width_bins)
                .flatten(start_dim=1)
            )
        if feature_set == "frame_diff_mean":
            diff = self._frame_diff(latent)
            spatial = torch.nn.functional.adaptive_avg_pool2d(
                diff.mean(dim=2),
                (height_bins, width_bins),
            )
            return (
                spatial.unsqueeze(2)
                .expand(batch, channels, temporal_bins, height_bins, width_bins)
                .flatten(start_dim=1)
            )
        if feature_set == "frame_diff_var":
            diff = self._frame_diff(latent)
            spatial = torch.nn.functional.adaptive_avg_pool2d(
                diff.var(dim=2, unbiased=False),
                (height_bins, width_bins),
            )
            return (
                spatial.unsqueeze(2)
                .expand(batch, channels, temporal_bins, height_bins, width_bins)
                .flatten(start_dim=1)
            )
        raise ValueError(
            f"Adaptive SeaCache inference got unsupported feature_set {feature_set!r}."
        )

    @staticmethod
    def _frame_diff(latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[2] <= 1:
            return torch.zeros_like(latent)
        diff = torch.zeros_like(latent)
        diff[:, :, 1:] = (latent[:, :, 1:] - latent[:, :, :-1]).abs()
        return diff

    @staticmethod
    def _normalize_latent_rank(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            return latent.unsqueeze(0)
        if latent.ndim == 5:
            return latent
        raise ValueError(
            f"Expected latent shape [C,T,H,W] or [B,C,T,H,W], got {tuple(latent.shape)}"
        )


OnlineCachedFeatureGate = OnlineAdaptiveThresholdGate


def _load_json_config(model_path: Path) -> dict[str, object]:
    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object in {config_path}, got {type(loaded).__name__}")
    return loaded


def _load_adaptive_gate_state(
    config: AdaptiveSeaCacheGateConfig,
) -> tuple[AdaptiveSeaCacheGateConfig, dict[str, torch.Tensor]]:
    state_or_checkpoint = torch.load(config.model_path, map_location="cpu")
    metadata: dict[str, object] = _load_json_config(config.model_path)
    feature_extractor: dict[str, object] = {}

    if isinstance(state_or_checkpoint, dict) and "model_state_dict" in state_or_checkpoint:
        state = state_or_checkpoint["model_state_dict"]
        metadata.update(state_or_checkpoint.get("args") or {})
        feature_extractor = state_or_checkpoint.get("feature_extractor") or {}
        checkpoint_model_type = str(
            state_or_checkpoint.get("model_type")
            or metadata.get("model_type")
            or config.model_type
        )
    elif isinstance(state_or_checkpoint, dict):
        state = state_or_checkpoint
        checkpoint_model_type = str(metadata.get("model_type") or config.model_type)
    else:
        raise ValueError(
            f"Expected state_dict or checkpoint dict in {config.model_path}, "
            f"got {type(state_or_checkpoint).__name__}."
        )

    model_type = "auto" if config.model_type == "auto" else config.model_type
    if model_type == "auto":
        if any(key.startswith("patch_embedding.") for key in state):
            model_type = "mini_dit_cls"
        elif any(key.startswith("fusion.") for key in state):
            model_type = "mlp_gated"
        else:
            model_type = "mlp"
    if model_type == "mini_dit_cls":
        input_shape = (
            config.dit_input_shape
            or _tuple_from_metadata(feature_extractor, "input_shape", 4)
            or (16, 12, 60, 104)
        )
        patch_size = (
            config.dit_patch_size
            or _tuple_from_metadata(feature_extractor, "patch_size", 3)
            or _tuple_from_metadata(metadata, "dit_patch_size", 3)
            or (3, 12, 8)
        )
        config = AdaptiveSeaCacheGateConfig(
            model_path=config.model_path,
            target_psnr=config.target_psnr,
            model_type="mini_dit_cls",
            feature_set=config.feature_set,
            feature_sets=config.feature_sets,
            hidden_dim=config.hidden_dim,
            feature_embedding_dim=config.feature_embedding_dim,
            feature_dim=config.feature_dim,
            grid_size=config.grid_size,
            dit_input_shape=input_shape,
            dit_patch_size=patch_size,
            dit_dim=int(metadata.get("dit_dim", config.dit_dim)),
            dit_layers=int(metadata.get("dit_layers", config.dit_layers)),
            dit_heads=int(metadata.get("dit_heads", config.dit_heads)),
            dit_mlp_ratio=float(metadata.get("dit_mlp_ratio", config.dit_mlp_ratio)),
            dit_dropout=float(metadata.get("dit_dropout", config.dit_dropout)),
            dit_gate_init=float(metadata.get("dit_gate_init", config.dit_gate_init)),
            psnr_min=float(metadata.get("psnr_min", config.psnr_min)),
            psnr_max=float(metadata.get("psnr_max", config.psnr_max)),
            min_threshold=float(metadata.get("min_threshold", config.min_threshold)),
            max_threshold=float(metadata.get("max_threshold", config.max_threshold)),
            device=config.device,
            measure_predictor_timing=config.measure_predictor_timing,
        )
    elif model_type == "mlp_gated":
        feature_sets_value = (
            feature_extractor.get("feature_sets")
            or metadata.get("feature_sets")
            or metadata.get("selected_feature_sets")
            or config.feature_sets
        )
        feature_sets = normalize_feature_sets(feature_sets_value)  # type: ignore[arg-type]
        config = AdaptiveSeaCacheGateConfig(
            model_path=config.model_path,
            target_psnr=config.target_psnr,
            model_type="mlp_gated",
            feature_set=config.feature_set,
            feature_sets=feature_sets,
            hidden_dim=int(metadata.get("hidden_dim", config.hidden_dim)),
            feature_embedding_dim=(
                int(
                    feature_extractor.get(
                        "feature_embedding_dim",
                        metadata.get(
                            "feature_embedding_dim",
                            metadata.get(
                                "resolved_feature_embedding_dim",
                                config.feature_embedding_dim or int(metadata.get("hidden_dim", config.hidden_dim)),
                            ),
                        ),
                    )
                )
            ),
            feature_dim=int(metadata.get("feature_dim", config.feature_dim)),
            grid_size=config.grid_size,
            dit_input_shape=config.dit_input_shape,
            dit_patch_size=config.dit_patch_size,
            dit_dim=config.dit_dim,
            dit_layers=config.dit_layers,
            dit_heads=config.dit_heads,
            dit_mlp_ratio=config.dit_mlp_ratio,
            dit_dropout=float(metadata.get("dit_dropout", config.dit_dropout)),
            dit_gate_init=config.dit_gate_init,
            psnr_min=float(metadata.get("psnr_min", config.psnr_min)),
            psnr_max=float(metadata.get("psnr_max", config.psnr_max)),
            min_threshold=float(metadata.get("min_threshold", config.min_threshold)),
            max_threshold=float(metadata.get("max_threshold", config.max_threshold)),
            device=config.device,
            measure_predictor_timing=config.measure_predictor_timing,
        )
    else:
        config = AdaptiveSeaCacheGateConfig(
            model_path=config.model_path,
            target_psnr=config.target_psnr,
            model_type="mlp",
            feature_set=str(metadata.get("feature_set", config.feature_set)),
            feature_sets=config.feature_sets,
            hidden_dim=int(metadata.get("hidden_dim", config.hidden_dim)),
            feature_embedding_dim=config.feature_embedding_dim,
            feature_dim=int(metadata.get("feature_dim", config.feature_dim)),
            grid_size=config.grid_size,
            dit_input_shape=config.dit_input_shape,
            dit_patch_size=config.dit_patch_size,
            dit_dim=config.dit_dim,
            dit_layers=config.dit_layers,
            dit_heads=config.dit_heads,
            dit_mlp_ratio=config.dit_mlp_ratio,
            dit_dropout=float(metadata.get("dit_dropout", config.dit_dropout)),
            dit_gate_init=config.dit_gate_init,
            psnr_min=float(metadata.get("psnr_min", config.psnr_min)),
            psnr_max=float(metadata.get("psnr_max", config.psnr_max)),
            min_threshold=float(metadata.get("min_threshold", config.min_threshold)),
            max_threshold=float(metadata.get("max_threshold", config.max_threshold)),
            device=config.device,
            measure_predictor_timing=config.measure_predictor_timing,
        )
    return config, state


def _tuple_from_metadata(
    metadata: dict[str, object],
    key: str,
    length: int,
) -> tuple[int, ...] | None:
    value = metadata.get(key)
    if value is None:
        return None
    values = tuple(int(item) for item in value)  # type: ignore[arg-type]
    if len(values) != length:
        raise ValueError(f"Expected {key} length {length}, got {values!r}")
    return values


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
        gate: OnlineAdaptiveThresholdGate,
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

    def clear_runtime_state(self) -> None:
        self.states.clear()
        self._key_states.clear()
        self._current_latents.clear()

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

    def clear_runtime_state(self) -> None:
        self.states.clear()
        self.decision_trace.clear()
        self.threshold_path.clear()

def build_adaptive_seacache_factory(
    gate_config: AdaptiveSeaCacheGateConfig,
):
    gate = OnlineAdaptiveThresholdGate.load(gate_config)
    logging.info(
        "Loaded adaptive SeaCache gate: model=%s model_type=%s target_psnr=%.3f "
        "feature_set=%s hidden_dim=%d",
        gate.config.model_path,
        gate.model_type,
        gate.config.target_psnr,
        gate.config.feature_set,
        gate.config.hidden_dim,
    )

    class AdaptiveSeaCacheFactory:
        def __init__(self) -> None:
            self.last_instance: AdaptiveSeaCacheTimestepCache | None = None

        def __call__(
            self,
            config: SeaCacheTimestepCacheConfig,
        ) -> AdaptiveSeaCacheTimestepCache:
            cache = AdaptiveSeaCacheTimestepCache(config, gate)
            self.last_instance = cache
            return cache

        def clear_last_instance(self) -> None:
            if self.last_instance is not None:
                self.last_instance.clear_runtime_state()
            self.last_instance = None

    return AdaptiveSeaCacheFactory()


def build_replay_seacache_factory(
    threshold_trace: dict[tuple[str, str, int], float],
):
    class ReplaySeaCacheFactory:
        def __init__(self) -> None:
            self.last_instance: ReplaySeaCacheTimestepCache | None = None

        def __call__(
            self,
            config: SeaCacheTimestepCacheConfig,
        ) -> ReplaySeaCacheTimestepCache:
            cache = ReplaySeaCacheTimestepCache(config, threshold_trace)
            self.last_instance = cache
            return cache

        def clear_last_instance(self) -> None:
            if self.last_instance is not None:
                self.last_instance.clear_runtime_state()
            self.last_instance = None

    return ReplaySeaCacheFactory()
