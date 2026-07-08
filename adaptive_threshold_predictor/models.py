from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


FEATURE_SETS = (
    "latent_pool",
    "temporal_mean",
    "temporal_var",
    "frame_diff_mean",
    "frame_diff_var",
)
DEFAULT_GATED_FEATURE_SETS = (
    "latent_pool",
    "temporal_mean",
    "temporal_var",
    "frame_diff_mean",
    "frame_diff_var",
)


def _threshold_from_unit(
    value: torch.Tensor,
    min_threshold: float,
    max_threshold: float,
) -> torch.Tensor:
    return min_threshold + value * (max_threshold - min_threshold)


def normalize_feature_sets(feature_sets: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(feature_sets, str):
        normalized = (feature_sets,)
    else:
        normalized = tuple(feature_sets)
    if not normalized:
        raise ValueError("At least one feature set is required")
    unknown = [feature_set for feature_set in normalized if feature_set not in FEATURE_SETS]
    if unknown:
        raise ValueError(
            f"Unknown feature_set(s) {unknown!r}; expected values from {FEATURE_SETS}"
        )
    return normalized


class ImprovedAdaCacheGate(nn.Module):
    """Lightweight timestep-cache threshold predictor.

    Inputs:
        latent: Either [B, C, T, H, W] or a single trace tensor [C, T, H, W].
        t: Normalized or raw timestep tensor, shape [B], [B, 1], or scalar.
        target_psnr: Normalized or raw target PSNR tensor, shape [B], [B, 1], or scalar.
        target_speedup: Normalized or raw target speedup tensor, shape [B], [B, 1],
            or scalar.

    Output:
        One threshold in [0, 1] for the current timestep-cache experiment,
        shape [B, 1].

    The condition path always receives timestep, target PSNR, and target speedup. The feature
    path is selected by ``feature_set``. Multi-feature runs use
    ``GatedMultiFeatureAdaCacheGate`` instead of direct feature concatenation.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_dim: int = 64,
        grid_size: tuple[int, int, int] = (2, 2, 2),
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        speedup_min: float = 1.0,
        speedup_max: float = 4.0,
        min_threshold: float = 0.10,
        max_threshold: float = 0.80,
        dropout: float = 0.0,
        feature_set: str = "latent_pool",
    ) -> None:
        super().__init__()
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature_set {feature_set!r}; expected one of {FEATURE_SETS}"
            )
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.normalize_inputs = normalize_inputs
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.speedup_min = speedup_min
        self.speedup_max = speedup_max
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.feature_set = feature_set

        self.pool = nn.AdaptiveAvgPool3d(grid_size)
        flat_latent_dim = latent_channels * grid_size[0] * grid_size[1] * grid_size[2]

        self.feature_proj = nn.Sequential(
            nn.Linear(flat_latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.cond_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.predict_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
    ) -> torch.Tensor:
        latent = self._normalize_latent_rank(latent)
        batch, channels, frames, _, _ = latent.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected latent channel count {self.latent_channels}, got {channels}"
            )

        latent = latent.float()
        feat_latent = self.feature_proj(
            self._extract_feature(latent, frames, self.feature_set)
        )
        feat_cond = self.cond_embed(
            self._prepare_condition(t, target_psnr, target_speedup, batch, latent.device)
        )
        fused = torch.cat([feat_latent, feat_cond], dim=-1)
        return _threshold_from_unit(
            torch.sigmoid(self.predict_head(fused)),
            self.min_threshold,
            self.max_threshold,
        )

    def _extract_feature(
        self,
        latent: torch.Tensor,
        frames: int,
        feature_set: str,
    ) -> torch.Tensor:
        if feature_set == "latent_pool":
            feature_volume = latent
        elif feature_set == "temporal_mean":
            feature_volume = latent.mean(dim=2, keepdim=True).expand_as(latent)
        elif feature_set == "temporal_var":
            feature_volume = latent.var(dim=2, keepdim=True, unbiased=False).expand_as(
                latent
            )
        elif feature_set == "frame_diff_mean":
            feature_volume = self._frame_diff(latent, frames).mean(
                dim=2, keepdim=True
            )
            feature_volume = feature_volume.expand_as(latent)
        elif feature_set == "frame_diff_var":
            feature_volume = self._frame_diff(latent, frames).var(
                dim=2, keepdim=True, unbiased=False
            )
            feature_volume = feature_volume.expand_as(latent)
        else:
            raise AssertionError(f"Unhandled feature_set: {feature_set}")
        return self.pool(feature_volume).flatten(start_dim=1)

    @staticmethod
    def _frame_diff(latent: torch.Tensor, frames: int) -> torch.Tensor:
        if frames <= 1:
            return torch.zeros_like(latent)
        diff = torch.abs(latent[:, :, 1:] - latent[:, :, :-1])
        pad = torch.zeros_like(latent[:, :, :1])
        return torch.cat([diff, pad], dim=2)

    @staticmethod
    def _normalize_latent_rank(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            return latent.unsqueeze(0)
        if latent.ndim == 5:
            return latent
        raise ValueError(
            f"Expected latent shape [C,T,H,W] or [B,C,T,H,W], got {tuple(latent.shape)}"
        )

    def _prepare_condition(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
        psnr_tensor = torch.as_tensor(
            target_psnr, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        speedup_tensor = torch.as_tensor(
            target_speedup, dtype=torch.float32, device=device
        ).reshape(-1, 1)

        if t_tensor.shape[0] == 1 and batch > 1:
            t_tensor = t_tensor.expand(batch, 1)
        if psnr_tensor.shape[0] == 1 and batch > 1:
            psnr_tensor = psnr_tensor.expand(batch, 1)
        if speedup_tensor.shape[0] == 1 and batch > 1:
            speedup_tensor = speedup_tensor.expand(batch, 1)
        if (
            t_tensor.shape[0] != batch
            or psnr_tensor.shape[0] != batch
            or speedup_tensor.shape[0] != batch
        ):
            raise ValueError(
                "Condition batch size mismatch: "
                f"latent batch={batch}, t={t_tensor.shape[0]}, "
                f"psnr={psnr_tensor.shape[0]}, speedup={speedup_tensor.shape[0]}"
            )

        if self.normalize_inputs:
            t_tensor = t_tensor.clamp(0.0, 1.0)
            psnr_tensor = (
                (psnr_tensor - self.psnr_min) / (self.psnr_max - self.psnr_min)
            ).clamp(0.0, 1.0)
            speedup_tensor = (
                (speedup_tensor - self.speedup_min)
                / (self.speedup_max - self.speedup_min)
            ).clamp(0.0, 1.0)

        return torch.cat([t_tensor, psnr_tensor, speedup_tensor], dim=-1)


class CachedFeatureAdaCacheGate(nn.Module):
    """Same prediction trunk as ImprovedAdaCacheGate, with precomputed features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        speedup_min: float = 1.0,
        speedup_max: float = 4.0,
        min_threshold: float = 0.10,
        max_threshold: float = 0.80,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.speedup_min = speedup_min
        self.speedup_max = speedup_max
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        feature: torch.Tensor,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
    ) -> torch.Tensor:
        feature = feature.float()
        batch = feature.shape[0]
        feat_latent = self.feature_proj(feature)
        feat_cond = self.cond_embed(
            self._prepare_condition(t, target_psnr, target_speedup, batch, feature.device)
        )
        return _threshold_from_unit(
            torch.sigmoid(self.predict_head(torch.cat([feat_latent, feat_cond], dim=-1))),
            self.min_threshold,
            self.max_threshold,
        )

    def _prepare_condition(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
        psnr_tensor = torch.as_tensor(
            target_psnr, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        speedup_tensor = torch.as_tensor(
            target_speedup, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        if t_tensor.shape[0] == 1 and batch > 1:
            t_tensor = t_tensor.expand(batch, 1)
        if psnr_tensor.shape[0] == 1 and batch > 1:
            psnr_tensor = psnr_tensor.expand(batch, 1)
        if speedup_tensor.shape[0] == 1 and batch > 1:
            speedup_tensor = speedup_tensor.expand(batch, 1)
        if self.normalize_inputs:
            t_tensor = t_tensor.clamp(0.0, 1.0)
            psnr_tensor = (
                (psnr_tensor - self.psnr_min) / (self.psnr_max - self.psnr_min)
            ).clamp(0.0, 1.0)
            speedup_tensor = (
                (speedup_tensor - self.speedup_min)
                / (self.speedup_max - self.speedup_min)
            ).clamp(0.0, 1.0)
        return torch.cat([t_tensor, psnr_tensor, speedup_tensor], dim=-1)


class GatedFeatureFusionAdaCacheGate(nn.Module):
    """Per-feature MLP encoders with condition-dependent softmax fusion."""

    def __init__(
        self,
        feature_dims: dict[str, int],
        hidden_dim: int = 64,
        feature_embedding_dim: int | None = None,
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        speedup_min: float = 1.0,
        speedup_max: float = 4.0,
        min_threshold: float = 0.10,
        max_threshold: float = 0.80,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_sets = normalize_feature_sets(tuple(feature_dims))
        self.feature_dims = {name: int(feature_dims[name]) for name in self.feature_sets}
        self.hidden_dim = hidden_dim
        self.feature_embedding_dim = (
            hidden_dim if feature_embedding_dim is None else feature_embedding_dim
        )
        self.normalize_inputs = normalize_inputs
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.speedup_min = speedup_min
        self.speedup_max = speedup_max
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.feature_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.feature_dims[name], hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.feature_embedding_dim),
                nn.SiLU(),
            )
            for name in self.feature_sets
        })
        self.cond_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(self.feature_sets)),
        )
        self.predict_head = nn.Sequential(
            nn.Linear(self.feature_embedding_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.last_gate_weights: torch.Tensor | None = None

    def forward(
        self,
        features: dict[str, torch.Tensor],
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
    ) -> torch.Tensor:
        first = features[self.feature_sets[0]]
        batch = first.shape[0]
        device = first.device
        encoded_features = []
        for name in self.feature_sets:
            feature = features[name].float()
            if feature.shape[0] != batch:
                raise ValueError(
                    f"Feature batch mismatch for {name}: "
                    f"expected {batch}, got {feature.shape[0]}"
                )
            flat = feature.flatten(start_dim=1)
            expected_dim = self.feature_dims[name]
            if flat.shape[1] != expected_dim:
                raise ValueError(
                    f"Feature dim mismatch for {name}: "
                    f"expected {expected_dim}, got {flat.shape[1]}"
                )
            encoded_features.append(self.feature_encoders[name](flat))

        encoded = torch.stack(encoded_features, dim=1)
        cond = self.cond_embed(
            self._prepare_condition(t, target_psnr, target_speedup, batch, device)
        )
        gate = torch.softmax(self.gate_head(cond), dim=-1)
        self.last_gate_weights = gate.detach()
        fused_feature = (encoded * gate.unsqueeze(-1)).sum(dim=1)
        return _threshold_from_unit(
            torch.sigmoid(self.predict_head(torch.cat([fused_feature, cond], dim=-1))),
            self.min_threshold,
            self.max_threshold,
        )

    def _prepare_condition(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
        psnr_tensor = torch.as_tensor(
            target_psnr, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        speedup_tensor = torch.as_tensor(
            target_speedup, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        if t_tensor.shape[0] == 1 and batch > 1:
            t_tensor = t_tensor.expand(batch, 1)
        if psnr_tensor.shape[0] == 1 and batch > 1:
            psnr_tensor = psnr_tensor.expand(batch, 1)
        if speedup_tensor.shape[0] == 1 and batch > 1:
            speedup_tensor = speedup_tensor.expand(batch, 1)
        if (
            t_tensor.shape[0] != batch
            or psnr_tensor.shape[0] != batch
            or speedup_tensor.shape[0] != batch
        ):
            raise ValueError(
                "Condition batch size mismatch: "
                f"feature batch={batch}, t={t_tensor.shape[0]}, "
                f"psnr={psnr_tensor.shape[0]}, speedup={speedup_tensor.shape[0]}"
            )
        if self.normalize_inputs:
            t_tensor = t_tensor.clamp(0.0, 1.0)
            psnr_tensor = (
                (psnr_tensor - self.psnr_min) / (self.psnr_max - self.psnr_min)
            ).clamp(0.0, 1.0)
            speedup_tensor = (
                (speedup_tensor - self.speedup_min)
                / (self.speedup_max - self.speedup_min)
            ).clamp(0.0, 1.0)
        return torch.cat([t_tensor, psnr_tensor, speedup_tensor], dim=-1)


class CachedGatedFeatureAdaCacheGate(nn.Module):
    """Gated feature fusion for precomputed pooled features."""

    def __init__(
        self,
        feature_dims: dict[str, int],
        hidden_dim: int = 64,
        feature_embedding_dim: int | None = None,
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        speedup_min: float = 1.0,
        speedup_max: float = 4.0,
        min_threshold: float = 0.10,
        max_threshold: float = 0.80,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fusion = GatedFeatureFusionAdaCacheGate(
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            feature_embedding_dim=feature_embedding_dim,
            normalize_inputs=normalize_inputs,
            psnr_min=psnr_min,
            psnr_max=psnr_max,
            speedup_min=speedup_min,
            speedup_max=speedup_max,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            dropout=dropout,
        )
        self.feature_sets = self.fusion.feature_sets
        self.feature_dims = self.fusion.feature_dims

    def forward(
        self,
        features: dict[str, torch.Tensor],
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
    ) -> torch.Tensor:
        return self.fusion(features, t, target_psnr, target_speedup)


class GatedMultiFeatureAdaCacheGate(nn.Module):
    """Gated feature fusion with raw-latent feature extraction inside the model."""

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_dim: int = 64,
        feature_embedding_dim: int | None = None,
        grid_size: tuple[int, int, int] = (2, 2, 2),
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        speedup_min: float = 1.0,
        speedup_max: float = 4.0,
        min_threshold: float = 0.10,
        max_threshold: float = 0.80,
        dropout: float = 0.0,
        feature_sets: Sequence[str] = DEFAULT_GATED_FEATURE_SETS,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.grid_size = grid_size
        self.feature_sets = normalize_feature_sets(feature_sets)
        self.pool = nn.AdaptiveAvgPool3d(grid_size)
        flat_feature_dim = (
            latent_channels * grid_size[0] * grid_size[1] * grid_size[2]
        )
        self.fusion = GatedFeatureFusionAdaCacheGate(
            feature_dims={name: flat_feature_dim for name in self.feature_sets},
            hidden_dim=hidden_dim,
            feature_embedding_dim=feature_embedding_dim,
            normalize_inputs=normalize_inputs,
            psnr_min=psnr_min,
            psnr_max=psnr_max,
            speedup_min=speedup_min,
            speedup_max=speedup_max,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            dropout=dropout,
        )

    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
    ) -> torch.Tensor:
        latent = ImprovedAdaCacheGate._normalize_latent_rank(latent)
        _, channels, frames, _, _ = latent.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected latent channel count {self.latent_channels}, got {channels}"
            )
        latent = latent.float()
        features = {
            name: self._extract_feature(latent, frames, name)
            for name in self.feature_sets
        }
        return self.fusion(features, t, target_psnr, target_speedup)

    def _extract_feature(
        self,
        latent: torch.Tensor,
        frames: int,
        feature_set: str,
    ) -> torch.Tensor:
        if feature_set == "latent_pool":
            feature_volume = latent
        elif feature_set == "temporal_mean":
            feature_volume = latent.mean(dim=2, keepdim=True).expand_as(latent)
        elif feature_set == "temporal_var":
            feature_volume = latent.var(dim=2, keepdim=True, unbiased=False).expand_as(
                latent
            )
        elif feature_set == "frame_diff_mean":
            feature_volume = ImprovedAdaCacheGate._frame_diff(latent, frames).mean(
                dim=2, keepdim=True
            )
            feature_volume = feature_volume.expand_as(latent)
        elif feature_set == "frame_diff_var":
            feature_volume = ImprovedAdaCacheGate._frame_diff(latent, frames).var(
                dim=2, keepdim=True, unbiased=False
            )
            feature_volume = feature_volume.expand_as(latent)
        else:
            raise AssertionError(f"Unhandled feature_set: {feature_set}")
        return self.pool(feature_volume).flatten(start_dim=1)


class ConditionOnlyAdaCacheGate(nn.Module):
    """Threshold predictor using only selected conditioning signals."""

    def __init__(
        self,
        hidden_dim: int = 64,
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        speedup_min: float = 1.0,
        speedup_max: float = 4.0,
        min_threshold: float = 0.10,
        max_threshold: float = 0.80,
        dropout: float = 0.0,
        condition_inputs: Sequence[str] = ("timestep", "target_psnr", "target_speedup"),
    ) -> None:
        super().__init__()
        valid_inputs = {"timestep", "target_psnr", "target_speedup"}
        unknown = set(condition_inputs) - valid_inputs
        if unknown:
            raise ValueError(f"Unknown condition inputs: {sorted(unknown)}")
        if "timestep" not in condition_inputs:
            raise ValueError("condition_inputs must include timestep")
        self.condition_inputs = tuple(condition_inputs)
        self.normalize_inputs = normalize_inputs
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.speedup_min = speedup_min
        self.speedup_max = speedup_max
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.cond_embed = nn.Sequential(
            nn.Linear(len(self.condition_inputs), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
        batch: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        cond = self._prepare_condition(t, target_psnr, target_speedup, batch, device)
        return _threshold_from_unit(
            torch.sigmoid(self.predict_head(self.cond_embed(cond))),
            self.min_threshold,
            self.max_threshold,
        )

    def _prepare_condition(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
        batch: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
        psnr_tensor = torch.as_tensor(
            target_psnr, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        speedup_tensor = torch.as_tensor(
            target_speedup, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        if batch is None:
            batch = max(t_tensor.shape[0], psnr_tensor.shape[0], speedup_tensor.shape[0])
        if t_tensor.shape[0] == 1 and batch > 1:
            t_tensor = t_tensor.expand(batch, 1)
        if psnr_tensor.shape[0] == 1 and batch > 1:
            psnr_tensor = psnr_tensor.expand(batch, 1)
        if speedup_tensor.shape[0] == 1 and batch > 1:
            speedup_tensor = speedup_tensor.expand(batch, 1)
        if (
            t_tensor.shape[0] != batch
            or psnr_tensor.shape[0] != batch
            or speedup_tensor.shape[0] != batch
        ):
            raise ValueError(
                "Condition batch size mismatch: "
                f"batch={batch}, t={t_tensor.shape[0]}, "
                f"psnr={psnr_tensor.shape[0]}, speedup={speedup_tensor.shape[0]}"
            )
        if self.normalize_inputs:
            t_tensor = t_tensor.clamp(0.0, 1.0)
            psnr_tensor = (
                (psnr_tensor - self.psnr_min) / (self.psnr_max - self.psnr_min)
            ).clamp(0.0, 1.0)
            speedup_tensor = (
                (speedup_tensor - self.speedup_min)
                / (self.speedup_max - self.speedup_min)
            ).clamp(0.0, 1.0)
        tensors = {
            "timestep": t_tensor,
            "target_psnr": psnr_tensor,
            "target_speedup": speedup_tensor,
        }
        return torch.cat([tensors[name] for name in self.condition_inputs], dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class GridMLPThresholdPredictor(nn.Module):
    """Capacity baseline over fixed 3D grid features.

    This model deliberately avoids attention. It is useful for checking whether
    MiniDiT gains come from token mixing or simply from using a higher-resolution
    grid feature than the old 2x2x2 pooled MLP.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int, int] = (16, 4, 5, 13),
        hidden_dim: int = 256,
        depth: int = 3,
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        speedup_min: float = 1.0,
        speedup_max: float = 4.0,
        min_threshold: float = 0.10,
        max_threshold: float = 0.80,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.grid_shape = grid_shape
        self.normalize_inputs = normalize_inputs
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.speedup_min = speedup_min
        self.speedup_max = speedup_max
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        input_dim = int(torch.tensor(grid_shape).prod().item()) + 3
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        grid_feature: torch.Tensor,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
    ) -> torch.Tensor:
        grid_feature = self._normalize_grid_rank(grid_feature).float()
        batch = grid_feature.shape[0]
        if tuple(grid_feature.shape[1:]) != tuple(self.grid_shape):
            raise ValueError(
                f"Expected grid feature shape {self.grid_shape}, "
                f"got {tuple(grid_feature.shape[1:])}"
            )
        cond = self._prepare_condition(
            t, target_psnr, target_speedup, batch, grid_feature.device
        )
        raw = self.net(torch.cat([grid_feature.flatten(start_dim=1), cond], dim=-1))
        return _threshold_from_unit(
            torch.sigmoid(raw),
            self.min_threshold,
            self.max_threshold,
        )

    @staticmethod
    def _normalize_grid_rank(grid_feature: torch.Tensor) -> torch.Tensor:
        if grid_feature.ndim == 4:
            return grid_feature.unsqueeze(0)
        if grid_feature.ndim == 5:
            return grid_feature
        raise ValueError(
            "Expected grid feature shape [C,T,H,W] or [B,C,T,H,W], "
            f"got {tuple(grid_feature.shape)}"
        )

    def _prepare_condition(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
        psnr_tensor = torch.as_tensor(
            target_psnr, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        speedup_tensor = torch.as_tensor(
            target_speedup, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        if t_tensor.shape[0] == 1 and batch > 1:
            t_tensor = t_tensor.expand(batch, 1)
        if psnr_tensor.shape[0] == 1 and batch > 1:
            psnr_tensor = psnr_tensor.expand(batch, 1)
        if speedup_tensor.shape[0] == 1 and batch > 1:
            speedup_tensor = speedup_tensor.expand(batch, 1)
        if (
            t_tensor.shape[0] != batch
            or psnr_tensor.shape[0] != batch
            or speedup_tensor.shape[0] != batch
        ):
            raise ValueError(
                "Condition batch size mismatch: "
                f"grid batch={batch}, t={t_tensor.shape[0]}, "
                f"psnr={psnr_tensor.shape[0]}, speedup={speedup_tensor.shape[0]}"
            )
        if self.normalize_inputs:
            t_tensor = t_tensor.clamp(0.0, 1.0)
            psnr_tensor = (
                (psnr_tensor - self.psnr_min) / (self.psnr_max - self.psnr_min)
            ).clamp(0.0, 1.0)
            speedup_tensor = (
                (speedup_tensor - self.speedup_min)
                / (self.speedup_max - self.speedup_min)
            ).clamp(0.0, 1.0)
        return torch.cat([t_tensor, psnr_tensor, speedup_tensor], dim=-1)


class MiniDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.05,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        if gate_init != 0.0:
            with torch.no_grad():
                self.modulation.bias[2 * dim:3 * dim].fill_(gate_init)
                self.modulation.bias[5 * dim:6 * dim].fill_(gate_init)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(cond).unsqueeze(1).chunk(6, dim=-1)
        )
        x_attn = self.norm1(x) * (1 + scale_attn) + shift_attn
        attn_out, _ = self.attn(x_attn, x_attn, x_attn, need_weights=False)
        x = x + gate_attn * attn_out
        x_mlp = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_mlp)
        return x


class MiniDiTCLSAdaptiveThresholdPredictor(nn.Module):
    """Small DiT-style predictor with learnable Conv3d patch embedding."""

    def __init__(
        self,
        input_shape: tuple[int, int, int, int] = (16, 12, 60, 104),
        patch_size: tuple[int, int, int] = (3, 12, 8),
        dim: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        speedup_min: float = 1.0,
        speedup_max: float = 4.0,
        min_threshold: float = 0.10,
        max_threshold: float = 0.80,
        dropout: float = 0.05,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        channels, frames, height, width = input_shape
        if (
            frames % patch_size[0] != 0
            or height % patch_size[1] != 0
            or width % patch_size[2] != 0
        ):
            raise ValueError(
                "input_shape must be divisible by patch_size: "
                f"input_shape={input_shape}, patch_size={patch_size}"
            )
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.grid_shape = (
            channels,
            frames // patch_size[0],
            height // patch_size[1],
            width // patch_size[2],
        )
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.normalize_inputs = normalize_inputs
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.speedup_min = speedup_min
        self.speedup_max = speedup_max
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.gate_init = gate_init

        self.patch_embedding = nn.Conv3d(
            channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, dim))
        _, grid_frames, grid_height, grid_width = self.grid_shape
        self.pos_t = nn.Parameter(torch.zeros(grid_frames, dim))
        self.pos_h = nn.Parameter(torch.zeros(grid_height, dim))
        self.pos_w = nn.Parameter(torch.zeros(grid_width, dim))
        self.pos_dropout = nn.Dropout(dropout)
        self.cond_embed = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            MiniDiTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                gate_init=gate_init,
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self._init_weights()

    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
    ) -> torch.Tensor:
        latent = self._normalize_latent_rank(latent).float()
        batch = latent.shape[0]
        if tuple(latent.shape[1:]) != tuple(self.input_shape):
            raise ValueError(
                f"Expected latent shape {self.input_shape}, "
                f"got {tuple(latent.shape[1:])}"
            )
        tokens = self.patch_embedding(latent)
        _, _, frames, height, width = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        pos = self._position_grid().unsqueeze(0).to(tokens.dtype)
        tokens = tokens + pos
        cls = self.cls_token.expand(batch, -1, -1) + self.cls_pos
        x = torch.cat([cls, tokens], dim=1)
        x = self.pos_dropout(x)
        cond = self.cond_embed(
            self._prepare_condition(t, target_psnr, target_speedup, batch, latent.device)
        )
        for block in self.blocks:
            x = block(x, cond)
        raw = self.head(x[:, 0])
        return _threshold_from_unit(
            torch.sigmoid(raw),
            self.min_threshold,
            self.max_threshold,
        )

    def _position_grid(self) -> torch.Tensor:
        _, frames, height, width = self.grid_shape
        pos = (
            self.pos_t[:, None, None, :]
            + self.pos_h[None, :, None, :]
            + self.pos_w[None, None, :, :]
        )
        return pos.reshape(frames * height * width, self.dim)

    @staticmethod
    def _normalize_latent_rank(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            return latent.unsqueeze(0)
        if latent.ndim == 5:
            return latent
        raise ValueError(
            "Expected latent shape [C,T,H,W] or [B,C,T,H,W], "
            f"got {tuple(latent.shape)}"
        )

    def _prepare_condition(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        target_speedup: torch.Tensor | float,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
        psnr_tensor = torch.as_tensor(
            target_psnr, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        speedup_tensor = torch.as_tensor(
            target_speedup, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        if t_tensor.shape[0] == 1 and batch > 1:
            t_tensor = t_tensor.expand(batch, 1)
        if psnr_tensor.shape[0] == 1 and batch > 1:
            psnr_tensor = psnr_tensor.expand(batch, 1)
        if speedup_tensor.shape[0] == 1 and batch > 1:
            speedup_tensor = speedup_tensor.expand(batch, 1)
        if (
            t_tensor.shape[0] != batch
            or psnr_tensor.shape[0] != batch
            or speedup_tensor.shape[0] != batch
        ):
            raise ValueError(
                "Condition batch size mismatch: "
                f"latent batch={batch}, t={t_tensor.shape[0]}, "
                f"psnr={psnr_tensor.shape[0]}, speedup={speedup_tensor.shape[0]}"
            )
        if self.normalize_inputs:
            t_tensor = t_tensor.clamp(0.0, 1.0)
            psnr_tensor = (
                (psnr_tensor - self.psnr_min) / (self.psnr_max - self.psnr_min)
            ).clamp(0.0, 1.0)
            speedup_tensor = (
                (speedup_tensor - self.speedup_min)
                / (self.speedup_max - self.speedup_min)
            ).clamp(0.0, 1.0)
        return torch.cat([t_tensor, psnr_tensor, speedup_tensor], dim=-1)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        nn.init.trunc_normal_(self.pos_h, std=0.02)
        nn.init.trunc_normal_(self.pos_w, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight.flatten(1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.modulation.weight)
            nn.init.zeros_(block.modulation.bias)
            if self.gate_init != 0.0:
                block.modulation.bias[2 * self.dim:3 * self.dim].data.fill_(
                    self.gate_init
                )
                block.modulation.bias[5 * self.dim:6 * self.dim].data.fill_(
                    self.gate_init
                )
