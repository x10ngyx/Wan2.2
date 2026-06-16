from __future__ import annotations

import torch
from torch import nn


FEATURE_SETS = (
    "latent_pool",
    "temporal_mean",
    "temporal_var",
    "frame_diff_mean",
    "frame_diff_var",
)


class ImprovedAdaCacheGate(nn.Module):
    """Lightweight timestep-cache threshold predictor.

    Inputs:
        latent: Either [B, C, T, H, W] or a single trace tensor [C, T, H, W].
        t: Normalized or raw timestep tensor, shape [B], [B, 1], or scalar.
        target_psnr: Normalized or raw target PSNR tensor, shape [B], [B, 1], or scalar.

    Output:
        One threshold in [0, 1] for the current timestep-cache experiment,
        shape [B, 1].

    The condition path always receives timestep and target PSNR. The feature
    path is selected by ``feature_set`` while keeping the same output dimension,
    so feature ablations compare input information rather than a different
    prediction head.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        hidden_dim: int = 64,
        grid_size: tuple[int, int, int] = (2, 2, 2),
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
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
            nn.Linear(2, hidden_dim),
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
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
    ) -> torch.Tensor:
        latent = self._normalize_latent_rank(latent)
        batch, channels, frames, _, _ = latent.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected latent channel count {self.latent_channels}, got {channels}"
            )

        latent = latent.float()
        feat_latent = self.feature_proj(self._extract_feature(latent, frames))
        feat_cond = self.cond_embed(
            self._prepare_condition(t, target_psnr, batch, latent.device)
        )
        fused = torch.cat([feat_latent, feat_cond], dim=-1)
        return self.predict_head(fused)

    def _extract_feature(self, latent: torch.Tensor, frames: int) -> torch.Tensor:
        if self.feature_set == "latent_pool":
            feature_volume = latent
        elif self.feature_set == "temporal_mean":
            feature_volume = latent.mean(dim=2, keepdim=True).expand_as(latent)
        elif self.feature_set == "temporal_var":
            feature_volume = latent.var(dim=2, keepdim=True, unbiased=False).expand_as(
                latent
            )
        elif self.feature_set == "frame_diff_mean":
            feature_volume = self._frame_diff(latent, frames).mean(
                dim=2, keepdim=True
            )
            feature_volume = feature_volume.expand_as(latent)
        elif self.feature_set == "frame_diff_var":
            feature_volume = self._frame_diff(latent, frames).var(
                dim=2, keepdim=True, unbiased=False
            )
            feature_volume = feature_volume.expand_as(latent)
        else:
            raise AssertionError(f"Unhandled feature_set: {self.feature_set}")
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
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
        psnr_tensor = torch.as_tensor(
            target_psnr, dtype=torch.float32, device=device
        ).reshape(-1, 1)

        if t_tensor.shape[0] == 1 and batch > 1:
            t_tensor = t_tensor.expand(batch, 1)
        if psnr_tensor.shape[0] == 1 and batch > 1:
            psnr_tensor = psnr_tensor.expand(batch, 1)
        if t_tensor.shape[0] != batch or psnr_tensor.shape[0] != batch:
            raise ValueError(
                "Condition batch size mismatch: "
                f"latent batch={batch}, t={t_tensor.shape[0]}, psnr={psnr_tensor.shape[0]}"
            )

        if self.normalize_inputs:
            t_tensor = t_tensor.clamp(0.0, 1.0)
            psnr_tensor = (
                (psnr_tensor - self.psnr_min) / (self.psnr_max - self.psnr_min)
            ).clamp(0.0, 1.0)

        return torch.cat([t_tensor, psnr_tensor], dim=-1)


class CachedFeatureAdaCacheGate(nn.Module):
    """Same prediction trunk as ImprovedAdaCacheGate, with precomputed features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
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
            nn.Sigmoid(),
        )

    def forward(
        self,
        feature: torch.Tensor,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
    ) -> torch.Tensor:
        feature = feature.float()
        batch = feature.shape[0]
        feat_latent = self.feature_proj(feature)
        feat_cond = self.cond_embed(
            self._prepare_condition(t, target_psnr, batch, feature.device)
        )
        return self.predict_head(torch.cat([feat_latent, feat_cond], dim=-1))

    def _prepare_condition(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
        psnr_tensor = torch.as_tensor(
            target_psnr, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        if t_tensor.shape[0] == 1 and batch > 1:
            t_tensor = t_tensor.expand(batch, 1)
        if psnr_tensor.shape[0] == 1 and batch > 1:
            psnr_tensor = psnr_tensor.expand(batch, 1)
        if self.normalize_inputs:
            t_tensor = t_tensor.clamp(0.0, 1.0)
            psnr_tensor = (
                (psnr_tensor - self.psnr_min) / (self.psnr_max - self.psnr_min)
            ).clamp(0.0, 1.0)
        return torch.cat([t_tensor, psnr_tensor], dim=-1)


class ConditionOnlyAdaCacheGate(nn.Module):
    """Threshold predictor using only timestep and target PSNR."""

    def __init__(
        self,
        hidden_dim: int = 64,
        normalize_inputs: bool = True,
        psnr_min: float = 10.0,
        psnr_max: float = 50.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.cond_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
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
            nn.Sigmoid(),
        )

    def forward(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        batch: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        cond = self._prepare_condition(t, target_psnr, batch, device)
        return self.predict_head(self.cond_embed(cond))

    def _prepare_condition(
        self,
        t: torch.Tensor | float,
        target_psnr: torch.Tensor | float,
        batch: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
        psnr_tensor = torch.as_tensor(
            target_psnr, dtype=torch.float32, device=device
        ).reshape(-1, 1)
        if batch is None:
            batch = max(t_tensor.shape[0], psnr_tensor.shape[0])
        if t_tensor.shape[0] == 1 and batch > 1:
            t_tensor = t_tensor.expand(batch, 1)
        if psnr_tensor.shape[0] == 1 and batch > 1:
            psnr_tensor = psnr_tensor.expand(batch, 1)
        if t_tensor.shape[0] != batch or psnr_tensor.shape[0] != batch:
            raise ValueError(
                "Condition batch size mismatch: "
                f"batch={batch}, t={t_tensor.shape[0]}, psnr={psnr_tensor.shape[0]}"
            )
        if self.normalize_inputs:
            t_tensor = t_tensor.clamp(0.0, 1.0)
            psnr_tensor = (
                (psnr_tensor - self.psnr_min) / (self.psnr_max - self.psnr_min)
            ).clamp(0.0, 1.0)
        return torch.cat([t_tensor, psnr_tensor], dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
