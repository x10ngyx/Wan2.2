from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class IQLModelConfig:
    input_dim: int
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.0


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class ValueNet(nn.Module):
    def __init__(self, config: IQLModelConfig) -> None:
        super().__init__()
        self.net = build_mlp(
            input_dim=config.input_dim,
            output_dim=1,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, config: IQLModelConfig) -> None:
        super().__init__()
        self.net = build_mlp(
            input_dim=config.input_dim,
            output_dim=2,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PolicyNet(nn.Module):
    def __init__(self, config: IQLModelConfig) -> None:
        super().__init__()
        self.net = build_mlp(
            input_dim=config.input_dim,
            output_dim=2,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def gather_action_values(q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return q_values.gather(1, actions.long().view(-1, 1)).squeeze(1)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(
        diff < 0,
        torch.full_like(diff, 1.0 - expectile),
        torch.full_like(diff, expectile),
    )
    return (weight * diff.pow(2)).mean()


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, rho: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(rho).add_(source_param.data, alpha=1.0 - rho)
