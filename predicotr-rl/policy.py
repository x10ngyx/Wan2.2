from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models import IQLModelConfig, PolicyNet  # noqa: E402


class SeaCacheRLPolicy:
    """Load an IQL policy checkpoint and return synchronized skip decisions.

    The wrapper intentionally accepts the same five latent feature tensors used
    by the previous adaptive predictor line.  It does not own SeaCache state;
    the caller passes the current progress counters so cond/uncond branches can
    share one decision per denoising step.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
        policy_threshold: float = 0.5,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.policy_threshold = policy_threshold
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.feature_sets = tuple(checkpoint["dataset_manifest"]["feature_sets"])
        self.reuse_cost_ratio = float(
            checkpoint["dataset_manifest"].get("reuse_cost_ratio", 0.081)
        )
        self.normalizer = {
            key: value.to(self.device).float()
            for key, value in checkpoint["normalizer"].items()
        }
        config = IQLModelConfig(**checkpoint["model_config"])
        self.policy_net = PolicyNet(config).to(self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.policy_net.eval()

    @torch.no_grad()
    def decide(
        self,
        features: Mapping[str, torch.Tensor],
        *,
        step_index: int,
        num_steps: int,
        target_speedup: float,
        reuse_count: int,
        recompute_count: int,
        consecutive_skip: int,
    ) -> dict[str, float | int]:
        state = self.build_state(
            features=features,
            step_index=step_index,
            num_steps=num_steps,
            target_speedup=target_speedup,
            reuse_count=reuse_count,
            recompute_count=recompute_count,
            consecutive_skip=consecutive_skip,
        )
        logits = self.policy_net(state)
        prob_skip = F.softmax(logits, dim=-1)[0, 1].item()
        action = int(prob_skip >= self.policy_threshold)
        return {
            "action": action,
            "prob_skip": prob_skip,
            "prob_recompute": 1.0 - prob_skip,
        }

    def build_state(
        self,
        features: Mapping[str, torch.Tensor],
        *,
        step_index: int,
        num_steps: int,
        target_speedup: float,
        reuse_count: int,
        recompute_count: int,
        consecutive_skip: int,
    ) -> torch.Tensor:
        feature_parts = []
        for feature_set in self.feature_sets:
            if feature_set not in features:
                raise KeyError(f"Missing RL policy feature {feature_set!r}")
            feature = features[feature_set].detach().to(self.device).float()
            feature_parts.append(feature.reshape(1, -1))
        feature_vector = torch.cat(feature_parts, dim=1)
        steps_seen = max(reuse_count + recompute_count, 0)
        if steps_seen == 0:
            current_speedup_proxy = 1.0
        else:
            actual_cost = recompute_count + self.reuse_cost_ratio * reuse_count
            remaining_baseline_cost = max(num_steps - steps_seen, 0)
            projected_full_cost = actual_cost + remaining_baseline_cost
            current_speedup_proxy = num_steps / max(projected_full_cost, 1e-6)
        scalar = torch.tensor(
            [[
                step_index / max(num_steps - 1, 1),
                float(target_speedup),
                float(current_speedup_proxy),
                consecutive_skip / max(num_steps, 1),
            ]],
            dtype=torch.float32,
            device=self.device,
        )
        state = torch.cat([feature_vector, scalar], dim=1)
        return (state - self.normalizer["mean"]) / self.normalizer["std"]
