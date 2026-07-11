from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


DEFAULT_FEATURE_SETS = (
    "latent_pool",
    "temporal_mean",
    "temporal_var",
    "frame_diff_mean",
    "frame_diff_var",
)


@dataclass
class IQLDatasetBundle:
    states: torch.Tensor
    next_states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    sample_ids: list[str]
    step_indices: torch.Tensor
    thresholds: torch.Tensor
    target_speedups: torch.Tensor
    achieved_speedups: torch.Tensor
    psnr: torch.Tensor
    latent_mse: torch.Tensor
    manifest: dict[str, object]


class TensorTransitionDataset(Dataset):
    def __init__(self, bundle: IQLDatasetBundle, indices: Sequence[int]) -> None:
        self.bundle = bundle
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        return {
            "state": self.bundle.states[index],
            "next_state": self.bundle.next_states[index],
            "action": self.bundle.actions[index],
            "reward": self.bundle.rewards[index],
            "done": self.bundle.dones[index],
        }


def load_summary_rows(data_root: Path) -> list[dict[str, str]]:
    summary_csv = data_root / "data" / "tables" / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary table: {summary_csv}")
    with summary_csv.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_step_list_map(serialized: str) -> set[int]:
    """Parse summary strings like ``('low', 'cond'):[1, 2] | ...``.

    The OpenVid SeaCache traces record the same per-step decision for cond and
    uncond branches.  We deliberately union all listed steps, which keeps the RL
    label synchronized and avoids branch-order assumptions.
    """

    steps: set[int] = set()
    if not serialized:
        return steps
    for part in serialized.split(" | "):
        if ":" not in part:
            continue
        _, value = part.split(":", 1)
        value = value.strip()
        if not value:
            continue
        parsed = ast.literal_eval(value)
        for step in parsed:
            steps.add(int(step))
    return steps


def build_iql_bundle(
    feature_cache: Path,
    data_root: Path,
    feature_sets: Sequence[str] = DEFAULT_FEATURE_SETS,
    target_speedup_offsets: Sequence[float] = (-0.3, -0.15, 0.0, 0.15, 0.3),
    min_target_speedup: float = 1.0,
    max_target_speedup: float = 4.0,
    max_examples: int | None = None,
    lambda_latent: float = 5.0,
    lambda_recompute: float = 0.04,
    lambda_psnr: float = 1.0,
    lambda_speedup: float = 30.0,
    reuse_cost_ratio: float = 0.081,
    latent_mse_cache: Path | None = None,
) -> IQLDatasetBundle:
    metadata_path = feature_cache / "metadata.pt"
    manifest_path = feature_cache / "manifest.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature-cache metadata: {metadata_path}")

    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    sample_ids = list(metadata["sample_id"])
    step_indices = metadata["step_index"].long()
    source_index = metadata["source_index"].long()
    thresholds = metadata["threshold"].float()
    feature_tensors = []
    for feature_set in feature_sets:
        path = feature_cache / f"features_{feature_set}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached feature tensor: {path}")
        feature_tensors.append(torch.load(path, map_location="cpu", weights_only=True).float())
    row_counts = {int(tensor.shape[0]) for tensor in feature_tensors}
    if len(row_counts) != 1:
        raise ValueError(f"Feature row-count mismatch: {[tuple(x.shape) for x in feature_tensors]}")
    features = torch.cat([tensor.flatten(start_dim=1) for tensor in feature_tensors], dim=1)

    if max_examples is not None:
        features = features[:max_examples]
        sample_ids = sample_ids[:max_examples]
        step_indices = step_indices[:max_examples]
        source_index = source_index[:max_examples]
        thresholds = thresholds[:max_examples]

    num_examples = int(features.shape[0])
    num_steps = int(step_indices.max().item()) + 1
    rows = load_summary_rows(data_root)
    candidate_count = len(rows)
    if candidate_count == 0:
        raise RuntimeError(f"No summary rows found under {data_root}")

    skip_sets = [parse_step_list_map(row.get("seacache_skipping_path", "")) for row in rows]
    psnr_by_candidate = torch.tensor(
        [float(row["mean_psnr"]) for row in rows],
        dtype=torch.float32,
    )
    speedup_by_candidate = torch.tensor(
        [float(row["speedup"]) for row in rows],
        dtype=torch.float32,
    )

    actions = torch.zeros(num_examples, dtype=torch.long)
    psnr = torch.zeros(num_examples, dtype=torch.float32)
    achieved_speedup = torch.zeros(num_examples, dtype=torch.float32)
    candidate_indices = torch.div(source_index[:num_examples], num_steps, rounding_mode="floor")
    if int(candidate_indices.max().item()) >= candidate_count:
        raise ValueError(
            "Feature metadata source_index does not match summary row count: "
            f"max candidate index {int(candidate_indices.max().item())}, rows {candidate_count}"
        )
    for index in range(num_examples):
        candidate_index = int(candidate_indices[index].item())
        step = int(step_indices[index].item())
        actions[index] = 1 if step in skip_sets[candidate_index] else 0
        psnr[index] = psnr_by_candidate[candidate_index]
        achieved_speedup[index] = speedup_by_candidate[candidate_index]
    proxy_achieved_speedup = compute_final_speedup_proxy(
        actions=actions,
        source_index=source_index[:num_examples],
        num_steps=num_steps,
        reuse_cost_ratio=reuse_cost_ratio,
    )

    latent_mse = load_or_compute_latent_mse(
        data_root=data_root,
        rows=rows,
        source_index=source_index[:num_examples],
        step_indices=step_indices[:num_examples],
        num_steps=num_steps,
        cache_path=latent_mse_cache,
    )

    target_offsets = torch.tensor(list(target_speedup_offsets), dtype=torch.float32)
    if int(target_offsets.numel()) == 0:
        raise ValueError("target_speedup_offsets must not be empty")
    target_speedups = build_local_target_speedups(
        achieved_speedup=achieved_speedup,
        offsets=target_offsets,
        min_target=min_target_speedup,
        max_target=max_target_speedup,
    )
    features = repeat_by_local_targets(features, target_offsets)
    actions = repeat_by_local_targets(actions, target_offsets)
    psnr = repeat_by_local_targets(psnr, target_offsets)
    achieved_speedup = repeat_by_local_targets(achieved_speedup, target_offsets)
    proxy_achieved_speedup = repeat_by_local_targets(proxy_achieved_speedup, target_offsets)
    latent_mse = repeat_by_local_targets(latent_mse, target_offsets)
    thresholds = repeat_by_local_targets(thresholds[:num_examples], target_offsets)
    step_indices = repeat_by_local_targets(step_indices[:num_examples], target_offsets)
    source_index = repeat_by_local_targets(source_index[:num_examples], target_offsets)
    sample_ids = [
        sample_id
        for _ in range(int(target_offsets.numel()))
        for sample_id in sample_ids[:num_examples]
    ]
    num_examples = int(features.shape[0])

    scalar_state = build_scalar_state(
        step_indices=step_indices,
        actions=actions,
        target_speedups=target_speedups,
        source_index=source_index,
        num_steps=num_steps,
        reuse_cost_ratio=reuse_cost_ratio,
    )
    states = torch.cat([features, scalar_state], dim=1)

    next_states = states.clone()
    dones = torch.zeros(num_examples, dtype=torch.float32)
    for index in range(num_examples):
        step = int(step_indices[index].item())
        if step >= num_steps - 1 or index + 1 >= num_examples:
            dones[index] = 1.0
            next_states[index] = states[index]
            continue
        same_target_copy = (
            int(source_index[index + 1].item()) == int(source_index[index].item()) + 1
            and abs(float(target_speedups[index + 1].item()) - float(target_speedups[index].item())) < 1e-6
        )
        if same_target_copy:
            next_states[index] = states[index + 1]
        else:
            dones[index] = 1.0
            next_states[index] = states[index]

    immediate = (
        -lambda_latent * latent_mse
        -lambda_recompute * (1.0 - actions.float())
    )
    terminal = (
        lambda_psnr * psnr
        -lambda_speedup * torch.abs(proxy_achieved_speedup - target_speedups)
    ) * dones
    rewards = immediate + terminal

    source_manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r") as handle:
            source_manifest = json.load(handle)
    manifest = {
        "feature_cache": str(feature_cache),
        "data_root": str(data_root),
        "feature_sets": list(feature_sets),
        "target_speedup_offsets": [float(value) for value in target_offsets.tolist()],
        "min_target_speedup": min_target_speedup,
        "max_target_speedup": max_target_speedup,
        "feature_dim": int(features.shape[1]),
        "scalar_dim": int(scalar_state.shape[1]),
        "state_dim": int(states.shape[1]),
        "num_examples": num_examples,
        "num_steps": num_steps,
        "lambda_latent": lambda_latent,
        "lambda_recompute": lambda_recompute,
        "lambda_psnr": lambda_psnr,
        "lambda_speedup": lambda_speedup,
        "reuse_cost_ratio": reuse_cost_ratio,
        "terminal_speedup_source": "final_action_cost_proxy",
        "latent_mse_cache": str(latent_mse_cache) if latent_mse_cache else None,
        "source_feature_manifest": source_manifest,
    }
    return IQLDatasetBundle(
        states=states,
        next_states=next_states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        sample_ids=sample_ids,
        step_indices=step_indices,
        thresholds=thresholds,
        target_speedups=target_speedups,
        achieved_speedups=achieved_speedup,
        psnr=psnr,
        latent_mse=latent_mse,
        manifest=manifest,
    )


def repeat_by_local_targets(values: torch.Tensor, target_offsets: torch.Tensor) -> torch.Tensor:
    repeat_count = int(target_offsets.numel())
    repeat_shape = [repeat_count] + [1] * (values.ndim - 1)
    return values.repeat(*repeat_shape)


def build_local_target_speedups(
    achieved_speedup: torch.Tensor,
    offsets: torch.Tensor,
    min_target: float,
    max_target: float,
) -> torch.Tensor:
    targets = []
    for offset in offsets.tolist():
        targets.append((achieved_speedup + float(offset)).clamp(min_target, max_target))
    return torch.cat(targets, dim=0)


def load_or_compute_latent_mse(
    data_root: Path,
    rows: list[dict[str, str]],
    source_index: torch.Tensor,
    step_indices: torch.Tensor,
    num_steps: int,
    cache_path: Path | None,
    progress_every: int = 0,
    save_every: int = 0,
) -> torch.Tensor:
    expected_count = int(source_index.numel())
    start_index = 0
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True).float()
        if int(cached.numel()) >= expected_count:
            return cached[:expected_count]
        start_index = int(cached.numel())
        values = torch.empty(expected_count, dtype=torch.float32)
        values[:start_index] = cached
    else:
        values = torch.empty(expected_count, dtype=torch.float32)

    for row_index in range(start_index, expected_count):
        candidate_index = int(source_index[row_index].item()) // num_steps
        step = int(step_indices[row_index].item())
        row = rows[candidate_index]
        candidate_step_root = data_root / row["seacache_step_inputs"]
        baseline_step_root = data_root / row["baseline_step_inputs"]
        if step + 1 < num_steps:
            candidate_latent = torch.load(
                candidate_step_root / f"step_{step + 1:03d}.pt",
                map_location="cpu",
                weights_only=True,
            )["latent"].float()
            baseline_latent = torch.load(
                baseline_step_root / f"step_{step + 1:03d}.pt",
                map_location="cpu",
                weights_only=True,
            )["latent"].float()
        else:
            candidate_latent = torch.load(
                candidate_step_root / "trace_done.pt",
                map_location="cpu",
                weights_only=True,
            )["final_latent"].float()
            baseline_latent = torch.load(
                baseline_step_root / "trace_done.pt",
                map_location="cpu",
                weights_only=True,
            )["final_latent"].float()
        values[row_index] = torch.mean((candidate_latent - baseline_latent).pow(2))
        completed = row_index + 1
        if cache_path is not None and save_every > 0 and completed % save_every == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(values[:completed], cache_path)
        if progress_every > 0 and (
            completed % progress_every == 0 or completed == expected_count
        ):
            print(
                json.dumps({
                    "latent_mse_completed": completed,
                    "latent_mse_total": expected_count,
                }),
                flush=True,
            )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(values, cache_path)
    return values


def compute_final_speedup_proxy(
    actions: torch.Tensor,
    source_index: torch.Tensor,
    num_steps: int,
    reuse_cost_ratio: float,
) -> torch.Tensor:
    candidate_indices = torch.div(source_index, num_steps, rounding_mode="floor")
    result = torch.empty_like(actions, dtype=torch.float32)
    unique_candidates = torch.unique(candidate_indices, sorted=True)
    for candidate in unique_candidates.tolist():
        mask = candidate_indices == int(candidate)
        candidate_actions = actions[mask]
        reuse_count = int(candidate_actions.sum().item())
        recompute_count = int(candidate_actions.numel()) - reuse_count
        total_cost = recompute_count + reuse_cost_ratio * reuse_count
        speedup = num_steps / max(total_cost, 1e-6)
        result[mask] = float(speedup)
    return result


def build_scalar_state(
    step_indices: torch.Tensor,
    actions: torch.Tensor,
    target_speedups: torch.Tensor,
    source_index: torch.Tensor,
    num_steps: int,
    reuse_cost_ratio: float,
) -> torch.Tensor:
    scalars = torch.zeros((int(step_indices.numel()), 4), dtype=torch.float32)
    current_traj_start = 0
    reuse_count = 0
    recompute_count = 0
    consecutive_skip = 0
    last_candidate = None
    for row_index in range(int(step_indices.numel())):
        candidate = int(source_index[row_index].item()) // num_steps
        if candidate != last_candidate:
            current_traj_start = row_index
            reuse_count = 0
            recompute_count = 0
            consecutive_skip = 0
            last_candidate = candidate
        step = int(step_indices[row_index].item())
        steps_seen = max(row_index - current_traj_start, 0)
        if steps_seen == 0:
            current_speedup_proxy = 1.0
        else:
            actual_cost = recompute_count + reuse_cost_ratio * reuse_count
            remaining_baseline_cost = max(num_steps - steps_seen, 0)
            projected_full_cost = actual_cost + remaining_baseline_cost
            current_speedup_proxy = num_steps / max(projected_full_cost, 1e-6)
        scalars[row_index] = torch.tensor(
            [
                step / max(num_steps - 1, 1),
                float(target_speedups[row_index]),
                current_speedup_proxy,
                consecutive_skip / max(num_steps, 1),
            ],
            dtype=torch.float32,
        )
        if int(actions[row_index].item()) == 1:
            reuse_count += 1
            consecutive_skip += 1
        else:
            recompute_count += 1
            consecutive_skip = 0
    return scalars


def split_indices_by_sample_id(
    sample_ids: Sequence[str],
    train_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    unique = sorted(set(sample_ids))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(unique), generator=generator).tolist()
    shuffled = [unique[index] for index in order]
    train_count = max(1, int(len(shuffled) * train_fraction))
    if train_count >= len(shuffled) and len(shuffled) > 1:
        train_count = len(shuffled) - 1
    train_samples = set(shuffled[:train_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for index, sample_id in enumerate(sample_ids):
        if sample_id in train_samples:
            train_indices.append(index)
        else:
            val_indices.append(index)
    return train_indices, val_indices


def compute_normalizer(states: torch.Tensor, indices: Sequence[int]) -> dict[str, torch.Tensor]:
    selected = states[torch.tensor(indices, dtype=torch.long)]
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {"mean": mean, "std": std}


def apply_normalizer(states: torch.Tensor, normalizer: dict[str, torch.Tensor]) -> torch.Tensor:
    return (states - normalizer["mean"]) / normalizer["std"]
