from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = Path("/hy-tmp/openvid_100_seacache_trace_data")
DEFAULT_TARGET_PSNRS = (20.0, 22.0, 25.0, 28.0, 30.0, 35.0)
DEFAULT_STEP_INDICES = tuple(range(50))
DATASET_MODES = ("candidate_inverse", "target_oracle")


@dataclass(frozen=True)
class GateExample:
    sample_id: str
    step_path: Path
    step_index: int
    num_steps: int
    timestep: float
    target_psnr: float
    threshold: float
    source_threshold: float
    mean_psnr: float
    speedup: float


def load_summary_rows(data_root: Path = DEFAULT_DATA_ROOT) -> list[dict[str, str]]:
    summary_csv = data_root / "data" / "tables" / "summary.csv"
    with summary_csv.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def make_oracle_threshold_examples(
    data_root: Path = DEFAULT_DATA_ROOT,
    target_psnrs: Iterable[float] = DEFAULT_TARGET_PSNRS,
    step_indices: Iterable[int] = DEFAULT_STEP_INDICES,
    max_examples: int | None = None,
) -> list[GateExample]:
    """Build direct-threshold labels from a threshold sweep table.

    For each sample and target PSNR, the label is the fastest threshold whose
    measured PSNR is at least the target. If no candidate reaches the target,
    the highest-PSNR candidate is used. This keeps the first-stage direct
    threshold predictor trainable with the existing SeaCache sweep data.
    """

    rows_by_sample: dict[str, list[dict[str, str]]] = {}
    for row in load_summary_rows(data_root):
        rows_by_sample.setdefault(row["sample_id"], []).append(row)

    examples: list[GateExample] = []
    for sample_id, rows in rows_by_sample.items():
        parsed = [
            {
                **row,
                "_threshold": float(row["threshold"]),
                "_mean_psnr": float(row["mean_psnr"]),
                "_speedup": float(row["speedup"]),
            }
            for row in rows
        ]
        step_root = data_root / "data" / "baseline" / "step_inputs" / sample_id
        meta_path = step_root / "meta.pt"
        meta = torch.load(meta_path, map_location="cpu")
        timesteps = meta["timesteps"]
        for target_psnr in target_psnrs:
            feasible = [row for row in parsed if row["_mean_psnr"] >= target_psnr]
            if feasible:
                best = max(feasible, key=lambda row: row["_speedup"])
            else:
                best = max(parsed, key=lambda row: row["_mean_psnr"])
            for step_index in step_indices:
                step_path = step_root / f"step_{step_index:03d}.pt"
                if not step_path.exists():
                    continue
                examples.append(
                    GateExample(
                        sample_id=sample_id,
                        step_path=step_path,
                        step_index=step_index,
                        num_steps=len(timesteps),
                        timestep=float(timesteps[step_index]),
                        target_psnr=float(target_psnr),
                        threshold=float(best["_threshold"]),
                        source_threshold=float(best["_threshold"]),
                        mean_psnr=float(best["_mean_psnr"]),
                        speedup=float(best["_speedup"]),
                    )
                )
                if max_examples is not None and len(examples) >= max_examples:
                    return examples
    return examples


def make_candidate_inverse_examples(
    data_root: Path = DEFAULT_DATA_ROOT,
    step_indices: Iterable[int] = DEFAULT_STEP_INDICES,
    max_examples: int | None = None,
) -> list[GateExample]:
    """Build candidate-wise inverse labels.

    Each measured threshold candidate contributes one example per selected
    denoising step:

        input: candidate run latent at step, step fraction, achieved PSNR
        label: threshold used by that candidate run

    With 100 samples, 10 threshold candidates, and 50 steps, this creates
    50,000 examples.
    """

    examples: list[GateExample] = []
    timesteps_cache: dict[Path, list[float]] = {}
    for row in load_summary_rows(data_root):
        sample_id = row["sample_id"]
        threshold = float(row["threshold"])
        achieved_psnr = float(row["mean_psnr"])
        speedup = float(row["speedup"])
        step_root = data_root / row["seacache_step_inputs"]
        if step_root not in timesteps_cache:
            meta_path = step_root / "meta.pt"
            meta = torch.load(meta_path, map_location="cpu")
            timesteps_cache[step_root] = meta["timesteps"]
        timesteps = timesteps_cache[step_root]
        for step_index in step_indices:
            step_path = step_root / f"step_{step_index:03d}.pt"
            if not step_path.exists():
                continue
            examples.append(
                GateExample(
                    sample_id=sample_id,
                    step_path=step_path,
                    step_index=step_index,
                    num_steps=len(timesteps),
                    timestep=float(timesteps[step_index]),
                    target_psnr=achieved_psnr,
                    threshold=threshold,
                    source_threshold=threshold,
                    mean_psnr=achieved_psnr,
                    speedup=speedup,
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    return examples


class TraceStepThresholdDataset(Dataset):
    def __init__(
        self,
        data_root: Path = DEFAULT_DATA_ROOT,
        dataset_mode: str = "candidate_inverse",
        target_psnrs: Iterable[float] = DEFAULT_TARGET_PSNRS,
        step_indices: Iterable[int] = DEFAULT_STEP_INDICES,
        max_examples: int | None = None,
    ) -> None:
        if dataset_mode not in DATASET_MODES:
            raise ValueError(
                f"Unknown dataset_mode {dataset_mode!r}; expected one of {DATASET_MODES}"
            )
        self.dataset_mode = dataset_mode
        if dataset_mode == "candidate_inverse":
            self.examples = make_candidate_inverse_examples(
                data_root=data_root,
                step_indices=step_indices,
                max_examples=max_examples,
            )
        else:
            self.examples = make_oracle_threshold_examples(
                data_root=data_root,
                target_psnrs=target_psnrs,
                step_indices=step_indices,
                max_examples=max_examples,
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        payload = torch.load(example.step_path, map_location="cpu")
        latent = payload["latent"].float()
        step_denominator = max(example.num_steps - 1, 1)
        step_fraction = example.step_index / step_denominator
        return {
            "sample_id": example.sample_id,
            "latent": latent,
            "timestep": torch.tensor([step_fraction], dtype=torch.float32),
            "target_psnr": torch.tensor([example.target_psnr], dtype=torch.float32),
            "threshold": torch.tensor([example.threshold], dtype=torch.float32),
        }


def collate_trace_steps(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, object]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "latent": torch.stack([item["latent"] for item in batch]),  # type: ignore[arg-type]
        "timestep": torch.stack([item["timestep"] for item in batch]),  # type: ignore[arg-type]
        "target_psnr": torch.stack([item["target_psnr"] for item in batch]),  # type: ignore[arg-type]
        "threshold": torch.stack([item["threshold"] for item in batch]),  # type: ignore[arg-type]
    }


def split_indices_by_sample_id(
    dataset: TraceStepThresholdDataset,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    sample_ids = sorted({example.sample_id for example in dataset.examples})
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(sample_ids), generator=generator).tolist()
    shuffled_sample_ids = [sample_ids[index] for index in permutation]
    train_sample_count = max(1, int(len(shuffled_sample_ids) * train_fraction))
    if train_sample_count >= len(shuffled_sample_ids) and len(shuffled_sample_ids) > 1:
        train_sample_count = len(shuffled_sample_ids) - 1

    train_samples = set(shuffled_sample_ids[:train_sample_count])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for index, example in enumerate(dataset.examples):
        if example.sample_id in train_samples:
            train_indices.append(index)
        else:
            val_indices.append(index)
    return train_indices, val_indices


class CachedFeatureThresholdDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        feature_set: str,
        max_examples: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.feature_set = feature_set
        feature_path = cache_dir / f"features_{feature_set}.pt"
        metadata_path = cache_dir / "metadata.pt"
        manifest_path = cache_dir / "manifest.json"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing cached feature file: {feature_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing cached metadata file: {metadata_path}")

        self.features = torch.load(feature_path, map_location="cpu").float()
        metadata = torch.load(metadata_path, map_location="cpu")
        self.timestep = metadata["timestep"].float()
        self.target_psnr = metadata["target_psnr"].float()
        self.threshold = metadata["threshold"].float()
        self.sample_ids = metadata["sample_id"]
        self.step_index = metadata["step_index"].long()
        self.source_index = metadata["source_index"].long()
        self.manifest = {}
        if manifest_path.exists():
            with manifest_path.open("r") as handle:
                self.manifest = json.load(handle)

        if max_examples is not None:
            self.features = self.features[:max_examples]
            self.timestep = self.timestep[:max_examples]
            self.target_psnr = self.target_psnr[:max_examples]
            self.threshold = self.threshold[:max_examples]
            self.sample_ids = self.sample_ids[:max_examples]
            self.step_index = self.step_index[:max_examples]
            self.source_index = self.source_index[:max_examples]
        self.examples = [
            type("CachedExample", (), {"sample_id": sample_id})()
            for sample_id in self.sample_ids
        ]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "sample_id": self.sample_ids[index],
            "feature": self.features[index],
            "timestep": self.timestep[index].view(1),
            "target_psnr": self.target_psnr[index].view(1),
            "threshold": self.threshold[index].view(1),
        }


def collate_cached_features(
    batch: list[dict[str, torch.Tensor | str]]
) -> dict[str, object]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "feature": torch.stack([item["feature"] for item in batch]),  # type: ignore[arg-type]
        "timestep": torch.stack([item["timestep"] for item in batch]),  # type: ignore[arg-type]
        "target_psnr": torch.stack([item["target_psnr"] for item in batch]),  # type: ignore[arg-type]
        "threshold": torch.stack([item["threshold"] for item in batch]),  # type: ignore[arg-type]
    }
