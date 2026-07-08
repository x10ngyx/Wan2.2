from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from adaptive_threshold_predictor.models import normalize_feature_sets


DEFAULT_DATA_ROOT = Path("/hy-tmp/openvid_100_seacache_trace_data")
DEFAULT_STEP_INDICES = tuple(range(50))


@dataclass(frozen=True)
class GateExample:
    sample_id: str
    step_path: Path
    step_index: int
    num_steps: int
    timestep: float
    target_psnr: float
    target_speedup: float
    threshold: float
    source_threshold: float
    mean_psnr: float
    speedup: float


def load_summary_rows(data_root: Path = DEFAULT_DATA_ROOT) -> list[dict[str, str]]:
    summary_csv = data_root / "data" / "tables" / "summary.csv"
    with summary_csv.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def make_candidate_inverse_examples(
    data_root: Path = DEFAULT_DATA_ROOT,
    step_indices: Iterable[int] = DEFAULT_STEP_INDICES,
    max_examples: int | None = None,
) -> list[GateExample]:
    """Build candidate-wise inverse labels.

    Each measured threshold candidate contributes one example per selected
    denoising step:

        input: candidate run latent at step, step fraction, achieved PSNR,
               achieved speedup
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
            meta = torch.load(meta_path, map_location="cpu", weights_only=True)
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
                    target_speedup=speedup,
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
        step_indices: Iterable[int] = DEFAULT_STEP_INDICES,
        max_examples: int | None = None,
    ) -> None:
        self.examples = make_candidate_inverse_examples(
            data_root=data_root,
            step_indices=step_indices,
            max_examples=max_examples,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        payload = torch.load(example.step_path, map_location="cpu", weights_only=True)
        latent = payload["latent"].float()
        step_denominator = max(example.num_steps - 1, 1)
        step_fraction = example.step_index / step_denominator
        return {
            "sample_id": example.sample_id,
            "latent": latent,
            "timestep": torch.tensor([step_fraction], dtype=torch.float32),
            "target_psnr": torch.tensor([example.target_psnr], dtype=torch.float32),
            "target_speedup": torch.tensor([example.target_speedup], dtype=torch.float32),
            "threshold": torch.tensor([example.threshold], dtype=torch.float32),
        }


def collate_trace_steps(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, object]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "latent": torch.stack([item["latent"] for item in batch]),  # type: ignore[arg-type]
        "timestep": torch.stack([item["timestep"] for item in batch]),  # type: ignore[arg-type]
        "target_psnr": torch.stack([item["target_psnr"] for item in batch]),  # type: ignore[arg-type]
        "target_speedup": torch.stack([item["target_speedup"] for item in batch]),  # type: ignore[arg-type]
        "threshold": torch.stack([item["threshold"] for item in batch]),  # type: ignore[arg-type]
    }


def _load_cache_json(cache_dir: Path, name: str) -> dict[str, object]:
    path = cache_dir / name
    if not path.exists():
        return {}
    with path.open("r") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object in {path}, got {type(loaded).__name__}")
    return loaded


def _target_speedup_from_metadata(
    metadata: dict[str, object],
    cache_dir: Path,
    expected_count: int,
) -> torch.Tensor:
    for key in ("target_speedup", "speedup"):
        if key in metadata:
            return metadata[key].float()  # type: ignore[union-attr]

    cache_config = _load_cache_json(cache_dir, "cache_config.json")
    manifest = _load_cache_json(cache_dir, "manifest.json")
    data_root = Path(str(
        cache_config.get("data_root")
        or manifest.get("data_root")
        or DEFAULT_DATA_ROOT
    ))
    source_index = metadata["source_index"].long()  # type: ignore[union-attr]
    examples = make_candidate_inverse_examples(data_root=data_root)
    values = torch.tensor(
        [examples[int(index)].target_speedup for index in source_index],
        dtype=torch.float32,
    )
    if int(values.numel()) != expected_count:
        raise ValueError(
            f"Recovered target_speedup count mismatch: {values.numel()} != {expected_count}"
        )
    return values


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


def split_indices_by_row(
    dataset: Dataset,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    row_count = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(row_count, generator=generator).tolist()
    train_count = max(1, int(row_count * train_fraction))
    if train_count >= row_count and row_count > 1:
        train_count = row_count - 1
    return permutation[:train_count], permutation[train_count:]


class PackedRawLatentThresholdDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        max_examples: int | None = None,
        preload: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.shards_dir = cache_dir / "shards"
        metadata_path = cache_dir / "metadata.pt"
        manifest_path = cache_dir / "manifest.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing packed metadata file: {metadata_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing packed manifest file: {manifest_path}")

        with manifest_path.open("r") as handle:
            self.manifest = json.load(handle)
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        self.sample_ids = metadata["sample_id"]
        self.timestep = metadata["timestep"].float()
        self.target_psnr = metadata["target_psnr"].float()
        self.target_speedup = _target_speedup_from_metadata(
            metadata,
            cache_dir,
            expected_count=len(self.sample_ids),
        )
        self.threshold = metadata["threshold"].float()
        self.step_index = metadata["step_index"].long()
        self.source_index = metadata["source_index"].long()
        self.shard_names = metadata["shard_name"]
        self.shard_offsets = metadata["shard_offset"].long()

        if max_examples is not None:
            self.sample_ids = self.sample_ids[:max_examples]
            self.timestep = self.timestep[:max_examples]
            self.target_psnr = self.target_psnr[:max_examples]
            self.target_speedup = self.target_speedup[:max_examples]
            self.threshold = self.threshold[:max_examples]
            self.step_index = self.step_index[:max_examples]
            self.source_index = self.source_index[:max_examples]
            self.shard_names = self.shard_names[:max_examples]
            self.shard_offsets = self.shard_offsets[:max_examples]

        self.examples = [
            SimpleNamespace(sample_id=sample_id)
            for sample_id in self.sample_ids
        ]
        self._shard_cache: dict[str, torch.Tensor] = {}
        if preload:
            unique_shards = sorted(set(self.shard_names))
            for shard_name in unique_shards:
                self._shard_cache[shard_name] = torch.load(
                    self.shards_dir / shard_name,
                    map_location="cpu",
                    weights_only=True,
                )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_shard(self, shard_name: str) -> torch.Tensor:
        shard = self._shard_cache.get(shard_name)
        if shard is None:
            shard = torch.load(
                self.shards_dir / shard_name,
                map_location="cpu",
                weights_only=True,
            )
            self._shard_cache = {shard_name: shard}
        return shard

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        shard_name = self.shard_names[index]
        shard_offset = int(self.shard_offsets[index])
        latent = self._load_shard(shard_name)[shard_offset].float()
        return {
            "sample_id": self.sample_ids[index],
            "latent": latent,
            "timestep": self.timestep[index].view(1),
            "target_psnr": self.target_psnr[index].view(1),
            "target_speedup": self.target_speedup[index].view(1),
            "threshold": self.threshold[index].view(1),
            "step_index": self.step_index[index].view(1),
        }


class CachedFeatureThresholdDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        feature_set: str | Sequence[str],
        max_examples: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.feature_sets = normalize_feature_sets(feature_set)
        self.feature_set = self.feature_sets[0]
        metadata_path = cache_dir / "metadata.pt"
        manifest_path = cache_dir / "manifest.json"
        feature_paths = [
            cache_dir / f"features_{feature_set}.pt"
            for feature_set in self.feature_sets
        ]
        missing_feature_paths = [path for path in feature_paths if not path.exists()]
        if missing_feature_paths:
            raise FileNotFoundError(
                "Missing cached feature file(s): "
                + ", ".join(str(path) for path in missing_feature_paths)
            )
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing cached metadata file: {metadata_path}")

        feature_tensors = [
            torch.load(path, map_location="cpu", weights_only=True).float()
            for path in feature_paths
        ]
        row_counts = {int(feature.shape[0]) for feature in feature_tensors}
        if len(row_counts) != 1:
            shapes = [tuple(feature.shape) for feature in feature_tensors]
            raise ValueError(f"Cached feature row-count mismatch: {shapes}")
        self.features = feature_tensors[0] if len(feature_tensors) == 1 else None
        self.features_by_name = {
            feature_set: feature
            for feature_set, feature in zip(self.feature_sets, feature_tensors)
        }
        self.feature_dims = {
            feature_set: int(feature.shape[1:].numel())
            for feature_set, feature in self.features_by_name.items()
        }
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        self.timestep = metadata["timestep"].float()
        self.target_psnr = metadata["target_psnr"].float()
        self.target_speedup = _target_speedup_from_metadata(
            metadata,
            cache_dir,
            expected_count=int(next(iter(self.features_by_name.values())).shape[0]),
        )
        self.threshold = metadata["threshold"].float()
        self.sample_ids = metadata["sample_id"]
        self.step_index = metadata["step_index"].long()
        self.source_index = metadata["source_index"].long()
        self.manifest = {}
        if manifest_path.exists():
            with manifest_path.open("r") as handle:
                self.manifest = json.load(handle)

        if max_examples is not None:
            if self.features is not None:
                self.features = self.features[:max_examples]
            self.features_by_name = {
                feature_set: feature[:max_examples]
                for feature_set, feature in self.features_by_name.items()
            }
            self.timestep = self.timestep[:max_examples]
            self.target_psnr = self.target_psnr[:max_examples]
            self.target_speedup = self.target_speedup[:max_examples]
            self.threshold = self.threshold[:max_examples]
            self.sample_ids = self.sample_ids[:max_examples]
            self.step_index = self.step_index[:max_examples]
            self.source_index = self.source_index[:max_examples]
        self.examples = [
            type("CachedExample", (), {"sample_id": sample_id})()
            for sample_id in self.sample_ids
        ]

    def __len__(self) -> int:
        return int(next(iter(self.features_by_name.values())).shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item: dict[str, torch.Tensor | str | dict[str, torch.Tensor]] = {
            "sample_id": self.sample_ids[index],
            "features": {
                feature_set: feature[index]
                for feature_set, feature in self.features_by_name.items()
            },
            "timestep": self.timestep[index].view(1),
            "target_psnr": self.target_psnr[index].view(1),
            "target_speedup": self.target_speedup[index].view(1),
            "threshold": self.threshold[index].view(1),
        }
        if self.features is not None:
            item["feature"] = self.features[index]
        return item


def collate_cached_features(
    batch: list[dict[str, torch.Tensor | str]]
) -> dict[str, object]:
    output: dict[str, object] = {
        "sample_id": [item["sample_id"] for item in batch],
        "features": {
            feature_set: torch.stack([
                item["features"][feature_set]  # type: ignore[index]
                for item in batch
            ])
            for feature_set in batch[0]["features"]  # type: ignore[index]
        },
        "timestep": torch.stack([item["timestep"] for item in batch]),  # type: ignore[arg-type]
        "target_psnr": torch.stack([item["target_psnr"] for item in batch]),  # type: ignore[arg-type]
        "target_speedup": torch.stack([item["target_speedup"] for item in batch]),  # type: ignore[arg-type]
        "threshold": torch.stack([item["threshold"] for item in batch]),  # type: ignore[arg-type]
    }
    if "feature" in batch[0]:
        output["feature"] = torch.stack([item["feature"] for item in batch])  # type: ignore[arg-type]
    return output


class GridFeatureThresholdDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        max_examples: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        feature_path = cache_dir / "grid_features.pt"
        metadata_path = cache_dir / "metadata.pt"
        manifest_path = cache_dir / "manifest.json"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing grid feature file: {feature_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing cached metadata file: {metadata_path}")

        self.grid_features = torch.load(
            feature_path, map_location="cpu", weights_only=True
        )
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        self.timestep = metadata["timestep"].float()
        self.target_psnr = metadata["target_psnr"].float()
        self.target_speedup = _target_speedup_from_metadata(
            metadata,
            cache_dir,
            expected_count=int(self.grid_features.shape[0]),
        )
        self.threshold = metadata["threshold"].float()
        self.sample_ids = metadata["sample_id"]
        self.step_index = metadata["step_index"].long()
        self.source_index = metadata["source_index"].long()
        self.manifest = {}
        if manifest_path.exists():
            with manifest_path.open("r") as handle:
                self.manifest = json.load(handle)

        if max_examples is not None:
            self.grid_features = self.grid_features[:max_examples]
            self.timestep = self.timestep[:max_examples]
            self.target_psnr = self.target_psnr[:max_examples]
            self.target_speedup = self.target_speedup[:max_examples]
            self.threshold = self.threshold[:max_examples]
            self.sample_ids = self.sample_ids[:max_examples]
            self.step_index = self.step_index[:max_examples]
            self.source_index = self.source_index[:max_examples]
        self.examples = [
            type("CachedExample", (), {"sample_id": sample_id})()
            for sample_id in self.sample_ids
        ]

    def __len__(self) -> int:
        return int(self.grid_features.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.grid_features.shape[1:])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "sample_id": self.sample_ids[index],
            "grid_feature": self.grid_features[index].float(),
            "timestep": self.timestep[index].view(1),
            "target_psnr": self.target_psnr[index].view(1),
            "target_speedup": self.target_speedup[index].view(1),
            "threshold": self.threshold[index].view(1),
            "step_index": self.step_index[index].view(1),
        }


def collate_grid_features(
    batch: list[dict[str, torch.Tensor | str]]
) -> dict[str, object]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "grid_feature": torch.stack([item["grid_feature"] for item in batch]),  # type: ignore[arg-type]
        "timestep": torch.stack([item["timestep"] for item in batch]),  # type: ignore[arg-type]
        "target_psnr": torch.stack([item["target_psnr"] for item in batch]),  # type: ignore[arg-type]
        "target_speedup": torch.stack([item["target_speedup"] for item in batch]),  # type: ignore[arg-type]
        "threshold": torch.stack([item["threshold"] for item in batch]),  # type: ignore[arg-type]
        "step_index": torch.stack([item["step_index"] for item in batch]),  # type: ignore[arg-type]
    }
