from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch

from adaptive_threshold_predictor.models import MiniDiTCLSAdaptiveThresholdPredictor


DEFAULT_CHECKPOINT = Path(
    "/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523/best_model_checkpoint.pt"
)
DEFAULT_PACKED_CACHE = Path(
    "/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805"
)
DEFAULT_SPLIT = Path(
    "/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523/split.json"
)


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe threshold predictor sensitivity to latent, timestep, PSNR, and speedup inputs."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--packed_cache", type=Path, default=DEFAULT_PACKED_CACHE)
    parser.add_argument("--split_json", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/hy-tmp/wan22_predictor_condition_sensitivity_probe"),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num_train_latents", type=int, default=3)
    parser.add_argument("--num_val_latents", type=int, default=3)
    parser.add_argument("--num_random_latents", type=int, default=3)
    parser.add_argument(
        "--bind_real_latent_timestep",
        action="store_true",
        help=(
            "Use each real train/val latent only with its source timestep. "
            "Random latents still sweep --timesteps and are reported as OOD."
        ),
    )
    parser.add_argument(
        "--step_indices",
        default="0,12,24,36,49",
        help="Comma-separated source step_index values for bound real-latent sampling.",
    )
    parser.add_argument(
        "--real_latents_per_step",
        type=int,
        default=2,
        help="Number of train and val real latents to sample per requested step_index.",
    )
    parser.add_argument("--condition_batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fixed_speedup", type=float, default=3.5)
    parser.add_argument("--fixed_psnr", type=float, default=45.0)
    parser.add_argument(
        "--timesteps",
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated normalized timestep values.",
    )
    parser.add_argument(
        "--psnrs",
        default="18,22,28,35,45",
        help="Comma-separated PSNR values for the fixed-speedup probe.",
    )
    parser.add_argument(
        "--speedups",
        default="1.1,1.4,1.7,2.0,2.5,2.8,3.2,3.5",
        help="Comma-separated target speedup values for the fixed-PSNR probe.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> MiniDiTCLSAdaptiveThresholdPredictor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    feature_extractor = checkpoint.get("feature_extractor", {})
    input_shape = tuple(int(value) for value in feature_extractor.get("input_shape", (16, 12, 60, 104)))
    patch_size = tuple(int(value) for value in args.get("dit_patch_size", (3, 12, 8)))
    model = MiniDiTCLSAdaptiveThresholdPredictor(
        input_shape=input_shape,
        patch_size=patch_size,
        dim=int(args.get("dit_dim", 96)),
        num_layers=int(args.get("dit_layers", 2)),
        num_heads=int(args.get("dit_heads", 4)),
        mlp_ratio=float(args.get("dit_mlp_ratio", 2.0)),
        psnr_min=float(args.get("psnr_min", 10.0)),
        psnr_max=float(args.get("psnr_max", 50.0)),
        speedup_min=float(args.get("speedup_min", 1.0)),
        speedup_max=float(args.get("speedup_max", 4.0)),
        min_threshold=float(args.get("min_threshold", 0.10)),
        max_threshold=float(args.get("max_threshold", 0.80)),
        dropout=float(args.get("dit_dropout", 0.05)),
        gate_init=float(args.get("dit_gate_init", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def choose_indices(split_path: Path, num_train: int, num_val: int, seed: int) -> tuple[list[int], list[int]]:
    split = json.loads(split_path.read_text())
    rng = random.Random(seed)
    train_indices = list(split["train_indices"])
    val_indices = list(split["val_indices"])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices[:num_train], val_indices[:num_val]


def choose_indices_by_step(
    split_path: Path,
    metadata: dict[str, object],
    step_indices: Iterable[int],
    per_step: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    split = json.loads(split_path.read_text())
    rng = random.Random(seed)
    source_steps = metadata["step_index"]

    def select(indices: list[int], split_name: str) -> list[int]:
        selected: list[int] = []
        for step_index in step_indices:
            candidates = [
                int(index)
                for index in indices
                if int(source_steps[int(index)]) == int(step_index)
            ]
            rng.shuffle(candidates)
            if len(candidates) < per_step:
                raise ValueError(
                    f"Not enough {split_name} rows for step_index={step_index}: "
                    f"need {per_step}, found {len(candidates)}"
                )
            selected.extend(candidates[:per_step])
        return selected

    return (
        select(list(split["train_indices"]), "train"),
        select(list(split["val_indices"]), "val"),
    )


def load_metadata(cache_root: Path) -> dict[str, object]:
    return torch.load(cache_root / "metadata.pt", map_location="cpu", weights_only=True)


def load_latent(cache_root: Path, metadata: dict[str, object], index: int) -> torch.Tensor:
    shard_names = metadata["shard_name"]
    shard_offsets = metadata["shard_offset"]
    shard_name = shard_names[index]
    shard_offset = int(shard_offsets[index])
    shard = torch.load(cache_root / "shards" / shard_name, map_location="cpu", weights_only=True)
    return shard[shard_offset].float()


def make_latent_specs(
    cache_root: Path,
    metadata: dict[str, object],
    train_indices: Iterable[int],
    val_indices: Iterable[int],
    num_random: int,
    seed: int,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    real_latents: list[torch.Tensor] = []
    sample_ids = metadata["sample_id"]
    step_indices = metadata["step_index"]
    thresholds = metadata["threshold"]
    target_psnr = metadata["target_psnr"]
    timestep = metadata["timestep"]

    for split_name, indices in (("train", train_indices), ("val", val_indices)):
        for rank, index in enumerate(indices):
            latent = load_latent(cache_root, metadata, index)
            real_latents.append(latent)
            specs.append(
                {
                    "latent_id": f"{split_name}_{rank}",
                    "latent_source": split_name,
                    "source_index": index,
                    "sample_id": sample_ids[index],
                    "source_step_index": int(step_indices[index]),
                    "source_timestep": float(timestep[index]),
                    "source_psnr": float(target_psnr[index]),
                    "source_threshold": float(thresholds[index]),
                    "timestep_mode": "source_bound",
                    "latent": latent,
                }
            )

    if not real_latents:
        raise ValueError("At least one real latent is required")
    stacked = torch.stack(real_latents)
    mean = float(stacked.mean())
    std = float(stacked.std().clamp_min(1e-6))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = tuple(int(value) for value in real_latents[0].shape)
    for rank in range(num_random):
        specs.append(
            {
                "latent_id": f"random_{rank}",
                "latent_source": "random_normal_matched",
                "source_index": "",
                "sample_id": "random_normal_matched",
                "source_step_index": "",
                "source_timestep": "",
                "source_psnr": "",
                "source_threshold": "",
                "timestep_mode": "random_ood_sweep",
                "latent": torch.randn(shape, generator=generator) * std + mean,
            }
        )
    return specs


def predict_thresholds_for_latent(
    model: MiniDiTCLSAdaptiveThresholdPredictor,
    latent: torch.Tensor,
    conditions: list[tuple[float, float, float]],
    device: torch.device,
    condition_batch_size: int,
) -> list[float]:
    """Predict many conditions for one latent while reusing patch tokens.

    MiniDiT's Conv3d patch embedding is independent of timestep/PSNR/speedup.
    Reusing it matters for CPU-only probes and produces the same values as
    calling model.forward once per condition while dropout is disabled by eval().
    """
    if not conditions:
        return []
    latent_batch = latent.unsqueeze(0).to(device)
    if condition_batch_size < 1:
        raise ValueError("--condition_batch_size must be >= 1")
    with torch.inference_mode():
        tokens = model.patch_embedding(latent_batch.float())
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + model._position_grid().unsqueeze(0).to(
            device=device, dtype=tokens.dtype
        )
        outputs: list[float] = []
        for start in range(0, len(conditions), condition_batch_size):
            chunk = conditions[start : start + condition_batch_size]
            timesteps = torch.tensor(
                [condition[0] for condition in chunk],
                dtype=torch.float32,
                device=device,
            )
            target_psnrs = torch.tensor(
                [condition[1] for condition in chunk],
                dtype=torch.float32,
                device=device,
            )
            target_speedups = torch.tensor(
                [condition[2] for condition in chunk],
                dtype=torch.float32,
                device=device,
            )
            batch = len(chunk)
            cls = model.cls_token.expand(batch, -1, -1) + model.cls_pos
            x = torch.cat([cls, tokens.expand(batch, -1, -1)], dim=1)
            x = model.pos_dropout(x)
            cond = model.cond_embed(
                model._prepare_condition(
                    timesteps, target_psnrs, target_speedups, batch, device
                )
            )
            for block in model.blocks:
                x = block(x, cond)
            raw = model.head(x[:, 0])
            pred = model.min_threshold + torch.sigmoid(raw) * (
                model.max_threshold - model.min_threshold
            )
            outputs.extend(float(value) for value in pred.detach().cpu().reshape(-1))
    return outputs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def summarize_values(rows: list[dict[str, object]], group_key: str | None = None) -> dict[str, object]:
    if group_key is None:
        values = [float(row["threshold_pred"]) for row in rows]
        return {
            "n": len(values),
            "mean": mean(values),
            "std": stdev(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(float(row["threshold_pred"]))
    return {
        key: {
            "n": len(values),
            "mean": mean(values),
            "std": stdev(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }
        for key, values in sorted(grouped.items())
    }


def add_common_fields(
    row: dict[str, object],
    latent_spec: dict[str, object],
    timestep: float,
    target_psnr: float,
    target_speedup: float,
    threshold_pred: float,
) -> dict[str, object]:
    return {
        **row,
        "latent_id": latent_spec["latent_id"],
        "latent_source": latent_spec["latent_source"],
        "source_index": latent_spec["source_index"],
        "sample_id": latent_spec["sample_id"],
        "source_step_index": latent_spec["source_step_index"],
        "source_timestep": latent_spec["source_timestep"],
        "source_psnr": latent_spec["source_psnr"],
        "source_threshold": latent_spec["source_threshold"],
        "timestep_mode": latent_spec["timestep_mode"],
        "timestep": timestep,
        "target_psnr": target_psnr,
        "target_speedup": target_speedup,
        "threshold_pred": threshold_pred,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint_model(args.checkpoint, device)
    metadata = load_metadata(args.packed_cache)
    step_indices = parse_csv_ints(args.step_indices)
    if args.bind_real_latent_timestep:
        train_indices, val_indices = choose_indices_by_step(
            args.split_json,
            metadata,
            step_indices,
            args.real_latents_per_step,
            args.seed,
        )
    else:
        train_indices, val_indices = choose_indices(
            args.split_json, args.num_train_latents, args.num_val_latents, args.seed
        )
    latent_specs = make_latent_specs(
        args.packed_cache,
        metadata,
        train_indices,
        val_indices,
        args.num_random_latents,
        args.seed,
    )
    timesteps = parse_csv_floats(args.timesteps)
    psnrs = parse_csv_floats(args.psnrs)
    speedups = parse_csv_floats(args.speedups)

    fixed_speedup_rows: list[dict[str, object]] = []
    for latent_spec in latent_specs:
        latent = latent_spec["latent"]
        assert isinstance(latent, torch.Tensor)
        latent_timesteps = timesteps
        if args.bind_real_latent_timestep and latent_spec["latent_source"] != "random_normal_matched":
            latent_timesteps = [float(latent_spec["source_timestep"])]
        conditions = [
            (timestep, target_psnr, args.fixed_speedup)
            for timestep in latent_timesteps
            for target_psnr in psnrs
        ]
        preds = predict_thresholds_for_latent(
            model, latent, conditions, device, args.condition_batch_size
        )
        for (timestep, target_psnr, target_speedup), pred in zip(conditions, preds):
            fixed_speedup_rows.append(
                add_common_fields(
                    {"experiment": "fixed_speedup_vary_latent_timestep_psnr"},
                    latent_spec,
                    timestep,
                    target_psnr,
                    target_speedup,
                    pred,
                )
            )

    fixed_psnr_rows: list[dict[str, object]] = []
    for latent_spec in latent_specs:
        latent = latent_spec["latent"]
        assert isinstance(latent, torch.Tensor)
        latent_timesteps = timesteps
        if args.bind_real_latent_timestep and latent_spec["latent_source"] != "random_normal_matched":
            latent_timesteps = [float(latent_spec["source_timestep"])]
        conditions = [
            (timestep, args.fixed_psnr, target_speedup)
            for timestep in latent_timesteps
            for target_speedup in speedups
        ]
        preds = predict_thresholds_for_latent(
            model, latent, conditions, device, args.condition_batch_size
        )
        for (timestep, target_psnr, target_speedup), pred in zip(conditions, preds):
            fixed_psnr_rows.append(
                add_common_fields(
                    {"experiment": "fixed_psnr_vary_latent_timestep_speedup"},
                    latent_spec,
                    timestep,
                    target_psnr,
                    target_speedup,
                    pred,
                )
            )

    fixed_speedup_csv = args.out_dir / "fixed_speedup_vary_latent_timestep_psnr.csv"
    fixed_psnr_csv = args.out_dir / "fixed_psnr_vary_latent_timestep_speedup.csv"
    write_csv(fixed_speedup_csv, fixed_speedup_rows)
    write_csv(fixed_psnr_csv, fixed_psnr_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "packed_cache": str(args.packed_cache),
        "split_json": str(args.split_json),
        "device": str(device),
        "fixed_speedup": args.fixed_speedup,
        "fixed_psnr": args.fixed_psnr,
        "timesteps": timesteps,
        "bind_real_latent_timestep": bool(args.bind_real_latent_timestep),
        "step_indices": step_indices,
        "real_latents_per_step": args.real_latents_per_step,
        "psnrs": psnrs,
        "speedups": speedups,
        "latent_ids": [
            {
                key: value
                for key, value in spec.items()
                if key != "latent"
            }
            for spec in latent_specs
        ],
        "fixed_speedup_overall": summarize_values(fixed_speedup_rows),
        "fixed_speedup_real_overall": summarize_values([
            row
            for row in fixed_speedup_rows
            if row["latent_source"] != "random_normal_matched"
        ]),
        "fixed_speedup_random_ood_overall": summarize_values([
            row
            for row in fixed_speedup_rows
            if row["latent_source"] == "random_normal_matched"
        ]),
        "fixed_speedup_by_psnr": summarize_values(fixed_speedup_rows, "target_psnr"),
        "fixed_speedup_real_by_psnr": summarize_values(
            [
                row
                for row in fixed_speedup_rows
                if row["latent_source"] != "random_normal_matched"
            ],
            "target_psnr",
        ),
        "fixed_speedup_by_timestep": summarize_values(fixed_speedup_rows, "timestep"),
        "fixed_speedup_real_by_source_step": summarize_values(
            [
                row
                for row in fixed_speedup_rows
                if row["latent_source"] != "random_normal_matched"
            ],
            "source_step_index",
        ),
        "fixed_speedup_by_latent_source": summarize_values(
            fixed_speedup_rows, "latent_source"
        ),
        "fixed_psnr_overall": summarize_values(fixed_psnr_rows),
        "fixed_psnr_real_overall": summarize_values([
            row
            for row in fixed_psnr_rows
            if row["latent_source"] != "random_normal_matched"
        ]),
        "fixed_psnr_random_ood_overall": summarize_values([
            row
            for row in fixed_psnr_rows
            if row["latent_source"] == "random_normal_matched"
        ]),
        "fixed_psnr_by_speedup": summarize_values(fixed_psnr_rows, "target_speedup"),
        "fixed_psnr_real_by_speedup": summarize_values(
            [
                row
                for row in fixed_psnr_rows
                if row["latent_source"] != "random_normal_matched"
            ],
            "target_speedup",
        ),
        "fixed_psnr_by_timestep": summarize_values(fixed_psnr_rows, "timestep"),
        "fixed_psnr_real_by_source_step": summarize_values(
            [
                row
                for row in fixed_psnr_rows
                if row["latent_source"] != "random_normal_matched"
            ],
            "source_step_index",
        ),
        "fixed_psnr_by_latent_source": summarize_values(
            fixed_psnr_rows, "latent_source"
        ),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    report_path = args.out_dir / "report.md"
    report_path.write_text(render_report(summary, fixed_speedup_csv, fixed_psnr_csv))
    print(f"Wrote {fixed_speedup_csv}")
    print(f"Wrote {fixed_psnr_csv}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


def format_stats(stats: dict[str, object]) -> str:
    return (
        f"n={stats['n']}, mean={float(stats['mean']):.6f}, "
        f"std={float(stats['std']):.6f}, min={float(stats['min']):.6f}, "
        f"max={float(stats['max']):.6f}, range={float(stats['range']):.6f}"
    )


def render_group_table(grouped: dict[str, object], key_name: str) -> str:
    lines = [
        f"| {key_name} | n | mean threshold | std | min | max | range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, stats_obj in grouped.items():
        stats = stats_obj
        assert isinstance(stats, dict)
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(stats["n"]),
                    f"{float(stats['mean']):.6f}",
                    f"{float(stats['std']):.6f}",
                    f"{float(stats['min']):.6f}",
                    f"{float(stats['max']):.6f}",
                    f"{float(stats['range']):.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_report(
    summary: dict[str, object], fixed_speedup_csv: Path, fixed_psnr_csv: Path
) -> str:
    fixed_speedup_overall = summary["fixed_speedup_overall"]
    fixed_psnr_overall = summary["fixed_psnr_overall"]
    fixed_speedup_real_overall = summary["fixed_speedup_real_overall"]
    fixed_psnr_real_overall = summary["fixed_psnr_real_overall"]
    fixed_speedup_random_ood_overall = summary["fixed_speedup_random_ood_overall"]
    fixed_psnr_random_ood_overall = summary["fixed_psnr_random_ood_overall"]
    assert isinstance(fixed_speedup_overall, dict)
    assert isinstance(fixed_psnr_overall, dict)
    assert isinstance(fixed_speedup_real_overall, dict)
    assert isinstance(fixed_psnr_real_overall, dict)
    assert isinstance(fixed_speedup_random_ood_overall, dict)
    assert isinstance(fixed_psnr_random_ood_overall, dict)
    return f"""# Predictor Condition Sensitivity Probe

Checkpoint: `{summary['checkpoint']}`

Packed latent cache: `{summary['packed_cache']}`

Device: `{summary['device']}`

Bind real latent timestep: `{summary['bind_real_latent_timestep']}`

## Experiment 1: fixed large target speedup

Fixed `target_speedup={summary['fixed_speedup']}`. Real train/val latents use their source timestep when binding is enabled. Random latents sweep normalized timesteps and are treated as OOD.

Real-bound overall: {format_stats(fixed_speedup_real_overall)}

{render_group_table(summary['fixed_speedup_real_by_psnr'], 'target_psnr')}

{render_group_table(summary['fixed_speedup_real_by_source_step'], 'source_step_index')}

{render_group_table(summary['fixed_speedup_by_latent_source'], 'latent_source')}

Random/OOD overall: {format_stats(fixed_speedup_random_ood_overall)}

CSV: `{fixed_speedup_csv}`

## Experiment 2: fixed large target PSNR

Fixed `target_psnr={summary['fixed_psnr']}`. Real train/val latents use their source timestep when binding is enabled. Random latents sweep normalized timesteps and are treated as OOD.

Real-bound overall: {format_stats(fixed_psnr_real_overall)}

{render_group_table(summary['fixed_psnr_real_by_speedup'], 'target_speedup')}

{render_group_table(summary['fixed_psnr_real_by_source_step'], 'source_step_index')}

{render_group_table(summary['fixed_psnr_by_latent_source'], 'latent_source')}

Random/OOD overall: {format_stats(fixed_psnr_random_ood_overall)}

CSV: `{fixed_psnr_csv}`
"""


if __name__ == "__main__":
    main()
