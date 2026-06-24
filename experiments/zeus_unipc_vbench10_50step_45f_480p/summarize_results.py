#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import re
from pathlib import Path


def read_elapsed(path: Path) -> float:
    text = path.read_text(encoding="utf-8").strip()
    match = re.search(r"elapsed_seconds=([0-9.]+)", text)
    if not match:
        raise ValueError(f"Missing elapsed_seconds in {path}")
    return float(match.group(1))


def read_cache_summary(path: Path):
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"ZEUS timestep cache summary: (\{.*\})", text)
    if not match:
        return {}
    return ast.literal_eval(match.group(1))


def sum_cache(summary, field: str) -> int:
    total = 0
    for item in summary.values():
        total += int(item.get(field, 0))
    return total


def count_unique_steps(summary, field: str) -> int:
    steps = set()
    for item in summary.values():
        steps.update(int(step) for step in item.get(field, []))
    return len(steps)


def collect_paths(summary, field: str) -> str:
    parts = []
    for key, item in sorted(summary.items()):
        parts.append(f"{key}:{item.get(field, [])}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.experiment_root)
    rows = []
    for baseline_video in sorted((root / "baseline").glob("*.mp4")):
        sample_id = baseline_video.stem
        zeus_video = root / "zeus" / f"{sample_id}.mp4"
        if not zeus_video.exists():
            continue

        baseline_elapsed = read_elapsed(root / "logs" / f"baseline_{sample_id}.time")
        zeus_elapsed = read_elapsed(root / "logs" / f"zeus_{sample_id}.time")
        psnr = json.loads((root / "psnr" / f"{sample_id}.json").read_text(encoding="utf-8"))
        baseline_ffprobe = json.loads((root / "ffprobe" / f"baseline_{sample_id}.json").read_text(encoding="utf-8"))
        zeus_ffprobe = json.loads((root / "ffprobe" / f"zeus_{sample_id}.json").read_text(encoding="utf-8"))
        cache_summary = read_cache_summary(root / "logs" / f"zeus_{sample_id}.log")

        rows.append({
            "sample_id": sample_id,
            "baseline_elapsed_seconds": baseline_elapsed,
            "zeus_elapsed_seconds": zeus_elapsed,
            "speedup": baseline_elapsed / zeus_elapsed if zeus_elapsed else "",
            "mean_psnr": psnr["mean_psnr"],
            "min_psnr": psnr["min_psnr"],
            "psnr_frames": psnr["frames"],
            "zeus_reuse_count": count_unique_steps(cache_summary, "skipping_path"),
            "zeus_recompute_count": count_unique_steps(cache_summary, "recompute_path"),
            "zeus_reuse_branch_call_count": sum_cache(cache_summary, "reuse"),
            "zeus_recompute_branch_call_count": sum_cache(cache_summary, "recompute"),
            "zeus_skipping_path": collect_paths(cache_summary, "skipping_path"),
            "zeus_recompute_path": collect_paths(cache_summary, "recompute_path"),
            "baseline_video": str(baseline_video),
            "zeus_video": str(zeus_video),
            "baseline_log": str(root / "logs" / f"baseline_{sample_id}.log"),
            "zeus_log": str(root / "logs" / f"zeus_{sample_id}.log"),
            "baseline_ffprobe": json.dumps(baseline_ffprobe, sort_keys=True),
            "zeus_ffprobe": json.dumps(zeus_ffprobe, sort_keys=True),
        })

    if not rows:
        raise SystemExit("No completed prompt pairs found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_baseline = sum(row["baseline_elapsed_seconds"] for row in rows)
    total_zeus = sum(row["zeus_elapsed_seconds"] for row in rows)
    aggregate = {
        "num_pairs": len(rows),
        "total_baseline_elapsed_seconds": total_baseline,
        "total_zeus_elapsed_seconds": total_zeus,
        "overall_speedup": total_baseline / total_zeus if total_zeus else None,
        "mean_psnr": sum(float(row["mean_psnr"]) for row in rows) / len(rows),
        "summary_csv": str(output),
    }
    (output.parent / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
