#!/usr/bin/env python3
import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path


PERFECT_PSNR_THRESHOLD = 100.0


def parse_psnr_stats(stats_path: Path):
    scores = []
    rows = []
    for line in stats_path.read_text(encoding="utf-8", errors="replace").splitlines():
        values = dict(re.findall(r"([a-zA-Z0-9_]+):([^ ]+)", line))
        if "psnr_avg" not in values:
            continue
        raw = values["psnr_avg"]
        score = float("inf") if raw.lower() == "inf" else float(raw)
        frame = int(values.get("n", len(rows) + 1))
        rows.append({"frame": frame, "psnr_avg": score})
        if math.isfinite(score) and score <= PERFECT_PSNR_THRESHOLD:
            scores.append(score)
    return rows, scores


def resolve_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        "/hy-tmp/env/Wan2.2/bin/ffmpeg",
        "/hy-tmp/miniconda3/envs/Wan2.2/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("ffmpeg not found; install ffmpeg/ffprobe in the Wan2.2 environment")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = Path(args.reference)
    candidate = Path(args.candidate)
    if not reference.exists():
        raise SystemExit(f"Missing reference video: {reference}")
    if not candidate.exists():
        raise SystemExit(f"Missing candidate video: {candidate}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path = output.with_suffix(output.suffix + ".ffmpeg_psnr.log")

    ffmpeg = resolve_ffmpeg()
    cmd = [
        ffmpeg,
        "-v", "error",
        "-i", str(reference),
        "-i", str(candidate),
        "-lavfi", f"psnr=stats_file={stats_path}",
        "-f", "null",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            "FFmpeg PSNR failed with status "
            f"{proc.returncode}: {proc.stderr.strip()}"
        )

    per_frame_rows, scores = parse_psnr_stats(stats_path)
    if not per_frame_rows:
        raise SystemExit(f"No PSNR rows parsed from {stats_path}")
    if not scores:
        mean_psnr = float("inf")
        min_psnr = float("inf")
        max_psnr = float("inf")
    else:
        mean_psnr = sum(scores) / len(scores)
        min_psnr = min(scores)
        max_psnr = max(scores)

    result = {
        "reference": str(reference),
        "candidate": str(candidate),
        "method": "ffmpeg_psnr_filter_psnr_avg_yuv_weighted",
        "perfect_psnr_threshold": PERFECT_PSNR_THRESHOLD,
        "frames": len(scores),
        "decoded_frames_total": len(per_frame_rows),
        "excluded_perfect_frames": len(per_frame_rows) - len(scores),
        "mean_psnr": mean_psnr,
        "min_psnr": min_psnr,
        "max_psnr": max_psnr,
        "per_frame_psnr": [row["psnr_avg"] for row in per_frame_rows],
        "ffmpeg_stats_file": str(stats_path),
    }

    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_frame_psnr"}, indent=2))


if __name__ == "__main__":
    main()
