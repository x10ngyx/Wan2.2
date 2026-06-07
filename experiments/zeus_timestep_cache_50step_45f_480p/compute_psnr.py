#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def frame_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ref = cv2.VideoCapture(args.reference)
    cand = cv2.VideoCapture(args.candidate)
    if not ref.isOpened():
        raise SystemExit(f"Failed to open reference video: {args.reference}")
    if not cand.isOpened():
        raise SystemExit(f"Failed to open candidate video: {args.candidate}")

    scores = []
    frame_index = 0
    while True:
        ok_ref, frame_ref = ref.read()
        ok_cand, frame_cand = cand.read()
        if not ok_ref and not ok_cand:
            break
        if ok_ref != ok_cand:
            raise SystemExit("Video frame counts differ before EOF")
        if frame_ref.shape != frame_cand.shape:
            raise SystemExit(f"Frame {frame_index} shape mismatch: {frame_ref.shape} vs {frame_cand.shape}")
        scores.append(frame_psnr(frame_ref, frame_cand))
        frame_index += 1

    if not scores:
        raise SystemExit("No frames decoded")

    finite_scores = [s for s in scores if math.isfinite(s)]
    mean_psnr = float("inf") if len(finite_scores) != len(scores) else float(np.mean(finite_scores))
    result = {
        "reference": args.reference,
        "candidate": args.candidate,
        "frames": len(scores),
        "mean_psnr": mean_psnr,
        "min_psnr": min(scores),
        "max_psnr": max(scores),
        "per_frame_psnr": scores,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_frame_psnr"}, indent=2))


if __name__ == "__main__":
    main()
