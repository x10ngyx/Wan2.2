#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


OUT = Path("reports/assets/predictor_speedup_training_loss_curves.svg")

RUNS = [
    {
        "title": "Transformer sample split",
        "speed": Path("/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_speedup_20260706_171523/epoch_metrics.csv"),
    },
    {
        "title": "Transformer row split",
        "speed": Path("/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523/epoch_metrics.csv"),
    },
    {
        "title": "5-feature sample split",
        "speed": Path("/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_speedup_samplesplit_long100_20260706_171523/epoch_metrics.csv"),
    },
    {
        "title": "5-feature row split",
        "speed": Path("/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_speedup_rowsplit_long100_20260706_171523/epoch_metrics.csv"),
    },
]


def read_curves(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train = [(int(row["epoch"]), float(row["train_loss"])) for row in rows]
    val = [(int(row["epoch"]), float(row["val_loss"])) for row in rows]
    return train, val


def points(curve: list[tuple[int, float]], x0: float, y0: float, w: float, h: float, ymin: float, ymax: float) -> str:
    max_epoch = max(epoch for epoch, _ in curve)
    span = max(ymax - ymin, 1e-9)
    coords = []
    for epoch, value in curve:
        x = x0 + (epoch - 1) / max(max_epoch - 1, 1) * w
        y = y0 + (ymax - value) / span * h
        coords.append(f"{x:.1f},{y:.1f}")
    return "M " + " L ".join(coords)


def best_epoch(curve: list[tuple[int, float]]) -> tuple[int, float]:
    return min(curve, key=lambda item: item[1])


def tick_values(ymin: float, ymax: float) -> list[float]:
    return [ymin + (ymax - ymin) * i / 4 for i in range(5)]


def main() -> None:
    curves = []
    for run in RUNS:
        train, val = read_curves(run["speed"])
        values = [v for _, v in train + val]
        ymax = max(values)
        ymin = min(values)
        pad = max((ymax - ymin) * 0.08, 0.001)
        curves.append((run["title"], train, val, max(0.0, ymin - pad), ymax + pad))

    width, height = 1120, 700
    margin_l, margin_t = 58, 58
    plot_w, plot_h = 480, 245
    gap_x, gap_y = 64, 76

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.title{font-size:18px;font-weight:700}.label{font-size:12px}.tick{font-size:10px;fill:#4b5563}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.train{fill:none;stroke:#2563eb;stroke-width:2}.val{fill:none;stroke:#dc2626;stroke-width:2}.best{stroke:#6b7280;stroke-width:1;stroke-dasharray:4 4}</style>",
        f'<text x="{width/2:.1f}" y="26" text-anchor="middle" class="title">Speedup-Conditioned Predictor Training Loss Curves</text>',
    ]

    for idx, (title, train, val, ymin, ymax) in enumerate(curves):
        col = idx % 2
        row = idx // 2
        x0 = margin_l + col * (plot_w + gap_x)
        y0 = margin_t + row * (plot_h + gap_y)

        for tick in tick_values(ymin, ymax):
            y = y0 + (ymax - tick) / (ymax - ymin) * plot_h
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x0 - 8}" y="{y + 3:.1f}" text-anchor="end" class="tick">{tick:.3f}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" class="axis"/>')
        parts.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" class="axis"/>')

        max_epoch = max(epoch for epoch, _ in val)
        parts.append(f'<text x="{x0:.1f}" y="{y0 + plot_h + 18}" text-anchor="middle" class="tick">1</text>')
        parts.append(f'<text x="{x0 + plot_w:.1f}" y="{y0 + plot_h + 18}" text-anchor="middle" class="tick">{max_epoch}</text>')

        parts.append(f'<path d="{points(train, x0, y0, plot_w, plot_h, ymin, ymax)}" class="train"/>')
        parts.append(f'<path d="{points(val, x0, y0, plot_w, plot_h, ymin, ymax)}" class="val"/>')

        val_best_epoch, _ = best_epoch(val)
        best_x = x0 + (val_best_epoch - 1) / max(max_epoch - 1, 1) * plot_w
        parts.append(f'<line x1="{best_x:.1f}" y1="{y0}" x2="{best_x:.1f}" y2="{y0 + plot_h}" class="best"/>')
        parts.append(f'<text x="{x0 + plot_w / 2:.1f}" y="{y0 - 14}" text-anchor="middle" class="label">{title}</text>')
        parts.append(f'<text x="{x0 + plot_w - 86:.1f}" y="{y0 + 16}" class="tick">best val {val_best_epoch}</text>')

    legend_y = height - 42
    parts.append(f'<line x1="{width/2 - 88}" y1="{legend_y}" x2="{width/2 - 62}" y2="{legend_y}" class="train"/><text x="{width/2 - 54}" y="{legend_y + 4}" class="tick">train loss</text>')
    parts.append(f'<line x1="{width/2 + 28}" y1="{legend_y}" x2="{width/2 + 54}" y2="{legend_y}" class="val"/><text x="{width/2 + 62}" y="{legend_y + 4}" class="tick">val/test loss</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height - 14}" text-anchor="middle" class="tick">x-axis: epoch within each speedup-conditioned run; y-axis: Smooth L1 loss. Dashed lines mark best validation-loss epoch.</text>')
    parts.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
