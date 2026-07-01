from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent / "assets"


STYLE = """
text { font-family: Arial, Helvetica, sans-serif; fill: #172033; }
.title { font-size: 30px; font-weight: 700; }
.subtitle { font-size: 15px; fill: #526078; }
.box-title { font-size: 17px; font-weight: 700; }
.shape { font-size: 14px; font-weight: 700; fill: #26364f; }
.detail { font-size: 12px; fill: #526078; }
.tiny { font-size: 11px; fill: #526078; }
.input { fill: #eef6ff; stroke: #2f73b8; }
.feature { fill: #edf9f1; stroke: #2c8b57; }
.cond { fill: #fff7e8; stroke: #b97813; }
.fusion { fill: #f2efff; stroke: #6a55b8; }
.output { fill: #fff0f0; stroke: #bd3b3b; }
.block { fill: #f8fbff; stroke: #65758d; }
.arrow { stroke: #34445f; stroke-width: 2.0; fill: none; marker-end: url(#arrowhead); }
.thin { stroke-width: 1.4; }
.dash { stroke-dasharray: 7 5; }
.control { stroke: #b97813; stroke-width: 1.8; stroke-dasharray: 7 5; fill: none; marker-end: url(#gatehead); }
"""


def text(x: int, y: int, value: str, cls: str = "") -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return f'<text x="{x}" y="{y}"{class_attr}>{escape(value)}</text>'


def rect(
    x: int,
    y: int,
    w: int,
    h: int,
    cls: str,
    title: str,
    lines: list[str],
    radius: int = 8,
) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" rx="{radius}" ry="{radius}" width="{w}" height="{h}" '
        f'class="{cls}" stroke-width="2"/>',
        text(x + 16, y + 28, title, "box-title"),
    ]
    yy = y + 54
    for index, line in enumerate(lines):
        parts.append(text(x + 16, yy, line, "shape" if index == 0 else "detail"))
        yy += 22
    return "\n".join(parts)


def arrow(x1: int, y1: int, x2: int, y2: int, extra: str = "") -> str:
    cls = "arrow" + (f" {extra}" if extra else "")
    if y1 == y2:
        d = f"M{x1} {y1} H{x2}"
    elif x1 == x2:
        d = f"M{x1} {y1} V{y2}"
    else:
        mid = (x1 + x2) // 2
        d = f"M{x1} {y1} C{mid} {y1} {mid} {y2} {x2} {y2}"
    return f'<path d="{d}" class="{cls}"/>'


def svg(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <style>{STYLE}</style>
    <marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#34445f"/>
    </marker>
    <marker id="gatehead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 Z" fill="#b97813"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
{body}
</svg>
"""


def make_gated_mlp() -> str:
    feature_names = [
        "latent_pool",
        "temporal_mean",
        "temporal_var",
        "frame_diff_mean",
        "frame_diff_var",
    ]
    parts = [
        text(48, 52, "5-Feature Gated MLP Threshold Predictor", "title"),
        text(
            48,
            80,
            "Separate feature encoders, condition-dependent softmax gate, fused feature readout.",
            "subtitle",
        ),
    ]

    y0 = 118
    for i, name in enumerate(feature_names):
        y = y0 + i * 76
        parts.append(rect(52, y, 196, 56, "input", name, ["[B,128]"]))
        parts.append(
            rect(
                292,
                y,
                246,
                56,
                "feature",
                f"feature encoder {i + 1}",
                ["128 -> 64", "Linear, SiLU, Dropout"],
            )
        )
        parts.append(arrow(248, y + 28, 292, y + 28, "thin"))
        parts.append(arrow(538, y + 28, 612, y + 28, "thin"))

    parts.append(
        rect(
            612,
            118,
            260,
            360,
            "feature",
            "encoded features",
            ["z1..z5, each [B,64]", "stack -> [B,5,64]"],
        )
    )
    parts.append(
        rect(
            52,
            560,
            248,
            86,
            "input",
            "conditioning inputs",
            ["step_fraction, target_psnr", "[B,2]"],
        )
    )
    parts.append(
        rect(
            364,
            560,
            240,
            86,
            "cond",
            "condition MLP",
            ["[B,2] -> [B,64]", "condition embedding c"],
        )
    )
    parts.append(arrow(300, 603, 364, 603))

    parts.append(
        rect(
            668,
            560,
            232,
            86,
            "cond",
            "gate head + softmax",
            ["c -> [B,5]", "g1..g5, sum=1"],
        )
    )
    parts.append(arrow(604, 603, 668, 603))

    parts.append(
        rect(
            950,
            238,
            278,
            116,
            "fusion",
            "gated feature fusion",
            ["sum_i g_i * z_i", "[B,64] fused feature", "gate controls feature weights"],
        )
    )
    parts.append(arrow(872, 296, 950, 296))
    parts.append('<path d="M784 560 C830 492 980 428 1089 354" class="control"/>')

    parts.append(
        rect(
            1276,
            238,
            300,
            116,
            "fusion",
            "concat with condition",
            ["fused feature [B,64]", "+ c [B,64] -> [B,128]"],
        )
    )
    parts.append(arrow(1228, 296, 1276, 296))
    parts.append(arrow(604, 603, 1276, 336, "thin dash"))

    parts.append(
        rect(
            1276,
            460,
            300,
            102,
            "fusion",
            "prediction head",
            ["128 -> 64 -> 64 -> 1", "LN, SiLU, Dropout, scaled Sigmoid"],
        )
    )
    parts.append(arrow(1426, 354, 1426, 460))
    parts.append(rect(1276, 632, 300, 78, "output", "predicted threshold", ["[B,1]", "range: [0.10,0.80]"]))
    parts.append(arrow(1426, 562, 1426, 632))

    return svg(1628, 760, "\n".join(parts))


def make_mini_dit() -> str:
    parts = [
        text(48, 52, "MiniDiT-CLS Adaptive Threshold Predictor", "title"),
        text(
            48,
            80,
            "Raw latent Conv3d patch tokens, learned factorized 3D position, AdaLN-conditioned Transformer blocks.",
            "subtitle",
        ),
        rect(54, 148, 250, 96, "input", "latent input", ["[B,16,12,60,104]", "Wan2.2 T2V latent trace"]),
        rect(374, 148, 284, 96, "feature", "Conv3d patch embedding", ["kernel=stride=(3,12,8)", "16 -> 96, grid [4,5,13]"]),
        rect(728, 148, 250, 96, "feature", "latent tokens", ["[B,260,96]", "flatten order: T, H, W"]),
        rect(1048, 148, 270, 96, "feature", "prepend CLS + add pos", ["[B,261,96]", "cls_pos + pos_t + pos_h + pos_w"]),
        arrow(304, 196, 374, 196),
        arrow(658, 196, 728, 196),
        arrow(978, 196, 1048, 196),
        rect(54, 384, 250, 92, "cond", "condition inputs", ["step_fraction", "target_psnr_norm"]),
        rect(374, 384, 284, 92, "cond", "condition MLP", ["[B,2] -> [B,96]", "SiLU MLP"]),
        arrow(304, 430, 374, 430),
        rect(728, 354, 590, 176, "block", "2 x conditioned Transformer blocks", ["AdaLN modulation: Linear(96 -> 6*96)", "shift/scale/gate for attention and MLP"]),
        text(762, 452, "x = x + gate_attn * SelfAttention(LN(x) * (1 + scale_attn) + shift_attn)", "tiny"),
        text(762, 478, "x = x + gate_mlp  * MLP(          LN(x) * (1 + scale_mlp)  + shift_mlp)", "tiny"),
        arrow(658, 430, 728, 430, "thin"),
        arrow(1183, 244, 1183, 354),
        rect(1048, 612, 270, 96, "fusion", "CLS readout head", ["LayerNorm, Linear, SiLU", "Dropout, Linear"]),
        rect(1048, 780, 270, 86, "output", "predicted threshold", ["0.10 + sigmoid(raw) * 0.70", "range: [0.10,0.80]"]),
        arrow(1183, 530, 1183, 612),
        arrow(1183, 708, 1183, 780),
    ]
    return svg(1380, 920, "\n".join(parts))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "gated_multifeature_mlp_architecture.svg": make_gated_mlp(),
        "mini_dit_cls_predictor_architecture.svg": make_mini_dit(),
    }
    for name, content in outputs.items():
        path = OUT_DIR / name
        path.write_text(content)
        print(path)


if __name__ == "__main__":
    main()
