from __future__ import annotations

import re
import textwrap
from pathlib import Path

from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "report_adaptive_predictor_training_curves.md"
OUT = ROOT / "reports" / "report_adaptive_predictor_training_curves.pdf"
ASSET_DIR = ROOT / "reports" / "assets" / "adaptive_training_curves"


FONT_CANDIDATES = (
    "/hy-tmp/fonts/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/hy-tmp/miniconda3/envs/Wan2.2/fonts/Ubuntu-R.ttf",
    "/hy-tmp/miniconda3/envs/Wan2.2/fonts/DejaVuSans.ttf",
)


FIGURES = {
    "feature_curves_2x2x2.svg": ASSET_DIR / "feature_curves_2x2x2.png",
    "grid_val_curves_by_feature.svg": ASSET_DIR / "grid_val_curves_by_feature.png",
    "best_val_loss_heatmap.svg": ASSET_DIR / "best_val_loss_heatmap.png",
    "long_run_curves.svg": ASSET_DIR / "long_run_curves.png",
}


def choose_font() -> FontProperties:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return FontProperties(fname=str(path))
    return FontProperties()


def clean_text(line: str) -> str:
    line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
    line = line.replace("`", "")
    return line.rstrip()


def add_text_page(
    pdf: PdfPages,
    title: str,
    lines: list[str],
    font: FontProperties,
    page_no: int,
) -> int:
    fig = plt.figure(figsize=(8.27, 11.69))
    axis = fig.add_axes([0, 0, 1, 1])
    axis.axis("off")
    y = 0.955
    if title:
        axis.text(
            0.06,
            y,
            title,
            fontproperties=font,
            fontsize=15,
            weight="bold",
            va="top",
        )
        y -= 0.045
    for raw in lines:
        if not raw.strip():
            y -= 0.012
            continue
        size = 9.1
        indent = 0.06
        text = raw
        if raw.startswith("# "):
            size = 17
            text = raw[2:]
            y -= 0.006
        elif raw.startswith("## "):
            size = 13.5
            text = raw[3:]
            y -= 0.006
        elif raw.startswith("- "):
            indent = 0.08
            text = "• " + raw[2:]
        elif re.match(r"\\d+\\. ", raw):
            indent = 0.08
        elif raw.startswith("|"):
            size = 7.0
            text = raw
        elif raw.startswith("```"):
            continue
        width = 94 if size >= 9 else 130
        wrapped = textwrap.wrap(text, width=width, replace_whitespace=False) or [""]
        for part in wrapped:
            axis.text(
                indent,
                y,
                part,
                fontproperties=font,
                fontsize=size,
                va="top",
            )
            y -= 0.018 if size <= 7.5 else 0.024
            if y < 0.055:
                axis.text(
                    0.5,
                    0.025,
                    f"{page_no}",
                    fontproperties=font,
                    fontsize=8,
                    ha="center",
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                page_no += 1
                fig = plt.figure(figsize=(8.27, 11.69))
                axis = fig.add_axes([0, 0, 1, 1])
                axis.axis("off")
                y = 0.955
    axis.text(
        0.5,
        0.025,
        f"{page_no}",
        fontproperties=font,
        fontsize=8,
        ha="center",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page_no + 1


def add_image_page(
    pdf: PdfPages,
    image_path: Path,
    title: str,
    font: FontProperties,
    page_no: int,
) -> int:
    image = Image.open(image_path)
    fig = plt.figure(figsize=(11.69, 8.27))
    axis = fig.add_axes([0.04, 0.08, 0.92, 0.84])
    axis.imshow(image)
    axis.axis("off")
    fig.text(
        0.5,
        0.955,
        title,
        fontproperties=font,
        fontsize=14,
        weight="bold",
        ha="center",
        va="top",
    )
    fig.text(0.5, 0.025, f"{page_no}", fontproperties=font, fontsize=8, ha="center")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page_no + 1


def main() -> None:
    font = choose_font()
    text = REPORT.read_text()
    raw_lines = text.splitlines()
    page_no = 1
    with PdfPages(OUT) as pdf:
        buffer: list[str] = []
        for line in raw_lines:
            figure_match = re.search(r"assets/adaptive_training_curves/([^)]\\.svg)", line)
            if figure_match:
                if buffer:
                    page_no = add_text_page(pdf, "", buffer, font, page_no)
                    buffer = []
                figure_name = figure_match.group(1)
                image_path = FIGURES[figure_name]
                page_no = add_image_page(
                    pdf,
                    image_path,
                    figure_name.replace("_", " ").replace(".svg", ""),
                    font,
                    page_no,
                )
                continue
            cleaned = clean_text(line)
            if cleaned.strip():
                buffer.append(cleaned)
            else:
                buffer.append("")
        if buffer:
            add_text_page(pdf, "", buffer, font, page_no)
    print(OUT)


if __name__ == "__main__":
    main()
