from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "gold": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "magenta": "#EA84DD",
    "blue": "#0F4D92",
    "orange": "#E9A6A1",
    "green": "#8BCF8B",
    "purple": "#9A4D8E",
    "gray": "#767676",
    "black": "#272727",
    "sky": "#3775BA",
}

CLASS_PALETTE = {
    "healthy": PALETTE["blue_main"],
    "lung_adenocarcinoma": PALETTE["red_strong"],
}


def apply_publication_style(font_size: float = 7, axes_linewidth: float = 0.8) -> None:
    """Apply Nature-style matplotlib settings with editable SVG/PDF text."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": max(font_size - 0.5, 5),
            "ytick.labelsize": max(font_size - 0.5, 5),
            "legend.fontsize": max(font_size - 0.5, 5),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": axes_linewidth,
            "axes.grid": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.5,
            "xtick.major.width": axes_linewidth,
            "ytick.major.width": axes_linewidth,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def style_axes(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=2.5, width=0.8, pad=2)
    if grid:
        ax.grid(True, color=PALETTE["neutral_light"], linewidth=0.35, alpha=0.65)


def style_colorbar(colorbar) -> None:
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(direction="out", length=2.0, width=0.6, pad=2)


def finalize_figure(
    fig: plt.Figure,
    out_path: Path,
    *,
    formats: tuple[str, ...] = ("svg", "pdf", "tiff", "png"),
    dpi: int = 600,
    close: bool = True,
) -> list[Path]:
    fig.tight_layout(pad=1)
    base = out_path.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        target = base.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        saved.append(target)
    if close:
        plt.close(fig)
    return saved
