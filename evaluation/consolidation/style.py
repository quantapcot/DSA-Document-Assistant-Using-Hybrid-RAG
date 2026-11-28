"""Publication-quality Vietnamese styling and Okabe-Ito visual grammar for consolidated figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from evaluation.consolidation import consolidated_png_dir, consolidated_svg_dir
from evaluation.path_resolver import sha256_file

# Okabe-Ito colorblind-safe palette for the 4 retrieval methods
METHOD_COLORS: dict[str, str] = {
    "bm25": "#0072B2",          # Blue
    "dense": "#E69F00",         # Orange
    "hybrid_rrf": "#009E73",    # Bluish Green
    "hybrid_rerank": "#D55E00", # Vermillion
}

COMBO_STYLE: dict[str, dict[str, Any]] = {
    "A1": {"color": "#0072B2", "ls": "-",  "marker": "o", "mfc": "#0072B2", "hatch": "",   "label": "A1 (Rec + BM25)"},
    "A2": {"color": "#E69F00", "ls": "-",  "marker": "s", "mfc": "#E69F00", "hatch": "",   "label": "A2 (Rec + Dense)"},
    "A3": {"color": "#009E73", "ls": "-",  "marker": "^", "mfc": "#009E73", "hatch": "",   "label": "A3 (Rec + RRF)"},
    "A4": {"color": "#D55E00", "ls": "-",  "marker": "D", "mfc": "#D55E00", "hatch": "",   "label": "A4 (Rec + Rerank)"},
    "B1": {"color": "#0072B2", "ls": "--", "marker": "o", "mfc": "white",   "hatch": "//", "label": "B1 (Adap + BM25)"},
    "B2": {"color": "#E69F00", "ls": "--", "marker": "s", "mfc": "white",   "hatch": "//", "label": "B2 (Adap + Dense)"},
    "B3": {"color": "#009E73", "ls": "--", "marker": "^", "mfc": "white",   "hatch": "//", "label": "B3 (Adap + RRF)"},
    "B4": {"color": "#D55E00", "ls": "--", "marker": "D", "mfc": "white",   "hatch": "//", "label": "B4 (Adap + Rerank)"},
    "NO_RAG": {"color": "#595959", "ls": ":", "marker": "X", "mfc": "#595959", "hatch": "..", "label": "NO_RAG (Không RAG)"},
}

COMBO_ORDER = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"]
PROFILE_COLORS = {"recursive_512_100": "#0072B2", "structure_adaptive": "#D55E00"}
K_VALUES = [1, 3, 5, 10, 15, 20, 30, 45, 60]
K_POS = list(range(len(K_VALUES)))


def apply_consolidated_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FAFAFA",
        "savefig.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#E2E6EA",
        "grid.linestyle": "-",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.85,
        "axes.axisbelow": True,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 13,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#D0D7DE",
        "svg.fonttype": "none",
    })


def shared_combo_handles(include_no_rag: bool = False) -> list[Line2D]:
    keys = COMBO_ORDER + (["NO_RAG"] if include_no_rag else [])
    handles: list[Line2D] = []
    for k in keys:
        st = COMBO_STYLE[k]
        handles.append(
            Line2D(
                [0], [0],
                color=st["color"],
                linestyle=st["ls"],
                linewidth=2.3,
                marker=st["marker"],
                markersize=7.5,
                markerfacecolor=st["mfc"],
                markeredgecolor=st["color"],
                markeredgewidth=1.6,
                label=st["label"],
            )
        )
    return handles


def style_axis_clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_consolidated_figure(fig: plt.Figure, root: Path, figure_id: str) -> dict[str, Any]:
    png_dir = consolidated_png_dir(root)
    svg_dir = consolidated_svg_dir(root)
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / f"{figure_id}.png"
    svg_path = svg_dir / f"{figure_id}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    return {
        "png_path": png_path.relative_to(root).as_posix(),
        "svg_path": svg_path.relative_to(root).as_posix(),
        "png_bytes": png_path.stat().st_size,
        "svg_bytes": svg_path.stat().st_size,
        "png_sha256": sha256_file(png_path),
        "svg_sha256": sha256_file(svg_path),
    }
