"""Academic publication styling and shared plotting utilities for Stage 05."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from evaluation.scripts.plot_retrieval_metrics import MARKERS as BASE_MARKERS
from evaluation.scripts.plot_retrieval_metrics import METHOD_LABELS as BASE_METHOD_LABELS

# Colorblind-safe Okabe-Ito-inspired palette distinguishing retrieval methods
METHOD_COLORS = {
    "bm25": "#0072B2",           # Blue
    "dense": "#D55E00",          # Vermilion
    "hybrid_rrf": "#009E73",     # Bluish Green
    "hybrid_rerank": "#7B2CBF",  # High-contrast Purple
    "none": "#4D4D4D",           # Dark Slate (NO_RAG)
}

COMBO_METHOD = {
    "A1": "bm25",
    "A2": "dense",
    "A3": "hybrid_rrf",
    "A4": "hybrid_rerank",
    "B1": "bm25",
    "B2": "dense",
    "B3": "hybrid_rrf",
    "B4": "hybrid_rerank",
    "NO_RAG": "none",
}

COMBO_PROFILE = {
    "A1": "recursive_512_100",
    "A2": "recursive_512_100",
    "A3": "recursive_512_100",
    "A4": "recursive_512_100",
    "B1": "structure_adaptive",
    "B2": "structure_adaptive",
    "B3": "structure_adaptive",
    "B4": "structure_adaptive",
    "NO_RAG": "none",
}

PROFILE_COLORS = {
    "recursive_512_100": "#1F5A8A",
    "structure_adaptive": "#D97706",
}

PROFILE_LINESTYLES = {
    "recursive_512_100": "-",
    "structure_adaptive": "--",
    "none": ":",
}

PROFILE_HATCHES = {
    "recursive_512_100": "",
    "structure_adaptive": "///",
    "none": "xx",
}

METHOD_MARKERS = dict(BASE_MARKERS)
METHOD_MARKERS["none"] = "X"
METHOD_LABELS = dict(BASE_METHOD_LABELS)
METHOD_LABELS["none"] = "No-RAG"


def resolve_font_family() -> tuple[str, list[str]]:
    """Use Arial when available on the host OS, otherwise document fallback."""
    available = {f.name for f in fm.fontManager.ttflist}
    if "Arial" in available:
        return "Arial", ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
    return "DejaVu Sans", ["DejaVu Sans", "Liberation Sans", "sans-serif"]


def configure_academic_style() -> dict[str, Any]:
    """Configure consistent publication-ready matplotlib defaults."""
    primary, fallback_list = resolve_font_family()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": fallback_list,
            "axes.unicode_minus": False,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9DEE3",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "axes.axisbelow": True,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.3,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#CBD5E1",
            "legend.framealpha": 0.94,
            "figure.titlesize": 13.0,
            "figure.titleweight": "bold",
            "svg.fonttype": "none",  # Keep text selectable in SVG
        }
    )
    return {
        "primary_font": primary,
        "fallback_chain": fallback_list,
        "dpi": 300,
    }


def combo_style(combo_id: str) -> dict[str, Any]:
    """Return consistent color, linestyle, marker, and hatch for a combo."""
    method = COMBO_METHOD[combo_id]
    profile = COMBO_PROFILE[combo_id]
    return {
        "color": METHOD_COLORS[method],
        "linestyle": PROFILE_LINESTYLES[profile],
        "marker": METHOD_MARKERS[method],
        "hatch": PROFILE_HATCHES[profile],
        "linewidth": 2.0 if profile == "recursive_512_100" else 1.85,
        "markersize": 5.5,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_figure(
    fig: plt.Figure,
    stem: str,
    png_dir: Path,
    svg_dir: Path,
    dpi: int = 300,
) -> dict[str, Any]:
    """Save figure as both 300 DPI PNG and vector SVG, then close it."""
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{stem}.png"
    svg_path = svg_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return {
        "png_path": png_path,
        "svg_path": svg_path,
        "png_bytes": png_path.stat().st_size,
        "svg_bytes": svg_path.stat().st_size,
        "png_sha256": sha256_file(png_path),
        "svg_sha256": sha256_file(svg_path),
    }


def add_protocol_footer(fig: plt.Figure, text: str, y: float = 0.005) -> None:
    """Attach a concise, muted protocol/source footer note at the bottom."""
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#4B5563",
        style="italic",
    )
