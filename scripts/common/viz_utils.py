"""Matplotlib helpers shared by figure-generation scripts.

* `apply_paper_style()`        — set matplotlib rcParams for journal-style figures.
* `save_fig(fig, path, ...)`   — write PDF + PNG with consistent DPI and tight bbox.

Migration note: several scripts under `scripts/visualization/` still inline
their own copies of these helpers. New scripts should use these versions; old
ones can be migrated opportunistically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


_PAPER_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_paper_style() -> None:
    """Apply journal-style matplotlib rcParams in place."""
    mpl.rcParams.update(_PAPER_RC)


def save_fig(
    fig: plt.Figure,
    path: str | Path,
    *,
    formats: Iterable[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save `fig` to `path` once per format. Returns the list of written paths."""
    base = Path(path)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in formats:
        target = base.with_suffix(f".{ext.lstrip('.')}")
        fig.savefig(target, dpi=dpi)
        written.append(target)
    return written
