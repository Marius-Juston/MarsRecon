"""
Generate ``convergence_curves.pdf`` and ``lunar_lambert_weight.pdf`` from a
single WandB run.

These two figures appear in the visual-results appendix of MarsFM
(§A.8 Training Dynamics). They are the only appendix figures that depend on
the training-time logs rather than on test-set inference, so they are produced
from the WandB run history instead of from ``depth_fm/visualization.py``.

Output (matching the file paths used in ``appendix_visual_results.tex``):

    <output-dir>/convergence_curves.pdf      # held-out RMSE, photo-SSIM, δ₁
    <output-dir>/lunar_lambert_weight.pdf    # learnable photometric blend L
    <output-dir>/convergence_curves.csv      # raw data for reproduction
    <output-dir>/lunar_lambert_weight.csv

Usage
-----

    python make_training_figures.py \\
        --entity   your-wandb-entity   \\
        --project  marsfm              \\
        --run-id   abc123def           \\
        --best-step 9476               \\
        --output-dir Figures/DTM/visual_results

If WandB is unreachable or the run isn't yours, ``--from-csv <dir>`` re-renders
both PDFs from a previously-saved CSV pair without hitting the API.

Requires
--------
    pip install wandb pandas matplotlib seaborn
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger("make_training_figures")

# ---------------------------------------------------------------------------
# Style — matches depth_fm/visualization.py::set_neurips_style so the figures
# slot into the appendix without visual discontinuity.
# ---------------------------------------------------------------------------

_NEURIPS_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
}


def _setup_style() -> None:
    mpl.rcParams.update(_NEURIPS_RC)
    sns.set_theme(style="ticks", rc=_NEURIPS_RC)


# Wandb metric keys logged by lightning_module.py
_VAL_KEYS = ("val/rmse_mean", "val/photo_consistency_mean", "val/delta_1_mean")
_LL_KEY = "train/lunar_lambert_weight"
_STEP_KEY = "_step"


# ---------------------------------------------------------------------------
# WandB I/O
# ---------------------------------------------------------------------------

def _fetch_run(entity: str, project: str, run_id: str):
    """Resolve a run handle. Imported lazily so --from-csv mode has no wandb dep."""
    import wandb
    api = wandb.Api(timeout=60)
    return api.run(f"{entity}/{project}/{run_id}")


def fetch_validation_metrics(run) -> pd.DataFrame:
    """Return a DataFrame with columns [step, val/rmse_mean, val/photo_consistency_mean, val/delta_1_mean].

    Uses ``run.history`` with a high sample cap because validation is logged at
    epoch boundaries — a single full epoch produces one row, so 5000 samples
    covers ~5000 validation epochs which is far more than any realistic run.
    """
    keys = list(_VAL_KEYS) + [_STEP_KEY]
    df = run.history(keys=list(_VAL_KEYS), samples=5000, pandas=True, x_axis=_STEP_KEY)
    # Newer wandb returns the step column as `_step` already; older versions
    # may name it differently. Normalise.
    if _STEP_KEY not in df.columns:
        # Fall back to whatever the index is.
        df = df.reset_index().rename(columns={"index": _STEP_KEY})
    df = df.dropna(subset=list(_VAL_KEYS), how="all").sort_values(_STEP_KEY)
    df = df.rename(columns={_STEP_KEY: "step"})
    return df.reset_index(drop=True)


def fetch_lunar_lambert_weight(run) -> pd.DataFrame:
    """Return a DataFrame [step, lunar_lambert_weight].

    The LL weight is logged every training step, so we use ``scan_history``
    (server-side iterator, no sampling) to recover every value rather than
    a downsampled trace.
    """
    rows = []
    for record in run.scan_history(keys=[_LL_KEY, _STEP_KEY]):
        if record.get(_LL_KEY) is None:
            continue
        rows.append({"step": int(record[_STEP_KEY]),
                     "lunar_lambert_weight": float(record[_LL_KEY])})
    if not rows:
        raise RuntimeError(
            f"No samples found for {_LL_KEY!r}. Was this run trained with "
            f"the photoclinometric loss enabled?"
        )
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Figure 1: validation-metric convergence
# ---------------------------------------------------------------------------

def plot_convergence_curves(
        df: pd.DataFrame,
        best_step: int | None = None,
        save_path: str | Path | None = None,
) -> plt.Figure:
    """Three side-by-side panels: held-out RMSE, photo-consistency SSIM, δ₁.

    The selected-checkpoint step (``best_step``) is annotated on each panel
    with a vertical dashed line and a small text label.
    """
    _setup_style()
    palette = sns.color_palette("flare", 3)

    panels = [
        ("val/rmse_mean", "Held-out RMSE", "RMSE (norm.)", "↓"),
        ("val/photo_consistency_mean", "Photo-consistency", "SSIM", "↑"),
        ("val/delta_1_mean", r"$\delta_1$", r"$\delta_1$ (\%)", "↑"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharex=True)

    for ax, (key, title, ylabel, direction), color in zip(axes, panels, palette):
        if key not in df.columns:
            ax.text(0.5, 0.5, f"{key}\nnot in run history",
                    transform=ax.transAxes, ha="center", va="center", color="gray")
            ax.set_title(title)
            ax.set_xlabel("Training step")
            continue

        sub = df[["step", key]].dropna()
        if sub.empty:
            ax.text(0.5, 0.5, f"{key}\nempty after dropna",
                    transform=ax.transAxes, ha="center", va="center", color="gray")
            continue

        # Light EMA smoothing for visual clarity; raw points overlaid.
        ema = sub[key].ewm(span=5, adjust=False).mean()
        ax.plot(sub["step"], sub[key], color=color, alpha=0.25, linewidth=0.8,
                marker="o", markersize=3, markeredgewidth=0)
        ax.plot(sub["step"], ema, color=color, linewidth=1.8, label="EMA (span=5)")

        # Best-step annotation — only meaningful on the panel whose direction
        # of "best" the user actually selected on (photo-consistency, in this
        # paper). We show it on all three for visual continuity.
        if best_step is not None:
            ax.axvline(best_step, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
            # Pull the value at (or near) best_step for the label.
            idx = (sub["step"] - best_step).abs().idxmin()
            val = sub.loc[idx, key]
            ax.annotate(
                f"step {best_step}\n{val:.3g}",
                xy=(best_step, val),
                xytext=(8, -8), textcoords="offset points",
                fontsize=7, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="black", alpha=0.85, linewidth=0.5),
            )

        ax.set_title(f"{title}  ({direction})")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle("Validation-set convergence", fontweight="bold")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Wrote %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Lunar–Lambert weight trajectory
# ---------------------------------------------------------------------------

def plot_lunar_lambert_weight(
        df: pd.DataFrame,
        best_step: int | None = None,
        save_path: str | Path | None = None,
) -> plt.Figure:
    """Single panel: L = σ(ℓ_LL) over training steps with init reference."""
    _setup_style()
    palette = sns.color_palette("flare", 3)

    fig, ax = plt.subplots(figsize=(7, 3.8))

    # Raw + EMA. The LL weight is logged every step, so the raw trace can be
    # noisy; the EMA reveals the underlying drift.
    ax.plot(df["step"], df["lunar_lambert_weight"],
            color=palette[2], alpha=0.20, linewidth=0.6, label="raw")
    ema = df["lunar_lambert_weight"].ewm(span=200, adjust=False).mean()
    ax.plot(df["step"], ema, color=palette[0], linewidth=1.8,
            label="EMA (span=200)")

    # Reference: initialisation at L = 0.5 (sigmoid of logit=0.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
               label=r"Init.\ $L_0 = 0.5$")

    # Reference: pure-Lambertian and pure-Lommel-Seeliger limits
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.text(df["step"].iloc[-1] * 0.99, 0.985, "pure Lambertian",
            ha="right", va="top", fontsize=7, color="gray", style="italic")
    ax.text(df["step"].iloc[-1] * 0.99, 0.015, "pure Lommel--Seeliger",
            ha="right", va="bottom", fontsize=7, color="gray", style="italic")

    if best_step is not None:
        ax.axvline(best_step, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        idx = (df["step"] - best_step).abs().idxmin()
        val_at_best = df.loc[idx, "lunar_lambert_weight"]
        ax.annotate(
            f"selected ckpt\nL = {val_at_best:.3f}",
            xy=(best_step, val_at_best),
            xytext=(10, 12), textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="black", alpha=0.9, linewidth=0.5),
        )

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Lunar--Lambert weight $L = \sigma(\ell_{\mathrm{LL}})$")
    ax.set_title("Learnable photometric blend over training",
                 fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Wrote %s", save_path)
    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %s  (%d rows)", path, len(df))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    src = parser.add_argument_group("source")
    src.add_argument("--entity", help="WandB entity (user or team)", default="marius-juston")
    src.add_argument("--project", help="WandB project name", default="mars-depthfm")
    src.add_argument("--run-id", help="WandB run ID (the 8-char hash, not the display name)", default="z8gomwmq")
    src.add_argument("--from-csv", type=Path,
                     help="Skip the WandB API and re-render from a previously-"
                          "saved <output-dir> containing the two CSVs.")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", type=Path,
                     default=Path("Figures/DTM/visual_results"),
                     help="Where to write the two PDFs and CSVs.")
    out.add_argument("--best-step", type=int, default=9476,
                     help="Step at which the reported checkpoint was selected; "
                          "annotated on both figures. (Default: 9476, the value "
                          "in the paper.)")

    misc = parser.add_argument_group("misc")
    misc.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metric data ──
    if args.from_csv is not None:
        logger.info("Loading data from CSV directory %s", args.from_csv)
        val_df = pd.read_csv(args.from_csv / "convergence_curves.csv")
        ll_df = pd.read_csv(args.from_csv / "lunar_lambert_weight.csv")
    else:
        if not all([args.entity, args.project, args.run_id]):
            parser.error("--entity, --project, and --run-id are required "
                         "unless --from-csv is given.")
        logger.info("Resolving wandb run %s/%s/%s", args.entity, args.project, args.run_id)
        run = _fetch_run(args.entity, args.project, args.run_id)
        logger.info("Run found: '%s' (state=%s)", run.name, run.state)

        logger.info("Pulling validation metrics …")
        val_df = fetch_validation_metrics(run)
        logger.info("  → %d validation points across keys %s",
                    len(val_df), list(_VAL_KEYS))

        logger.info("Pulling Lunar-Lambert weight (per-step, may take a moment) …")
        ll_df = fetch_lunar_lambert_weight(run)
        logger.info("  → %d weight samples", len(ll_df))

        # Persist raw data so reviewers / co-authors can re-render without
        # WandB credentials.
        _save_csv(val_df, out_dir / "convergence_curves.csv")
        _save_csv(ll_df, out_dir / "lunar_lambert_weight.csv")

    # ── Render figures ──
    plot_convergence_curves(
        val_df, best_step=args.best_step,
        save_path=out_dir / "convergence_curves.pdf",
    )
    plot_lunar_lambert_weight(
        ll_df, best_step=args.best_step,
        save_path=out_dir / "lunar_lambert_weight.pdf",
    )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
