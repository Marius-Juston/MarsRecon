"""
Generate the qualitative results panel for the MarsFM paper.

Produces a 3-row × 7-column composite PDF:
    rows    = three representative test scenes (smooth | mid-frequency | rough)
    columns = ortho input | predicted DTM | GT DTM | signed error (post-affine) |
              pred normals | pred render | gt render

USAGE
-----
    python scripts/make_qualitative_panel.py \
        --config configs/train_hirise_large.yaml \
        --checkpoint outputs/.../depthfm-best-photo-step=9476-...ckpt \
        --out Figures/results/qualitative_panel.pdf \
        --scene-batches 30 12 47        # (optional) explicit batch indices
        --num-steps 2 --ensemble-size 4

If `--scene-batches` is omitted, the script samples 50 test patches, ranks them
by a per-patch slope-RMSE proxy on the GT, and picks the 10th, 50th, and 90th
percentile to span "smooth / mid / rough" terrain automatically.

DESIGN NOTES
------------
- We deliberately do NOT use the lightning_module's internal multiprocess
  visualization workers; this is a single-shot render on rank 0.
- The plot uses the same affine-alignment routine as metrics.py so the
  signed-error column is consistent with the test-set numbers.
- One A6000 is sufficient; we never broadcast across GPUs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import TwoSlopeNorm
from omegaconf import OmegaConf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter

# Allow `python scripts/...` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from depth_fm.training.train_lightning import build_dataloaders
from depth_fm.training.lightning_module import DepthFMLightningModule as MarsDepthFMLightning  # noqa: E402
from depth_fm.objectives.metrics import affine_align  # noqa: E402
from depth_fm.objectives.losses import PhotoclinometricLoss  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style — matches typical IEEE conference figure conventions
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.linewidth": 0.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile_stretch(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0,
                        mask: np.ndarray | None = None) -> np.ndarray:
    """Robust contrast stretch for grayscale display."""
    src = arr if mask is None else arr[mask.astype(bool)]
    if src.size == 0:
        return np.zeros_like(arr)
    a, b = np.percentile(src, [lo, hi])
    if b - a < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - a) / (b - a), 0.0, 1.0)


def _slope_proxy(dtm: np.ndarray, mask: np.ndarray) -> float:
    """Cheap slope-roughness summary used to bin patches into terrain classes."""
    gy, gx = np.gradient(dtm)
    g = np.hypot(gx, gy)
    return float(g[mask.astype(bool)].std())


@torch.no_grad()
def _predict_one(model, ortho: torch.Tensor, num_steps: int, ensemble: int) -> np.ndarray:
    """Run DepthFM inference on a single (3,H,W) ortho patch in [-1,1]."""
    model.eval()
    pred = model.predict_depth(
        ortho.unsqueeze(0),                       # (1,3,H,W)
        num_steps=num_steps,
        ensemble_size=ensemble,
    )                                              # (1,1,H,W) in [0,1]
    return pred.squeeze().cpu().numpy()


def _plot_panel(scenes: list[dict], out_path: Path, dtm_unit: str = "") -> None:
    """Render the 3-row × 7-col composite."""
    n_rows = len(scenes)
    fig, axes = plt.subplots(
        n_rows, 7,
        # Bumped width to 15.0 and wspace to 0.35 to give the text breathing room
        figsize=(13.0, 1.95 * n_rows),
        gridspec_kw={"wspace": 0.05, "hspace": 0.10}, 
    )
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = [
        "HiRISE RED ortho", "MarsFM prediction", "GT DTM",
        "Signed error", "Pred Normals", 
        "Pred LL Render", "GT LL Render"
    ]

    # Shared depth colour scale per row to keep pred/GT visually comparable
    for i, scene in enumerate(scenes):
        ortho = scene["ortho"]
        pred = scene["pred_aligned"]
        gt = scene["gt"]
        err = scene["err"]
        mask = scene["mask"].astype(bool)

        # vmin/vmax: 1st/99th percentile of GT in mask
        depth_lo, depth_hi = np.percentile(gt[mask], [1, 99])
        emax = max(abs(np.percentile(err[mask], [1, 99])).max(), 1e-6)

        # Col 0 — ortho (grayscale, percentile-stretched)
        axes[i, 0].imshow(_percentile_stretch(ortho, mask=mask),
                          cmap="gray", vmin=0, vmax=1)
        # Col 1 — prediction
        axes[i, 1].imshow(np.where(mask, pred, np.nan),
                          cmap="terrain", vmin=depth_lo, vmax=depth_hi)
        # Col 2 — GT
        im_gt = axes[i, 2].imshow(np.where(mask, gt, np.nan),
                                  cmap="terrain", vmin=depth_lo, vmax=depth_hi)
        # Col 3 — signed error
        im_err = axes[i, 3].imshow(np.where(mask, err, np.nan),
                                   cmap="RdBu_r",
                                   norm=TwoSlopeNorm(vmin=-emax, vcenter=0.0, vmax=emax))

        # Col 4 — Pred Normals
        n_disp = np.clip((scene["pred_normals"] + 1.0) / 2.0, 0.0, 1.0)
        n_disp[~mask] = np.nan
        axes[i, 4].imshow(n_disp)

        # Col 5 — Pred LL Render
        pr_disp = _percentile_stretch(scene["pred_render"], mask=mask)
        pr_disp = np.where(mask, pr_disp, np.nan)
        axes[i, 5].imshow(pr_disp, cmap="gray", vmin=0, vmax=1)

        # Col 6 — GT LL Render
        gr_disp = _percentile_stretch(scene["gt_render"], mask=mask)
        gr_disp = np.where(mask, gr_disp, np.nan)
        axes[i, 6].imshow(gr_disp, cmap="gray", vmin=0, vmax=1)

        # Row label
        axes[i, 0].set_ylabel(scene["label"], fontsize=9)

        for j in range(7):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(col_titles[j])

        # --- THE FIX: Per-row colourbars using inset_axes ---
        # Anchors the colorbar 4% to the right of the axis bounding box
        # --- Per-row colourbars using inset_axes ---
        # cax_gt = inset_axes(axes[i, 2], width="6%", height="100%", loc='lower left',
        #                     bbox_to_anchor=(1.06, 0., 1, 1), bbox_transform=axes[i, 2].transAxes, borderpad=0)
        # Added format string to keep labels predictably short
        # cbar_gt = fig.colorbar(im_gt, cax=cax_gt, format=FormatStrFormatter('%.2f'))
        # cbar_gt.ax.tick_params(labelsize=6, length=0)

        # cax_err = inset_axes(axes[i, 3], width="6%", height="100%", loc='lower left',
        #                      bbox_to_anchor=(1.06, 0., 1, 1), bbox_transform=axes[i, 3].transAxes, borderpad=0)
        # cbar_err = fig.colorbar(im_err, cax=cax_err, format=FormatStrFormatter('%.2f'))
        # cbar_err.ax.tick_params(labelsize=6, length=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    log.info("wrote %s and .png", out_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path,
                   help="Same YAML used for training (drives test dataloader).")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Lightning .ckpt path (use the best-photo or best-rmse one).")
    p.add_argument("--out", type=Path,
                   default=Path("Figures/results/qualitative_panel.pdf"))
    p.add_argument("--scene-batches", type=int, nargs=3, default=None,
                   help="Optional explicit (smooth, mid, rough) test-loader indices.")
    p.add_argument("--candidate-pool", type=int, default=50,
                   help="Patches to score when auto-picking scenes.")
    p.add_argument("--num-steps", type=int, default=2)
    p.add_argument("--ensemble-size", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    cfg = OmegaConf.create(yaml.safe_load(args.config.read_text()))

    log.info("loading checkpoint %s", args.checkpoint)
    module = MarsDepthFMLightning.load_from_checkpoint(
        args.checkpoint, config=cfg, map_location=args.device, strict=False, weights_only=False
    ).to(args.device)
    module.eval()

    log.info("building test dataloader")
    loaders = build_dataloaders(cfg, split_seed=42, parallel=False)
    loader = loaders["test"]

    # Pull out the photo_loss from the loaded checkpoint so we keep the trained `lunar_lambert_weight` scalar
    photo_loss = getattr(module.loss_fn, "photo_loss", PhotoclinometricLoss()).to(args.device).eval()

    # ---- Pass 1: pick scenes -----------------------------------------------
    selected: list[dict] = []
    if args.scene_batches is not None:
        wanted = set(args.scene_batches)
        for idx, batch in enumerate(loader):
            if idx in wanted:
                selected.append({"idx": idx, "batch": batch})
            if len(selected) == 3:
                break
        # order: smooth -> mid -> rough as user supplied
        order = {b: i for i, b in enumerate(args.scene_batches)}
        selected.sort(key=lambda s: order[s["idx"]])
    else:
        log.info("auto-selecting scenes from %d candidates by slope spread",
                 args.candidate_pool)
        scored: list[tuple[float, int, dict]] = []
        for idx, batch in enumerate(loader):
            gt = batch["dtm"][0, 0].numpy()
            mask = batch.get("confidence", torch.ones_like(batch["dtm"]))[0, 0].numpy()
            scored.append((_slope_proxy(gt, mask), idx, batch))
            if idx + 1 >= args.candidate_pool:
                break
        scored.sort(key=lambda r: r[0])
        # 10th, 50th, 90th percentile rank
        ranks = [int(0.1 * len(scored)),
                 int(0.5 * len(scored)),
                 int(0.9 * len(scored))]
        for r in ranks:
            score, idx, batch = scored[r]
            selected.append({"idx": idx, "batch": batch})
        log.info("picked test indices %s (slope proxies %s)",
                 [s["idx"] for s in selected],
                 [scored[r][0] for r in ranks])

    labels = ["Smooth dunes", "Cratered plateau", "Rough ejecta"]

    # ---- Pass 2: predict + assemble ----------------------------------------
    scenes: list[dict] = []
    for label, item in zip(labels, selected):
        batch = item["batch"]
        ortho = batch["image"][0].to(args.device)            # (3,H,W) in [-1,1]
        gt = batch["dtm"][0, 0].numpy()
        mask = batch.get("confidence", torch.ones_like(batch["dtm"]))[0, 0].numpy()
        mask = mask > 0.5

        pred = _predict_one(module.model, ortho,
                            num_steps=args.num_steps,
                            ensemble=args.ensemble_size)     # (H,W) in [0,1]
        # Convert pred from [0,1] to the same signed range as GT, then affine-align
        pred_signed = 2.0 * pred - 1.0                       # match GT scale
        pred_aligned, scale, shift = affine_align(pred_signed, gt, mask)
        err = pred_aligned - gt

        # Use the RED channel (index 0 after replicate) for display
        ortho_disp = ortho[0].cpu().numpy()

        # -------------------------------------------------------------
        # Photoclinometric Renders and Surface Normals Extraction
        # -------------------------------------------------------------
        # Gather physics parameters to feed the renderer
        sun_vec = batch.get("sun_vector", torch.tensor([[0., 0., 1.]])).to(args.device)[0:1]
        intensity = batch.get("intensity", torch.tensor([1.0])).to(args.device).view(-1, 1)[0:1]
        ambient = batch.get("ambient", torch.tensor([0.0])).to(args.device).view(-1, 1)[0:1]

        # Elevate pred & gt to tensor shapes for the renderer
        pred_t = torch.from_numpy(pred_aligned).float().unsqueeze(0).unsqueeze(0).to(args.device)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0).to(args.device)

        with torch.no_grad():
            pred_render, pred_normals = photo_loss.render_from_depth(pred_t, sun_vec, intensity, ambient)
            gt_render, _ = photo_loss.render_from_depth(gt_t, sun_vec, intensity, ambient)

        # Push to cpu mapped numpy arrays for display
        pred_normals_np = pred_normals[0].cpu().numpy().transpose(1, 2, 0)
        pred_render_np = pred_render[0, 0].cpu().numpy()
        gt_render_np = gt_render[0, 0].cpu().numpy()
        # -------------------------------------------------------------

        scenes.append({
            "label": label,
            "ortho": ortho_disp,
            "pred_aligned": pred_aligned,
            "gt": gt,
            "err": err,
            "mask": mask,
            "pred_normals": pred_normals_np,
            "pred_render": pred_render_np,
            "gt_render": gt_render_np,
        })
        log.info("[%s] idx=%d  scale=%.3f  shift=%+.3f  RMSE=%.4f",
                 label, item["idx"], scale, shift,
                 float(np.sqrt(np.mean((err[mask.astype(bool)]) ** 2))))

    _plot_panel(scenes, args.out)


if __name__ == "__main__":
    main()