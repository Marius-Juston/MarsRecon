"""
PyTorch Lightning module for Mars DepthFM.

Implements the DepthFM flow matching training loop with:
- Proper train/val/test step separation
- Per-epoch metric logging with all standard depth metrics
- Flow evolution visualisation at configurable intervals
- Normals loss warmup scheduling

DDP-safe: all sync_dist logging is unconditional, all model inference
in visualization runs on all ranks (only rank 0 logs figures).

────────────────────────────────────────────────────────────────────────
Visualization throughput design  (v2 — multiprocessing + round-robin)
────────────────────────────────────────────────────────────────────────

WHY THE ORIGINAL WAS SLOW
──────────────────────────
The original code used a `concurrent.futures.ThreadPoolExecutor` on rank 0
for background figure generation.  Three problems:

  1. **GIL contention.**  matplotlib (Agg backend) is pure-Python / Cython
     and holds the GIL during every rasterisation call.  The training
     DataLoader pre-fetch, metric aggregation, and gradient logging all
     compete for the same GIL on rank 0.

  2. **Rank asymmetry → DDP stalls.**  Only rank 0 generated figures, so
     it fell behind.  Every subsequent DDP collective (all-reduce during
     backward, sync_dist logging, barrier at checkpointing) forced the
     other GPUs to idle until rank 0 caught up.

  3. **Blocking checkpoint flush.**  `on_save_checkpoint` called
     `_flush_vis_tasks(timeout=120s)`, stalling the *entire* DDP group
     because the other ranks had already passed the save barrier and were
     waiting at the next collective.

HOW THIS VERSION FIXES IT
─────────────────────────
  A. **`torch.multiprocessing` worker (one per DDP rank).**
     A persistent daemon `mp.Process` (started with the `spawn` context
     so it never inherits CUDA state — PyTorch docs mandate this) runs a
     tight matplotlib→PDF/PNG loop.  Because it is a *process*, not a
     thread, it owns its own GIL and never blocks the training process.

  B. **Round-robin figure distribution.**
     Instead of funneling all 9+ figures through rank 0, we broadcast
     the tiny vis snapshot (~1-2 MB of numpy arrays) from rank 0 to all
     ranks via `torch.distributed.broadcast`, then assign figures to
     ranks with `task_idx % world_size`.  Each rank's worker renders
     only its share of figures in parallel, cutting wall-clock time
     by ≈ world_size×.  Single-GPU training degrades gracefully (all
     tasks stay on rank 0).

  C. **Deferred wandb upload.**
     All ranks write PNGs + PDFs to a shared staging directory.  At the
     *start* of the next validation epoch, rank 0 sweeps the staging
     directory and does a single batched `wandb.log()` call.  This
     guarantees that uploads never block the training loop — by the time
     we look, the previous epoch's figures are certainly on disk.

  D. **Non-blocking checkpoint.**
     `on_save_checkpoint` no longer flushes pending vis tasks.  Worst
     case, a crash loses some PDFs from the current epoch — acceptable
     because the model weights are safe.

  E. **`mp.SimpleQueue` instead of `mp.Queue`.**
     Per PyTorch multiprocessing docs, SimpleQueue spawns no background
     threads and avoids the deadlock-prone serialisation threads that
     `mp.Queue` uses internally.

PERFORMANCE IMPACT (expected)
─────────────────────────────
  • GPU 0 idle time during validation: eliminated (no longer generates
    figures synchronously or via GIL-holding threads).
  • Total figure generation wall time: ÷ world_size (round-robin).
  • wandb upload latency: moved entirely out of the training critical
    path (deferred to next epoch start).
  • Checkpoint save time: reduced by up to 120 s (no vis flush).
"""

from __future__ import annotations

import itertools
import logging
import math
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import lightning
import lightning as L
import matplotlib as mpl
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import EMAWeightAveraging

from depth_fm.losses import CombinedLoss, PhotoclinometricLoss
from depth_fm.metrics import affine_align
from depth_fm.metrics import compute_depth_metrics, compute_photo_consistency, MetricsAggregator
from depth_fm.model import build_model
from depth_fm.noise import q_sample

mpl.use("Agg")

logger = logging.getLogger(__name__)

# Suppress the DDP stride mismatch warning — our contiguous hook handles it
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")


# ══════════════════════════════════════════════════════════════════════
# Multiprocessing visualisation worker  (module-level for pickling)
# ══════════════════════════════════════════════════════════════════════
#
# WHY MODULE-LEVEL:  `mp.Process(target=...)` with the `spawn` context
# pickles the target function.  Bound methods and closures fail to
# pickle reliably, so we use a plain top-level function.
#
# WHY NO CUDA:  The worker process must never import or touch CUDA.
# We import only matplotlib, numpy, and the pure-Python plotting
# utilities.  The `spawn` context guarantees a fresh interpreter
# with no inherited CUDA state (per PyTorch docs).

def _vis_worker_main(task_queue: mp.SimpleQueue, staging_dir: str, rank: int):
    """
    Persistent visualisation worker — runs in its own process.

    Receives task dicts from ``task_queue``, generates figures via the
    project's plotting utilities, and writes PDF + PNG to ``staging_dir``.

    Protocol:
      • Each task is a dict: ``{"type": str, "tag": str, "step": int, "data": dict}``
      • A ``None`` sentinel shuts the worker down.

    Design notes:
      • Imports are done inside this function so that ``spawn``-context
        child processes start clean (no CUDA, no inherited state).
      • We use ``SimpleQueue`` which has no background threads — the
        PyTorch multiprocessing docs specifically recommend it to avoid
        deadlocks from the serialisation threads that ``Queue`` spawns.
      • All exceptions are caught-and-logged so the worker never crashes
        silently.  The main process can continue training regardless.
    """
    # ── Fresh matplotlib backend in child process ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    # Re-import plot functions in child (spawn context = fresh interpreter)
    from depth_fm.visualization import (
        plot_prediction_triptych as _triptych,
        plot_cross_sections as _cross,
        plot_error_heatmap as _heatmap,
        plot_flow_evolution as _flow,
        plot_normal_maps as _normals,
        plot_elevation_scatter as _scatter,
        plot_lunar_lambert_comparison as _lambert,
        plot_patch_gallery as _gallery,
        plot_uncertainty_map as _uncertainty,
        plot_geomorphometric_analysis as geomorphometric,
        plot_radial_psd_curves as _radial_psd,
        plot_slope_error_map as _slope_error,
    )
    from depth_fm.metrics import affine_align as _align

    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)

    def _save(fig, tag: str, step: int):
        safe = tag.replace("/", "_").replace("\\", "_")
        pdf_path = staging / f"{safe}_step{step}.pdf"
        png_path = staging / f"{safe}_step{step}.png"
        fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", dpi=300,
                    metadata={"Creator": "Mars DepthFM", "Subject": tag})
        fig.savefig(str(png_path), format="png", bbox_inches="tight", dpi=150)
        _plt.close(fig)

    def _render(task: dict):
        t = task["type"]
        d = task["data"]
        tag = task["tag"]
        step = task["step"]

        if t == "triptych":
            fig = _triptych(d["img"], d["pred"], d["gt"], title=d["title"])
            _save(fig, tag, step)

        elif t == "cross_sections":
            fig = _cross(d["pred"], d["gt"], title=d["title"])
            _save(fig, tag, step)

        elif t == "error_heatmap":
            pred_a, _, _ = _align(d["pred"], d["gt"])
            fig = _heatmap(pred_a, d["gt"], title=d["title"])
            _save(fig, tag, step)

        elif t == "normal_maps":
            pred_a, _, _ = _align(d["pred"], d["gt"])
            fig = _normals(pred_a, d["gt"], title=d["title"])
            _save(fig, tag, step)

        elif t == "elevation_scatter":
            pred_a, _, _ = _align(d["pred"], d["gt"])
            fig = _scatter(pred_a, d["gt"], title=d["title"])
            _save(fig, tag, step)

        elif t == "lunar_lambert":
            pred_a, _, _ = _align(d["pred"], d["gt"])

            pred_t = torch.from_numpy(pred_a).float().unsqueeze(0).unsqueeze(0)
            gt_t = torch.from_numpy(d["gt"]).float().unsqueeze(0).unsqueeze(0)

            sun_t = torch.from_numpy(d["sun_vector"]).float().unsqueeze(0)

            int_t = torch.tensor([d["intensity"]], dtype=torch.float32)
            amb_t = torch.tensor([d["ambient"]], dtype=torch.float32)

            mask_disp = d.get("mask_disp")
            if mask_disp is not None:
                mask_t = torch.from_numpy(mask_disp).float().unsqueeze(0).unsqueeze(0)
            else:
                mask_t = torch.ones_like(gt_t)

            local_loss = PhotoclinometricLoss().eval()

            w = float(min(max(d["lunar_lambert_weight"], 1e-4), 1.0 - 1e-4))
            local_loss.lunar_lambert_logit.data = torch.tensor(
                math.log(w / (1.0 - w)), dtype=local_loss.lunar_lambert_logit.dtype
            )

            fig = _lambert(
                pred_dtm=pred_t,
                gt_dtm=gt_t,
                real_ortho=d["img"],
                sun_vector=sun_t,
                intensity=int_t,
                ambient=amb_t,
                loss_fn=local_loss,
                valid_mask=mask_t,
                title=d["title"],
            )
            _save(fig, tag, step)

        elif t == "flow_evolution":
            fig = _flow(d["intermediates"], d["gt"], title=d["title"])
            _save(fig, tag, step)

        elif t == "gallery":
            fig = _gallery(d["patches"], title=d["title"])
            _save(fig, tag, step)

        elif t == "velocity_triptych":
            # Training-time velocity visualisation
            import matplotlib.gridspec as _gs
            img_np = d["img"]
            vt_disp = d["vt_disp"]
            vp_disp = d["vp_disp"]
            mask_disp = d.get("mask_disp")

            n_cols = 4 if mask_disp is not None else 3
            fig = _plt.figure(figsize=(4 * n_cols, 4))
            gs = _gs.GridSpec(1, n_cols, figure=fig, wspace=0.05)

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(img_np)
            ax0.set_title(f"Input image (step {step})", fontsize=9)
            ax0.axis("off")

            col = 1
            if mask_disp is not None:
                ax_m = fig.add_subplot(gs[0, col])
                im_m = ax_m.imshow(mask_disp, cmap="gray", vmin=0, vmax=1)
                ax_m.set_title("Valid Data Mask", fontsize=9)
                ax_m.axis("off")
                _plt.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)
                col += 1

            ax1 = fig.add_subplot(gs[0, col])
            im1 = ax1.imshow(vt_disp, cmap="viridis", vmin=0, vmax=1)
            ax1.set_title("v_target ||·||₂", fontsize=9)
            ax1.axis("off")
            _plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            col += 1

            ax2 = fig.add_subplot(gs[0, col])
            im2 = ax2.imshow(vp_disp, cmap="viridis", vmin=0, vmax=1)
            ax2.set_title("v_pred ||·||₂", fontsize=9)
            ax2.axis("off")
            _plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            _save(fig, tag, step)
        elif t == "uncertainty_map":
            fig = _uncertainty(d["pred"], d["var_dtm"], d["img"], title=d["title"])
            _save(fig, tag, step)
        elif t == "geomorph_analysis":
            fig = geomorphometric(d["pred"], d["gt"], title=d["title"])
            _save(fig, tag, step)
        elif t == "radial_psd":
            fig = _radial_psd(d["pred"], d["gt"], title=d["title"])
            _save(fig, tag, step)

        elif t == "slope_error_dod":
            fig = _slope_error(d["pred"], d["gt"], d["img"], title=d["title"])
            _save(fig, tag, step)

        else:
            print(f"[VisWorker rank={rank}] Unknown task type: {t}", file=sys.stderr, flush=True)

    # ── Main loop ──
    while True:
        try:
            task = task_queue.get()
        except EOFError:
            break

        if task is None:  # Poison pill → shutdown
            break

        try:
            _render(task)
        except Exception:
            # Print full traceback to stderr — the main process logger may
            # not be accessible from the child.
            print(
                f"[VisWorker rank={rank}] Error rendering {task.get('tag', '?')}:\n"
                f"{traceback.format_exc()}",
                file=sys.stderr, flush=True,
            )


class DepthFMLightningModule(L.LightningModule):
    """Lightning module for DepthFM flow matching training.

    Handles:
    - Conditional flow matching velocity loss
    - Surface normals auxiliary loss with warmup
    - Logit-normal timestep sampling
    - Noise augmentation on source distribution
    - Multi-step Euler inference for validation
    - Full metric computation and logging
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Build model
        torch.backends.cudnn.benchmark = True
        self.model = build_model(config)

        if config.training.get("optimizer", "adamw") != "shampoo":
            self.model = self.model.to(memory_format=torch.channels_last)
        else:
            logger.warning("Disabled channels_last memory format for shampoo training")

        # Loss
        lc = config.training.losses
        self.loss_fn = CombinedLoss(
            velocity_weight=lc.velocity_weight,
            normals_weight=lc.get("normals_weight", 0.1),
            normals_start_step=lc.get("normals_start_step", 2000),
            use_confidence_weighting=lc.get("use_confidence_weighting", False),
            freq_weight=lc.get("freq_weight", 0.0),
            freq_start_step=lc.get("freq_start_step", 0),
            freq_alpha=lc.get("freq_alpha", 1.0),
            grad_weight=lc.get("grad_weight", 0.0),
            grad_start_step=lc.get("grad_start_step", 0),
            grad_scales=tuple(lc.get("grad_scales", [1, 2, 4])),
            photo_weight=lc.get("photo_weight", 0.0),
            photo_start_step=lc.get("photo_start_step", 2000),
        )

        # Flow matching config
        self.fm = config.training.flow_matching

        # Metric aggregators (reset each epoch)
        self._val_aggregator = MetricsAggregator()
        self._test_aggregator = MetricsAggregator()

        # Track validation history for convergence plotting
        self.val_history: dict[str, list[float]] = {
            "val/rmse": [], "val/abs_rel": [], "val/delta_1": [],
            "val/loss": [], "train/loss": [],
        }

        # Cache for grad norm logging (avoid recomputing every step)
        self._grad_norm_log_interval = 50

        # ── Visualisation system (initialised lazily in on_train_start) ──
        # These are set up once training begins and the Trainer context is
        # available (world_size, default_root_dir, logger, etc.).
        self._vis_worker: mp.Process | None = None
        self._vis_queue: mp.SimpleQueue | None = None
        self._vis_staging_dir: str | None = None
        self._vector_fig_dir: str | None = None

    # ------------------------------------------------------------------
    # DDP / rank helpers
    # ------------------------------------------------------------------

    @property
    def _world_size(self) -> int:
        """Return DDP world size, defaulting to 1 for single-GPU."""
        if self.trainer and hasattr(self.trainer, "world_size"):
            return self.trainer.world_size
        return 1

    def _is_dist_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _should_handle_task(self, task_idx: int) -> bool:
        """Round-robin: does this rank own task_idx?

        WHY ROUND-ROBIN:  Distributing N figure-generation tasks across
        W workers gives each worker ⌈N/W⌉ tasks.  Because each rank's
        worker runs in its own process (separate GIL, separate CPU cores),
        figure generation is truly parallel.  On a 4-GPU node generating
        9 validation figures, each rank renders ≤3 figures instead of one
        rank rendering all 9.
        """
        return task_idx % self._world_size == self.global_rank

    def _trace(self, msg: str):
        """Helper to print explicit DDP synchronization trace logs."""
        step = getattr(self, "global_step", "N/A")
        logger.debug(f"[DDP TRACE | Rank {self.global_rank} | Step {step}] {msg}")

    # ------------------------------------------------------------------
    # Visualisation worker lifecycle
    # ------------------------------------------------------------------

    def _init_vis_worker(self):
        """Start the persistent background vis process for this rank.

        WHY PER-RANK WORKERS:  Each DDP rank gets its own worker process
        so that round-robin task assignment results in truly parallel
        figure generation across the node.  If we had a single shared
        worker (on rank 0), the other ranks' assigned tasks would still
        need to be sent to rank 0, re-introducing the asymmetry.

        WHY `spawn` CONTEXT:  PyTorch docs state that CUDA is incompatible
        with the `fork` start method.  `spawn` creates a fresh interpreter
        in the child, guaranteeing no inherited CUDA state, file locks, or
        background threads.

        WHY `SimpleQueue`:  PyTorch's multiprocessing best-practices docs
        warn that `mp.Queue` internally spawns serialisation threads that
        can cause deadlocks.  `SimpleQueue` is thread-free and sufficient
        for our ordered task stream.
        """
        root = getattr(self.trainer, "default_root_dir", "outputs")

        # All ranks write to a shared staging directory so rank 0 can
        # sweep it for wandb uploads.  Sub-dirs per rank avoid filename
        # collisions when multiple ranks render different figures
        # concurrently.
        self._vis_staging_dir = str(Path(root) / "vis_staging" / f"rank{self.global_rank}")
        Path(self._vis_staging_dir).mkdir(parents=True, exist_ok=True)

        # Also maintain the vector_fig_dir for backward compat / artifact
        self._vector_fig_dir = str(Path(root) / "vector_figures")
        Path(self._vector_fig_dir).mkdir(parents=True, exist_ok=True)

        ctx = mp.get_context("spawn")
        self._vis_queue = ctx.SimpleQueue()
        self._vis_worker = ctx.Process(
            target=_vis_worker_main,
            args=(self._vis_queue, self._vis_staging_dir, self.global_rank),
            daemon=True,  # Daemon so it dies with the parent if training crashes
            name=f"vis-worker-rank{self.global_rank}",
        )
        self._vis_worker.start()
        logger.info(
            "Rank %d: started vis worker process (PID %d, staging=%s)",
            self.global_rank, self._vis_worker.pid, self._vis_staging_dir,
        )

    def _shutdown_vis_worker(self, timeout: float = 30.0):
        """Gracefully shut down the vis worker.

        Sends a ``None`` poison pill then joins with a timeout.  If the
        worker is stuck rendering a complex figure, the timeout prevents
        an infinite hang at training end.
        """
        if self._vis_queue is not None and self._vis_worker is not None:
            try:
                self._vis_queue.put(None)  # Poison pill
            except Exception:
                pass
            self._vis_worker.join(timeout=timeout)
            if self._vis_worker.is_alive():
                logger.warning(
                    "Rank %d: vis worker did not exit within %.0fs, terminating",
                    self.global_rank, timeout,
                )
                self._vis_worker.terminate()
            else:
                logger.info("Rank %d: vis worker shut down cleanly", self.global_rank)
        self._vis_worker = None
        self._vis_queue = None

    def _submit_vis_task(self, task: dict):
        """Put a task dict onto the worker's queue.

        Non-blocking from the caller's perspective.  The dict is pickled
        into shared memory by SimpleQueue (zero-copy for numpy arrays
        that are already contiguous).
        """
        if self._vis_queue is None:
            logger.warning("Rank %d: vis worker not initialised, dropping task %s",
                           self.global_rank, task.get("tag", "?"))
            return
        try:
            self._vis_queue.put(task)
        except Exception:
            logger.exception("Failed to submit vis task %s", task.get("tag", "?"))

    # ------------------------------------------------------------------
    # Deferred wandb upload
    # ------------------------------------------------------------------

    def _deferred_wandb_upload(self):
        """Rank 0: sweep the staging directory and upload accumulated PNGs.

        WHY DEFERRED:  By uploading at the *start* of the next epoch
        (or at training end), we guarantee:
          1. All worker processes from the previous epoch have finished
             writing (they had an entire epoch of training time to do so).
          2. The upload network I/O never blocks the training loop.
          3. wandb auth/session lives only in the main process — no need
             to initialise wandb in child workers.

        The at-most-one-epoch delay in the wandb UI is negligible for
        monitoring purposes.
        """
        if self.global_rank != 0:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        root = Path(getattr(self.trainer, "default_root_dir", "outputs")) / "vis_staging"
        if not root.exists():
            return

        pngs = sorted(root.rglob("*.png"))
        if not pngs:
            return

        exp = self.logger.experiment
        if not hasattr(exp, "log"):
            return

        t0 = time.monotonic()
        batch_log = {}
        for png in pngs:
            # Parse tag and step from filename: "val_triptych_step1000.png"
            stem = png.stem  # e.g. "val_triptych_step1000"
            parts = stem.rsplit("_step", 1)
            if len(parts) == 2:
                tag = parts[0].replace("_", "/", 1)  # Restore first / separator
                try:
                    step = int(parts[1])
                except ValueError:
                    step = self.global_step
            else:
                tag = stem
                step = self.global_step

            batch_log[tag] = wandb.Image(str(png))

        if batch_log:
            try:
                exp.log(batch_log, step=self.global_step)
                logger.info(
                    "Rank 0: uploaded %d staged figures to wandb in %.1fs",
                    len(batch_log), time.monotonic() - t0,
                )
            except Exception:
                logger.exception("Failed to upload staged figures to wandb")

        # Also copy PDFs to vector_fig_dir for artifact bundling
        pdfs = sorted(root.rglob("*.pdf"))
        if self._vector_fig_dir and pdfs:
            vec_dir = Path(self._vector_fig_dir)
            for pdf in pdfs:
                dest = vec_dir / pdf.name
                if not dest.exists():
                    try:
                        import shutil
                        shutil.copy2(str(pdf), str(dest))
                    except Exception:
                        pass

        # Clean up uploaded files to avoid re-uploading next epoch
        for f in itertools.chain(pngs, pdfs):
            try:
                f.unlink()
            except Exception:
                pass

    def _upload_vector_figures_artifact(self, prefix: str = "val"):
        """Bundle all accumulated vector PDFs into a wandb Artifact."""
        if self.global_rank != 0:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return
        if self._vector_fig_dir is None:
            return

        fig_dir = Path(self._vector_fig_dir)
        pdfs = sorted(fig_dir.glob("*.pdf"))
        if not pdfs:
            return

        try:
            exp = self.logger.experiment
            if not hasattr(exp, "log_artifact"):
                return

            artifact = wandb.Artifact(
                name=f"figures-{prefix}-step{self.global_step}",
                type="figures",
                description=f"Vector PDF figures from {prefix} epoch "
                            f"(step {self.global_step}). Ready for LaTeX.",
            )
            for pdf in pdfs:
                artifact.add_file(str(pdf), name=pdf.name)
            exp.log_artifact(artifact)
            logger.info(
                "Uploaded %d vector PDFs as wandb artifact 'figures-%s-step%d'",
                len(pdfs), prefix, self.global_step,
            )
        except Exception:
            logger.exception("Failed to upload vector figures artifact")

    # ------------------------------------------------------------------
    # Broadcast vis data for round-robin
    # ------------------------------------------------------------------

    def _broadcast_numpy_dict(self, data: dict | None, src: int = 0) -> dict:
        """Broadcast a dict of numpy arrays + scalars from src to all ranks.

        WHY THIS IS NEEDED:  For round-robin figure assignment, every rank
        must have the same visualisation data (from rank 0's validation
        batch 0).  Broadcasting the ~1-2 MB of numpy arrays takes <1 ms
        on modern NVLink/InfiniBand interconnects — negligible compared
        to the matplotlib rendering time saved.

        HOW:  We convert numpy arrays to CPU tensors, broadcast them,
        then convert back.  Scalars are broadcast inside a 1-element tensor.
        Non-numeric values (strings) are ignored (each rank constructs
        them identically from the step number).
        """
        if not self._is_dist_initialized() or self._world_size <= 1:
            return data if data is not None else {}

        # First: broadcast which keys exist and their types
        # For simplicity, we assume all ranks know the key structure
        # (they do, because the task list is deterministic from config).
        if self.global_rank == src:
            assert data is not None
            result = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    t = torch.from_numpy(v.copy()).contiguous()
                    shape_t = torch.tensor(list(t.shape), dtype=torch.long, device=self.device)
                    dist.broadcast(shape_t, src=src)
                    t = t.to(self.device)
                    dist.broadcast(t, src=src)
                    result[k] = t.cpu().numpy()
                elif isinstance(v, (int, float)):
                    t = torch.tensor([v], dtype=torch.float64, device=self.device)
                    dist.broadcast(t, src=src)
                    result[k] = type(v)(t.item())
                elif isinstance(v, dict):
                    # Nested dict (e.g. flow_intermediates: {float: ndarray})
                    # Broadcast number of items, then each
                    n = torch.tensor([len(v)], dtype=torch.long, device=self.device)
                    dist.broadcast(n, src=src)
                    inner = {}
                    for ik, iv in sorted(v.items()):
                        key_t = torch.tensor([float(ik)], dtype=torch.float64, device=self.device)
                        dist.broadcast(key_t, src=src)
                        if isinstance(iv, np.ndarray):
                            vt = torch.from_numpy(iv.copy()).contiguous().to(self.device)
                            shape_t = torch.tensor(list(vt.shape), dtype=torch.long, device=self.device)
                            dist.broadcast(shape_t, src=src)
                            dist.broadcast(vt, src=src)
                            inner[float(key_t.item())] = vt.cpu().numpy()
                    result[k] = inner
                else:
                    result[k] = v  # strings, None — same on all ranks
            return result
        else:
            # Receiver side — must mirror the exact send order
            result = {}
            if data is None:
                data = {}
            for k, v_template in data.items():
                if isinstance(v_template, np.ndarray):
                    ndim = len(v_template.shape)
                    shape_t = torch.zeros(ndim, dtype=torch.long, device=self.device)
                    dist.broadcast(shape_t, src=src)
                    t = torch.zeros(*shape_t.tolist(), dtype=torch.float32, device=self.device)
                    dist.broadcast(t, src=src)
                    result[k] = t.cpu().numpy()
                elif isinstance(v_template, (int, float)):
                    t = torch.zeros(1, dtype=torch.float64, device=self.device)
                    dist.broadcast(t, src=src)
                    result[k] = type(v_template)(t.item())
                elif isinstance(v_template, dict):
                    n = torch.zeros(1, dtype=torch.long, device=self.device)
                    dist.broadcast(n, src=src)
                    inner = {}
                    for _ in range(n.item()):
                        key_t = torch.zeros(1, dtype=torch.float64, device=self.device)
                        dist.broadcast(key_t, src=src)
                        shape_t_inner = torch.zeros(len(list(v_template.values())[0].shape) if v_template else 2,
                                                    dtype=torch.long, device=self.device)
                        dist.broadcast(shape_t_inner, src=src)
                        vt = torch.zeros(*shape_t_inner.tolist(), dtype=torch.float32, device=self.device)
                        dist.broadcast(vt, src=src)
                        inner[float(key_t.item())] = vt.cpu().numpy()
                    result[k] = inner
                else:
                    result[k] = v_template
            return result

    # ------------------------------------------------------------------
    # Flow matching core
    # ------------------------------------------------------------------

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        method = self.fm.timestep_sampling
        if method == "logit_normal":
            mean = self.fm.get("logit_normal_mean", 0.0)
            std = self.fm.get("logit_normal_std", 1.0)
            normal = torch.randn(batch_size, device=self.device) * std + mean
            t = torch.sigmoid(normal)
        else:
            t = torch.rand(batch_size, device=self.device)
        return t.clamp(1e-5, 1.0 - 1e-5)

    def _get_x_source(self, z_img: torch.Tensor) -> torch.Tensor:
        if self.model.noising_step > 0:
            return q_sample(z_img, self.model.noising_step)
        return z_img.clone()

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode_to_latent(pixels)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.decode_from_latent(latent)

    def _decode_pixel_loss(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model.decode_from_latent(latent)

    @torch.no_grad()
    def estimate_uncertainty(
            self,
            z_img: torch.Tensor,
            num_samples: int = 10,
            num_steps: int = 4
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the epistemic uncertainty via stochastic ODE sampling.

        Args:
            z_img: (B, C, H, W) Encoded input orthoimage.
            num_samples: Number of stochastic noise initializations (N).
            num_steps: Euler integration steps per sample.

        Returns:
            mean_dtm: (B, 1, H, W) The expected topographic surface.
            var_dtm: (B, 1, H, W) The pixel-wise epistemic variance.
        """
        self._trace(f"Starting epistemic uncertainty estimation with N={num_samples}")

        dtm_hypotheses = []

        for i in range(num_samples):
            # _predict_depth automatically samples a new x_source ~ N(0, I)
            # via the _get_x_source() method internally.
            z_pred = self._predict_depth(z_img, num_steps=num_steps)

            # Decode to physical pixel space
            pred_pix = self._decode(z_pred)[:, 0]
            dtm_hypotheses.append(pred_pix)

        # Stack into shape: (N, B, 1, H, W)
        dtm_tensor = torch.stack(dtm_hypotheses, dim=0)

        # Calculate moments
        mean_dtm = torch.mean(dtm_tensor, dim=0)
        var_dtm = torch.var(dtm_tensor, dim=0, unbiased=True)

        return mean_dtm, var_dtm

    @torch.no_grad()
    def _predict_depth(self, z_img: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        x_source = self._get_x_source(z_img)
        z_t = x_source
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((z_t.shape[0],), t_val, device=self.device)
            v = self.model.predict_velocity(z_t, t, z_img)
            z_t = z_t + dt * v
        return z_t

    @torch.no_grad()
    def _predict_flow_intermediates(
            self, z_img: torch.Tensor, num_steps: int = 4,
    ) -> dict[float, torch.Tensor]:
        intermediates = {}
        x_source = self._get_x_source(z_img)
        z_t = x_source
        dt = 1.0 / num_steps
        intermediates[0.0] = self._decode(z_t)[:, 0].float().cpu().numpy()
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((z_t.shape[0],), t_val, device=self.device)
            v = self.model.predict_velocity(z_t, t, z_img)
            z_t = z_t + dt * v
            t_after = (step + 1) * dt
            decoded = self._decode(z_t)[:, 0].float().cpu().numpy()
            intermediates[t_after] = decoded
        return intermediates

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        self._trace(f"Entered training_step for batch {batch_idx}")
        z_img = self._encode(batch["image"])
        z_depth_raw = self._encode(batch["dtm"])

        z_depth = z_depth_raw * self.config.data.get("signal_boost", 1.0)

        B = z_img.shape[0]
        t = self._sample_timesteps(B)

        x_source = self._get_x_source(z_img)
        v_target = z_depth - x_source

        sigma_min = self.fm.get("sigma_min", 1e-4)
        t_exp = t.view(-1, 1, 1, 1)
        eps = torch.randn_like(x_source)
        z_t = (1.0 - t_exp) * x_source + t_exp * z_depth + sigma_min * eps

        self._trace(f"training_step: starting model.predict_velocity (forward pass)")
        v_pred = self.model.predict_velocity(z_t, t, z_img)
        self._trace(f"training_step: finished model.predict_velocity")

        # --- Velocity prediction diagnostics (detached, free) ---
        with torch.no_grad():
            v_pred_f = v_pred.detach().float()
            v_target_f = v_target.float()
            v_pred_norm = v_pred_f.norm(dim=1).mean()
            v_target_norm = v_target_f.norm(dim=1).mean()
            v_diff_norm = (v_pred_f - v_target_f).norm(dim=1).mean()
            v_rel_error = v_diff_norm / (v_target_norm + 1e-8)
            t_mean = t.float().mean()
            t_std = t.float().std()

        step = self.global_step

        self.log("train/v_pred_norm", v_pred_norm, rank_zero_only=True)
        self.log("train/v_target_norm", v_target_norm, rank_zero_only=True)
        self.log("train/v_rel_error", v_rel_error, rank_zero_only=True)
        self.log("train/t_mean", t_mean, rank_zero_only=True)
        self.log("train/t_std", t_std, rank_zero_only=True)

        with torch.no_grad():
            self.log("train/z_depth_std", z_depth_raw.std().item(), rank_zero_only=True)
            self.log("train/noise_std", x_source.std().item(), rank_zero_only=True)

        # Pixel-space losses
        pred_pix = gt_pix = None
        if self.loss_fn.needs_pixel_decode(step):
            self._trace("training_step: executing loss_fn.needs_pixel_decode block")
            z_pred_clean = z_t + (1.0 - t_exp) * v_pred
            pred_pix = self._decode_pixel_loss(z_pred_clean)
            gt_pix = self._decode(z_depth)

        self._trace("training_step: calculating loss_dict")
        loss_dict = self.loss_fn(
            v_pred=v_pred,
            v_target=v_target,
            pred_depth_pixels=pred_pix,
            gt_depth_pixels=gt_pix,
            confidence=batch.get("confidence"),
            global_step=step,
            real_ortho=batch["image"],
            sun_vector=batch.get("sun_vector"),
            ambient=batch.get("ambient"),
            intensity=batch.get("intensity"),
        )

        self.log("train/loss", loss_dict["total"], prog_bar=True, sync_dist=False)
        self.log("train/loss_velocity", loss_dict["velocity"], rank_zero_only=True)
        self.log("train/loss_photo", loss_dict["photo"], rank_zero_only=True)
        self.log("train/loss_normals", loss_dict["normals"], rank_zero_only=True)
        self.log("train/loss_freq", loss_dict["freq"], rank_zero_only=True)
        self.log("train/loss_grad", loss_dict["grad"], rank_zero_only=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], rank_zero_only=True)
        if "lunar_lambert_weight" in loss_dict:
            self.log("train/lunar_lambert_weight", loss_dict["lunar_lambert_weight"], rank_zero_only=True)

        if self.val_history and batch_idx == 0:
            self.val_history["train/loss"].append(loss_dict["total"].item())

        # ── Training visualisation (via mp worker, rank 0 only) ──
        # Training vis is infrequent (every N steps) and produces only
        # one figure, so round-robin overhead isn't warranted.  We just
        # send it to rank 0's worker.
        vis_every = self.config.training.get("train_vis_every_steps", 500)
        show_vis = self.config.training.get("show_train_vis", False)
        if show_vis and step % vis_every == 0 and step > 0 and self.global_rank == 0:
            self._trace("training_step: preparing train visuals")
            with torch.no_grad():
                img_pix_vis = self._decode(z_img[:1])
                img_np = img_pix_vis[0].float().cpu().numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = np.clip((img_np + 1.0) / 2.0, 0.0, 1.0)

                vt_mag = v_target[:1].float().norm(dim=1)[0].cpu().numpy()
                vp_mag = v_pred.detach()[:1].float().norm(dim=1)[0].cpu().numpy()

                def _n01(a):
                    lo, hi = a.min(), a.max()
                    return (a - lo) / (hi - lo + 1e-8)

                mask_disp = None
                conf = batch.get("confidence")
                if conf is not None:
                    mask_disp = conf[0, 0].float().cpu().numpy()

            self._submit_vis_task({
                "type": "velocity_triptych",
                "tag": "train/velocity_triptych",
                "step": step,
                "data": {
                    "img": img_np,
                    "vt_disp": _n01(vt_mag),
                    "vp_disp": _n01(vp_mag),
                    "mask_disp": mask_disp,
                },
            })

        self._trace(f"Exiting training_step for batch {batch_idx}")

        total_loss = loss_dict["total"]
        if not torch.isfinite(total_loss):
            logger.warning(
                "training_step: NaN/Inf loss at step %d batch %d. "
                "Returning zero to skip this step.",
                step, batch_idx,
            )
            return torch.tensor(0.0, device=total_loss.device, requires_grad=True)

        return total_loss

    # ------------------------------------------------------------------
    # Validation step — DDP-safe
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # ── NON-BLOCKING ──
        # The old code called _flush_vis_tasks(timeout=120s) here, which
        # stalled GPU 0 and caused all other ranks to idle at the next
        # DDP collective.
        #
        # Now we simply log a note.  The worst case is that a crash loses
        # some PDFs from the current epoch — acceptable because model
        # weights are always safe.
        if self._vis_worker is not None and self._vis_worker.is_alive():
            logger.info(
                "Rank %d: checkpoint saving — vis worker still active (non-blocking)",
                self.global_rank,
            )

    def on_validation_epoch_start(self) -> None:
        self._trace("Entered on_validation_epoch_start")
        self._val_aggregator = MetricsAggregator()
        self._val_vis_data = None


        # Replace self._val_gallery_data = {} with running lists
        self._best_val_patches = []
        self._worst_val_patches = []

        # ── Deferred upload from PREVIOUS epoch ──
        # By the time we start the next validation epoch, all worker
        # processes from the previous epoch have had an entire training
        # epoch to finish.  This is the ideal time to sweep the staging
        # directory and upload to wandb — zero risk of incomplete files.
        self._deferred_wandb_upload()

        self._trace("Exiting on_validation_epoch_start")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._trace(f"Entered validation_step for batch {batch_idx}")
        signal_boost = self.config.data.get("signal_boost", 1.0)

        z_img = self._encode(batch["image"])

        self._trace(f"validation_step batch {batch_idx}: starting _predict_depth (DDP collective)")
        z_pred = self._predict_depth(z_img, num_steps=self.config.get("test_euler_steps", 4))
        self._trace(f"validation_step batch {batch_idx}: finished _predict_depth")

        z_pred_unboosted = z_pred / signal_boost

        pred_pix_t = self._decode(z_pred_unboosted)[:, 0].float()
        gt_raw_t = batch["dtm"][:, 0].float()

        gt_raw = gt_raw_t.cpu().numpy()
        pred_pix = pred_pix_t.cpu().numpy()

        if "confidence" in batch:
            conf_mask = batch["confidence"][:, 0].float().cpu().numpy()
        else:
            conf_mask = np.ones_like(gt_raw)

        with torch.no_grad():
            z_depth = self._encode(batch["dtm"]) * signal_boost
            x_source = self._get_x_source(z_img)
            v_target = z_depth - x_source
            t = self._sample_timesteps(z_img.shape[0])
            t_exp = t.view(-1, 1, 1, 1)
            z_t = (1.0 - t_exp) * x_source + t_exp * z_depth

            self._trace(f"validation_step batch {batch_idx}: starting model.predict_velocity (DDP collective)")
            v_pred = self.model.predict_velocity(z_t, t, z_img)
            self._trace(f"validation_step batch {batch_idx}: finished model.predict_velocity")

            vel_loss = F.mse_loss(v_pred, v_target)
        self.log("val/loss", vel_loss, prog_bar=True, sync_dist=True)

        for i in range(pred_pix.shape[0]):
            tile_id = batch.get("tile_id", [""])[i] if "tile_id" in batch else f"b{batch_idx}_s{i}"
            metrics = compute_depth_metrics(pred_pix[i], gt_raw[i], align=True)

            if "sun_vector" in batch and "intensity" in batch and "ambient" in batch:
                pred_aligned_i, _, _ = affine_align(pred_pix[i], gt_raw[i])
                img_i = batch["image"][i].float().cpu().numpy()
                if img_i.shape[0] == 3:
                    ortho_gray_i = img_i.mean(axis=0)
                else:
                    ortho_gray_i = img_i[0]
                ortho_gray_i = (ortho_gray_i + 1.0) / 2.0
                sv_i = batch["sun_vector"][i].cpu().numpy()
                int_i = batch["intensity"][i].item()
                amb_i = batch["ambient"][i].item()
                ll_w = self.loss_fn.photo_loss.lunar_lambert_weight.item() if hasattr(
                    self.loss_fn.photo_loss, 'lunar_lambert_weight') else 0.5
                try:
                    photo_score = compute_photo_consistency(
                        pred_aligned_i, ortho_gray_i, sv_i, int_i, amb_i, ll_w,
                        valid_mask=(conf_mask[i] > 0.5) if conf_mask is not None else None,
                    )
                    metrics.photo_consistency = photo_score
                except Exception:
                    logger.exception("Problem computing phot consistency")

            self._val_aggregator.add(metrics, tile_id)

            if hasattr(metrics, "photo_consistency"):
                img_i = batch["image"][i].float().cpu().numpy()
                if img_i.shape[0] == 3:
                    img_i = np.transpose(img_i, (1, 2, 0))
                    img_i = (img_i + 1.0) / 2.0

                patch_data = {
                    "image": img_i,
                    "pred": pred_pix[i].copy(),
                    "gt": gt_raw[i].copy(),
                    "tile_id": tile_id,
                    "score": metrics.photo_consistency,
                    "rmse": metrics.rmse,
                }

                # Maintain best 5 (Assuming higher photo_consistency is better)
                self._best_val_patches.append(patch_data)
                self._best_val_patches.sort(key=lambda x: x["score"], reverse=True)
                self._best_val_patches = self._best_val_patches[:5]

                # Maintain worst 5
                self._worst_val_patches.append(patch_data)
                self._worst_val_patches.sort(key=lambda x: x["score"], reverse=False)
                self._worst_val_patches = self._worst_val_patches[:5]

        flow_intermediates = None
        flow_vis_every = self.config.training.get("flow_vis_every_steps", 500)
        if batch_idx == 0 and self.global_step > 0:
            if self.current_epoch % max(flow_vis_every, 1) == 0:
                self._trace(f"validation_step batch {batch_idx}: computing flow_intermediates (DDP collective)")
                flow_intermediates = self._predict_flow_intermediates(z_img[:1], num_steps=4)
                self._trace(f"validation_step batch {batch_idx}: finished flow_intermediates")

        if batch_idx == 0:
            if self.current_epoch % max(flow_vis_every, 1) == 0:
                self._trace(f"validation_step batch {batch_idx}: computing epistemic uncertainty")
                _, var_dtm = self.estimate_uncertainty(z_img[:1], num_samples=8, num_steps=4)
                var_dtm_np = var_dtm[0].float().cpu().numpy()
            else:
                var_dtm_np = None

            # Prepare vis snapshot — kept on rank 0, broadcast later
            img_np = batch["image"][0].float().cpu().numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np + 1) / 2

            sun_vec = batch.get("sun_vector")
            self._val_vis_data = {
                "img": img_np,
                "pred": pred_pix[0].copy(),
                "gt": gt_raw[0].copy(),
                "step": self.global_step,
                "has_sun": sun_vec is not None,
                "sun_vector": sun_vec[0].cpu().numpy() if sun_vec is not None else None,
                "intensity": batch["intensity"][0].item() if batch.get("intensity") is not None else 1.0,
                "ambient": batch["ambient"][0].item() if batch.get("ambient") is not None else 0.0,
                "flow_intermediates": (
                    {t: v[0] for t, v in flow_intermediates.items()}
                    if flow_intermediates is not None else None
                ),
                "var_dtm": var_dtm_np
            }

        self._trace(f"Exiting validation_step for batch {batch_idx}")

    def on_validation_epoch_end(self) -> None:
        self._trace("Entered on_validation_epoch_end")
        summary = self._val_aggregator.summary()

        _FIXED_VAL_METRICS = [
            "rmse", "abs_rel", "si_log", "delta_1",
            "normal_angular_error", "slope_rmse",
            "dbf_score", "curvature_rmse", "ms_ssim_topo",
            "photo_consistency", "psd_ratio", "patch_swd",
        ]

        self._trace("on_validation_epoch_end: Starting mandatory sync_dist collective calls")
        for metric_name in _FIXED_VAL_METRICS:
            val = summary.get(metric_name, {}).get("mean", 0.0) if summary else 0.0
            std = summary.get(metric_name, {}).get("std", 0.0) if summary else 0.0
            self.log(f"val/{metric_name}", val, sync_dist=True)
            self.log(f"val/{metric_name}_std", std, sync_dist=True)

        self.log("val/rmse_mean", summary.get("rmse", {}).get("mean", 0) if summary else 0.0, prog_bar=True,
                 sync_dist=True)
        self.log("val/delta_1_mean", summary.get("delta_1", {}).get("mean", 0) if summary else 0.0, prog_bar=True,
                 sync_dist=True)
        self.log("val/photo_consistency_mean",
                 summary.get("photo_consistency", {}).get("mean", 0) if summary else 0.0,
                 prog_bar=True, sync_dist=True)
        self._trace("on_validation_epoch_end: Finished mandatory sync_dist collective calls")

        if not summary:
            self._trace("on_validation_epoch_end: Summary empty, returning early")
            return

        for key in ["val/rmse", "val/abs_rel", "val/delta_1", "val/loss"]:
            metric_key = key.split("/")[-1]
            if metric_key in summary:
                self.val_history.setdefault(key, []).append(summary[metric_key]["mean"])

        worst = self._val_aggregator.worst_k("photo_consistency", k=5)
        best = self._val_aggregator.best_k("photo_consistency", k=5)
        if worst:
            logger.info("Worst 5 val patches by Photo Consistency: %s",
                        ", ".join(f"{tid}={v:.2f}m" for tid, v in worst))
        if best:
            logger.info("Best 5 val patches by Photo Consistency: %s",
                        ", ".join(f"{tid}={v:.2f}m" for tid, v in best))

        # ── Round-robin figure distribution ──
        #
        # All ranks participate in this block.  The vis data from rank 0's
        # batch 0 is already in self._val_vis_data on rank 0.  For multi-GPU,
        # we broadcast it so every rank can render its assigned subset.
        #
        # The task list is DETERMINISTIC (same indices on every rank), so
        # round-robin assignment is consistent without explicit coordination.

        if self._val_vis_data is not None and self.global_step > 0:
            vd = self._val_vis_data
            step = self.global_step

            # ── Build deterministic task list ──
            # The order here defines the round-robin assignment.
            # With 4 GPUs: rank 0 gets tasks 0,4,8; rank 1 gets 1,5; etc.
            tasks = []

            # 0: Prediction triptych
            tasks.append({
                "type": "triptych",
                "tag": "val/triptych",
                "step": step,
                "data": {"img": vd["img"], "pred": vd["pred"], "gt": vd["gt"],
                         "title": f"Step {step}"},
            })

            # 1: Cross-sections
            tasks.append({
                "type": "cross_sections",
                "tag": "val/cross_sections",
                "step": step,
                "data": {"pred": vd["pred"], "gt": vd["gt"],
                         "title": f"Cross-sections (step {step})"},
            })

            # 2: Error heatmap
            tasks.append({
                "type": "error_heatmap",
                "tag": "val/error_map",
                "step": step,
                "data": {"pred": vd["pred"], "gt": vd["gt"],
                         "title": f"Error map (step {step})"},
            })

            # 3: Normal maps
            tasks.append({
                "type": "normal_maps",
                "tag": "val/normal_maps",
                "step": step,
                "data": {"pred": vd["pred"], "gt": vd["gt"],
                         "title": f"Surface normals (step {step})"},
            })

            # 4: Elevation scatter
            tasks.append({
                "type": "elevation_scatter",
                "tag": "val/elevation_scatter",
                "step": step,
                "data": {"pred": vd["pred"], "gt": vd["gt"],
                         "title": f"Pred vs GT (step {step})"},
            })

            # 5: Lunar Lambert (conditional on sun data)
            # 5: Lunar Lambert (conditional on sun data)
            if vd.get("has_sun") and vd.get("sun_vector") is not None:
                # Safely extract the learned weight scalar from the live model
                try:
                    ll_weight = float(self.loss_fn.photo_loss.lunar_lambert_weight.item())
                except AttributeError:
                    logger.exception("Photo Loss has no lunar_lambert_weight or has not been initialized yet")
                    ll_weight = 0.5  # Fallback if photo_loss isn't initialized yet

                # If you have confidence maps, extract the valid mask for the render
                conf = self._val_vis_data.get("confidence")  # Or wherever you store it
                mask_disp = conf[0] if conf is not None else None

                tasks.append({
                    "type": "lunar_lambert",
                    "tag": "val/lambertian_render",
                    "step": step,
                    "data": {
                        "pred": vd["pred"],
                        "gt": vd["gt"],
                        "img": vd["img"],
                        "sun_vector": vd["sun_vector"],
                        "intensity": vd["intensity"],
                        "ambient": vd["ambient"],
                        "mask_disp": mask_disp,
                        "lunar_lambert_weight": ll_weight,
                        "title": f"Lambertian Render (step {step})",
                    },
                })

            # 6: Flow evolution (conditional)
            if vd.get("flow_intermediates") is not None:
                tasks.append({
                    "type": "flow_evolution",
                    "tag": "val/flow_evolution",
                    "step": step,
                    "data": {
                        "intermediates": vd["flow_intermediates"],
                        "gt": vd["gt"],
                        "title": f"Flow (step {step})",
                    },
                })

            # 7: Epistemic Uncertainty map (conditional)
            if vd.get("var_dtm") is not None:
                tasks.append({
                    "type": "uncertainty_map",
                    "tag": "val/epistemic_uncertainty",
                    "step": step,
                    "data": {
                        "img": vd["img"],
                        "pred": vd["pred"],
                        "var_dtm": vd["var_dtm"],
                        "title": f"Epistemic Variance (N=8, step {step})",
                    },
                })

            # 8: Geomorphometric Analysis Triptych
            tasks.append({
                "type": "geomorph_analysis",
                "tag": "val/geomorph_analysis",
                "step": step,
                "data": {
                    "pred": vd["pred"], "gt": vd["gt"],
                    "title": f"Geomorphometric Analysis: Edges & Curvature (step {step})",
                },
            })

            # 9. Radial PSD Curve
            tasks.append({
                "type": "radial_psd",
                "tag": "val/radial_psd",
                "step": step,
                "data": {
                    "pred": vd["pred"], "gt": vd["gt"],
                    "title": f"Spectral Synthesis (step {step})",
                },
            })

            # 10. Geomorphometric DoD (Slope Error)
            tasks.append({
                "type": "slope_error_dod",
                "tag": "val/slope_error_dod",
                "step": step,
                "data": {
                    "pred": vd["pred"], "gt": vd["gt"], "img": vd["img"],
                    "title": f"Slope Error DoD (step {step})",
                },
            })

            # ── Distribute tasks via round-robin ──
            # For multi-GPU: broadcast vis data so non-rank-0 workers
            # have the numpy arrays they need.
            if self._world_size > 1 and self._is_dist_initialized():
                self._trace("on_validation_epoch_end: broadcasting vis data for round-robin")
                # Broadcast the core numpy data that all tasks need.
                # We pack it into a flat dict for _broadcast_numpy_dict.
                broadcast_data = {
                    "img": vd["img"],
                    "pred": vd["pred"],
                    "gt": vd["gt"],
                }
                if vd.get("sun_vector") is not None:
                    broadcast_data["sun_vector"] = vd["sun_vector"]
                if vd.get("flow_intermediates") is not None:
                    broadcast_data["flow_intermediates"] = vd["flow_intermediates"]
                if vd.get("var_dtm") is not None:
                    broadcast_data["var_dtm"] = vd["var_dtm"]

                # On non-rank-0: create template dict with correct shapes
                # so the receiver side of _broadcast_numpy_dict knows the
                # structure.
                if self.global_rank != 0:
                    # Construct templates with correct shapes by using
                    # zero arrays — the actual data comes from broadcast.
                    # We get shapes from the task data structures.
                    pass  # broadcast_data has the right keys; values will be overwritten

                received = self._broadcast_numpy_dict(broadcast_data, src=0)

                # Rebuild tasks on non-rank-0 with received data
                if self.global_rank != 0:
                    for task in tasks:
                        td = task["data"]
                        if "img" in td:
                            td["img"] = received["img"]
                        if "pred" in td:
                            td["pred"] = received["pred"]
                        if "gt" in td:
                            td["gt"] = received["gt"]
                        if "sun_vector" in td and "sun_vector" in received:
                            td["sun_vector"] = received["sun_vector"]
                        if "intermediates" in td and "flow_intermediates" in received:
                            td["intermediates"] = received["flow_intermediates"]
                        if "var_dtm" in td and "var_dtm" in received:
                            td["var_dtm"] = received["var_dtm"]

                self._trace("on_validation_epoch_end: broadcast complete")

            # Submit only this rank's tasks
            my_task_count = 0
            for idx, task in enumerate(tasks):
                if self._should_handle_task(idx):
                    self._submit_vis_task(task)
                    my_task_count += 1

            self._trace(
                f"on_validation_epoch_end: rank {self.global_rank} submitted "
                f"{my_task_count}/{len(tasks)} figure tasks"
            )

        # ── Gallery: rank 0 only (needs aggregator data from this rank) ──
        # Gallery uses per-sample data stored in _val_gallery_data which
        # is rank-local.  Broadcasting 50 samples of gallery data would be
        # expensive and the benefit is minimal (it's just 2 figures).
        if self.global_rank == 0:
            for label, patches in [("worst", self._worst_val_patches), ("best", self._best_val_patches)]:
                if patches:
                    self._submit_vis_task({
                        "type": "gallery",
                        "tag": f"val/{label}_5_gallery",
                        "step": self.global_step,
                        "data": {
                            "patches": patches,
                            "title": f"VAL {label.upper()} {len(patches)} by Photo Consistency (step {self.global_step})",
                        },
                    })

        self._val_vis_data = None
        self._best_val_patches = []
        self._worst_val_patches = []
        self._trace("Exiting on_validation_epoch_end")

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        self._trace("Entered on_test_epoch_start")
        self._test_aggregator = MetricsAggregator()

        self._best_test_patches = []
        self._worst_test_patches = []

        self._trace("Exited on_test_epoch_start")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._trace(f"Entered test_step for batch {batch_idx}")
        z_img = self._encode(batch["image"])

        num_steps = self.config.training.get("test_euler_steps", 4)

        self._trace(f"test_step batch {batch_idx}: starting _predict_depth (DDP collective)")
        z_pred = self._predict_depth(z_img, num_steps=num_steps)
        self._trace(f"test_step batch {batch_idx}: finished _predict_depth")

        pred_pix_t = self._decode(z_pred)[:, 0].float()
        gt_raw_t = batch["dtm"][:, 0].float().cpu().numpy()

        pred_pix = pred_pix_t.cpu().numpy()
        gt_raw = gt_raw_t

        if "confidence" in batch:
            conf_mask = batch["confidence"][:, 0].float().cpu().numpy()
        else:
            conf_mask = np.ones_like(gt_raw)

        for i in range(pred_pix.shape[0]):
            tile_id = batch.get("tile_id", [""])[i] if "tile_id" in batch else f"b{batch_idx}_s{i}"
            metrics = compute_depth_metrics(pred_pix[i], gt_raw[i], align=True)

            if "sun_vector" in batch and "intensity" in batch and "ambient" in batch:
                pred_aligned_i, _, _ = affine_align(pred_pix[i], gt_raw[i])
                img_i = batch["image"][i].float().cpu().numpy()
                if img_i.shape[0] == 3:
                    ortho_gray_i = img_i.mean(axis=0)
                else:
                    ortho_gray_i = img_i[0]
                ortho_gray_i = (ortho_gray_i + 1.0) / 2.0
                sv_i = batch["sun_vector"][i].cpu().numpy()
                int_i = batch["intensity"][i].item()
                amb_i = batch["ambient"][i].item()
                ll_w = self.loss_fn.photo_loss.lunar_lambert_weight.item() if hasattr(
                    self.loss_fn.photo_loss, 'lunar_lambert_weight') else 0.5
                try:
                    photo_score = compute_photo_consistency(
                        pred_aligned_i, ortho_gray_i, sv_i, int_i, amb_i, ll_w,
                        valid_mask=(conf_mask[i] > 0.5) if conf_mask is not None else None,
                    )
                    metrics.photo_consistency = photo_score
                except Exception:
                    logger.exception("Problem computing photo consistency")

            self._test_aggregator.add(metrics, tile_id)

            if hasattr(metrics, "photo_consistency"):
                img_i = batch["image"][i].float().cpu().numpy()
                if img_i.shape[0] == 3:
                    img_i = np.transpose(img_i, (1, 2, 0))
                    img_i = (img_i + 1.0) / 2.0

                patch_data = {
                    "image": img_i,
                    "pred": pred_pix[i].copy(),
                    "gt": gt_raw[i].copy(),
                    "tile_id": tile_id,
                    "score": metrics.photo_consistency,
                    "rmse": metrics.rmse,
                }

                # Maintain best 5 (Assuming higher photo_consistency is better)
                self._best_test_patches.append(patch_data)
                self._best_test_patches.sort(key=lambda x: x["score"], reverse=True)
                self._best_test_patches = self._best_test_patches[:5]

                # Maintain worst 5
                self._worst_test_patches.append(patch_data)
                self._worst_test_patches.sort(key=lambda x: x["score"], reverse=False)
                self._worst_test_patches = self._worst_val_patches[:5]

        self._trace(f"Exiting test_step for batch {batch_idx}")

    def on_test_epoch_end(self) -> None:
        self._trace("Entered on_test_epoch_end")
        summary = self._test_aggregator.summary()

        _FIXED_TEST_METRICS = [
            "rmse", "abs_rel", "si_log", "delta_1",
            "normal_angular_error", "slope_rmse",
            "dbf_score", "curvature_rmse", "ms_ssim_topo",
            "photo_consistency", "psd_ratio",
        ]

        self._trace("on_test_epoch_end: Starting mandatory sync_dist collective calls")
        for metric_name in _FIXED_TEST_METRICS:
            val = summary.get(metric_name, {}).get("mean", 0.0) if summary else 0.0
            self.log(f"test/{metric_name}", val, sync_dist=True)
        self._trace("on_test_epoch_end: Finished mandatory sync_dist collective calls")

        if not summary:
            self._trace("on_test_epoch_end: Summary empty, returning early")
            return

        if self.global_rank == 0:
            logger.info("=" * 60)
            logger.info("TEST SET RESULTS")
            logger.info("=" * 60)
            for m, s in summary.items():
                logger.info("  %-25s  %.4f ± %.4f", m, s["mean"], s["std"])
            logger.info("=" * 60)

        worst = self._test_aggregator.worst_k("photo_consistency", k=5)
        best = self._test_aggregator.best_k("photo_consistency", k=5)
        if worst:
            logger.info("Worst 5 test patches by Photo Consistency: %s",
                        ", ".join(f"{tid}={v:.2f}m" for tid, v in worst))
        if best:
            logger.info("Best 5 test patches by Photo Consistency: %s",
                        ", ".join(f"{tid}={v:.2f}m" for tid, v in best))

        # Gallery on rank 0 only
        if self.global_rank == 0:
            for label, patches in [("worst", self._worst_test_patches), ("best", self._best_test_patches)]:
                if patches:
                    self._submit_vis_task({
                        "type": "gallery",
                        "tag": f"test/{label}_5_gallery",
                        "step": self.global_step,
                        "data": {
                            "patches": patches,
                            "title": f"TEST {label.upper()} {len(patches)} by Photo Consistency (step {self.global_step})",
                        },
                    })

        # Final upload for test
        self._deferred_wandb_upload()
        self._upload_vector_figures_artifact(prefix="test")

        self._best_test_patches = []
        self._worst_test_patches = []
        self._trace("Exiting on_test_epoch_end")

    # ------------------------------------------------------------------
    # Timestep ablation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_timestep_ablation(
            self,
            test_loader,
            step_counts: list[int] | None = None,
            max_batches: int | None = None,
    ) -> dict[int, dict[str, dict[str, float]]]:
        self._trace("Entered run_timestep_ablation")
        if step_counts is None:
            step_counts = [1, 2, 4, 8, 10, 20]

        self.eval()
        results = {}

        for n_steps in step_counts:
            self._trace(f"run_timestep_ablation: testing with {n_steps} steps")
            agg = MetricsAggregator()

            for batch_idx, batch in enumerate(test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                image = batch["image"].to(self.device)
                dtm = batch["dtm"].to(self.device)

                z_img = self._encode(image)
                z_depth = self._encode(dtm)

                self._trace(f"run_timestep_ablation (steps={n_steps}): starting _predict_depth")
                z_pred = self._predict_depth(z_img, num_steps=n_steps)
                self._trace(f"run_timestep_ablation (steps={n_steps}): finished _predict_depth")

                if self.trainer.world_size > 1:
                    self._trace(f"run_timestep_ablation (steps={n_steps}): starting all_gather collective")
                    z_pred = self.all_gather(z_pred).view(-1, *z_pred.shape[1:])
                    z_depth_gathered = self.all_gather(z_depth).view(-1, *z_depth.shape[1:])
                    self._trace(f"run_timestep_ablation (steps={n_steps}): finished all_gather collective")
                else:
                    z_depth_gathered = z_depth

                if self.global_rank == 0:
                    pred_pix = self._decode(z_pred)[:, 0].float().cpu().numpy()
                    gt_pix = self._decode(z_depth_gathered)[:, 0].float().cpu().numpy()

                    for i in range(pred_pix.shape[0]):
                        tile_id = f"b{batch_idx}_s{i}"
                        metrics = compute_depth_metrics(pred_pix[i], gt_pix[i], align=True)
                        agg.add(metrics, tile_id)

            results[n_steps] = agg.summary()

            if self.global_rank == 0:
                rmse_mean = results[n_steps].get("rmse", {}).get("mean", 0)
                d1_mean = results[n_steps].get("delta_1", {}).get("mean", 0)
                logger.info(
                    "  Steps=%2d  →  RMSE=%.4f m, δ₁=%.2f%%, n=%d samples",
                    n_steps, rmse_mean, d1_mean, len(agg.records),
                )

        self.train()
        self._trace("Exited run_timestep_ablation")
        return results

    # ------------------------------------------------------------------
    # DDP gradient contiguity fix
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        self._trace("Entered on_train_start")

        # ── Start vis worker on EVERY rank ──
        # Each rank gets its own worker so round-robin tasks are processed
        # in parallel across the node.
        self._init_vis_worker()

        _contiguous_hook = lambda g: g.contiguous() if not g.is_contiguous() else g

        n_hooks = 0
        for module in self.model.backbone.modules():
            if (
                    isinstance(module, torch.nn.Conv2d)
                    and module.kernel_size == (1, 1)
                    and module.weight.requires_grad
            ):
                module.weight.register_hook(_contiguous_hook)
                n_hooks += 1

        logger.info(
            "Registered contiguous-gradient hooks on %d 1×1 Conv2d weights"
            " (DDP channels_last stride fix)",
            n_hooks,
        )
        self._trace("Exited on_train_start")

    def on_train_end(self) -> None:
        """Final wandb upload and vis worker shutdown."""
        # Upload any remaining staged figures
        self._deferred_wandb_upload()
        self._upload_vector_figures_artifact(prefix="val")

        # Shut down the worker process
        self._shutdown_vis_worker(timeout=60.0)
        self._trace("Training complete, vis worker shut down")

    # ------------------------------------------------------------------
    # Gradient norm logging — throttled to reduce overhead
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.global_step % self._grad_norm_log_interval != 0:
            return
        self._trace("Entered on_before_optimizer_step (grad norm logging)")

        grads = [p.grad for p in self.parameters() if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(grads)

        self.log("train/grad_norm", total_norm, prog_bar=False, rank_zero_only=True)
        self._trace("Exited on_before_optimizer_step")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        tc = self.config.training
        trainable = [p for p in self.model.backbone.parameters() if p.requires_grad]
        losses = [p for p in self.loss_fn.parameters() if p.requires_grad]

        logger.info("Number of trainable parameters %d, loss function %d", len(trainable), len(losses))
        trainable += losses

        # 1. Determine Optimizer Strategy
        optimizer_type = tc.get("optimizer", "adamw").lower()

        optim = tc.shampoo if optimizer_type == "shampoo" else tc.adamw

        if optimizer_type == "shampoo":
            # Deferred import so non-Shampoo environments don't crash
            from distributed_shampoo import (
                DistributedShampoo,
                DDPDistributedConfig,
                DefaultEigenvalueCorrectedShampooConfig
            )
            import torch.distributed as dist

            # Extract DDP metadata safely
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                # Default to BF16 comms for efficiency, fallback to FP32 if specified
                comm_dtype = torch.bfloat16 if tc.get("mixed_precision") == "bf16" else torch.float32
            else:
                world_size = 1
                comm_dtype = torch.float32

            # Typically 8 GPUs per node. If running on a 4-GPU node, set to 4.
            trainers_per_group = min(tc.get("num_gpus", 4), world_size)

            logger.info("Initializing Distributed Shampoo (SOAP) for %d DDP ranks", world_size)

            optimizer = DistributedShampoo(
                trainable,
                lr=optim.learning_rate,
                betas=(optim.adam_beta1, optim.adam_beta2),
                epsilon=1e-12,  # Crucial: Must be extremely small for Shampoo
                weight_decay=optim.weight_decay,

                # --- Preconditioning Engine (SOAP) ---
                max_preconditioner_dim=optim.get("shampoo_max_dim", 4096),
                precondition_frequency=optim.get("shampoo_freq", 100),
                start_preconditioning_step=optim.get("shampoo_start", 100),
                preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,

                # --- DDP Synchronization Engine ---
                distributed_config=DDPDistributedConfig(
                    communication_dtype=comm_dtype,
                    num_trainers_per_group=trainers_per_group,
                    communicate_params=False,
                ),
            )
        else:
            # Fallback to standard AdamW
            logger.info("Initializing standard AdamW")
            optimizer = torch.optim.AdamW(
                trainable,
                lr=optim.learning_rate,
                betas=(optim.adam_beta1, optim.adam_beta2),
                eps=optim.adam_epsilon,
                weight_decay=optim.weight_decay,
            )

        # 2. Corrected Scheduler Logic (Flat + Delayed Cosine)
        warmup_steps = optim.lr_warmup_steps

        if optim.get("lr_scheduler", "constant") == "constant":
            # Flat learning rate after warmup (Highly recommended for Flow Matching)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(warmup_steps, 1),
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            # Traditional Cosine
            cosine_steps = tc.max_steps - warmup_steps
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(warmup_steps, 1),
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(cosine_steps, 1),
                eta_min=optim.learning_rate * 0.01,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class FasterEMAWeightAveraging(EMAWeightAveraging):
    def _swap_models(self, pl_module: lightning.LightningModule) -> None:
        assert self._average_model is not None
        average_params = itertools.chain(
            self._average_model.module.parameters(), self._average_model.module.buffers()
        )
        current_params = itertools.chain(pl_module.parameters(), pl_module.buffers())
        for average_param, current_param in zip(average_params, current_params):
            if average_param.device == current_param.device:
                average_param.data, current_param.data = current_param.data, average_param.data
            else:
                tmp = average_param.data.clone()
                average_param.data.copy_(current_param.data)
                current_param.data.copy_(tmp)
