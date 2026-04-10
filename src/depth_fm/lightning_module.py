"""
PyTorch Lightning module for Mars DepthFM.

Implements the DepthFM flow matching training loop with:
- Proper train/val/test step separation
- Per-epoch metric logging with all standard depth metrics
- Flow evolution visualisation at configurable intervals
- EMA weight averaging
- Normals loss warmup scheduling
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from depth_fm.losses import CombinedLoss
from depth_fm.metrics import compute_depth_metrics, MetricsAggregator
from depth_fm.model import build_model
from depth_fm.noise import q_sample

logger = logging.getLogger(__name__)


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
        self.model = build_model(config)

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

        # EMA
        self._ema_decay = config.training.get("ema_decay", 0.9999)
        self._ema_shadow: dict[str, torch.Tensor] = {}
        self._ema_initialised = False

        # Metric aggregators (reset each epoch)
        self._val_aggregator = MetricsAggregator()
        self._test_aggregator = MetricsAggregator()

        # Track validation history for convergence plotting
        self.val_history: dict[str, list[float]] = {
            "val/rmse": [], "val/abs_rel": [], "val/delta_1": [],
            "val/loss": [], "train/loss": [],
        }

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
        """
        Apply cosine-schedule noise to the image latent to produce the ODE
        starting distribution, matching the original DepthFM exactly.

        Mirrors dfm.py:
            if self.noising_step > 0:
                x_source = q_sample(x_source, self.noising_step)
        """
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
        """Decode to pixel space WITH gradient tracking.

        Used for pixel-space losses (FFL, gradient matching, normals) so that
        gradients flow back through the frozen VAE decoder to the predicted
        velocity.  The VAE parameters are frozen (requires_grad=False) so they
        are never updated; only the backbone gradients accumulate.
        """
        return self.model.decode_from_latent(latent)

    @torch.no_grad()
    def _predict_depth(
            self, z_img: torch.Tensor, num_steps: int = 1
    ) -> torch.Tensor:
        """
        Multi-step Euler ODE solve from image latent to depth latent.

        Matches dfm.py exactly:
          - ODE starts from q_sample(z_img, noising_step)  (noised source)
          - Context (channel-concat) is the CLEAN z_img
        """
        x_source = self._get_x_source(z_img)  # q_sample noise on starting point
        z_t = x_source
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((z_t.shape[0],), t_val, device=self.device)
            v = self.model.predict_velocity(z_t, t, z_img)  # clean image as context
            z_t = z_t + dt * v
        return z_t

    @torch.no_grad()
    def _predict_flow_intermediates(
            self, z_img: torch.Tensor, num_steps: int = 4,
    ) -> dict[float, torch.Tensor]:
        """Return decoded depth at intermediate ODE timesteps."""
        intermediates = {}
        x_source = self._get_x_source(z_img)  # noised starting point
        z_t = x_source
        dt = 1.0 / num_steps
        intermediates[0.0] = self._decode(z_t)[:, 0].float().cpu().numpy()
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((z_t.shape[0],), t_val, device=self.device)
            v = self.model.predict_velocity(z_t, t, z_img)  # clean image as context
            z_t = z_t + dt * v
            t_after = (step + 1) * dt
            decoded = self._decode(z_t)[:, 0].float().cpu().numpy()
            intermediates[t_after] = decoded
        return intermediates

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    # def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
    #     """
    #     Signals to the CUDAGraphs memory allocator that a new execution
    #     step is beginning BEFORE the compiled model is invoked.
    #     This perfectly protects the gradient checkpointing buffers.
    #     """
    #     if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
    #         torch.compiler.cudagraph_mark_step_begin()

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        z_img = self._encode(batch["image"])
        z_depth_raw = self._encode(batch["dtm"])

        z_depth = z_depth_raw * self.config.data.get("signal_boost", 1.0)

        B = z_img.shape[0]
        t = self._sample_timesteps(B)

        # Apply cosine noise to get the source distribution — matches dfm.py:
        #   x_source = q_sample(ims_z, self.noising_step)
        # The velocity field maps x_source → z_depth, so the target and
        # interpolation are both relative to the noised source, not clean z_img.
        x_source = self._get_x_source(z_img)
        v_target = z_depth - x_source

        sigma_min = self.fm.get("sigma_min", 1e-4)
        t_exp = t.view(-1, 1, 1, 1)
        eps = torch.randn_like(x_source)
        z_t = (1.0 - t_exp) * x_source + t_exp * z_depth + sigma_min * eps

        # Conditioning is the CLEAN image latent (channel-concat inside UNet),
        # matching dfm.py: context = ims_z  (not the noised x_source)
        v_pred = self.model.predict_velocity(z_t, t, z_img)

        # --- Velocity prediction diagnostics (detached, free) ---
        with torch.no_grad():
            v_pred_f = v_pred.detach().float()
            v_target_f = v_target.float()
            v_pred_norm = v_pred_f.norm(dim=1).mean()
            v_pred_norm = v_pred_f.norm(dim=1).mean()
            v_target_norm = v_target_f.norm(dim=1).mean()
            v_diff_norm = (v_pred_f - v_target_f).norm(dim=1).mean()
            v_rel_error = v_diff_norm / (v_target_norm + 1e-8)
            t_mean = t.float().mean()
            t_std = t.float().std()

        if self.global_step % 10 == 0:
            a = z_depth_raw.std().item()
            b = x_source.std().item()
            c = b / a

            self.log("train/z_depth_std", a, sync_dist=True)
            self.log("train/noise_std", b, sync_dist=True)
            self.log("train/ideal_ratio", c, sync_dist=True)

        self.log("train/v_pred_norm", v_pred_norm, sync_dist=True)
        self.log("train/v_target_norm", v_target_norm, sync_dist=True)
        self.log("train/v_rel_error", v_rel_error, sync_dist=True)
        self.log("train/t_mean", t_mean, sync_dist=False)
        self.log("train/t_std", t_std, sync_dist=False)

        # Pixel-space losses: project velocity → clean depth estimate (x₀ prediction)
        # x₀_pred = z_t + (1 − t) × v_pred  (rectified flow identity)
        # Pixel-space losses: project velocity → clean depth estimate (x₀ prediction)
        # x₀_pred = z_t + (1 − t) × v_pred  (rectified flow identity)
        pred_pix = gt_pix = None
        step = self.global_step
        if self.loss_fn.needs_pixel_decode(step):
            z_pred_clean = z_t + (1.0 - t_exp) * v_pred
            pred_pix = self._decode_pixel_loss(z_pred_clean)
            gt_pix = self._decode(z_depth)  # GT never needs gradients

        loss_dict = self.loss_fn(
            v_pred=v_pred,
            v_target=v_target,
            pred_depth_pixels=pred_pix,
            gt_depth_pixels=gt_pix,
            confidence=batch.get("confidence"),
            global_step=step,
            real_ortho=batch["image"],
            sun_vector=batch.get("sun_vector"),
            ambient=batch.get("ambient"),  # <--- ADDED
            intensity=batch.get("intensity")  # <--- ADDED
        )

        # Logging
        self.log("train/loss", loss_dict["total"], prog_bar=True, sync_dist=True)
        self.log("train/loss_velocity", loss_dict["velocity"], sync_dist=True)
        self.log("train/loss_normals", loss_dict["normals"], sync_dist=True)
        self.log("train/loss_freq", loss_dict["freq"], sync_dist=True)
        self.log("train/loss_grad", loss_dict["grad"], sync_dist=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], sync_dist=False)

        if self.val_history and batch_idx == 0:
            self.val_history["train/loss"].append(loss_dict["total"].item())

        # Periodic training-time velocity triptych
        vis_every = self.config.training.get("train_vis_every_steps", 500)
        show_vis = self.config.training.get("show_train_vis", False)
        if self.global_rank == 0:
            if show_vis and step % vis_every == 0 and step > 0 and self.logger and hasattr(self.logger, "experiment"):
                self._log_training_visuals(z_img, v_target, v_pred, step)

        return loss_dict["total"]

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["ema_shadow"] = self._ema_shadow
        checkpoint["ema_initialised"] = self._ema_initialised

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        loaded_shadow = checkpoint.get("ema_shadow", {})
        self._ema_shadow = {k: v.to(self.device) for k, v in loaded_shadow.items()}
        self._ema_initialised = checkpoint.get("ema_initialised", False)

    def on_validation_epoch_start(self) -> None:
        self._val_aggregator = MetricsAggregator()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        signal_boost = self.config.data.get("signal_boost", 1.0)

        z_img = self._encode(batch["image"])
        z_depth = self._encode(batch["dtm"]) * signal_boost

        # 1-step Euler prediction
        z_pred = self._predict_depth(z_img, num_steps=self.config.get("test_euler_steps", 4))

        z_pred_unboosted = z_pred / signal_boost
        z_depth_unboosted = z_depth / signal_boost

        # Decode predictions and the ground truth latent to pixel space
        pred_pix = self._decode(z_pred_unboosted)[:, 0].float().cpu().numpy()
        gt_decoded = self._decode(z_depth_unboosted)[:, 0].float().cpu().numpy()

        # Get the ACTUAL raw DTM from the dataloader for true metric comparison
        gt_raw = batch["dtm"][:, 0].float().cpu().numpy()

        vae_reconstruction_error = float(np.abs(gt_decoded - gt_raw).mean())
        self.log("val/vae_reconstruction_error", vae_reconstruction_error, sync_dist=True)

        if "confidence" in batch:
            conf_mask = batch["confidence"][:, 0].float().cpu().numpy()
        else:
            conf_mask = np.ones_like(gt_raw)

        # Velocity loss for logging — must match training_step
        x_source = self._get_x_source(z_img)
        v_target = z_depth - x_source
        t = self._sample_timesteps(z_img.shape[0])
        t_exp = t.view(-1, 1, 1, 1)
        z_t = (1.0 - t_exp) * x_source + t_exp * z_depth
        v_pred = self.model.predict_velocity(z_t, t, z_img)  # clean z_img
        vel_loss = F.mse_loss(v_pred, v_target)
        self.log("val/loss", vel_loss, prog_bar=True, sync_dist=True)

        # Per-sample metrics
        for i in range(pred_pix.shape[0]):
            tile_id = batch.get("tile_id", [""])[i] if "tile_id" in batch else f"b{batch_idx}_s{i}"
            metrics = compute_depth_metrics(pred_pix[i], gt_raw[i], align=True)
            self._val_aggregator.add(metrics, tile_id)

        if self.global_rank == 0:
            # Flow evolution plot (first batch only, first sample)
            if batch_idx == 0 and self.logger and hasattr(self.logger, "experiment"):
                self._log_validation_visuals(batch, z_img, z_depth, pred_pix, gt_raw, conf_mask)

    def on_validation_epoch_end(self) -> None:
        summary = self._val_aggregator.summary()
        if not summary:
            return

        for metric_name, stats in summary.items():
            self.log(f"val/{metric_name}", stats["mean"], sync_dist=True)
            self.log(f"val/{metric_name}_std", stats["std"], sync_dist=True)

        self.log("val/rmse_mean", summary.get("rmse", {}).get("mean", 0), prog_bar=True, sync_dist=True)
        self.log("val/delta_1_mean", summary.get("delta_1", {}).get("mean", 0), prog_bar=True, sync_dist=True)

        # Track for convergence plotting
        for key in ["val/rmse", "val/abs_rel", "val/delta_1", "val/loss"]:
            metric_key = key.split("/")[-1]
            if metric_key in summary:
                self.val_history.setdefault(key, []).append(summary[metric_key]["mean"])

        # Log worst patches
        worst = self._val_aggregator.worst_k("rmse", k=5)
        if worst:
            logger.info("Worst 5 val patches by RMSE: %s",
                        ", ".join(f"{tid}={v:.2f}m" for tid, v in worst))

    def _log_validation_visuals(self, batch, z_img, z_depth, pred_pix, gt_pix, conf_mask):
        """Log visualisation figures to wandb/tensorboard."""
        if self.global_rank != 0:
            return

        try:
            from depth_fm.visualization import (
                plot_prediction_triptych,
                plot_cross_sections,
                plot_error_heatmap,
                plot_flow_evolution,
                plot_normal_maps,
                plot_elevation_scatter,
                plot_hillshade_comparison,
            )
            import matplotlib.pyplot as plt

            # Take first sample
            img_np = batch["image"][0].float().cpu().numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np + 1) / 2  # [-1,1] → [0,1]
            pred = pred_pix[0]
            gt = gt_pix[0]

            step = self.global_step

            # Triptych
            fig = plot_prediction_triptych(img_np, pred, gt, title=f"Step {step}")
            self._log_figure("val/triptych", fig, step)
            plt.close(fig)

            # Cross-sections
            fig = plot_cross_sections(pred, gt, title=f"Cross-sections (step {step})")
            self._log_figure("val/cross_sections", fig, step)
            plt.close(fig)

            # Error heatmap
            from depth_fm.metrics import affine_align
            pred_aligned, _, _ = affine_align(pred, gt)
            fig = plot_error_heatmap(pred_aligned, gt, title=f"Error map (step {step})")
            self._log_figure("val/error_map", fig, step)
            plt.close(fig)

            # Hillshade comparison
            fig = plot_hillshade_comparison(
                pred, gt, azimuth=315.0, altitude=45.0, z_factor=2.0,
                title=f"Hillshade (step {step})",
            )
            self._log_figure("val/hillshade", fig, step)
            plt.close(fig)

            # Normal maps
            fig = plot_normal_maps(pred_aligned, gt, title=f"Surface normals (step {step})")
            self._log_figure("val/normal_maps", fig, step)
            plt.close(fig)

            # Elevation scatter
            fig = plot_elevation_scatter(pred_aligned, gt, title=f"Pred vs GT (step {step})")
            self._log_figure("val/elevation_scatter", fig, step)
            plt.close(fig)

            # Flow evolution (every N val epochs to save compute)
            if step > 0 and self.current_epoch % (self.config.training.get("flow_vis_every_steps", 2)) == 0:
                intermediates = self._predict_flow_intermediates(z_img[:1], num_steps=4)
                # Convert single-sample intermediates
                single_intermediates = {t: v[0] for t, v in intermediates.items()}
                fig = plot_flow_evolution(single_intermediates, gt, title=f"Flow (step {step})")
                self._log_figure("val/flow_evolution", fig, step)
                plt.close(fig)

        except Exception as e:
            logger.warning("Visual logging failed: %s", e)

    def _log_training_visuals(
            self,
            z_img: torch.Tensor,
            v_target: torch.Tensor,
            v_pred: torch.Tensor,
            step: int,
    ) -> None:
        """Log a training-time triptych: decoded input | v_target magnitude | v_pred magnitude.

        All ops are no-grad; failures are silenced so training is never interrupted.
        """
        if self.global_rank != 0:
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import numpy as np

            with torch.no_grad():
                # Decode input image from latent (first sample only)
                img_pix = self._decode(z_img[:1])  # (1, 3, H, W) in [-1, 1]
                img_np = img_pix[0].float().cpu().numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = np.clip((img_np + 1.0) / 2.0, 0.0, 1.0)

                # Per-pixel L2 velocity magnitude in latent space (h, w)
                vt_mag = v_target[:1].float().norm(dim=1)[0].cpu().numpy()
                vp_mag = v_pred.detach()[:1].float().norm(dim=1)[0].cpu().numpy()

                def _norm01(arr):
                    lo, hi = arr.min(), arr.max()
                    return (arr - lo) / (hi - lo + 1e-8)

                vt_disp = _norm01(vt_mag)
                vp_disp = _norm01(vp_mag)

            fig = plt.figure(figsize=(12, 4))
            gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(img_np)
            ax0.set_title(f"Input image (step {step})", fontsize=9)
            ax0.axis("off")

            ax1 = fig.add_subplot(gs[0, 1])
            im1 = ax1.imshow(vt_disp, cmap="viridis", vmin=0, vmax=1)
            ax1.set_title("v_target ||·||₂", fontsize=9)
            ax1.axis("off")
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            ax2 = fig.add_subplot(gs[0, 2])
            im2 = ax2.imshow(vp_disp, cmap="viridis", vmin=0, vmax=1)
            ax2.set_title("v_pred ||·||₂", fontsize=9)
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            self._log_figure("train/velocity_triptych", fig, step)
            plt.close(fig)

        except Exception as e:
            logger.warning("Training visual logging failed: %s", e)

    def _log_figure(self, tag: str, fig, step: int):
        """Log a matplotlib figure to the active logger."""
        if self.logger is None:
            return
        try:
            if hasattr(self.logger, "experiment"):
                exp = self.logger.experiment
                # wandb
                if hasattr(exp, "log"):
                    import wandb
                    exp.log({tag: wandb.Image(fig)}, step=step)
                # tensorboard
                elif hasattr(exp, "add_figure"):
                    exp.add_figure(tag, fig, global_step=step)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        self._test_aggregator = MetricsAggregator()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        z_img = self._encode(batch["image"])
        z_depth = self._encode(batch["dtm"])

        # Multi-step inference for best test quality
        num_steps = self.config.training.get("test_euler_steps", 4)
        z_pred = self._predict_depth(z_img, num_steps=num_steps)

        pred_pix = self._decode(z_pred)[:, 0].float().cpu().numpy()
        gt_pix = self._decode(z_depth)[:, 0].float().cpu().numpy()

        for i in range(pred_pix.shape[0]):
            tile_id = batch.get("tile_id", [""])[i] if "tile_id" in batch else f"b{batch_idx}_s{i}"
            metrics = compute_depth_metrics(pred_pix[i], gt_pix[i], align=True)
            self._test_aggregator.add(metrics, tile_id)

    def on_test_epoch_end(self) -> None:
        summary = self._test_aggregator.summary()
        if not summary:
            return

        for metric_name, stats in summary.items():
            self.log(f"test/{metric_name}", stats["mean"], sync_dist=True)

        # Log summary table
        logger.info("=" * 60)
        logger.info("TEST SET RESULTS")
        logger.info("=" * 60)
        for m, s in summary.items():
            logger.info("  %-25s  %.4f ± %.4f", m, s["mean"], s["std"])
        logger.info("=" * 60)

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
        """Evaluate test set at multiple Euler step counts.

        Runs the full test set (or ``max_batches`` batches) through the
        ODE solver at each step count in ``step_counts``, computing all
        metrics.  Returns a dict mapping step_count → summary stats,
        ready for :func:`plot_timestep_ablation`.

        Args:
            test_loader: DataLoader for the test split.
            step_counts: list of Euler step counts to evaluate.
                Defaults to [1, 2, 4, 8, 10, 20].
            max_batches: cap the number of batches per step count
                (useful for large test sets; None = full test set).

        Returns:
            dict mapping step_count → MetricsAggregator.summary()
        """
        if step_counts is None:
            step_counts = [1, 2, 4, 8, 10, 20]

        self.eval()
        results = {}

        for n_steps in step_counts:
            agg = MetricsAggregator()

            for batch_idx, batch in enumerate(test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # Move to device
                image = batch["image"].to(self.device)
                dtm = batch["dtm"].to(self.device)

                z_img = self._encode(image)
                z_depth = self._encode(dtm)

                z_pred = self._predict_depth(z_img, num_steps=n_steps)

                if self.trainer.world_size > 1:
                    z_pred = self.all_gather(z_pred).view(-1, *z_pred.shape[1:])
                    z_depth_gathered = self.all_gather(z_depth).view(-1, *z_depth.shape[1:])
                    # Also gather tile_ids if you are using them (requires string gathering or indices)
                else:
                    z_depth_gathered = z_depth

                if self.global_rank == 0:
                    pred_pix = self._decode(z_pred)[:, 0].float().cpu().numpy()
                    gt_pix = self._decode(z_depth_gathered)[:, 0].float().cpu().numpy()

                    for i in range(pred_pix.shape[0]):
                        tile_id = f"b{batch_idx}_s{i}"  # Simplified for gathered data
                        metrics = compute_depth_metrics(pred_pix[i], gt_pix[i], align=True)
                        agg.add(metrics, tile_id)

            results[n_steps] = agg.summary()

            rmse_mean = results[n_steps].get("rmse", {}).get("mean", 0)
            d1_mean = results[n_steps].get("delta_1", {}).get("mean", 0)
            logger.info(
                "  Steps=%2d  →  RMSE=%.4f m, δ₁=%.2f%%, n=%d samples",
                n_steps, rmse_mean, d1_mean, len(agg.records),
            )

        self.train()
        return results

    # ------------------------------------------------------------------
    # DDP gradient contiguity fix
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Register backward hooks on the specific Conv2d weights that produce
        channels_last gradients under torch.compile(mode='max-autotune').

        Root cause: max-autotune selects Triton kernels for 1×1 Conv2d backward
        that write weight gradients in channels_last layout, e.g.:
          grad strides   = [960, 1, 960, 960]  (channels_last NHWC)
          bucket strides = [960, 1, 1, 1]      (contiguous NCHW expected by DDP)
        DDP falls back to a slower copy path and warns:
          "Grad strides do not match bucket view strides."

        Scope: only 1×1 Conv2d weights — the sole source of channels_last grads.
        All other parameters (Linear, GroupNorm, 3×3 Conv) produce contiguous
        gradients so registering hooks on them would be pure overhead.

        Single-GPU training (strategy='auto') has no DDP reducer at all so this
        is entirely a no-op in that case, but registering a handful of hooks is
        still harmless.
        """
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

    # ------------------------------------------------------------------
    # Gradient norm logging
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log backbone gradient norm before clipping (after backward pass)."""
        total_norm_sq = 0.0
        num_params = 0
        for p in self.model.backbone.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().float().norm().item() ** 2
                num_params += 1
        if num_params > 0:
            self.log("train/grad_norm", total_norm_sq ** 0.5, prog_bar=False, sync_dist=False)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        tc = self.config.training
        trainable = [p for p in self.model.backbone.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable,
            lr=tc.learning_rate,
            betas=(tc.adam_beta1, tc.adam_beta2),
            eps=tc.adam_epsilon,
            weight_decay=tc.weight_decay,
        )

        warmup_steps = tc.lr_warmup_steps
        cosine_steps = tc.max_steps - warmup_steps

        # Phase 1: linear ramp from lr/warmup_steps → lr over warmup_steps steps
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(warmup_steps, 1),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        # Phase 2: cosine decay from lr → lr*0.01 over remaining steps
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cosine_steps, 1),
            eta_min=tc.learning_rate * 0.01,
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

    # ------------------------------------------------------------------
    # EMA (manual, called by callback)
    # ------------------------------------------------------------------

    def ema_update(self):
        if not self._ema_initialised:
            for n, p in self.model.backbone.named_parameters():
                if p.requires_grad:
                    self._ema_shadow[n] = p.data.clone()
            self._ema_initialised = True
            return

        for n, p in self.model.backbone.named_parameters():
            if p.requires_grad and n in self._ema_shadow:
                if self._ema_shadow[n].device != p.device:
                    self._ema_shadow[n] = self._ema_shadow[n].to(p.device)

                self._ema_shadow[n].mul_(self._ema_decay).add_(
                    p.data, alpha=1.0 - self._ema_decay
                )

    def load_ema_weights(self):
        self._ema_backup = {}
        for n, p in self.model.backbone.named_parameters():
            if n in self._ema_shadow:
                self._ema_backup[n] = p.data.clone()
                p.data.copy_(self._ema_shadow[n])

    def restore_training_weights(self):
        for n, p in self.model.backbone.named_parameters():
            if n in self._ema_backup:
                p.data.copy_(self._ema_backup[n])
        self._ema_backup = {}
        gc.collect()


class EMACallback(L.Callback):
    """Update EMA weights after each training step; use EMA for val/test."""

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        pl_module.ema_update()

    def on_validation_epoch_start(self, trainer, pl_module):
        if pl_module._ema_initialised:
            pl_module.load_ema_weights()

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module._ema_initialised:
            pl_module.restore_training_weights()

    def on_test_epoch_start(self, trainer, pl_module):
        if pl_module._ema_initialised:
            pl_module.load_ema_weights()

    def on_test_epoch_end(self, trainer, pl_module):
        if pl_module._ema_initialised:
            pl_module.restore_training_weights()
