"""
PyTorch Lightning module for Mars DepthFM.

Implements the DepthFM flow matching training loop with:
- Proper train/val/test step separation
- Per-epoch metric logging with all standard depth metrics
- Flow evolution visualisation at configurable intervals
- EMA weight averaging
- Normals loss warmup scheduling

DDP-safe: all sync_dist logging is unconditional, all model inference
in visualization runs on all ranks (only rank 0 logs figures).
"""

from __future__ import annotations

import gc
import logging
import time
import warnings
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from depth_fm.losses import CombinedLoss
from depth_fm.metrics import compute_depth_metrics, compute_photo_consistency, MetricsAggregator
from depth_fm.model import build_model
from depth_fm.noise import q_sample

logger = logging.getLogger(__name__)

# Suppress the DDP stride mismatch warning — our contiguous hook handles it
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")


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

        # Cache for grad norm logging (avoid recomputing every step)
        self._grad_norm_log_interval = 50

        # Directory for vector-format (PDF) figures for paper use.
        # Resolved lazily from trainer.default_root_dir once training starts.
        self._vector_fig_dir: str | None = None

    def _trace(self, msg: str):
        """Helper to print explicit DDP synchronization trace logs."""
        # Using INFO level so it passes standard logging filters.
        # You can grep for "[DDP TRACE]" in your console output.
        step = getattr(self, "global_step", "N/A")
        logger.debug(f"[DDP TRACE | Rank {self.global_rank} | Step {step}] {msg}")

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
        """Decode to pixel space WITH gradient tracking for pixel-space losses."""
        return self.model.decode_from_latent(latent)

    @torch.no_grad()
    def _predict_depth(
            self, z_img: torch.Tensor, num_steps: int = 1
    ) -> torch.Tensor:
        """Multi-step Euler ODE solve from image latent to depth latent."""
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
        """Return decoded depth at intermediate ODE timesteps."""
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

        vis_every = self.config.training.get("train_vis_every_steps", 500)
        show_vis = self.config.training.get("show_train_vis", False)
        if show_vis and step % vis_every == 0 and step > 0:
            self._trace("training_step: preparing train visuals")
            with torch.no_grad():
                img_pix_vis = self._decode(z_img[:1])
            if self.global_rank == 0 and self.logger and hasattr(self.logger, "experiment"):
                self._trace("training_step: rank 0 logging train visuals")
                start_vis = time.perf_counter()
                self._log_training_visuals(
                    img_pix_vis=img_pix_vis,
                    v_target=v_target,
                    v_pred=v_pred,
                    confidence=batch.get("confidence"),
                    step=step
                )
                vis_dur = time.perf_counter() - start_vis
                self._trace(f"training_step: rank 0 finished train visuals in {vis_dur:.2f}s")

        self._trace(f"Exiting training_step for batch {batch_idx}")

        # Final NaN guard — prevent NaN from reaching the optimizer.
        # Under DDP + gradient accumulation, a single NaN poisons all
        # accumulated grads across all ranks.
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
        checkpoint["ema_shadow"] = self._ema_shadow
        checkpoint["ema_initialised"] = self._ema_initialised

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        loaded_shadow = checkpoint.get("ema_shadow", {})
        self._ema_shadow = {k: v.to(self.device) for k, v in loaded_shadow.items()}
        self._ema_initialised = checkpoint.get("ema_initialised", False)

    def on_validation_epoch_start(self) -> None:
        self._trace("Entered on_validation_epoch_start")
        self._val_aggregator = MetricsAggregator()
        self._val_vis_data = None
        # Store per-sample data for worst/best gallery plots
        self._val_gallery_data: dict[str, dict] = {}
        self._trace("Exiting on_validation_epoch_start")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._trace(f"Entered validation_step for batch {batch_idx}")
        signal_boost = self.config.data.get("signal_boost", 1.0)

        z_img = self._encode(batch["image"])
        z_depth = self._encode(batch["dtm"]) * signal_boost

        self._trace(f"validation_step batch {batch_idx}: starting _predict_depth (DDP collective)")
        z_pred = self._predict_depth(z_img, num_steps=self.config.get("test_euler_steps", 4))
        self._trace(f"validation_step batch {batch_idx}: finished _predict_depth")

        z_pred_unboosted = z_pred / signal_boost
        z_depth_unboosted = z_depth / signal_boost

        pred_pix = self._decode(z_pred_unboosted)[:, 0].float().cpu().numpy()
        gt_decoded = self._decode(z_depth_unboosted)[:, 0].float().cpu().numpy()
        gt_raw = batch["dtm"][:, 0].float().cpu().numpy()

        vae_reconstruction_error = float(np.abs(gt_decoded - gt_raw).mean())
        self.log("val/vae_reconstruction_error", vae_reconstruction_error, sync_dist=False)

        if "confidence" in batch:
            conf_mask = batch["confidence"][:, 0].float().cpu().numpy()
        else:
            conf_mask = np.ones_like(gt_raw)

        with torch.no_grad():
            x_source = self._get_x_source(z_img)
            v_target = z_depth - x_source
            t = self._sample_timesteps(z_img.shape[0])
            t_exp = t.view(-1, 1, 1, 1)
            z_t = (1.0 - t_exp) * x_source + t_exp * z_depth

            self._trace(f"validation_step batch {batch_idx}: starting model.predict_velocity (DDP collective)")
            v_pred = self.model.predict_velocity(z_t, t, z_img)
            self._trace(f"validation_step batch {batch_idx}: finished model.predict_velocity")

            vel_loss = F.mse_loss(v_pred, v_target)
        self.log("val/loss", vel_loss, prog_bar=True, sync_dist=False)

        # Per-sample metrics (including photo consistency when sun data available)
        for i in range(pred_pix.shape[0]):
            tile_id = batch.get("tile_id", [""])[i] if "tile_id" in batch else f"b{batch_idx}_s{i}"
            metrics = compute_depth_metrics(pred_pix[i], gt_raw[i], align=True)

            # Compute photometric consistency if sun data is available
            if "sun_vector" in batch and "intensity" in batch and "ambient" in batch:
                from depth_fm.metrics import affine_align
                pred_aligned_i, _, _ = affine_align(pred_pix[i], gt_raw[i])
                img_i = batch["image"][i].float().cpu().numpy()
                if img_i.shape[0] == 3:
                    ortho_gray_i = img_i.mean(axis=0)
                else:
                    ortho_gray_i = img_i[0]
                # Denormalize from [-1,1] to [0,1]
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
                    pass

            self._val_aggregator.add(metrics, tile_id)

            # Store per-sample data for worst/best gallery (limit memory: keep max 50)
            if len(self._val_gallery_data) < 50:
                img_i = batch["image"][i].float().cpu().numpy()
                if img_i.shape[0] == 3:
                    img_i = np.transpose(img_i, (1, 2, 0))
                    img_i = (img_i + 1.0) / 2.0
                self._val_gallery_data[tile_id] = {
                    "image": img_i,
                    "pred": pred_pix[i].copy(),
                    "gt": gt_raw[i].copy(),
                    "rmse": metrics.rmse,
                }

        flow_intermediates = None
        flow_vis_every = self.config.training.get("flow_vis_every_steps", 500)
        if batch_idx == 0 and self.global_step > 0:
            if self.current_epoch % max(flow_vis_every, 1) == 0:
                self._trace(f"validation_step batch {batch_idx}: computing flow_intermediates (DDP collective)")
                flow_intermediates = self._predict_flow_intermediates(z_img[:1], num_steps=4)
                self._trace(f"validation_step batch {batch_idx}: finished flow_intermediates")

        if batch_idx == 0:
            self._val_vis_data = {
                "batch": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
                "z_img": z_img.detach(),
                "z_depth": z_depth.detach(),
                "pred_pix": pred_pix,
                "gt_raw": gt_raw,
                "conf_mask": conf_mask,
                "flow_intermediates": flow_intermediates,
            }

        self._trace(f"Exiting validation_step for batch {batch_idx}")

    def on_validation_epoch_end(self) -> None:
        self._trace("Entered on_validation_epoch_end")
        summary = self._val_aggregator.summary()

        # State-of-the-art metric set:
        # SILog (Eigen 2014, KITTI primary) replaces delta_2/delta_3 as more informative
        # photo_consistency captures resolution-independent quality
        _FIXED_VAL_METRICS = [
            "rmse", "abs_rel", "si_log", "delta_1",
            "normal_angular_error", "slope_rmse",
            "photo_consistency", "psd_ratio",
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
        # Photo consistency as prog_bar metric for dual-checkpoint monitoring
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

        worst = self._val_aggregator.worst_k("rmse", k=5)
        best = self._val_aggregator.best_k("rmse", k=5)
        if worst:
            logger.info("Worst 5 val patches by RMSE: %s",
                        ", ".join(f"{tid}={v:.2f}m" for tid, v in worst))
        if best:
            logger.info("Best 5 val patches by RMSE: %s",
                        ", ".join(f"{tid}={v:.2f}m" for tid, v in best))

        if self.global_rank == 0 and self._val_vis_data is not None:
            if self.logger and hasattr(self.logger, "experiment"):
                self._trace("on_validation_epoch_end: Rank 0 starting visual logging uploads")
                start_vis = time.perf_counter()
                vd = self._val_vis_data
                self._log_validation_visuals(
                    vd["batch"], vd["z_img"], vd["z_depth"],
                    vd["pred_pix"], vd["gt_raw"], vd["conf_mask"],
                    flow_intermediates=vd.get("flow_intermediates"),
                )

                # Log worst/best 5 gallery for val
                self._log_gallery("val", worst, best)

                vis_dur = time.perf_counter() - start_vis
                self._trace(f"on_validation_epoch_end: Rank 0 finished visual logging uploads in {vis_dur:.2f}s")

            # Upload all vector PDFs as a downloadable wandb artifact
            self._upload_vector_figures_artifact(prefix="val")

        self._val_vis_data = None
        self._val_gallery_data = {}
        self._trace("Exiting on_validation_epoch_end")

    def _log_validation_visuals(
            self, batch, z_img, z_depth, pred_pix, gt_pix, conf_mask,
            flow_intermediates=None,
    ):
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
                plot_lunar_lambert_comparison,
            )
            import matplotlib.pyplot as plt

            img_np = batch["image"][0].float().cpu().numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np + 1) / 2
            pred = pred_pix[0]
            gt = gt_pix[0]
            step = self.global_step

            fig = plot_prediction_triptych(img_np, pred, gt, title=f"Step {step}")
            self._log_figure("val/triptych", fig, step)
            plt.close(fig)

            fig = plot_cross_sections(pred, gt, title=f"Cross-sections (step {step})")
            self._log_figure("val/cross_sections", fig, step)
            plt.close(fig)

            from depth_fm.metrics import affine_align
            pred_aligned, _, _ = affine_align(pred, gt)
            fig = plot_error_heatmap(pred_aligned, gt, title=f"Error map (step {step})")
            self._log_figure("val/error_map", fig, step)
            plt.close(fig)

            sun_vec = batch.get("sun_vector")
            intensity = batch.get("intensity")
            ambient = batch.get("ambient")

            if sun_vec is not None:
                from depth_fm.metrics import affine_align
                pred_aligned, _, _ = affine_align(pred, gt)

                sv = sun_vec[0].cpu().numpy()
                int_val = intensity[0].item() if intensity is not None else 1.0
                amb_val = ambient[0].item() if ambient is not None else 0.0

                fig = plot_lunar_lambert_comparison(
                    pred_aligned, gt,
                    sun_vector=sv,
                    intensity=int_val,
                    ambient=amb_val,
                    lunar_lambert_weight=0.5,  # FIXME actually train using correct metric
                    title=f"Lambertian Render (step {step})",
                )
                self._log_figure("val/lambertian_render", fig, step)
                plt.close(fig)

            # Re-generate normal maps using aligned data
            pred_aligned, _, _ = affine_align(pred, gt)
            fig = plot_normal_maps(pred_aligned, gt, title=f"Surface normals (step {step})")
            self._log_figure("val/normal_maps", fig, step)
            plt.close(fig)

            fig = plot_elevation_scatter(pred_aligned, gt, title=f"Pred vs GT (step {step})")
            self._log_figure("val/elevation_scatter", fig, step)
            plt.close(fig)

            if flow_intermediates is not None:
                single_intermediates = {t: v[0] for t, v in flow_intermediates.items()}
                fig = plot_flow_evolution(single_intermediates, gt, title=f"Flow (step {step})")
                self._log_figure("val/flow_evolution", fig, step)
                plt.close(fig)

        except Exception as e:
            logger.exception("Visual logging failed")

    def _log_training_visuals(
            self,
            img_pix_vis: torch.Tensor,
            v_target: torch.Tensor,
            v_pred: torch.Tensor,
            confidence: torch.Tensor | None,
            step: int,
    ) -> None:
        """Log a training-time triptych, expanded to include the confidence mask."""
        if self.global_rank != 0:
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            with torch.no_grad():
                img_np = img_pix_vis[0].float().cpu().numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = np.clip((img_np + 1.0) / 2.0, 0.0, 1.0)

                vt_mag = v_target[:1].float().norm(dim=1)[0].cpu().numpy()
                vp_mag = v_pred.detach()[:1].float().norm(dim=1)[0].cpu().numpy()

                def _norm01(arr):
                    lo, hi = arr.min(), arr.max()
                    return (arr - lo) / (hi - lo + 1e-8)

                vt_disp = _norm01(vt_mag)
                vp_disp = _norm01(vp_mag)

                mask_disp = None
                if confidence is not None:
                    # Extract the (H, W) mask from the first item in the batch
                    mask_disp = confidence[0, 0].float().cpu().numpy()

            # Dynamically size the figure based on whether the mask exists
            n_cols = 4 if mask_disp is not None else 3
            fig = plt.figure(figsize=(4 * n_cols, 4))
            gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.05)

            # 1. Input Image
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(img_np)
            ax0.set_title(f"Input image (step {step})", fontsize=9)
            ax0.axis("off")

            col_idx = 1

            # 2. Confidence Mask (If available)
            if mask_disp is not None:
                ax_mask = fig.add_subplot(gs[0, col_idx])
                # Plot binary mask in high-contrast gray
                im_mask = ax_mask.imshow(mask_disp, cmap="gray", vmin=0, vmax=1)
                ax_mask.set_title("Valid Data Mask", fontsize=9)
                ax_mask.axis("off")
                plt.colorbar(im_mask, ax=ax_mask, fraction=0.046, pad=0.04)
                col_idx += 1

            # 3. Target Velocity
            ax1 = fig.add_subplot(gs[0, col_idx])
            im1 = ax1.imshow(vt_disp, cmap="viridis", vmin=0, vmax=1)
            ax1.set_title("v_target ||·||₂", fontsize=9)
            ax1.axis("off")
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            col_idx += 1

            # 4. Predicted Velocity
            ax2 = fig.add_subplot(gs[0, col_idx])
            im2 = ax2.imshow(vp_disp, cmap="viridis", vmin=0, vmax=1)
            ax2.set_title("v_pred ||·||₂", fontsize=9)
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            self._log_figure("train/velocity_triptych", fig, step)
            plt.close(fig)

        except Exception as e:
            logger.exception("Training visual logging failed")

    def _get_vector_fig_dir(self) -> str:
        """Return (and create) the directory for vector-format PDF figures."""
        if self._vector_fig_dir is None:
            from pathlib import Path
            root = getattr(self.trainer, "default_root_dir", "outputs")
            d = Path(root) / "vector_figures"
            d.mkdir(parents=True, exist_ok=True)
            self._vector_fig_dir = str(d)
        return self._vector_fig_dir

    def _log_figure(self, tag: str, fig, step: int):
        """Log a matplotlib figure as BOTH a raster preview and a vector PDF.

        Strategy for publication-quality outputs:
          1. Save as PDF locally in vector_figures/ — lossless vector format
             ready for LaTeX/Overleaf inclusion.
          2. Log a raster preview to wandb via wandb.Image() for quick
             browsing in the web UI.

        The PDFs are additionally bundled into a wandb Artifact at the end
        of each validation/test epoch for bulk download.
        """
        from pathlib import Path

        if self.logger is None and self.global_rank != 0:
            return

        # --- 1. Save vector PDF locally ---
        try:
            fig_dir = self._get_vector_fig_dir()
            # Sanitise tag for filename: "val/triptych" → "val_triptych"
            safe_name = tag.replace("/", "_").replace("\\", "_")
            pdf_path = Path(fig_dir) / f"{safe_name}_step{step}.pdf"
            fig.savefig(
                str(pdf_path),
                format="pdf",
                bbox_inches="tight",
                dpi=300,           # embedded raster elements (images) at 300 DPI
                metadata={"Creator": "Mars DepthFM", "Subject": tag},
            )
        except Exception:
            logger.exception(f"Failed to save vector PDF for {tag}")

        # --- 2. Log raster preview to wandb / tensorboard ---
        try:
            if self.logger is not None and hasattr(self.logger, "experiment"):
                exp = self.logger.experiment
                if hasattr(exp, "log"):
                    import wandb
                    exp.log({tag: wandb.Image(fig)}, step=step)
                elif hasattr(exp, "add_figure"):
                    exp.add_figure(tag, fig, global_step=step)
        except Exception:
            logger.exception(f"Failed to log raster preview for {tag}")

    def _upload_vector_figures_artifact(self, prefix: str = "val"):
        """Bundle all accumulated vector PDFs into a wandb Artifact for download.

        Called at the end of each val/test epoch so all figures from that
        epoch are available as a single downloadable artifact in the wandb UI.

        Navigate to:  Artifacts → "figures-{prefix}" → Files tab → Download
        """
        from pathlib import Path

        if self.global_rank != 0:
            return
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return

        fig_dir = Path(self._get_vector_fig_dir())
        pdfs = sorted(fig_dir.glob("*.pdf"))
        if not pdfs:
            return

        try:
            import wandb
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

    def _log_gallery(self, prefix: str, worst: list, best: list):
        """Log worst/best 5 gallery plots to wandb for troubleshooting.

        Args:
            prefix: "val" or "test"
            worst: list of (tile_id, rmse_value) for worst patches
            best: list of (tile_id, rmse_value) for best patches
        """
        if self.global_rank != 0 or self.logger is None:
            return

        gallery_data = getattr(self, f"_{prefix}_gallery_data", {})
        if not gallery_data:
            return

        try:
            from depth_fm.visualization import plot_patch_gallery
            import matplotlib.pyplot as plt

            step = self.global_step

            # Build worst gallery patches
            worst_patches = []
            for tile_id, rmse_val in worst[:5]:
                if tile_id in gallery_data:
                    d = gallery_data[tile_id]
                    worst_patches.append({
                        "image": d["image"],
                        "pred": d["pred"],
                        "gt": d["gt"],
                        "tile_id": tile_id,
                        "rmse": rmse_val,
                    })

            if worst_patches:
                fig = plot_patch_gallery(
                    worst_patches,
                    title=f"{prefix.upper()} Worst {len(worst_patches)} by RMSE (step {step})",
                )
                self._log_figure(f"{prefix}/worst_5_gallery", fig, step)
                plt.close(fig)

            # Build best gallery patches
            best_patches = []
            for tile_id, rmse_val in best[:5]:
                if tile_id in gallery_data:
                    d = gallery_data[tile_id]
                    best_patches.append({
                        "image": d["image"],
                        "pred": d["pred"],
                        "gt": d["gt"],
                        "tile_id": tile_id,
                        "rmse": rmse_val,
                    })

            if best_patches:
                fig = plot_patch_gallery(
                    best_patches,
                    title=f"{prefix.upper()} Best {len(best_patches)} by RMSE (step {step})",
                )
                self._log_figure(f"{prefix}/best_5_gallery", fig, step)
                plt.close(fig)

        except Exception:
            logger.exception(f"Gallery logging failed for {prefix}")

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        self._trace("Entered on_test_epoch_start")
        self._test_aggregator = MetricsAggregator()
        self._test_gallery_data: dict[str, dict] = {}
        self._trace("Exited on_test_epoch_start")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._trace(f"Entered test_step for batch {batch_idx}")
        z_img = self._encode(batch["image"])
        z_depth = self._encode(batch["dtm"])

        num_steps = self.config.training.get("test_euler_steps", 4)

        self._trace(f"test_step batch {batch_idx}: starting _predict_depth (DDP collective)")
        z_pred = self._predict_depth(z_img, num_steps=num_steps)
        self._trace(f"test_step batch {batch_idx}: finished _predict_depth")

        pred_pix = self._decode(z_pred)[:, 0].float().cpu().numpy()
        gt_pix = self._decode(z_depth)[:, 0].float().cpu().numpy()
        gt_raw = batch["dtm"][:, 0].float().cpu().numpy()

        if "confidence" in batch:
            conf_mask = batch["confidence"][:, 0].float().cpu().numpy()
        else:
            conf_mask = np.ones_like(gt_raw)

        for i in range(pred_pix.shape[0]):
            tile_id = batch.get("tile_id", [""])[i] if "tile_id" in batch else f"b{batch_idx}_s{i}"
            metrics = compute_depth_metrics(pred_pix[i], gt_raw[i], align=True)

            # Compute photometric consistency if sun data available
            if "sun_vector" in batch and "intensity" in batch and "ambient" in batch:
                from depth_fm.metrics import affine_align
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
                    pass

            self._test_aggregator.add(metrics, tile_id)

            # Store per-sample data for worst/best gallery (limit memory)
            if len(self._test_gallery_data) < 50:
                img_i = batch["image"][i].float().cpu().numpy()
                if img_i.shape[0] == 3:
                    img_i = np.transpose(img_i, (1, 2, 0))
                    img_i = (img_i + 1.0) / 2.0
                self._test_gallery_data[tile_id] = {
                    "image": img_i,
                    "pred": pred_pix[i].copy(),
                    "gt": gt_raw[i].copy(),
                    "rmse": metrics.rmse,
                }

        self._trace(f"Exiting test_step for batch {batch_idx}")

    def on_test_epoch_end(self) -> None:
        self._trace("Entered on_test_epoch_end")
        summary = self._test_aggregator.summary()

        _FIXED_TEST_METRICS = [
            "rmse", "abs_rel", "si_log", "delta_1",
            "normal_angular_error", "slope_rmse",
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

            # Log worst/best 5 gallery for test
            worst = self._test_aggregator.worst_k("rmse", k=5)
            best = self._test_aggregator.best_k("rmse", k=5)
            if worst:
                logger.info("Worst 5 test patches by RMSE: %s",
                            ", ".join(f"{tid}={v:.2f}m" for tid, v in worst))
            if best:
                logger.info("Best 5 test patches by RMSE: %s",
                            ", ".join(f"{tid}={v:.2f}m" for tid, v in best))

            self._log_gallery("test", worst, best)

            # Upload all vector PDFs as a downloadable wandb artifact
            self._upload_vector_figures_artifact(prefix="test")

        self._test_gallery_data = {}
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
        """Evaluate test set at multiple Euler step counts."""
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

    # ------------------------------------------------------------------
    # Gradient norm logging — throttled to reduce overhead
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.global_step % self._grad_norm_log_interval != 0:
            return
        self._trace("Entered on_before_optimizer_step (grad norm logging)")
        total_norm_sq = 0.0
        for p in self.model.backbone.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().float().norm().item() ** 2
        self.log("train/grad_norm", total_norm_sq ** 0.5, prog_bar=False, rank_zero_only=True)
        self._trace("Exited on_before_optimizer_step")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        tc = self.config.training
        trainable = [p for p in self.model.backbone.parameters() if p.requires_grad]
        losses = [p for p in self.loss_fn.parameters() if p.requires_grad]

        logger.info("Number of trainable parameters parameters %d, loss function %d", len(trainable), len(losses))
        trainable += losses

        optimizer = torch.optim.AdamW(
            trainable,
            lr=tc.learning_rate,
            betas=(tc.adam_beta1, tc.adam_beta2),
            eps=tc.adam_epsilon,
            weight_decay=tc.weight_decay,
        )

        warmup_steps = tc.lr_warmup_steps
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