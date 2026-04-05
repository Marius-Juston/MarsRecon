"""
Loss functions for Mars DepthFM training.

Primary loss: Conditional Flow Matching velocity regression
    L_FM = || v_theta(z_t, t; z_img) - (z_depth - z_img) ||^2

Auxiliary losses (all operate in pixel space on the predicted clean depth):
    Surface normals consistency  — angular error between Sobel normals
    Focal Frequency Loss (FFL)   — penalises frequency-domain discrepancies
    Multi-Scale Gradient Loss    — L1 slope error at 1×, 2×, 4× downsampling
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: focal-frequency-loss package
# Install with: pip install focal-frequency-loss
# ---------------------------------------------------------------------------
try:
    from focal_frequency_loss import FocalFrequencyLoss as _FFL

    _HAS_FFL = True
except ImportError:
    _HAS_FFL = False


class FlowMatchingVelocityLoss(nn.Module):
    """
    Conditional flow matching velocity loss.

    Given a predicted velocity v_pred and the target velocity (z_depth - z_img),
    compute the L2 loss, optionally weighted by confidence maps.
    """

    def __init__(self, use_confidence_weighting: bool = False):
        super().__init__()
        self.use_confidence = use_confidence_weighting

    def forward(
            self,
            v_pred: torch.Tensor,
            v_target: torch.Tensor,
            confidence: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            v_pred: predicted velocity (B, C, h, w)
            v_target: target velocity z_depth - z_img (B, C, h, w)
            confidence: optional weight map (B, 1, H, W) — will be downsampled

        Returns:
            scalar loss
        """
        sq_error = (v_pred.float() - v_target.float()).pow(2)

        if self.use_confidence and confidence is not None:
            if confidence.shape[-2:] != v_pred.shape[-2:]:
                confidence = F.interpolate(
                    confidence,
                    size=v_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            weight = confidence.clamp(0.1, 1.0)
            sq_error = sq_error * weight

        return sq_error.mean()


class SurfaceNormalsLoss(nn.Module):
    """
    Surface normals consistency loss.

    Computes surface normals from the spatial gradients (Sobel) of predicted and
    GT depth maps, then measures angular difference via cosine similarity.

    This operates in PIXEL space (decoded depth), not latent space.

    DepthFM uses the "normal consistency" formulation from:
        Fan et al., "Three-filters-to-normal", IEEE RA-L 2021
    which computes normals using finite differences and compares with cosine loss.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _compute_normals(self, depth: torch.Tensor) -> torch.Tensor:
        # Pad by replicating the edge pixels so gradients are 0 at the boundary
        padded_depth = F.pad(depth, (1, 1, 1, 1), mode='replicate')
        dz_dx = F.conv2d(padded_depth, self.sobel_x, padding=0)
        dz_dy = F.conv2d(padded_depth, self.sobel_y, padding=0)
        ones = torch.ones_like(dz_dx)
        normals = torch.cat([-dz_dx, -dz_dy, ones], dim=1)
        return F.normalize(normals, p=2, dim=1)

    def forward(
            self,
            pred_depth: torch.Tensor,
            gt_depth: torch.Tensor,
    ) -> torch.Tensor:
        if pred_depth.shape[1] == 3:
            pred_depth = pred_depth[:, :1]
        if gt_depth.shape[1] == 3:
            gt_depth = gt_depth[:, :1]

        pred_normals = self._compute_normals(pred_depth)
        gt_normals = self._compute_normals(gt_depth)

        cos_sim = (pred_normals * gt_normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        return (1.0 - cos_sim).mean().clamp(min=1e-6)


class MultiScaleGradientLoss(nn.Module):
    """Multi-scale gradient matching loss for high-frequency topography detail.

    Penalises L1 differences between predicted and GT spatial gradients at
    multiple downsampling scales.  This is inherently scale-invariant (derivative
    differences, not absolute values) and forces the model to learn crisp slopes,
    crater rims, and dune ripples regardless of residual absolute-scale errors.

    Loss:
        L_grad = mean over scales of:
            mean(|∇x_pred - ∇x_gt| + |∇y_pred - ∇y_gt|)

    Args:
        scales: tuple of integer downsampling factors (applied as average pooling).
    """

    def __init__(self, scales: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.scales = scales

    def _gradients(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (dy, dx) finite-difference gradients of a (B, 1, H, W) map."""
        dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]  # (B, 1, H-1, W)
        dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]  # (B, 1, H, W-1)
        return dy, dx

    def forward(
            self,
            pred_depth: torch.Tensor,
            gt_depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_depth: (B, 1, H, W) or (B, 3, H, W)
            gt_depth:   (B, 1, H, W) or (B, 3, H, W)
        Returns:
            scalar loss
        """
        if pred_depth.shape[1] == 3:
            pred_depth = pred_depth[:, :1]
        if gt_depth.shape[1] == 3:
            gt_depth = gt_depth[:, :1]

        total = torch.tensor(0.0, device=pred_depth.device)
        for s in self.scales:
            if s > 1:
                p = F.avg_pool2d(pred_depth, kernel_size=s, stride=s)
                g = F.avg_pool2d(gt_depth, kernel_size=s, stride=s)
            else:
                p, g = pred_depth, gt_depth

            dy_p, dx_p = self._gradients(p)
            dy_g, dx_g = self._gradients(g)

            total = total + (dy_p - dy_g).abs().mean() + (dx_p - dx_g).abs().mean()

        return total / len(self.scales)


class FocalFrequencyLoss(nn.Module):
    """Wrapper around the focal-frequency-loss package.

    Penalises discrepancies in the Fourier domain, explicitly forcing the
    network to recover high-frequency details that MSE smooths over.

    Requires: pip install focal-frequency-loss
    Paper: https://arxiv.org/pdf/2012.12821

    Args:
        loss_weight: Overall weight for this loss term.
        alpha: Focal factor — higher values focus more on hard frequencies.
    """

    def __init__(self, loss_weight: float = 1.0, alpha: float = 1.0):
        super().__init__()
        if not _HAS_FFL:
            raise ImportError(
                "focal-frequency-loss is not installed. "
                "Run: pip install focal-frequency-loss"
            )
        self._ffl = _FFL(loss_weight=loss_weight, alpha=alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._ffl(pred, target)


class CombinedLoss(nn.Module):
    """Combined training loss for Mars DepthFM.

    L_total = w_vel * L_FM
            + w_norm * L_normals       (after normals_start_step)
            + w_freq * L_FFL           (after freq_start_step)
            + w_grad * L_grad          (after grad_start_step)

    Pixel-space losses (normals, FFL, grad) require the caller to pass
    ``pred_depth_pixels`` and ``gt_depth_pixels`` decoded WITHOUT no_grad so
    that gradients flow back through the decoder to v_pred.
    """

    def __init__(
            self,
            velocity_weight: float = 1.0,
            normals_weight: float = 0.1,
            normals_start_step: int = 2000,
            use_confidence_weighting: bool = False,
            freq_weight: float = 0.0,
            freq_start_step: int = 0,
            freq_alpha: float = 1.0,
            grad_weight: float = 0.0,
            grad_start_step: int = 0,
            grad_scales: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        self.velocity_loss = FlowMatchingVelocityLoss(use_confidence_weighting)
        self.normals_loss = SurfaceNormalsLoss()
        self.grad_loss = MultiScaleGradientLoss(scales=grad_scales) if grad_weight > 0 else None
        self.vel_weight = velocity_weight
        self.norm_weight = normals_weight
        self.norm_start = normals_start_step
        self.freq_weight = freq_weight
        self.freq_start = freq_start_step
        self.grad_weight = grad_weight
        self.grad_start = grad_start_step

        if freq_weight > 0:
            if not _HAS_FFL:
                logger.warning(
                    "freq_weight=%.2f requested but focal-frequency-loss is not installed. "
                    "FFL will be skipped. Run: pip install focal-frequency-loss",
                    freq_weight,
                )
                self.ffl = None
            else:
                self.ffl = FocalFrequencyLoss(loss_weight=1.0, alpha=freq_alpha)
        else:
            self.ffl = None

    def needs_pixel_decode(self, global_step: int) -> bool:
        """Return True if any pixel-space loss is active at this step."""
        return (
                (self.norm_weight > 0 and global_step >= self.norm_start) or
                (self.freq_weight > 0 and self.ffl is not None and global_step >= self.freq_start) or
                (self.grad_weight > 0 and self.grad_loss is not None and global_step >= self.grad_start)
        )

    def forward(
            self,
            v_pred: torch.Tensor,
            v_target: torch.Tensor,
            pred_depth_pixels: torch.Tensor = None,
            gt_depth_pixels: torch.Tensor = None,
            confidence: torch.Tensor = None,
            global_step: int = 0,
    ) -> dict:
        """
        Args:
            v_pred: predicted velocity in latent space (B, C, h, w)
            v_target: target velocity (B, C, h, w)
            pred_depth_pixels: decoded predicted clean depth (B, 3, H, W)
            gt_depth_pixels: decoded GT depth (B, 3, H, W)
            confidence: optional confidence map (B, 1, H, W)
            global_step: current training step

        Returns:
            dict with "total", "velocity", "normals", "freq", "grad" losses
        """
        l_vel = self.velocity_loss(v_pred, v_target, confidence)

        loss_dict = {
            "velocity": l_vel,
            "normals": torch.tensor(0.0, device=l_vel.device),
            "freq": torch.tensor(0.0, device=l_vel.device),
            "grad": torch.tensor(0.0, device=l_vel.device),
        }

        total = self.vel_weight * l_vel
        has_pixels = pred_depth_pixels is not None and gt_depth_pixels is not None

        if self.norm_weight > 0 and global_step >= self.norm_start and has_pixels:
            l_norm = self.normals_loss(pred_depth_pixels, gt_depth_pixels)
            loss_dict["normals"] = l_norm
            total = total + self.norm_weight * l_norm

        if (
                self.freq_weight > 0
                and self.ffl is not None
                and global_step >= self.freq_start
                and has_pixels
        ):
            l_freq = self.ffl(pred_depth_pixels, gt_depth_pixels)
            loss_dict["freq"] = l_freq
            total = total + self.freq_weight * l_freq

        if (
                self.grad_weight > 0
                and self.grad_loss is not None
                and global_step >= self.grad_start
                and has_pixels
        ):
            l_grad = self.grad_loss(pred_depth_pixels, gt_depth_pixels)
            loss_dict["grad"] = l_grad
            total = total + self.grad_weight * l_grad

        loss_dict["total"] = total
        return loss_dict
