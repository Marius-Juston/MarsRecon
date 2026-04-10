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

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotoclinometricLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]) / 8.0

        self.register_buffer("kernel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", sobel_y.view(1, 1, 3, 3))

    def _get_surface_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """Calculates (Nx, Ny, Nz) unit normals from the depth map."""
        B, C, H, W = depth.shape

        # Spatial scaling factor
        spatial_scale = max(H, W) / 2.0

        padded = F.pad(depth, (1, 1, 1, 1), mode='replicate')

        dz_dx = F.conv2d(padded, self.kernel_x) * spatial_scale
        dz_dy = F.conv2d(padded, self.kernel_y) * spatial_scale

        n_x = -dz_dx
        n_y = -dz_dy
        n_z = torch.ones_like(n_x)

        normals = torch.cat([n_x, n_y, n_z], dim=1)
        return F.normalize(normals, p=2, dim=1)

    def forward(self, pred_depth: torch.Tensor, real_ortho: torch.Tensor, mask: torch.Tensor,
                sun_vectors: torch.Tensor, ambient: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:

        if pred_depth.shape[1] == 3:
            pred_depth = pred_depth[:, :1]

        B = pred_depth.shape[0]

        if real_ortho.shape[1] == 3:
            ortho_gray = real_ortho.mean(dim=1, keepdim=True)
        else:
            ortho_gray = real_ortho

        # 1. Calculate Normals
        normals = self._get_surface_normals(pred_depth)

        # 2. Reshape lighting parameters for spatial broadcasting
        l_dir = F.normalize(sun_vectors, p=2, dim=1).view(B, 3, 1, 1)
        intensity = intensity.view(B, 1, 1, 1)
        ambient = ambient.view(B, 1, 1, 1)

        # 3. Lambertian Render with Real-World Lighting
        render = torch.sum(normals * l_dir, dim=1, keepdim=True)
        render = (render * intensity) + ambient

        # Clamp to mimic actual camera sensor bounds (-1 to 1 based on your dataloader)
        render = torch.clamp(render, min=-1.0, max=1.0)

        # 4. Masked Pearson Correlation (Computed Per-Image in Batch)
        valid_mask = mask.bool()

        # Get valid pixel counts per image. Shape: (B, 1, 1, 1)
        valid_counts = valid_mask.view(B, -1).sum(dim=1).view(B, 1, 1, 1)

        # Identify which batch items have enough valid pixels to compute correlation
        valid_batch_items = (valid_counts.squeeze() > 100)

        if not valid_batch_items.any():
            return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)

        # Zero out invalid pixels so they don't affect the sum
        r_masked = render * valid_mask
        o_masked = ortho_gray * valid_mask

        # Compute means per-image
        r_mean = r_masked.view(B, -1).sum(dim=1).view(B, 1, 1, 1) / valid_counts
        o_mean = o_masked.view(B, -1).sum(dim=1).view(B, 1, 1, 1) / valid_counts

        # Center the variables (only for valid pixels)
        r_centered = (render - r_mean) * valid_mask
        o_centered = (ortho_gray - o_mean) * valid_mask

        # Covariance and Variance per-image. Shape: (B,)
        cov = (r_centered * o_centered).view(B, -1).sum(dim=1)
        var_r = (r_centered ** 2).view(B, -1).sum(dim=1)
        var_o = (o_centered ** 2).view(B, -1).sum(dim=1)

        # Calculate Pearson per-image
        denominator = torch.sqrt(var_r * var_o)
        correlation = cov / (denominator + 1e-8)  # Epsilon prevents divide-by-zero

        # Average the loss only over valid batch items
        loss = 1.0 - correlation[valid_batch_items].mean()

        return loss


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
            # Nearest neighbor interpolation ensures binary masks remain sharp
            mask = F.interpolate(
                confidence.float(),
                size=v_pred.shape[-2:],
                mode="area",
            )
            sq_error = sq_error * mask

            # Normalize strictly by the active area to maintain gradient scale
            active_elements = mask.sum() * v_pred.shape[1]
            return sq_error.sum() / (active_elements + 1e-8)

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
        # Extract dynamic image dimensions
        B, C, H, W = depth.shape

        # Calculate the exact norm to map pixel-space Sobel to [-1, 1] physical space
        spatial_scale = max(H, W) / 2.0

        # Pad by replicating the edge pixels so gradients are 0 at the boundary
        padded_depth = F.pad(depth, (1, 1, 1, 1), mode='replicate')
        dz_dx = F.conv2d(padded_depth, self.sobel_x, padding=0) * spatial_scale
        dz_dy = F.conv2d(padded_depth, self.sobel_y, padding=0) * spatial_scale
        ones = torch.ones_like(dz_dx)
        normals = torch.cat([-dz_dx, -dz_dy, ones], dim=1)
        return F.normalize(normals, p=2, dim=1)

    def forward(
            self,
            pred_depth: torch.Tensor,
            gt_depth: torch.Tensor,
            confidence: torch.Tensor = None,
    ) -> torch.Tensor:
        if pred_depth.shape[1] == 3:
            pred_depth = pred_depth[:, :1]
        if gt_depth.shape[1] == 3:
            gt_depth = gt_depth[:, :1]

        pred_normals = self._compute_normals(pred_depth)
        gt_normals = self._compute_normals(gt_depth)

        cos_sim = (pred_normals * gt_normals).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        loss = 1.0 - cos_sim

        if confidence is not None:
            # Erode the mask by 1 pixel (radius of 3x3 Sobel) to prevent
            # artificial gradients at the boundary of nodata regions.
            # Min pooling achieved via inverted max pooling.
            eroded_mask = -F.max_pool2d(-confidence.float(), kernel_size=3, stride=1, padding=1)

            loss = loss * eroded_mask
            return loss.sum() / (eroded_mask.sum() + 1e-8)

        return loss.mean().clamp(min=1e-6)


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
            confidence: torch.Tensor = None,
    ) -> torch.Tensor:
        if pred_depth.shape[1] == 3:
            pred_depth = pred_depth[:, :1]
        if gt_depth.shape[1] == 3:
            gt_depth = gt_depth[:, :1]

        total_loss = torch.tensor(0.0, device=pred_depth.device)

        for s in self.scales:
            if s > 1:
                p = F.avg_pool2d(pred_depth, kernel_size=s, stride=s)
                g = F.avg_pool2d(gt_depth, kernel_size=s, stride=s)
                m = F.avg_pool2d(confidence.float(), kernel_size=s, stride=s) if confidence is not None else None
            else:
                p, g, m = pred_depth, gt_depth, (confidence.float() if confidence is not None else None)

            dy_p, dx_p = self._gradients(p)
            dy_g, dx_g = self._gradients(g)

            err_y = (dy_p - dy_g).abs() / s
            err_x = (dx_p - dx_g).abs() / s

            if m is not None:
                # Gradient is only valid if both pixels involved in the difference are valid
                m_dy = m[:, :, 1:, :] * m[:, :, :-1, :]
                m_dx = m[:, :, :, 1:] * m[:, :, :, :-1]

                loss_y = (err_y * m_dy).sum() / (m_dy.sum() + 1e-8)
                loss_x = (err_x * m_dx).sum() / (m_dx.sum() + 1e-8)
                total_loss = total_loss + loss_y + loss_x
            else:
                total_loss = total_loss + err_y.mean() + err_x.mean()

        return total_loss / len(self.scales)


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

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            confidence: torch.Tensor = None
    ) -> torch.Tensor:

        if confidence is not None:
            # GT Injection: Replace predicted masked regions with GT to ensure
            # zero discrepancy in the Fourier domain for invalid regions, preventing
            # the Gibbs phenomenon (ringing artifacts) at mask boundaries.
            pred = pred * confidence + target * (1.0 - confidence)

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
            photo_weight: float = 0.0,
            photo_start_step: int = 2000
    ):
        super().__init__()
        self.photo_start_step = photo_start_step
        self.photo_weight = photo_weight
        self.photo_loss = PhotoclinometricLoss()
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
        self.use_confidence_weighting = use_confidence_weighting

        if self.use_confidence_weighting:
            logging.info("Using confidence weighting to improve nodata region filtering")

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
                (self.grad_weight > 0 and self.grad_loss is not None and global_step >= self.grad_start) or
                (self.photo_weight > 0 and self.photo_loss is not None and global_step >= self.photo_start_step)
        )

    def forward(
            self,
            v_pred: torch.Tensor,
            v_target: torch.Tensor,
            pred_depth_pixels: torch.Tensor = None,
            gt_depth_pixels: torch.Tensor = None,
            confidence: torch.Tensor = None,
            real_ortho: torch.Tensor | None = None,
            sun_vector: torch.Tensor | None = None,
            global_step: int = 0,
            ambient: torch.Tensor = None,
            intensity: torch.Tensor = None,
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
        active_conf = confidence if self.use_confidence_weighting else None

        l_vel = self.velocity_loss(v_pred, v_target, active_conf)

        loss_dict = {
            "velocity": l_vel,
            "normals": torch.tensor(0.0, device=l_vel.device),
            "freq": torch.tensor(0.0, device=l_vel.device),
            "grad": torch.tensor(0.0, device=l_vel.device),
            "photo": torch.tensor(0.0, device=v_pred.device)
        }

        total = self.vel_weight * l_vel
        has_pixels = pred_depth_pixels is not None and gt_depth_pixels is not None

        if self.norm_weight > 0 and global_step >= self.norm_start and has_pixels:
            l_norm = self.normals_loss(pred_depth_pixels, gt_depth_pixels, active_conf)
            loss_dict["normals"] = l_norm
            total = total + self.norm_weight * l_norm

        if (
                self.freq_weight > 0
                and self.ffl is not None
                and global_step >= self.freq_start
                and has_pixels
        ):
            l_freq = self.ffl(pred_depth_pixels, gt_depth_pixels, active_conf)
            loss_dict["freq"] = l_freq
            total = total + self.freq_weight * l_freq

        if (
                self.grad_weight > 0
                and self.grad_loss is not None
                and global_step >= self.grad_start
                and has_pixels
        ):
            l_grad = self.grad_loss(pred_depth_pixels, gt_depth_pixels, active_conf)
            loss_dict["grad"] = l_grad
            total = total + self.grad_weight * l_grad

        if (
                self.photo_weight > 0
                and self.photo_loss is not None
                and global_step >= self.photo_start_step
                and has_pixels
        ):
            if sun_vector is None:
                # Default to pointing diagonally down
                sun_vector = torch.tensor([[0.5, -0.5, 1.0]], device=pred_depth_pixels.device)
                sun_vector = sun_vector.expand(pred_depth_pixels.shape[0], -1)

            l_photo = self.photo_loss(
                pred_depth=pred_depth_pixels,
                real_ortho=real_ortho,
                mask=confidence if confidence is not None else torch.ones_like(pred_depth_pixels[:, :1]),
                sun_vectors=sun_vector,
                ambient=ambient,
                intensity=intensity,
            )

            loss_dict["photo"] = l_photo
            total = total + self.photo_weight * l_photo

        loss_dict["total"] = total
        return loss_dict
