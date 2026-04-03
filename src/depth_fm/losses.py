"""
Loss functions for Mars DepthFM training.

Primary loss: Conditional Flow Matching velocity regression
    L_FM = || v_theta(z_t, t; z_img) - (z_depth - z_img) ||^2

Auxiliary loss: Surface normals consistency
    Computed from spatial gradients of predicted depth vs GT depth.
    DepthFM showed this significantly improves depth quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Per-element squared error
        sq_error = (v_pred - v_target).pow(2)

        if self.use_confidence and confidence is not None:
            # Downsample confidence to latent resolution
            if confidence.shape[-2:] != v_pred.shape[-2:]:
                confidence = F.interpolate(
                    confidence,
                    size=v_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            # Weight the loss: high confidence = full weight, low = reduced
            # Clamp to [0.1, 1.0] so no region is fully ignored
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
        # Sobel kernels for spatial gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # Register as buffers (moved to device automatically)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _compute_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute surface normals from a depth map using Sobel gradients.

        Args:
            depth: (B, 1, H, W) depth map

        Returns:
            normals: (B, 3, H, W) unit normal vectors
        """
        # Compute gradients
        dz_dx = F.conv2d(depth, self.sobel_x, padding=1)
        dz_dy = F.conv2d(depth, self.sobel_y, padding=1)

        # Normal = (-dz/dx, -dz/dy, 1), then normalize
        ones = torch.ones_like(dz_dx)
        normals = torch.cat([-dz_dx, -dz_dy, ones], dim=1)  # (B, 3, H, W)

        # L2 normalize to unit vectors
        normals = F.normalize(normals, p=2, dim=1)
        return normals

    def forward(
            self,
            pred_depth: torch.Tensor,
            gt_depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_depth: predicted depth (B, 1, H, W) or (B, 3, H, W)
            gt_depth: ground truth depth (B, 1, H, W) or (B, 3, H, W)

        Returns:
            scalar loss: 1 - mean(cosine_similarity)
        """
        # Take first channel if multi-channel
        if pred_depth.shape[1] == 3:
            pred_depth = pred_depth[:, :1]
        if gt_depth.shape[1] == 3:
            gt_depth = gt_depth[:, :1]

        pred_normals = self._compute_normals(pred_depth)
        gt_normals = self._compute_normals(gt_depth)

        # Cosine similarity per pixel
        cos_sim = (pred_normals * gt_normals).sum(dim=1, keepdim=True)
        cos_sim = cos_sim.clamp(-1.0, 1.0)

        # Loss: 1 - cosine_similarity (0 = perfect alignment, 2 = opposite)
        loss = (1.0 - cos_sim).mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined training loss for Mars DepthFM.

    L_total = w_vel * L_FM + w_norm * L_normals

    The normals loss is only activated after a warmup period to let the
    velocity field stabilize first.
    """

    def __init__(
            self,
            velocity_weight: float = 1.0,
            normals_weight: float = 0.1,
            normals_start_step: int = 2000,
            use_confidence_weighting: bool = False,
    ):
        super().__init__()
        self.velocity_loss = FlowMatchingVelocityLoss(use_confidence_weighting)
        self.normals_loss = SurfaceNormalsLoss()
        self.vel_weight = velocity_weight
        self.norm_weight = normals_weight
        self.norm_start = normals_start_step

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
        Compute combined loss.

        Args:
            v_pred: predicted velocity in latent space (B, C, h, w)
            v_target: target velocity (B, C, h, w)
            pred_depth_pixels: decoded predicted depth (B, 3, H, W) — only needed
                               if normals loss is active
            gt_depth_pixels: decoded GT depth (B, 3, H, W)
            confidence: optional confidence map (B, 1, H, W)
            global_step: current training step

        Returns:
            dict with "total", "velocity", "normals" losses
        """
        # Primary velocity loss (always active)
        l_vel = self.velocity_loss(v_pred, v_target, confidence)

        loss_dict = {
            "velocity": l_vel,
            "normals": torch.tensor(0.0, device=l_vel.device),
        }

        total = self.vel_weight * l_vel

        # Auxiliary normals loss (activated after warmup)
        if (
                self.norm_weight > 0
                and global_step >= self.norm_start
                and pred_depth_pixels is not None
                and gt_depth_pixels is not None
        ):
            l_norm = self.normals_loss(pred_depth_pixels, gt_depth_pixels)
            loss_dict["normals"] = l_norm
            total = total + self.norm_weight * l_norm

        loss_dict["total"] = total
        return loss_dict
