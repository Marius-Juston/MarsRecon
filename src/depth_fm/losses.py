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
from functools import lru_cache

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
    """State-of-the-art photoclinometric loss for planetary DTM estimation.

    Improvements over simple Lambertian:
      1. Lunar-Lambert reflectance model (McEwen 1991; Hapke 2012):
         Blends Lambertian and Lommel-Seeliger components with a learnable
         weight, correctly modeling the limb-darkening behaviour of regolith.
      2. Multi-scale rendering at 1×, 2×, 4× — enforces macro-scale
         topographic consistency alongside fine detail.
      3. Combined Pearson + z-normalised SSIM loss.
         Pearson captures global correlation; SSIM captures local
         structural similarity (Wang et al. 2004). Both components are
         applied to per-image z-score normalised inputs, making the loss
         invariant to global luminance and contrast mismatch between the
         render and the ortho. This matters here because the scene
         intensity/ambient parameters are fitted with a pure Lambertian
         OLS, while the render uses Lunar-Lambert — so a systematic
         luminance offset is expected and should not be penalised.
      4. Unclamped render. Because both loss terms are scale- and
         shift-invariant, clamping the render to [-1, 1] would only
         destroy gradient information in saturated shadow pixels — which
         is exactly where photoclinometric information is densest.
      5. Variance floor and proper epsilon handling to prevent NaN
         gradients when renders are near-flat (e.g. shadow regions).

    The loss is:
        L = (1 - α) * (1 - Pearson) + α * (1 - SSIM_z)
    averaged over scales, where α = 0.5 by default and SSIM_z is SSIM
    computed on z-score normalised inputs.

    NaN-safety:
        Every code path is hardened against NaN under DDP + gradient
        accumulation + mixed precision. Key invariants:
        - NEVER early-return with a detached zero — always flow through
          self.lunar_lambert_logit so DDP gradient sync sees every param.
        - All internal math forced to float32 to avoid bf16/fp16 overflow
          from the spatial_scale multiplication (up to 256×).
        - sqrt() always gets eps inside to prevent ∞ gradients.
        - F.normalize always gets eps > 0.
        - Lommel-Seeliger cos_e clamped to prevent denominator collapse.
    """

    def __init__(self, ssim_weight: float = 0.5, scales: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.scales = scales

        # Sobel kernels for computing surface gradients
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]]) / 8.0

        self.register_buffer("kernel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", sobel_y.view(1, 1, 3, 3))

        # Learnable Lunar-Lambert blend: sigmoid(raw) → L ∈ (0, 1)
        # L=1 → pure Lambertian, L=0 → pure Lommel-Seeliger
        # Initialised at 0.0 → sigmoid(0)=0.5 (equal blend)
        self.lunar_lambert_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def lunar_lambert_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.lunar_lambert_logit)

    def _zero_loss(self, B: int = 1, reduction: str = 'mean') -> torch.Tensor:
        """Return a zero loss that is CONNECTED to all parameters.

        Critical for DDP: if any rank skips the real loss (e.g. no valid
        pixels), it must still produce gradients for every parameter that
        other ranks are updating. A plain `torch.tensor(0.0)` would cause
        a DDP gradient sync mismatch → hang or NaN.

        The `0.0 * sigmoid(logit)` produces zero loss with a zero gradient
        for the logit, which is compatible with the real gradient on other
        ranks.
        """
        z = 0.0 * self.lunar_lambert_weight
        return z.expand(B) if reduction == 'none' else z

    def _get_surface_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """Calculates (Nx, Ny, Nz) unit normals from the depth map.

        All math in float32 to prevent bf16 overflow: spatial_scale can be
        up to 256 and Sobel gradients multiply by that, easily exceeding
        fp16 max (~65504) with large depth values.
        """
        B, C, H, W = depth.shape
        spatial_scale = max(H, W) / 2.0

        # Force float32 for numerical safety
        depth_f32 = depth.float()
        kx = self.kernel_x.float()
        ky = self.kernel_y.float()

        padded = F.pad(depth_f32, (1, 1, 1, 1), mode='replicate')
        dz_dx = F.conv2d(padded, kx) * spatial_scale
        dz_dy = F.conv2d(padded, ky) * spatial_scale

        # Clamp extreme gradients to prevent downstream overflow
        grad_clamp = 1e4
        dz_dx = dz_dx.clamp(-grad_clamp, grad_clamp)
        dz_dy = dz_dy.clamp(-grad_clamp, grad_clamp)

        n_x = -dz_dx
        n_y = -dz_dy
        n_z = torch.ones_like(n_x)
        normals = torch.cat([n_x, n_y, n_z], dim=1)
        # eps=1e-6 prevents NaN from zero-norm vectors (degenerate surfaces)
        return F.normalize(normals, p=2, dim=1, eps=1e-6)

    def _lunar_lambert_render(
            self, normals: torch.Tensor, l_dir: torch.Tensor,
            intensity: torch.Tensor, ambient: torch.Tensor,
    ) -> torch.Tensor:
        """Full Lunar-Lambert reflectance (McEwen 1991), unclamped.

        NaN-safe: cos_e is clamped to [1e-4, 1] to prevent Lommel-Seeliger
        denominator collapse. Even at extreme incidence angles where
        cos_i ≈ 0 and cos_e ≈ 0, the division stays bounded.

        No final clamp on the render: downstream Pearson and z-SSIM are
        scale- and shift-invariant, so clamping would only saturate
        gradients in exactly the shadow pixels where photoclinometric
        information is densest.
        """
        # Cosine of incidence angle
        cos_i = torch.sum(normals * l_dir, dim=1, keepdim=True)
        cos_i_clamped = torch.clamp(cos_i, min=0.0)

        # Cosine of emission angle (Z-normal for nadir view)
        # Clamp to [1e-4, 1] — cos_e near zero means the surface is edge-on
        # to the camera, which is physically degenerate for nadir imagery.
        cos_e = normals[:, 2:3, :, :].clamp(min=1e-4)

        # Lambertian and Lommel-Seeliger components
        lambert = cos_i_clamped
        # With cos_e ≥ 1e-4, denominator ≥ 1e-4, so gradient stays bounded
        lommel_seeliger = cos_i_clamped / (cos_i_clamped + cos_e + 1e-6)

        # Learnable blend
        L = self.lunar_lambert_weight
        render = (L * lambert + (1.0 - L) * lommel_seeliger) * intensity + ambient
        return render

    # ------------------------------------------------------------------
    # Public helpers — used by visualization utilities so they share the
    # exact same math as the training loss. If the render/normal math
    # ever changes, the viz updates automatically.
    # ------------------------------------------------------------------

    def surface_normals(self, depth: torch.Tensor) -> torch.Tensor:
        """Public: compute unit surface normals from a depth map."""
        return self._get_surface_normals(depth.float())

    def render_from_depth(
            self, depth: torch.Tensor, sun_vector: torch.Tensor,
            intensity: torch.Tensor, ambient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Public: render a depth map under the current Lunar-Lambert model.

        Handles input normalisation/broadcasting so callers can pass
        per-sample sun vectors and scalar exposure parameters directly.

        Returns:
            render:  (B, 1, H, W) unclamped radiance.
            normals: (B, 3, H, W) unit surface normals.
        """
        B = depth.shape[0]
        depth = depth.float()
        normals = self._get_surface_normals(depth)
        l_dir = F.normalize(sun_vector.float(), p=2, dim=1, eps=1e-6).view(B, 3, 1, 1)
        intensity_b = intensity.float().reshape(B, 1, 1, 1)
        ambient_b = ambient.float().reshape(B, 1, 1, 1)
        render = self._lunar_lambert_render(normals, l_dir, intensity_b, ambient_b)
        return render, normals

    @staticmethod
    def _zscore(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Per-image z-score normalisation within the valid mask.

        After this, values over valid pixels have mean ≈ 0 and std ≈ 1.
        This makes downstream local statistics (for SSIM) invariant to:
          - Global luminance shift (expected: Lunar-Lambert render has a
            systematic offset vs. a Lambertian-fitted intensity/ambient).
          - Global contrast scale (expected: LS component compresses
            dynamic range relative to pure Lambert).
        Local structural similarity is preserved.
        """
        B = x.shape[0]
        vc = valid_mask.view(B, -1).sum(dim=1).view(B, 1, 1, 1).clamp(min=1.0)
        mean = (x * valid_mask).view(B, -1).sum(dim=1).view(B, 1, 1, 1) / vc
        centered = (x - mean) * valid_mask
        var = (centered ** 2).view(B, -1).sum(dim=1).view(B, 1, 1, 1) / vc
        std = torch.sqrt(var + 1e-6)
        return (x - mean) / std

    def _masked_pearson(
            self, render: torch.Tensor, ortho: torch.Tensor,
            valid_mask: torch.Tensor, B: int, reduction: str = 'mean'
    ) -> torch.Tensor:
        """Pearson correlation loss, NaN-safe for DDP.

        Returns scalar loss connected to the computation graph.
        If no valid pixels exist, returns self._zero_loss() instead of
        a detached tensor.
        """
        valid_counts = valid_mask.view(B, -1).sum(dim=1).view(B, 1, 1, 1).float()
        H, W = render.shape[-2:]
        min_pixels = max(100, int(H * W * 0.05))

        # Handle scalar vs 1-element tensor from squeeze
        vc_squeezed = valid_counts.view(B)
        valid_batch = (vc_squeezed > min_pixels)

        # Removed: if not valid_batch.any(): ...

        r_masked = render * valid_mask
        o_masked = ortho * valid_mask

        r_mean = r_masked.view(B, -1).sum(dim=1).view(B, 1, 1, 1) / (valid_counts + 1e-8)
        o_mean = o_masked.view(B, -1).sum(dim=1).view(B, 1, 1, 1) / (valid_counts + 1e-8)

        r_centered = (render - r_mean) * valid_mask
        o_centered = (ortho - o_mean) * valid_mask

        cov = (r_centered * o_centered).view(B, -1).sum(dim=1)
        var_r = (r_centered ** 2).view(B, -1).sum(dim=1)
        var_o = (o_centered ** 2).view(B, -1).sum(dim=1)

        # Variance floor: skip near-flat renders / uniform ortho patches
        min_var = 1e-4
        has_variance = (var_r > min_var) & (var_o > min_var)
        final_valid = valid_batch & has_variance

        # Removed: if not final_valid.any(): ...

        # eps INSIDE sqrt to prevent ∞ gradient at var→0
        denom = torch.sqrt(var_r * var_o + 1e-8)
        correlation = cov / (denom + 1e-8)

        # Clamp correlation to [-1, 1] — floating point can exceed this
        correlation = correlation.clamp(-1.0, 1.0)

        # Calculate raw loss
        raw_loss = 1.0 - correlation

        # VECTORIZED ROUTING: If valid, use raw_loss. If invalid, use DDP-safe zero loss.
        zero_fallback = self._zero_loss(B, reduction='none')
        loss = torch.where(final_valid, raw_loss, zero_fallback)

        if reduction == 'none':
            return loss.clamp(0.0, 2.0)
        else:
            # Safely compute the mean over only the valid items
            return (loss.sum() / final_valid.float().sum().clamp(min=1.0)).clamp(0.0, 2.0)

    def _masked_ssim(
            self, render: torch.Tensor, ortho: torch.Tensor,
            valid_mask: torch.Tensor, window_size: int = 7, reduction: str = 'mean'
    ) -> torch.Tensor:
        """Differentiable SSIM loss on z-score normalised inputs, NaN-safe.

        Z-scoring the inputs makes SSIM focus on local structural
        similarity and effectively ignores global luminance / contrast
        mismatch. This is important because:
          - The render uses Lunar-Lambert while `intensity`/`ambient` are
            fitted with a pure Lambertian OLS — a luminance offset is
            expected and uninformative.
          - The learnable L blend further shifts absolute luminance each
            step; without z-scoring, SSIM would chase that rather than
            the topography.

        Constants scaled for z-scored data (effective data_range ≈ 6).
        """
        B = render.shape[0]

        # Wang 2004: C1 = (K1 * L)², C2 = (K2 * L)², with L = data range.
        # Pre-z-score L ≈ 2 (inputs in [-1, 1]); post-z-score L ≈ 6 (≈ ±3σ).
        data_range = 6.0
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        # nan_to_num before any stats: degenerate normals can produce NaN
        render_safe = torch.nan_to_num(render, nan=0.0, posinf=6.0, neginf=-6.0)
        ortho_safe = torch.nan_to_num(ortho, nan=0.0, posinf=6.0, neginf=-6.0)

        # Per-image z-score within valid mask
        render_z = self._zscore(render_safe, valid_mask)
        ortho_z = self._zscore(ortho_safe, valid_mask)

        # Fill invalid regions with 0 (= the z-score mean) to avoid
        # discontinuities at the mask boundary contaminating local stats.
        r_filled = render_z * valid_mask
        o_filled = ortho_z * valid_mask

        # Gaussian kernel for local statistics
        pad = window_size // 2
        kernel = self._gaussian_kernel(window_size, 1.5, render.device, render.dtype)

        mu_r = F.conv2d(r_filled, kernel, padding=pad)
        mu_o = F.conv2d(o_filled, kernel, padding=pad)

        mu_r_sq = mu_r ** 2
        mu_o_sq = mu_o ** 2
        mu_ro = mu_r * mu_o

        sigma_r_sq = F.conv2d(r_filled ** 2, kernel, padding=pad) - mu_r_sq
        sigma_o_sq = F.conv2d(o_filled ** 2, kernel, padding=pad) - mu_o_sq
        sigma_ro = F.conv2d(r_filled * o_filled, kernel, padding=pad) - mu_ro

        # Clamp variances for numerical stability
        sigma_r_sq = torch.clamp(sigma_r_sq, min=0.0)
        sigma_o_sq = torch.clamp(sigma_o_sq, min=0.0)

        numerator = (2 * mu_ro + C1) * (2 * sigma_ro + C2)
        denominator = (mu_r_sq + mu_o_sq + C1) * (sigma_r_sq + sigma_o_sq + C2)

        # denominator should always be > 0 due to C1, C2, but clamp to be safe
        ssim_map = numerator / (denominator + 1e-8)

        # Clamp SSIM map to [-1, 1] — prevents runaway values
        ssim_map = ssim_map.clamp(-1.0, 1.0)

        # Erode mask for border safety
        mask_eroded = F.avg_pool2d(valid_mask.float(), window_size, stride=1, padding=pad)
        mask_eroded = (mask_eroded > 0.99).float()

        # Calculate batched SSIM values
        valid_items = mask_eroded.view(B, -1).sum(dim=1) >= 10

        # Removed: if not valid_items.any(): ...

        ssim_num = (ssim_map * mask_eroded).view(B, -1).sum(dim=1)
        ssim_den = mask_eroded.view(B, -1).sum(dim=1) + 1e-8
        ssim_val = ssim_num / ssim_den

        raw_loss = 1.0 - ssim_val

        # VECTORIZED ROUTING
        zero_fallback = self._zero_loss(B, reduction='none')
        loss = torch.where(valid_items, raw_loss, zero_fallback)

        if reduction == 'none':
            return loss.clamp(0.0, 2.0)
        else:
            return (loss.sum() / valid_items.float().sum().clamp(min=1.0)).clamp(0.0, 2.0)

    @staticmethod
    @lru_cache
    def _gaussian_kernel(size: int, sigma: float, device, dtype=None) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        return kernel.view(1, 1, size, size)

    def forward(self, pred_depth: torch.Tensor, real_ortho: torch.Tensor, mask: torch.Tensor,
                sun_vectors: torch.Tensor, ambient: torch.Tensor, intensity: torch.Tensor,
                reduction: str = 'mean') -> torch.Tensor:

        if pred_depth.shape[1] == 3:
            pred_depth = pred_depth[:, :1]

        B = pred_depth.shape[0]

        if real_ortho.shape[1] == 3:
            ortho_gray = real_ortho.mean(dim=1, keepdim=True)
        else:
            ortho_gray = real_ortho

        # ---- Force float32 for entire computation ----
        # bf16/fp16 WILL overflow: spatial_scale * sobel_grad can reach
        # 256 * depth_range, easily >65504 (fp16 max).
        # Force numerical stability defensively to avoid .isfinite() control flow
        
        # 1. Detect invalid pixels BEFORE altering them
        valid_depth = torch.isfinite(pred_depth)

        # 2. Safely zero them out so math doesn't propagate NaNs
        pred_depth = torch.where(valid_depth, pred_depth, torch.zeros_like(pred_depth))

        # 3. Erode the valid depth mask by 1 pixel (because a 3x3 Sobel 
        #    kernel will contaminate 1 adjacent pixel in every direction)
        valid_depth_eroded = F.avg_pool2d(valid_depth.float(), kernel_size=3, stride=1, padding=1)
        valid_depth_eroded = (valid_depth_eroded > 0.99).float()

        # 4. Update the global mask so the loss function ignores the contaminated zones
        mask = mask.float() * valid_depth_eroded

        ortho_gray = ortho_gray.float()
        sun_vectors = sun_vectors.float()
        ambient = ambient.float()
        intensity = intensity.float()

        # eps=1e-6 prevents NaN when sun_vector is all-zeros
        l_dir = F.normalize(sun_vectors, p=2, dim=1, eps=1e-6).view(B, 3, 1, 1)
        intensity = intensity.view(B, 1, 1, 1)
        ambient = ambient.view(B, 1, 1, 1)
        valid_mask = mask

        # Check for degenerate inputs: NaN/Inf depth from VAE decoder
        # Removed: if not torch.isfinite(pred_depth).all(): ...

        total_loss = self._zero_loss(B, reduction='none')

        for s in self.scales:
            if s > 1:
                d_s = F.avg_pool2d(pred_depth, kernel_size=s, stride=s)
                o_s = F.avg_pool2d(ortho_gray, kernel_size=s, stride=s)
                m_s = F.avg_pool2d(valid_mask, kernel_size=s, stride=s)
                # Binarise: only pixels fully valid at this scale
                m_s = (m_s > 0.99).float()
            else:
                d_s, o_s = pred_depth, ortho_gray
                # Binarise at scale 1 as well, so soft confidence maps are
                # treated consistently with the s > 1 branch. Without this,
                # a confidence of 0.6 would count as valid at s=1 and
                # invalid at s=2, producing scale-dependent artifacts.
                m_s = (valid_mask > 0.99).float()

            # Skip scale if mask is nearly empty (< 5% valid)
            # VECTORIZED THRESHOLD: Do not use python `if`
            valid_ratio = m_s.view(B, -1).mean(dim=1)
            scale_is_valid = valid_ratio >= 0.05

            normals = self._get_surface_normals(d_s)
            render = self._lunar_lambert_render(normals, l_dir, intensity, ambient)

            # Pearson component (scale- and shift-invariant by construction)
            # Enforce 'none' reduction internally to maintain tensor shapes for torch.where
            pearson_loss = self._masked_pearson(render, o_s, m_s, B, reduction='none')

            # SSIM component (on z-score normalised inputs)
            ssim_loss = self._masked_ssim(render, o_s, m_s, window_size=7, reduction='none')

            alpha = self.ssim_weight
            # Combined: (1-α)*Pearson + α*SSIM
            scale_loss = (1.0 - alpha) * pearson_loss + alpha * ssim_loss

            # Apply the scale validity mask
            zero_fallback = self._zero_loss(B, reduction='none')
            scale_loss = torch.where(scale_is_valid, scale_loss, zero_fallback)

            total_loss = total_loss + scale_loss

        result = total_loss / len(self.scales)

        # Removed final .isfinite check, replacing with nan_to_num mapping
        result = torch.nan_to_num(result, nan=0.0)

        if reduction == 'mean':
            return result.mean()
        return result


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
            # Area interpolation gives a fractional-valid weight per latent
            # cell: if 3 of 4 original pixels in the receptive field are
            # valid, the weight is 0.75. This is more faithful than nearest
            # neighbor because a single valid pixel shouldn't cause the
            # whole latent cell to count as fully valid.
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

        return loss.mean()


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


class AbsoluteDepthLoss(nn.Module):
    """Direct Huber regression in [-1, 1] depth space.

    Fills the 'absolute accuracy' gap: every other pixel-space loss in the
    pipeline is scale/shift-invariant (normals, gradient, photoclinometric).
    Without a direct loss on absolute values, a model could have perfect
    slopes and renders but systematically wrong elevations.

    Huber combines:
      - L2 behaviour near convergence (|r| <= delta): smooth gradients, fast
        final-stage descent.
      - L1 behaviour in the tails (|r| > delta): robust to outliers from
        stereo GT artefacts and nodata boundaries.

    For inputs bounded in [-1, 1], delta=0.1 treats errors under ~10 %
    of the data range as 'near optimum' (L2) and larger errors as
    'outliers' (L1).
    """

    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = delta

    # -- Public helper: viz uses this directly so the plotted map is the
    # -- exact quantity the training loss averages over ----------------
    def per_pixel_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Un-reduced Huber loss map, shape (B, 1, H, W)."""
        if pred.shape[1] == 3:
            pred = pred[:, :1]
        if gt.shape[1] == 3:
            gt = gt[:, :1]
        return F.huber_loss(
            pred.float(), gt.float(), reduction="none", delta=self.delta
        )

    def forward(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            confidence: torch.Tensor = None,
    ) -> torch.Tensor:
        loss = self.per_pixel_loss(pred, gt)
        if confidence is not None:
            conf = confidence.float()
            return (loss * conf).sum() / (conf.sum() + 1e-8)
        return loss.mean()


class LaplacianLoss(nn.Module):
    """Second-order curvature consistency via discrete Laplacian.

    Captures shape information that first-order gradients miss. Critical
    for Mars because its defining landforms are curvature features:
      - Crater bowls (negative curvature)
      - Central peaks (positive curvature)
      - Volcanic calderas, rims, scarps

    A model matching all first-order gradients could still get crater
    concavity qualitatively wrong; the Laplacian forces correct
    concave/convex structure.

        L = mean_over_valid(|∇²pred - ∇²gt|)
    """

    def __init__(self):
        super().__init__()
        # Discrete 5-point Laplacian
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    # -- Public helpers -----------------------------------------------------
    def laplacian(self, depth: torch.Tensor) -> torch.Tensor:
        """Discrete Laplacian (∇²d) of a depth map, shape (B, 1, H, W)."""
        if depth.shape[1] == 3:
            depth = depth[:, :1]
        padded = F.pad(depth.float(), (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, self.kernel.float())

    def per_pixel_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Absolute Laplacian difference, shape (B, 1, H, W)."""
        return (self.laplacian(pred) - self.laplacian(gt)).abs()

    def forward(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            confidence: torch.Tensor = None,
    ) -> torch.Tensor:
        loss = self.per_pixel_loss(pred, gt)
        if confidence is not None:
            # Erode mask by 1 px since the Laplacian uses a 3x3 neighbourhood
            eroded = -F.max_pool2d(
                -confidence.float(), kernel_size=3, stride=1, padding=1
            )
            return (loss * eroded).sum() / (eroded.sum() + 1e-8)
        return loss.mean()


class OrdinalRankingLoss(nn.Module):
    """Relative-ordering consistency over sampled pixel pairs.

    For each sampled pair (i, j): if gt[i] - gt[j] > margin then pred[i]
    should exceed pred[j]. Uses a hinge loss:

        L_pair = ReLU( -sign(gt_i - gt_j) * (pred_i - pred_j) )

    averaged over pairs where |gt_i - gt_j| > margin (i.e. pairs whose GT
    ordering is unambiguous).

    Why this helps on Mars
    ----------------------
    Mars DTM ground truth from stereo reconstruction has systematic noise
    (registration, interpolation, crater-wall occlusion). Absolute values
    in noisy regions are unreliable, but pairwise ORDERING is robust — if
    crater floor is lower than rim in the GT, it is almost certainly lower
    in reality even if the absolute depths are off. Ordinal loss gives a
    supervision signal that degrades gracefully with GT noise.

    Based on: Chen et al., 'Single-Image Depth Perception in the Wild',
    NeurIPS 2016 (DIW) — adapted here to dense GT by random pair sampling.
    """

    def __init__(self, margin: float = 0.02, num_pairs: int = 10000):
        super().__init__()
        self.margin = margin
        self.num_pairs = num_pairs

    # -- Public helpers -----------------------------------------------------
    def sample_pairs(
            self,
            B: int,
            N: int,
            device: torch.device,
            generator: torch.Generator | None = None,
            confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (idx_i, idx_j) flat indices, each shape (B, num_pairs).

        Deterministic if a generator is provided — used by viz for
        reproducible figures.
        """
        if confidence is not None:
            # Flatten to (B, N) and ensure non-negative weights
            weights = torch.clamp(confidence.reshape(B, N).float(), min=0.0)

            # Add a tiny epsilon. If a batch element has completely zero confidence,
            # this prevents a multinomial crash by falling back to uniform sampling.
            # The invalid pairs will still be safely ignored in pair_stats.
            weights = weights + 1e-8

            idx = torch.multinomial(
                weights,
                self.num_pairs * 2,
                replacement=True,
                generator=generator
            )
        else:
            if generator is not None:
                idx = torch.randint(
                    N, (B, self.num_pairs * 2), device=device, generator=generator
                )
            else:
                idx = torch.randint(N, (B, self.num_pairs * 2), device=device)

        return idx[:, : self.num_pairs], idx[:, self.num_pairs:]

    def pair_stats(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            idx_i: torch.Tensor,
            idx_j: torch.Tensor,
            confidence: torch.Tensor = None,
    ) -> dict:
        """Per-pair losses, ordering labels, violation flags.

        Returns dict with:
          losses      (B, num_pairs) — hinge loss, 0 for ambiguous pairs
          ordered     (B, num_pairs) bool — |gt_diff| > margin and valid
          violations  (B, num_pairs) bool — ordered and pred disagrees
          gt_diff     (B, num_pairs) — gt_i - gt_j
          pr_diff     (B, num_pairs) — pred_i - pred_j
        """
        if pred.shape[1] == 3:
            pred = pred[:, :1]
        if gt.shape[1] == 3:
            gt = gt[:, :1]

        B = pred.shape[0]
        pred_flat = pred.reshape(B, -1).float()
        gt_flat = gt.reshape(B, -1).float()

        gt_i = gt_flat.gather(1, idx_i)
        gt_j = gt_flat.gather(1, idx_j)
        pr_i = pred_flat.gather(1, idx_i)
        pr_j = pred_flat.gather(1, idx_j)

        gt_diff = gt_i - gt_j
        pr_diff = pr_i - pr_j
        sign_gt = torch.sign(gt_diff)

        ordered = gt_diff.abs() > self.margin

        # We keep this check even with multinomial sampling to catch
        # the 1e-8 epsilon fallback edge-case where a batch is totally unconfident.
        if confidence is not None:
            conf_flat = confidence.reshape(B, -1).float()
            c_i = conf_flat.gather(1, idx_i)
            c_j = conf_flat.gather(1, idx_j)
            ordered = ordered & (c_i > 0.5) & (c_j > 0.5)

        losses = F.relu(-sign_gt * pr_diff) * ordered.float()
        # 'violation' = ordered AND predicted difference has wrong sign
        violations = ordered & (sign_gt * pr_diff <= 0)

        return {
            "losses": losses,
            "ordered": ordered,
            "violations": violations,
            "gt_diff": gt_diff,
            "pr_diff": pr_diff,
        }

    def forward(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            confidence: torch.Tensor = None,
    ) -> torch.Tensor:
        B = pred.shape[0]
        N = pred.shape[-2] * pred.shape[-1]

        # Pass confidence explicitly to drive the sampling distribution
        idx_i, idx_j = self.sample_pairs(B, N, pred.device, confidence=confidence)
        stats = self.pair_stats(pred, gt, idx_i, idx_j, confidence)

        n_ordered = stats["ordered"].float().sum()
        if n_ordered < 1:
            # Param-connected zero so DDP gradient sync stays consistent
            return 0.0 * pred.float().mean()
        return stats["losses"].sum() / (n_ordered + 1e-8)


class CombinedLoss(nn.Module):
    """Combined training loss for Mars DepthFM.

    L_total = w_vel    * L_FM
            + w_norm   * L_normals     (after normals_start_step)
            + w_freq   * L_FFL         (after freq_start_step)
            + w_grad   * L_grad        (after grad_start_step)
            + w_photo  * L_photo       (after photo_start_step)
            + w_huber  * L_Huber       (after huber_start_step)     [NEW]
            + w_lap    * L_Laplacian   (after laplacian_start_step) [NEW]
            + w_ord    * L_Ordinal     (after ordinal_start_step)   [NEW]

    Pixel-space losses (normals, FFL, grad, photo, huber, laplacian,
    ordinal) require the caller to pass ``pred_depth_pixels`` and
    ``gt_depth_pixels`` decoded WITHOUT no_grad so that gradients flow
    back through the decoder to v_pred.
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
            photo_start_step: int = 2000,
            huber_weight: float = 3.0,
            huber_start_step: int = 0,
            huber_delta: float = 0.1,
            laplacian_weight: float = 0.05,
            laplacian_start_step: int = 0,
            ordinal_weight: float = 1.0,
            ordinal_start_step: int = 0,
            ordinal_margin: float = 0.02,
            ordinal_num_pairs: int = 10000,
    ):
        super().__init__()
        # ---- Photo / velocity / existing aux ----------------------------
        self.photo_start_step = photo_start_step
        self.photo_weight = photo_weight
        self.photo_loss = PhotoclinometricLoss(
            ssim_weight=0.5,
            scales=(1, 2, 4),
        )
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

        # ---- New losses --------------------------------------------------
        self.huber_weight = huber_weight
        self.huber_start = huber_start_step
        self.huber_loss = AbsoluteDepthLoss(delta=huber_delta) if huber_weight > 0 else None

        self.laplacian_weight = laplacian_weight
        self.laplacian_start = laplacian_start_step
        self.laplacian_loss = LaplacianLoss() if laplacian_weight > 0 else None

        self.ordinal_weight = ordinal_weight
        self.ordinal_start = ordinal_start_step
        self.ordinal_loss = (
            OrdinalRankingLoss(margin=ordinal_margin, num_pairs=ordinal_num_pairs)
            if ordinal_weight > 0 else None
        )

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
                (self.photo_weight > 0 and self.photo_loss is not None and global_step >= self.photo_start_step) or
                (self.huber_weight > 0 and self.huber_loss is not None and global_step >= self.huber_start) or
                (
                        self.laplacian_weight > 0 and self.laplacian_loss is not None and global_step >= self.laplacian_start) or
                (self.ordinal_weight > 0 and self.ordinal_loss is not None and global_step >= self.ordinal_start)
        )

    def forward(
            self,
            v_pred: torch.Tensor,
            v_target: torch.Tensor,
            pred_depth_pixels: torch.Tensor = None,
            gt_depth_pixels: torch.Tensor = None,
            pred_depth_physical: torch.Tensor = None,
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
            gt_depth_pixels:   decoded GT depth (B, 3, H, W)
            confidence: optional confidence map (B, 1, H, W)
            global_step: current training step

        Returns:
            dict with component losses and "total"
        """
        active_conf = confidence if self.use_confidence_weighting else None

        l_vel = self.velocity_loss(v_pred, v_target, active_conf)

        loss_dict = {
            "velocity": l_vel,
            "normals": torch.tensor(0.0, device=l_vel.device),
            "freq": torch.tensor(0.0, device=l_vel.device),
            "grad": torch.tensor(0.0, device=l_vel.device),
            "photo": torch.tensor(0.0, device=v_pred.device),
            "huber": torch.tensor(0.0, device=v_pred.device),
            "laplacian": torch.tensor(0.0, device=v_pred.device),
            "ordinal": torch.tensor(0.0, device=v_pred.device),
        }

        total = self.vel_weight * l_vel
        has_pixels = pred_depth_pixels is not None and gt_depth_pixels is not None

        # ---- Existing pixel-space losses --------------------------------
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
                sun_vector = torch.tensor([[0.5, -0.5, 1.0]], device=pred_depth_pixels.device)
                sun_vector = sun_vector.expand(pred_depth_pixels.shape[0], -1)

            B_photo = pred_depth_pixels.shape[0]
            if ambient is None:
                ambient = torch.zeros(B_photo, 1, device=pred_depth_pixels.device)
            if intensity is None:
                intensity = torch.ones(B_photo, 1, device=pred_depth_pixels.device)

            pred_for_photo = pred_depth_physical if pred_depth_physical is not None else pred_depth_pixels
            l_photo = self.photo_loss(
                pred_depth=pred_for_photo,
                real_ortho=real_ortho,
                # Use active_conf (gated by use_confidence_weighting) for
                # consistency with all other aux losses. If the switch is
                # off, fall back to all-ones so photo loss still runs.
                mask=active_conf if active_conf is not None
                else torch.ones_like(pred_depth_pixels[:, :1]),
                sun_vectors=sun_vector,
                ambient=ambient,
                intensity=intensity,
            )

            loss_dict["photo"] = l_photo
            total = total + self.photo_weight * l_photo

        # ---- New pixel-space losses -------------------------------------
        if (
                self.huber_weight > 0
                and self.huber_loss is not None
                and global_step >= self.huber_start
                and has_pixels
        ):
            l_huber = self.huber_loss(pred_depth_pixels, gt_depth_pixels, active_conf)
            loss_dict["huber"] = l_huber
            total = total + self.huber_weight * l_huber

        if (
                self.laplacian_weight > 0
                and self.laplacian_loss is not None
                and global_step >= self.laplacian_start
                and has_pixels
        ):
            l_lap = self.laplacian_loss(pred_depth_pixels, gt_depth_pixels, active_conf)
            loss_dict["laplacian"] = l_lap
            total = total + self.laplacian_weight * l_lap

        if (
                self.ordinal_weight > 0
                and self.ordinal_loss is not None
                and global_step >= self.ordinal_start
                and has_pixels
        ):
            l_ord = self.ordinal_loss(pred_depth_pixels, gt_depth_pixels, active_conf)
            loss_dict["ordinal"] = l_ord
            total = total + self.ordinal_weight * l_ord

        loss_dict["total"] = total

        # ---- Final NaN guard --------------------------------------------
        if not torch.isfinite(total):
            logger.warning(
                "CombinedLoss: non-finite total detected at step %d. "
                "Components: vel=%.4f norm=%.4f freq=%.4f grad=%.4f photo=%.4f "
                "huber=%.4f lap=%.4f ord=%.4f. "
                "Falling back to velocity-only loss.",
                global_step,
                loss_dict["velocity"].item() if torch.is_tensor(loss_dict["velocity"]) else 0,
                loss_dict["normals"].item() if torch.is_tensor(loss_dict["normals"]) else 0,
                loss_dict["freq"].item() if torch.is_tensor(loss_dict["freq"]) else 0,
                loss_dict["grad"].item() if torch.is_tensor(loss_dict["grad"]) else 0,
                loss_dict["photo"].item() if torch.is_tensor(loss_dict["photo"]) else 0,
                loss_dict["huber"].item() if torch.is_tensor(loss_dict["huber"]) else 0,
                loss_dict["laplacian"].item() if torch.is_tensor(loss_dict["laplacian"]) else 0,
                loss_dict["ordinal"].item() if torch.is_tensor(loss_dict["ordinal"]) else 0,
            )
            total = self.vel_weight * l_vel
            if not torch.isfinite(total):
                total = torch.tensor(0.0, device=l_vel.device, requires_grad=True)
            loss_dict["total"] = total

        # Expose learned photometric parameter for logging
        if self.photo_loss is not None:
            loss_dict["lunar_lambert_weight"] = self.photo_loss.lunar_lambert_weight.detach()

        return loss_dict
