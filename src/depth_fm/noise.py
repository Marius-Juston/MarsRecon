"""
DepthFM noise utilities — shared between training and inference.

Copied verbatim from depth-fm/depthfm/dfm.py so that both codepaths
are provably identical to the reference implementation.
"""

import einops
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Cosine noise schedule (from dfm.py)
# ---------------------------------------------------------------------------

def cosine_log_snr(t: float, eps: float = 1e-5) -> float:
    return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def cosine_alpha_bar(t: float) -> float:
    """ᾱ_t under the cosine schedule."""
    return 1 / (1 + np.exp(-cosine_log_snr(t)))


@torch.no_grad()
def q_sample(
        x_start: torch.Tensor,
        t: int,
        noise: torch.Tensor = None,
        n_diffusion_timesteps: int = 1000,
) -> torch.Tensor:
    """
    Diffuse x_start for t steps: sample from q(x_t | x_0).

    Matches dfm.py exactly:
        alpha_bar_t = cosine_alpha_bar(t / n_diffusion_timesteps)
        return sqrt(alpha_bar_t)*x_start + sqrt(1-alpha_bar_t)*noise

    Args:
        x_start: latent to noise, shape (B, C, H, W).
        t: integer diffusion step (e.g. model.noising_step = 400).
        noise: optional pre-drawn noise; drawn fresh if None.
        n_diffusion_timesteps: total diffusion steps (default 1000).

    Returns:
        Noised latent, same shape as x_start.
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    alpha_bar_t = cosine_alpha_bar(t / n_diffusion_timesteps)
    alpha_bar_t = torch.tensor(alpha_bar_t, device=x_start.device, dtype=x_start.dtype)
    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise


# ---------------------------------------------------------------------------
# Depth post-processing (from dfm.py)
# ---------------------------------------------------------------------------

def per_sample_min_max_normalization(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each sample in a batch independently to [0, 1].

    Matches dfm.py exactly.
    """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)
