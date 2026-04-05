# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import math
from typing import Tuple, Iterable, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from einops import repeat


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from a 1D tensor 'a' using indices 't', and reshape them
    to broadcast across 'x_shape'.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def checkpoint(func: callable, inputs: Iterable[torch.Tensor], params: Iterable[torch.Tensor], flag: bool) -> Any:
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    Note: 'params' is kept in the signature for backwards compatibility with
    older calling code, but PyTorch 2.x handles parameter gradients natively.
    """
    if flag:
        # use_reentrant=False safely bridges gradient checkpointing with torch.compile()
        return cp.checkpoint(func, *inputs, use_reentrant=False)

    return func(*inputs)


def timestep_embedding(
        timesteps: torch.Tensor,
        dim: int,
        max_period: int = 10000,
        repeat_only: bool = False
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if repeat_only:
        return repeat(timesteps, 'b -> b d', d=dim)

    half = dim // 2

    # Create frequencies directly on the target device to avoid memory transfers
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )

    args = timesteps[:, None].to(torch.float32) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    """
    Scale the parameters of a module and return it.
    """
    with torch.no_grad():
        for p in module.parameters():
            p.mul_(scale)
    return module


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Take the mean over all non-batch dimensions.
    """
    # flatten(1) squashes all dimensions after the batch dimension (dim 0) into a single dimension.
    return tensor.flatten(start_dim=1).mean(dim=1)


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to standard float32 for stable internal variance calculation,
        # then return to the original mixed-precision dtype. Clone prevents CUDAGraph aliases.
        return super().forward(x.to(torch.float32)).to(dtype=x.dtype).clone()


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
