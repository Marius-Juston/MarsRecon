import math
from typing import Optional, Union, List, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .util import checkpoint


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Union[Callable, Any]) -> Any:
    if exists(val):
        return val
    return d() if callable(d) else d


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module


def Normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        project_in = (
            GEGLU(dim, inner_dim) if glu
            else nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # Reshape to (Batch, Heads=1, SeqLen, Dim) for PyTorch Native SDPA
        q = rearrange(q, "b c h w -> b 1 (h w) c")
        k = rearrange(k, "b c h w -> b 1 (h w) c")
        v = rearrange(v, "b c h w -> b 1 (h w) c")

        # Native PyTorch SDPA handles memory-efficient attention and the 1/sqrt(c) scaling automatically
        out = F.scaled_dot_product_attention(q, k, v)

        # Reshape back to image format
        out = rearrange(out, "b 1 (h w) c -> b c h w", h=h)
        return x + self.proj_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.dim_head = dim_head
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                rescale_attention: bool = True) -> torch.Tensor:
        is_self_attention = context is None
        n_tokens = x.shape[1]

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Use (b h) n d layout to match original exactly (same SDPA kernel path)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), (q, k, v))

        scale = (math.log(n_tokens) / math.log(n_tokens * 4) / self.dim_head) ** 0.5 \
            if (rescale_attention and is_self_attention) else None

        out = F.scaled_dot_product_attention(q, k, v, scale=scale)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            n_heads: int,
            d_head: int,
            dropout: float = 0.0,
            context_dim: Optional[int] = None,
            gated_ff: bool = True,
            checkpoint: bool = True,
            disable_self_attn: bool = False,
    ):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.checkpoint = checkpoint

        # Uses standard CrossAttention which is now backed by native PyTorch SDPA
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, projects the input (aka embedding) and reshapes to b, t, d.
    Then applies standard transformer blocks.
    Finally, reshapes back to image dimensions.
    """
    _is_spatial_transformer = True  # tag for compile-friendly dispatch

    def __init__(
            self,
            in_channels: int,
            n_heads: int,
            d_head: int,
            depth: int = 1,
            dropout: float = 0.0,
            context_dim: Optional[Union[int, List[int]]] = None,
            disable_self_attn: bool = False,
            use_linear: bool = False,
            use_checkpoint: bool = True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]

        self.in_channels = in_channels
        self.use_linear = use_linear
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim[d] if context_dim is not None else None,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
            )
            for d in range(depth)
        ])

        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))

    def forward(self, x: torch.Tensor,
                context: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> torch.Tensor:
        if context is not None and not isinstance(context, list):
            context = [context]

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        if not self.use_linear:
            x = self.proj_in(x)

        x = rearrange(x, "b c h w -> b (h w) c").contiguous()

        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            ctx = context[i] if context is not None else None
            x = block(x, context=ctx)

        if self.use_linear:
            x = self.proj_out(x)

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        if not self.use_linear:
            x = self.proj_out(x)

        return x + x_in