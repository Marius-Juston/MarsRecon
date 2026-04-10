import torch.nn as nn


class DebugUNet(nn.Module):
    """
    A tiny CNN backbone to test the Flow Matching training loop.
    Replaces the heavy CompVis UNet to isolate data/loop bugs.
    """

    def __init__(self, in_channels=8, out_channels=4):
        super().__init__()
        # 8 channels in: 4 for z_t (noise), 4 for context (clean image)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # Downsample
            nn.GroupNorm(16, 128),
            nn.SiLU(),
        )

        # Simple continuous timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, context, context_ca=None, **kwargs):
        """
        Matches the signature called by MarsDepthFM.predict_velocity.
        x: (B, 4, h, w)
        t: (B,)
        context: (B, 4, h, w)
        context_ca: (B, seq_len, dim) - Ignored in debug model
        """
        # 1. Channel concatenation (mimics what CompVis UNet does internally)
        h = torch.cat([x, context], dim=1)  # (B, 8, h, w)

        # 2. Extract spatial features
        h = self.feature_extractor(h)  # (B, 128, h/2, w/2)

        # 3. Inject continuous timestep
        t_emb = self.time_embed(t.view(-1, 1)).view(-1, 128, 1, 1)
        h = h + t_emb

        # 4. Decode to velocity prediction
        v_pred = self.decoder(h)  # (B, 4, h, w)

        return v_pred


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Maps continuous time t in [0, 1] to a high-dimensional frequency space."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # t is shape (B,), embeddings becomes (B, half_dim)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FiLMResBlock(nn.Module):
    """
    Residual Block with Feature-wise Linear Modulation (FiLM).
    The time embedding controls the scale (gamma) and shift (beta) of the features.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # MLPs to generate scale and shift from the time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

        # 1x1 conv to match channels if they change in the residual connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        # First convolution
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM Modulation: time tells the network HOW to look at the features
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        # Reshape to broadcast over spatial dimensions (B, C, 1, 1)
        h = h * (1.0 + scale[..., None, None]) + shift[..., None, None]

        # Second convolution
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class SelfAttention2D(nn.Module):
    """Global spatial self-attention to understand macroscopic terrain features."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        # Generate Queries, Keys, Values
        qkv = self.qkv(h).view(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each is (B, C, N)

        # Scaled Dot-Product Attention
        attn = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))  # (B, C, N)
        out = out.view(B, C, H, W)

        return x + self.proj(out)


class ModulatedMicroFlowNet(nn.Module):
    """
    A modernized, lightweight Flow Matching backbone.
    Replaces the heavy CompVis UNet with a FiLM-modulated, attention-augmented CNN.
    """

    def __init__(self, in_channels=8, out_channels=4, base_dim=64, time_dim=256):
        super().__init__()

        # 1. Time Processing
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # 2. Input Projection (8 channels in: 4 for z_t, 4 for image context)
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # 3. Downsample Pathway
        self.down1 = FiLMResBlock(base_dim, base_dim * 2, time_dim)
        self.pool1 = nn.Conv2d(base_dim * 2, base_dim * 2, 4, stride=2, padding=1)

        self.down2 = FiLMResBlock(base_dim * 2, base_dim * 4, time_dim)
        self.pool2 = nn.Conv2d(base_dim * 4, base_dim * 4, 4, stride=2, padding=1)

        # 4. Bottleneck (Deepest features + Global Attention)
        self.mid1 = FiLMResBlock(base_dim * 4, base_dim * 4, time_dim)
        self.attn = SelfAttention2D(base_dim * 4)
        self.mid2 = FiLMResBlock(base_dim * 4, base_dim * 4, time_dim)

        # 5. Upsample Pathway
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        # channels multiply by 2 because of skip connections (concat)
        self.up_res1 = FiLMResBlock(base_dim * 4 + base_dim * 4, base_dim * 2, time_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_res2 = FiLMResBlock(base_dim * 2 + base_dim * 2, base_dim, time_dim)

        # 6. Output Projection
        self.norm_out = nn.GroupNorm(8, base_dim)
        self.conv_out = nn.Conv2d(base_dim, out_channels, 3, padding=1)

    def forward(self, x, t, context, context_ca=None, **kwargs):
        """
        x: (B, 4, h, w) - The noisy latent z_t
        t: (B,) - Continuous timestep in [0, 1]
        context: (B, 4, h, w) - Clean orthoimage conditioning
        """
        # Embed time
        t_emb = self.time_embed(t)

        # Channel concatenate state and condition
        h = torch.cat([x, context], dim=1)  # (B, 8, h, w)

        # Encode
        h0 = self.conv_in(h)

        h1 = self.down1(h0, t_emb)
        h1_pool = self.pool1(h1)

        h2 = self.down2(h1_pool, t_emb)
        h2_pool = self.pool2(h2)

        # Bottleneck
        m = self.mid1(h2_pool, t_emb)
        m = self.attn(m)
        m = self.mid2(m, t_emb)

        # Decode with Skip Connections
        u1 = self.up1(m)
        u1 = torch.cat([u1, h2], dim=1)
        u1 = self.up_res1(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, h1], dim=1)
        u2 = self.up_res2(u2, t_emb)

        # Output Velocity Field
        out = self.conv_out(F.silu(self.norm_out(u2)))
        return out
