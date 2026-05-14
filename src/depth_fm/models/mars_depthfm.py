"""
DepthFM model wrapper using the actual CompVis LDM UNetModel architecture.

Loads from the official DepthFM checkpoint format:
    {noising_step, ldm_hparams, empty_text_embedding, state_dict}

The UNet has in_channels=8 because image conditioning is concatenated INSIDE
the forward pass (context → channel concat before input_blocks). The output is
always 4 channels (velocity field).

VAE is SD 1.5 with hardcoded scale_factor=0.18215.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

from depth_fm.models.experimental import DebugUNet, ModulatedMicroFlowNet

logger = logging.getLogger(__name__)

# SD 1.5 VAE scale factor — hardcoded, matches the DepthFM paper
_SCALE_FACTOR = 0.18215


def load_sd21_backend(depthfm_checkpoint: str, vae_id: str, device: str = "cpu", use_checkpoint=True) -> Tuple[
    nn.Module, AutoencoderKL, int, torch.Tensor]:
    """
    Load the actual DepthFM model from the official checkpoint.

    Checkpoint format:
        noising_step          — int, e.g. 200
        ldm_hparams           — dict of UNetModel constructor kwargs
        empty_text_embedding  — (1, seq_len, context_dim) null text conditioning
        state_dict            — UNetModel weights (keys match CompVis naming)

    Returns:
        unet: trainable UNetModel
        vae: frozen SD 1.5 AutoencoderKL
        noising_step: int
        empty_text_embed: (1, seq_len, context_dim) tensor
    """
    from depth_fm.models.unet import UNetModel

    ckpt = torch.load(depthfm_checkpoint, map_location=device, weights_only=False)

    noising_step = ckpt["noising_step"]
    empty_text_embed = torch.from_numpy(ckpt["empty_text_embedding"]).to(device=device)  # (1, seq_len, 1024)
    ldm_hparams = dict(ckpt["ldm_hparams"])  # copy to avoid mutation
    ldm_hparams["use_checkpoint"] = use_checkpoint

    unet = UNetModel(**ldm_hparams)
    missing, unexpected = unet.load_state_dict(ckpt["state_dict"], strict=True)
    if missing:
        logger.warning("Missing keys loading DepthFM UNet: %d", len(missing))
        for k in missing[:5]:
            logger.warning("  %s", k)
    if unexpected:
        logger.warning("Unexpected keys loading DepthFM UNet: %d", len(unexpected))
        for k in unexpected[:5]:
            logger.warning("  %s", k)

    # SD 1.5 VAE — frozen encoder/decoder
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(vae_id, subfolder="vae").to(device=device)
    vae.eval()
    vae.requires_grad_(False)

    n_params = sum(p.numel() for p in unet.parameters()) / 1e6
    logger.info(
        "Loaded DepthFM: UNet=%.1fM params, noising_step=%d, "
        "empty_text_embed shape=%s",
        n_params, noising_step, tuple(empty_text_embed.shape),
    )

    return unet, vae, noising_step, empty_text_embed


class MarsDepthFM(nn.Module):
    """
    Wrapper for DepthFM flow matching depth estimation.

    Exposes a clean interface for the training loop:
        encode_to_latent   — pixel → latent (frozen VAE, no grad)
        decode_from_latent — latent → pixel (frozen VAE, grad allowed for losses)
        predict_velocity   — core UNet forward
    """

    def __init__(
            self,
            backbone: nn.Module,
            vae: AutoencoderKL,
            noising_step: int,
            empty_text_embed: torch.Tensor,
            freeze_encoder: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.vae = vae
        self.noising_step = noising_step
        # Register as buffer so it moves with .to(device) and gets saved
        self.register_buffer("empty_text_embed", empty_text_embed)

        self.scale_factor = _SCALE_FACTOR
        self.latent_channels = 4  # SD 1.5 VAE always outputs 4-channel latents

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze the encoder (input_blocks) of the UNet, train only decoder."""
        for name, param in self.backbone.named_parameters():
            if "input_blocks" in name or "middle_block" in name:
                param.requires_grad_(False)

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        logger.info("Encoder frozen: %.1fM / %.1fM trainable", trainable / 1e6, total / 1e6)

    @torch.no_grad()
    def encode_to_latent(self, pixel_images: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel-space images (B, 3, H, W) ∈ [-1, 1] to latent space.
        Returns: (B, 4, h, w) scaled latents.
        """
        posterior = self.vae.encode(pixel_images).latent_dist
        latent = posterior.mode()
        return latent * self.scale_factor

    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent (B, 4, h, w) to pixel space (B, 3, H, W) ∈ [-1, 1].

        Called WITHOUT torch.no_grad() when used for pixel-space training losses
        so gradients flow back through the frozen VAE decoder to v_pred.
        """
        latent_unscaled = latent / self.scale_factor
        return self.vae.decode(latent_unscaled).sample

    # ------------------------------------------------------------------
    # Inference API — mirrors DepthFM.forward() / predict_depth() exactly
    # ------------------------------------------------------------------

    def forward(
            self,
            ims: torch.Tensor,
            num_steps: int = 2,
            ensemble_size: int = 4,
    ) -> torch.Tensor:
        """
        Full inference forward — mirrors DepthFM.forward() line-for-line.

        Args:
            ims: (1, 3, H, W) in [-1, 1]
            num_steps: Euler ODE steps
            ensemble_size: repeat image in batch; average depth at end

        Returns:
            depth: (1, 1, H, W) in [0, 1]
        """
        from depth_fm.flow.noise import q_sample, per_sample_min_max_normalization

        if ensemble_size > 1:
            ims = ims.repeat(ensemble_size, 1, 1, 1)

        bs = ims.shape[0]
        device = ims.device

        # Encode image latent (mode, no sampling) — mirrors DepthFM.encode(sample_posterior=False)
        ims_z = self.encode_to_latent(ims)

        # Null text conditioning (cross-attention) — matches dfm.py repeat semantics
        conditioning = self.empty_text_embed.to(device=device, dtype=ims_z.dtype).expand(bs, -1, -1)

        # Clean image latent for channel-concat conditioning
        context = ims_z

        # Noise-augment the ODE starting point — matches dfm.py exactly:
        #   if self.noising_step > 0: x_source = q_sample(x_source, self.noising_step)
        x_source = ims_z.clone()
        if self.noising_step > 0:
            x_source = q_sample(x_source, self.noising_step)

        # Euler ODE: x_source → depth_z  (equivalent to torchdiffeq Euler in dfm.py)
        z_t = x_source
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((bs,), t_val, device=device, dtype=z_t.dtype)
            # UNet forward: channel-concat with context (clean image), cross-attn with conditioning
            v = self.backbone(x=z_t, t=t, context=context, context_ca=conditioning)
            z_t = z_t + dt * v

        depth_z = z_t

        # Decode + channel mean — matches dfm.py: depth.mean(dim=1, keepdim=True)
        depth = self.decode_from_latent(depth_z)  # (E, 3, H, W)
        depth = depth.mean(dim=1, keepdim=True)  # (E, 1, H, W)

        if ensemble_size > 1:
            depth = depth.mean(dim=0, keepdim=True)  # (1, 1, H, W)

        # exp() then per-sample min-max normalize → [0, 1]  — matches dfm.py
        depth = per_sample_min_max_normalization(depth.exp())

        return depth

    @torch.no_grad()
    def predict_depth(
            self,
            ims: torch.Tensor,
            num_steps: int = 2,
            ensemble_size: int = 4,
    ) -> torch.Tensor:
        """
        Public inference API — mirrors DepthFM.predict_depth() exactly.

        Returns:
            depth: (1, 1, H, W) in [0, 1]
        """
        return self.forward(ims, num_steps, ensemble_size)

    def predict_velocity(
            self,
            z_t: torch.Tensor,
            t: torch.Tensor,
            z_img_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity field v_θ(z_t, t; z_img).

        The UNet concatenates z_img_cond onto z_t INTERNALLY before input_blocks,
        so we pass them as separate arguments (x and context).

        Args:
            z_t:        (B, 4, h, w) — noisy interpolant on the flow path
            t:          (B,) in [0, 1] — continuous timestep
            z_img_cond: (B, 4, h, w) — clean image latent (conditioning)

        Returns:
            v_pred: (B, 4, h, w) — predicted velocity
        """

        B = z_t.size(0)

        # Expand stored null text embedding to batch size and cast to current dtype
        null_ca = self.empty_text_embed.to(dtype=z_t.dtype).expand(B, -1, -1)

        # UNet forward: x=z_t, context=z_img_cond (channel-concat inside),
        # context_ca=null_ca (cross-attention in transformer blocks)
        # t is passed as-is in [0,1]; timestep_embedding handles sinusoidal encoding
        # (B, 4, h, w) — no slicing needed, out_channels=4
        return self.backbone(
            x=z_t,
            t=t,
            context=z_img_cond,
            context_ca=null_ca,
        )


def build_model(config) -> MarsDepthFM:
    """
    Factory function: build the MarsDepthFM model from config.

    Only supports backend="sd21" (the actual DepthFM architecture).
    """
    backend = config.model.backend
    if backend != "sd21":
        raise ValueError(
            f"Unsupported backend: '{backend}'. Only 'sd21' is supported "
            "(the original DepthFM CompVis LDM UNet checkpoint)."
        )

    backbone, vae, noising_step, empty_text_embed = load_sd21_backend(
        depthfm_checkpoint=config.model.depthfm_checkpoint,
        vae_id=config.model.vae_id,
        use_checkpoint=config.model.use_checkpoint
    )

    model_type = config.model.get("model_type", "debug")

    if model_type == "debug":
        backbone = DebugUNet()

    elif model_type == "modulated":
        backbone = ModulatedMicroFlowNet()

    model = MarsDepthFM(
        backbone=backbone,
        vae=vae,
        noising_step=noising_step,
        empty_text_embed=empty_text_embed,
        freeze_encoder=config.model.get("freeze_encoder", False),
    )

    # Gradient checkpointing — UNetModel supports use_checkpoint per-block,
    # but it's set at construction time via ldm_hparams. If requested post-hoc,
    # patch the flag on all ResBlock/AttentionBlock instances.
    if config.model.get("gradient_checkpointing", False):
        from depth_fm.models.unet.openaimodel import ResBlock, AttentionBlock
        count = 0
        for module in model.backbone.modules():
            if isinstance(module, (ResBlock, AttentionBlock)):
                module.use_checkpoint = True
                count += 1
        logger.info("Gradient checkpointing enabled on %d blocks", count)

    # torch.compile for throughput — use "reduce-overhead" to avoid the extra
    # VRAM that "max-autotune" reserves for CUDA-graph autotuning. This mode
    # still captures CUDA graphs but skips expensive kernel search.
    # disable=True skips compile entirely (useful for debugging).
    if config.model.get("compile_backbone", False):
        compile_mode = config.model.get("compile_mode", "max-autotune")
        logger.info("Compiling backbone with mode=%s", compile_mode)
        model.backbone = torch.compile(
            model.backbone,
            mode=compile_mode,
            fullgraph=False,  # allow graph breaks rather than erroring
        )

    return model
