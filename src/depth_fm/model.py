"""
Model wrapper for Mars DepthFM.

Supports two backends:
  - SD 2.1: UNet from DepthFM checkpoint. Input channels doubled (4->8) for
    concatenated image conditioning. This is the DepthFM architecture.
  - SD 3.5: MMDiT-X transformer. Input channels doubled (16->32). Natively
    uses rectified flow. Text conditioning zeroed out (image-only task).

Both backends share the same flow matching training interface.
"""

import logging

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)


# ─────────────────────────────── SD 2.1 Backend ───────────────────────────────

def load_sd21_backend(depthfm_checkpoint: str, vae_id: str, device: str = "cpu"):
    """
    Load the DepthFM model (SD 2.1 UNet + VAE).

    The DepthFM checkpoint contains a UNet that has been:
    1. Initialized from SD 2.1's v-prediction UNet
    2. Modified to accept 8 input channels (4 noisy depth + 4 image conditioning)
    3. Fine-tuned with flow matching on synthetic depth data

    Returns:
        unet: the trainable UNet
        vae: frozen VAE encoder/decoder
    """
    from diffusers import UNet2DConditionModel

    # Load VAE from SD 2.1 (frozen, used for encode/decode only)
    vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae")
    vae.eval()
    vae.requires_grad_(False)

    # Load DepthFM checkpoint
    ckpt = torch.load(depthfm_checkpoint, map_location=device, weights_only=False)

    # The DepthFM checkpoint may store the state dict under different keys
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Filter to UNet keys (DepthFM stores the UNet under 'unet.' prefix)
    unet_state = {}
    for k, v in state_dict.items():
        # Remove common prefixes
        clean_key = k
        for prefix in ["unet.", "model.diffusion_model.", "model."]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break
        unet_state[clean_key] = v

    # Create UNet with 8 input channels (DepthFM's architecture)
    # SD 2.1 UNet config but with in_channels=8
    unet = UNet2DConditionModel.from_pretrained(
        vae_id,
        subfolder="unet",
        in_channels=8,  # 4 (noisy depth) + 4 (image conditioning)
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )

    # Load DepthFM weights (may have mismatched sizes for the input conv)
    missing, unexpected = unet.load_state_dict(unet_state, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading DepthFM UNet: {len(missing)} keys")
        for k in missing[:5]:
            logger.warning(f"  {k}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)} keys")

    # The UNet expects cross-attention conditioning, but DepthFM doesn't use
    # text prompts. We pass zero embeddings. SD 2.1 cross_attention_dim = 1024.
    cross_attn_dim = unet.config.cross_attention_dim

    logger.info(
        f"Loaded SD 2.1 backend: UNet params={sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M, "
        f"VAE latent_channels={vae.config.latent_channels}, "
        f"cross_attention_dim={cross_attn_dim}"
    )

    return unet, vae, cross_attn_dim


# ─────────────────────────────── SD 3.5 Backend ───────────────────────────────

def load_sd35_backend(model_id: str, device: str = "cpu"):
    """
    Load SD 3.5 Medium's MMDiT-X transformer and VAE.

    The transformer's patch embedding is modified to accept 32 input channels
    (16 noisy depth latent + 16 image conditioning latent).

    SD 3.5 natively uses rectified flow (flow matching), so the objective
    aligns perfectly — no need to convert from v-prediction.

    Returns:
        transformer: the trainable MMDiT-X
        vae: frozen VAE
    """
    from diffusers import SD3Transformer2DModel

    # Load VAE
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae.eval()
    vae.requires_grad_(False)

    # Load transformer
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer"
    )

    # Modify patch embedding to accept doubled input channels
    # SD 3.5 Medium: pos_embed.proj is Conv2d(16, inner_dim, patch_size, stride=patch_size)
    old_proj = transformer.pos_embed.proj
    in_ch = old_proj.in_channels  # 16
    new_in_ch = in_ch * 2  # 32

    new_proj = nn.Conv2d(
        new_in_ch,
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None,
    )

    # Initialize: copy old weights for first 16 channels, duplicate for next 16
    with torch.no_grad():
        new_proj.weight[:, :in_ch] = old_proj.weight
        new_proj.weight[:, in_ch:] = old_proj.weight  # conditioning channels
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    transformer.pos_embed.proj = new_proj
    # Update config to reflect new in_channels
    transformer.config["in_channels"] = new_in_ch

    # For text conditioning: we'll pass null embeddings during training
    # SD 3.5 uses joint_attention_dim=4096 for the combined CLIP+T5 embeddings
    cross_attn_dim = transformer.config.get("joint_attention_dim", 4096)

    logger.info(
        f"Loaded SD 3.5 backend: Transformer params="
        f"{sum(p.numel() for p in transformer.parameters()) / 1e6:.1f}M, "
        f"VAE latent_channels={vae.config.latent_channels}, "
        f"modified in_channels={new_in_ch}"
    )

    return transformer, vae, cross_attn_dim


# ──────────────────────────── Unified Model Wrapper ───────────────────────────

class MarsDepthFM(nn.Module):
    """
    Unified wrapper for flow matching depth estimation.

    Handles both SD 2.1 (UNet) and SD 3.5 (MMDiT) backends with a consistent
    interface for the training loop.
    """

    def __init__(
            self,
            backbone: nn.Module,
            vae: AutoencoderKL,
            backend: str,
            cross_attn_dim: int,
            use_lora: bool = False,
            lora_rank: int = 64,
            lora_alpha: int = 64,
            freeze_encoder: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.vae = vae
        self.backend = backend  # "sd21" or "sd35"
        self.cross_attn_dim = cross_attn_dim
        self.latent_channels = vae.config.latent_channels  # 4 for SD2.1, 16 for SD3.5

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)

        # Freeze encoder half of UNet if requested (SD 2.1 only)
        if freeze_encoder and backend == "sd21":
            self._freeze_encoder()

    def _apply_lora(self, rank: int, alpha: int):
        """Apply LoRA adapters to the backbone."""
        try:
            from peft import LoraConfig, get_peft_model
            target_modules = []
            if self.backend == "sd21":
                target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
            else:
                target_modules = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]

            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            logger.info(
                f"LoRA applied: rank={rank}, alpha={alpha}, "
                f"trainable params={sum(p.numel() for p in self.backbone.parameters() if p.requires_grad) / 1e6:.1f}M"
            )
        except ImportError:
            logger.warning("peft not installed, skipping LoRA. pip install peft")

    def _freeze_encoder(self):
        """Freeze the encoder (down blocks) of the UNet, train only decoder."""
        for name, param in self.backbone.named_parameters():
            if "down_blocks" in name or "mid_block" in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        logger.info(f"Encoder frozen: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M trainable")

    @torch.no_grad()
    def encode_to_latent(self, pixel_images: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel-space images (B, 3, H, W) to latent space using frozen VAE.
        Returns: (B, C, h, w) where C=4 (SD2.1) or C=16 (SD3.5)
        """
        # VAE expects input in [-1, 1]
        posterior = self.vae.encode(pixel_images).latent_dist
        latent = posterior.mode()  # deterministic encoding (use .sample() for stochastic)
        # Scale by VAE scaling factor
        latent = latent * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent back to pixel space. Returns (B, 3, H, W) in [-1, 1]."""
        latent = latent / self.vae.config.scaling_factor
        decoded = self.vae.decode(latent).sample
        return decoded

    def _get_null_encoder_hidden_states(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """
        Create null text conditioning for the backbone.
        SD 2.1: needs (B, seq_len, 1024) cross-attention input
        SD 3.5: needs (B, seq_len, joint_attention_dim) + pooled projections
        """
        if self.backend == "sd21":
            # SD 2.1 UNet expects encoder_hidden_states of shape (B, seq_len, 1024)
            # Use seq_len=77 (standard CLIP), all zeros
            return torch.zeros(batch_size, 77, self.cross_attn_dim, device=device, dtype=dtype)

        elif self.backend == "sd35":
            # SD 3.5 MMDiT expects:
            #   encoder_hidden_states: (B, seq_len, joint_attention_dim)
            #   pooled_projections: (B, pooled_dim)
            encoder_hidden_states = torch.zeros(
                batch_size, 77, self.cross_attn_dim, device=device, dtype=dtype
            )
            # pooled_projections dim = 2048 for SD 3.5 Medium
            pooled_projections = torch.zeros(batch_size, 2048, device=device, dtype=dtype)
            return encoder_hidden_states, pooled_projections

    def predict_velocity(
            self,
            z_t: torch.Tensor,
            t: torch.Tensor,
            z_img_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: predict velocity field v_theta(z_t, t; z_img).

        Args:
            z_t: noisy interpolant (B, C, h, w) — the point on the flow path
            t: timestep (B,) in [0, 1]
            z_img_cond: clean image latent (B, C, h, w) — the conditioning signal

        Returns:
            v_pred: predicted velocity (B, C, h, w)
        """
        batch_size = z_t.shape[0]
        device = z_t.device
        dtype = z_t.dtype

        # Concatenate noisy depth latent with clean image conditioning
        # This is the core DepthFM design: channel-wise concatenation
        model_input = torch.cat([z_t, z_img_cond], dim=1)  # (B, 2C, h, w)

        if self.backend == "sd21":
            # SD 2.1 UNet expects timestep as integer in [0, 1000]
            # Map continuous t in [0,1] to discrete timestep
            timestep = (t * 1000.0).long()

            null_text = self._get_null_encoder_hidden_states(batch_size, device, dtype)
            output = self.backbone(
                model_input,
                timestep,
                encoder_hidden_states=null_text,
                return_dict=True,
            )
            v_pred = output.sample

        elif self.backend == "sd35":
            # SD 3.5 MMDiT expects continuous timestep
            # SD 3.5 uses sigma-based timestep internally
            timestep = t * 1000.0  # scale to match SD3.5 convention

            enc_hidden, pooled = self._get_null_encoder_hidden_states(
                batch_size, device, dtype
            )
            output = self.backbone(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=enc_hidden,
                pooled_projections=pooled,
                return_dict=True,
            )
            v_pred = output.sample

        # The model outputs in the concatenated channel space (2C);
        # we only want the depth prediction channels (first C)
        v_pred = v_pred[:, :self.latent_channels]

        return v_pred


def build_model(config) -> MarsDepthFM:
    """
    Factory function: build the model from config.

    Args:
        config: OmegaConf config with model.backend, model.depthfm_checkpoint, etc.

    Returns:
        MarsDepthFM instance
    """
    backend = config.model.backend

    if backend == "sd21":
        backbone, vae, cross_attn_dim = load_sd21_backend(
            config.model.depthfm_checkpoint,
            config.model.vae_id,
        )
    elif backend == "sd35":
        backbone, vae, cross_attn_dim = load_sd35_backend(
            config.model.sd35_model_id,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    model = MarsDepthFM(
        backbone=backbone,
        vae=vae,
        backend=backend,
        cross_attn_dim=cross_attn_dim,
        use_lora=config.model.get("use_lora", False),
        lora_rank=config.model.get("lora_rank", 64),
        lora_alpha=config.model.get("lora_alpha", 64),
        freeze_encoder=config.model.get("freeze_encoder", False),
    )

    # Gradient checkpointing
    if config.model.get("gradient_checkpointing", False):
        if hasattr(model.backbone, "enable_gradient_checkpointing"):
            model.backbone.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")

    return model
