"""
Inference script for Mars DepthFM.

Given a single monocular RED camera image, generates a DTM prediction
using the trained flow matching model with 1-step or multi-step Euler ODE solving.

Usage:
    python scripts/inference.py \
        --checkpoint checkpoints/mars_depthfm/final/ema_backbone.pt \
        --input_image test_image.tif \
        --output_dtm predicted_dtm.tif \
        --backend sd21 \
        --num_steps 1

Multi-step Euler increases quality at the cost of speed:
    --num_steps 1:  fastest, ~90% quality (DepthFM's main selling point)
    --num_steps 4:  good balance
    --num_steps 10: highest quality, slower

Ensemble mode: run N times with different noise augmentations, average results.
This is what DepthFM uses for benchmark evaluation.
    --ensemble_size 10
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_image(path: str, resolution: int = 512) -> torch.Tensor:
    """Load and preprocess a single image to (1, 3, H, W) in [-1, 1]."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dtm_dataset import load_geotiff
        img = load_geotiff(path).astype(np.float32)
    except Exception:
        img = np.array(Image.open(path).convert("RGB")).astype(np.float32)
        img = img.transpose(2, 0, 1)  # HWC -> CHW

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=0)
    elif img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)

    # Normalize to [-1, 1]
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        img = 2.0 * (img - img_min) / (img_max - img_min) - 1.0

    img_tensor = torch.from_numpy(img).unsqueeze(0).float()

    # Resize to target resolution
    if img_tensor.shape[-2] != resolution or img_tensor.shape[-1] != resolution:
        img_tensor = torch.nn.functional.interpolate(
            img_tensor, size=(resolution, resolution),
            mode="bilinear", align_corners=False,
        )

    return img_tensor


@torch.no_grad()
def predict_dtm(
        model,
        image_tensor: torch.Tensor,
        num_steps: int = 1,
        ensemble_size: int = 1,
) -> torch.Tensor:
    """
    Predict DTM from a single image using flow matching ODE solving.

    Multi-step Euler: solve dz/dt = v_theta(z_t, t) from t=0 (image) to t=1 (depth)
        z_{t+dt} = z_t + dt * v_theta(z_t, t)

    Ensemble: run with different noise augmentations, average latent predictions.
    """
    device = next(model.backbone.parameters()).device
    image_tensor = image_tensor.to(device)

    # Encode image to latent space
    z_img = model.encode_to_latent(image_tensor)

    predictions = []
    for _ in range(ensemble_size):
        z_t = z_img.clone()

        # Multi-step Euler integration from t=0 to t=1
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((z_t.shape[0],), t_val, device=device)

            # Add slight noise augmentation to conditioning (same as training)
            z_cond = z_img + 0.003 * torch.randn_like(z_img)  # sqrt(1 - 0.997)

            v_pred = model.predict_velocity(z_t, t, z_cond)
            z_t = z_t + dt * v_pred

        predictions.append(z_t)

    # Average ensemble predictions in latent space
    z_depth = torch.stack(predictions).mean(dim=0)

    # Decode to pixel space
    depth_pixels = model.decode_from_latent(z_depth)

    # Take first channel (all 3 are identical for depth)
    depth_map = depth_pixels[0, 0].float().cpu().numpy()

    return depth_map


def main():
    parser = argparse.ArgumentParser(description="Mars DepthFM Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to EMA backbone .pt file")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_dtm", type=str, required=True)
    parser.add_argument("--backend", type=str, default="sd21", choices=["sd21", "sd35"])
    parser.add_argument("--vae_id", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=1, help="Euler ODE steps (1=fastest, 10=best)")
    parser.add_argument("--ensemble_size", type=int, default=1, help="Number of ensemble members")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Set VAE defaults
    if args.vae_id is None:
        args.vae_id = "stabilityai/stable-diffusion-2-1" if args.backend == "sd21" \
            else "stabilityai/stable-diffusion-3.5-medium"

    # Load model components
    from diffusers import AutoencoderKL
    print(f"Loading VAE from {args.vae_id}...")
    vae = AutoencoderKL.from_pretrained(args.vae_id, subfolder="vae")
    vae.eval().requires_grad_(False)

    if args.backend == "sd21":
        from diffusers import UNet2DConditionModel
        backbone = UNet2DConditionModel.from_pretrained(
            args.vae_id, subfolder="unet",
            in_channels=8, low_cpu_mem_usage=False, ignore_mismatched_sizes=True,
        )
    else:
        from diffusers import SD3Transformer2DModel
        backbone = SD3Transformer2DModel.from_pretrained(
            args.vae_id, subfolder="transformer"
        )

    # Load trained weights
    print(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict, strict=False)

    from depth_fm.models.mars_depthfm import MarsDepthFM
    model = MarsDepthFM(
        backbone=backbone,
        vae=vae,
        backend=args.backend,
        cross_attn_dim=1024 if args.backend == "sd21" else 4096,
    )
    model = model.to(args.device)
    model.eval()

    # Load and process image
    print(f"Processing {args.input_image}...")
    image_tensor = load_image(args.input_image, args.resolution)

    # Predict
    depth_map = predict_dtm(
        model, image_tensor,
        num_steps=args.num_steps,
        ensemble_size=args.ensemble_size,
    )

    # Save output
    os.makedirs(os.path.dirname(args.output_dtm) or ".", exist_ok=True)

    if args.output_dtm.endswith((".tif", ".tiff")):
        try:
            import tifffile
            tifffile.imwrite(args.output_dtm, depth_map.astype(np.float32))
        except ImportError:
            np.save(args.output_dtm.replace(".tif", ".npy"), depth_map)
            print(f"tifffile not installed, saved as .npy instead")
    else:
        np.save(args.output_dtm, depth_map)

    print(f"Saved DTM prediction to {args.output_dtm}")
    print(f"  Shape: {depth_map.shape}")
    print(f"  Range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")


if __name__ == "__main__":
    main()
