"""
Pre-compute VAE latents for all training images and DTMs.

This eliminates the VAE forward pass during training (the VAE is frozen anyway),
roughly doubling training throughput. Uses all 128 CPU cores for parallel I/O
and a single GPU for batched VAE encoding.

Usage:
    python scripts/precompute_latents.py \
        --data_dir data/train \
        --output_dir data/latents/train \
        --backend sd21 \
        --batch_size 16 \
        --num_workers 128
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depth_fm.dataset import MarsDTMDataset


def encode_batch(vae, images, device, dtype):
    """Encode a batch of pixel images to latent space."""
    with torch.no_grad():
        images = images.to(device=device, dtype=dtype)
        posterior = vae.encode(images).latent_dist
        latents = posterior.mode() * vae.config.scaling_factor
    return latents.cpu()


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data dir with images/ and dtms/ subdirs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for latent .pt files")
    parser.add_argument("--backend", type=str, default="sd21", choices=["sd21", "sd35"])
    parser.add_argument("--vae_id", type=str, default=None, help="VAE model ID (defaults based on backend)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dtm_normalization", type=str, default="minmax")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=128, help="DataLoader workers (use all CPU cores)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine VAE
    if args.vae_id is None:
        if args.backend == "sd21":
            args.vae_id = "runwayml/stable-diffusion-v1-5"
        else:
            args.vae_id = "stabilityai/stable-diffusion-3.5-medium"

    # Load VAE
    from diffusers import AutoencoderKL
    print(f"Loading VAE from {args.vae_id}...")
    vae = AutoencoderKL.from_pretrained(args.vae_id, subfolder="vae")
    vae = vae.to(args.device)
    vae.eval()
    vae.requires_grad_(False)
    dtype = torch.float32  # VAE encoding needs full precision for quality

    # Build dataset (no augmentation — we want deterministic encoding)
    image_dir = os.path.join(args.data_dir, "images")
    dtm_dir = os.path.join(args.data_dir, "dtms")
    confidence_dir = os.path.join(args.data_dir, "confidence")
    if not os.path.isdir(confidence_dir):
        confidence_dir = None

    dataset = MarsDTMDataset(
        image_dir=image_dir,
        dtm_dir=dtm_dir,
        confidence_dir=confidence_dir,
        resolution=args.resolution,
        dtm_normalization=args.dtm_normalization,
        use_stereo_augmentation=True,  # encode both L and R
        random_horizontal_flip=False,
        random_vertical_flip=False,
        random_rotation_degrees=0,
        brightness_jitter=0,
        contrast_jitter=0,
        is_validation=True,  # no augmentation
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(args.num_workers, len(dataset)),
        pin_memory=True,
        prefetch_factor=4,
        drop_last=False,
        persistent_workers=True,
    )

    print(f"Encoding {len(dataset)} samples to {args.output_dir}...")
    print(f"Using {min(args.num_workers, len(dataset))} workers for data loading")

    processed = 0
    for batch in tqdm(loader, desc="Encoding"):
        images = batch["image"]  # (B, 3, H, W)
        dtms = batch["dtm"]  # (B, 3, H, W)
        confs = batch["confidence"]
        tile_ids = batch["tile_id"]

        # Encode images
        image_latents = encode_batch(vae, images, args.device, dtype)
        # Encode DTMs
        dtm_latents = encode_batch(vae, dtms, args.device, dtype)

        # Save individual latent files
        for i in range(len(tile_ids)):
            tile_id = tile_ids[i]
            torch.save(image_latents[i], os.path.join(args.output_dir, f"{tile_id}_image.pt"))
            torch.save(dtm_latents[i], os.path.join(args.output_dir, f"{tile_id}_dtm.pt"))

            # Downsample confidence to latent resolution and save
            conf = confs[i:i + 1]  # (1, 1, H, W)
            conf_small = torch.nn.functional.interpolate(
                conf, size=image_latents.shape[-2:], mode="bilinear", align_corners=False
            )
            torch.save(conf_small[0], os.path.join(args.output_dir, f"{tile_id}_conf.pt"))

            processed += 1

    print(f"Done. Encoded {processed} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
