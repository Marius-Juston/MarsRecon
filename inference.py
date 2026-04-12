"""
DepthFM inference — thin wrapper over MarsDepthFM.predict_depth().

The full inference logic now lives in MarsDepthFM.forward() / predict_depth()
so training and inference share one authoritative implementation.

Usage:
    uv run python inference.py --img assets/dog.png --ckpt checkpoints/depthfm-v1.ckpt
"""

import argparse
import numpy as np
import os
import sys

import einops
import torch
import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import Resampling

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# ---------------------------------------------------------------------------
# Image loading — identical to depth-fm/inference.py
# ---------------------------------------------------------------------------

def get_dtype_from_str(dtype_str: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]


def resize_max_res(img: Image.Image, max_edge_resolution: int,
                   resample_method=Resampling.BILINEAR):
    original_width, original_height = img.size
    downscale_factor = min(max_edge_resolution / original_width,
                           max_edge_resolution / original_height)
    new_width  = int(original_width  * downscale_factor)
    new_height = int(original_height * downscale_factor)
    new_width  = round(new_width  / 64) * 64
    new_height = round(new_height / 64) * 64
    print(f"Resizing image from {original_width}x{original_height} to {new_width}x{new_height}")
    return img.resize((new_width, new_height), resample=resample_method), (original_width, original_height)


def load_im(fp: str, processing_res: int = -1):
    assert os.path.exists(fp), f"File not found: {fp}"
    im = Image.open(fp).convert("RGB")
    if processing_res < 0:
        processing_res = max(im.size)
    im, orig_res = resize_max_res(im, processing_res)
    x = np.array(im)
    x = einops.rearrange(x, "h w c -> c h w")
    x = x / 127.5 - 1
    x = torch.tensor(x, dtype=torch.float32)[None]
    return x, orig_res


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str):
    from depth_fm.model import MarsDepthFM, load_sd21_backend

    backbone, vae, noising_step, empty_text_embed = load_sd21_backend(
        depthfm_checkpoint=ckpt_path,
        vae_id="runwayml/stable-diffusion-v1-5",
        device=device,
        use_checkpoint=False,   # inference: no gradient checkpointing (matches original)
    )
    model = MarsDepthFM(
        backbone=backbone,
        vae=vae,
        noising_step=noising_step,
        empty_text_embed=empty_text_embed,
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f"{'Input':<10}: {args.img}")
    print(f"{'Steps':<10}: {args.num_steps}")
    print(f"{'Ensemble':<10}: {args.ensemble_size}")

    device = f"cuda:{args.device}"

    print("Loading model...")
    model = load_model(args.ckpt, device=device)

    im, orig_res = load_im(args.img, args.processing_res)
    im = im.to(device)

    # Mirror original: set backbone dtype attribute + autocast
    dtype = get_dtype_from_str(args.dtype)
    model.backbone.dtype = dtype

    with torch.autocast(device_type="cuda", dtype=dtype):
        depth = model.predict_depth(im,
                                    num_steps=args.num_steps,
                                    ensemble_size=args.ensemble_size)

    depth = depth.squeeze(0).squeeze(0).cpu().numpy()   # (H, W) in [0, 1]

    if args.no_color:
        depth_arr = (depth * 255).astype(np.uint8)
    else:
        depth_arr = plt.get_cmap("magma")(depth, bytes=True)[..., :3]

    depth_fp = args.img + "_depth.png"
    depth_img = Image.fromarray(depth_arr)
    if depth_img.size != orig_res:
        depth_img = depth_img.resize(orig_res, Resampling.BILINEAR)
    depth_img.save(depth_fp)
    print(f"==> Saved depth map to {depth_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DepthFM Inference (MarsDepthFM)")
    parser.add_argument("--img",           type=str, default="assets/dog.png")
    parser.add_argument("--ckpt",          type=str, default="checkpoints/depthfm-v1.ckpt")
    parser.add_argument("--num_steps",     type=int, default=2)
    parser.add_argument("--ensemble_size", type=int, default=4)
    parser.add_argument("--no_color",      action="store_true")
    parser.add_argument("--device",        type=int, default=0)
    parser.add_argument("--processing_res",type=int, default=-1)
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16", "fp16"], default="fp16")
    args = parser.parse_args()

    main(args)
