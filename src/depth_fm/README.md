# Mars DepthFM — Flow Matching DTM Generation

## Architecture Choice: SD 2.1 vs SD 3.5

**Critical note:** The DepthFM checkpoint (`depthfm-v1.ckpt`) is built on the
**SD 2.1 UNet** (865M params, 4-channel VAE latent). SD 3.5 Medium uses a
completely different architecture — an **MMDiT-X transformer** (2.5B params,
16-channel VAE latent). These are **not compatible**. You cannot load DepthFM
weights into SD 3.5.

Additionally, the link `stabilityai/stable-diffusion-3.5-medium-tensorrt` is a
**TensorRT inference-optimized** model — it cannot be used for training. The
trainable version is `stabilityai/stable-diffusion-3.5-medium`.

This codebase supports **both backends**:

|                             | SD 2.1 + DepthFM ckpt          | SD 3.5 Medium                            |
|-----------------------------|--------------------------------|------------------------------------------|
| **Backbone**                | UNet (865M)                    | MMDiT-X (2.5B)                           |
| **VAE latent**              | 4 channels                     | 16 channels                              |
| **Pre-trained depth prior** | Yes (DepthFM)                  | No (image-only)                          |
| **Native flow matching**    | No (adapted from v-pred)       | Yes (rectified flow)                     |
| **VRAM per GPU**            | ~18 GB                         | ~35 GB                                   |
| **Recommendation**          | Start here for fastest results | Better long-term if you have enough data |

**Our recommendation**: Start with SD 2.1 + DepthFM checkpoint. It already has
learned depth priors from synthetic data. Fine-tuning for Mars domain is much
faster than teaching SD 3.5 depth estimation from scratch.

## Setup

```bash
# Environment
conda create -n mars_depthfm python=3.10 -y
conda activate mars_depthfm
pip install -r requirements.txt

# DepthFM checkpoint (SD 2.1 path)
wget https://ommer-lab.com/files/depthfm/depthfm-v1.ckpt -P checkpoints/

# SD 3.5 path (requires HuggingFace login + license acceptance)
# huggingface-cli login
# python -c "from diffusers import SD3Transformer2DModel; SD3Transformer2DModel.from_pretrained('stabilityai/stable-diffusion-3.5-medium', subfolder='transformer')"

# Configure accelerate for 4 GPUs
accelerate config  # select multi-GPU, 4 processes, bf16
```

## Data Layout

```
data/
├── train/
│   ├── images/          # RED camera images (.tif or .png)
│   │   ├── tile_0001.tif
│   │   └── ...
│   ├── dtms/            # Ground truth DTMs (.tif)
│   │   ├── tile_0001.tif
│   │   └── ...
│   └── confidence/      # Optional: stereo matching confidence maps
│       ├── tile_0001.tif
│       └── ...
├── val/
│   ├── images/
│   └── dtms/
```

Both left and right stereo images can be placed in `images/` — each paired with
the same DTM. Name them `tile_0001_L.tif` / `tile_0001_R.tif` and the dataset
class will pair both with `tile_0001.tif` from `dtms/`.

## Training

```bash
# Step 1: Pre-compute VAE latents (uses all 128 CPU cores)
python scripts/precompute_latents.py \
    --data_dir data/train \
    --output_dir data/latents/train \
    --backend sd21 \
    --num_workers 128

# Step 2: Train (4x A6000)
bash scripts/launch_train.sh
```

## Inference

```bash
python scripts/inference.py \
    --checkpoint checkpoints/mars_depthfm_best.pt \
    --input_image test_image.tif \
    --output_dtm predicted_dtm.tif \
    --backend sd21
```
