"""
Recompute sun vectors in an existing LitData cache.

Reads the old LitData cache, recalculates sun_vector/intensity/ambient
using the current estimate_sun_vector_ols (CPU-side Python math), and
writes a new LitData cache with corrected values.

Two-phase approach (same as build_litdata.py):
  Phase 1: Read old cache sequentially → write corrected .npz files (no GDAL)
  Phase 2: litdata.optimize repacks .npz into new optimized chunks

Usage:
    python patch_sun_vectors_litdata.py \
        --input /scratch/mars_hirise_dtm/litdata_cache_d737fb4b0c75695d \
        --output /scratch/mars_hirise_dtm/litdata_cache_d737fb4b0c75695d_patched
"""
import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from litdata import StreamingDataset, optimize
from tqdm import tqdm

from depth_fm.depthfm_adapter import estimate_sun_vector_ols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Top-level function for litdata.optimize (must be picklable for spawn) ──

def _repack_npz(npz_path: str) -> dict:
    """Read a temp .npz and return dict for litdata chunking."""
    import numpy as np
    data = np.load(npz_path)
    return {
        "image": data["image"],
        "dtm": data["dtm"],
        "confidence": data["confidence"],
        "sun_vector": data["sun_vector"],
        "intensity": data["intensity"],
        "ambient": data["ambient"],
    }


def patch_split(input_dir: Path, output_dir: Path, split: str, workers: int = 32):
    """Read one split from old LitData cache, recompute sun vectors, write new cache."""
    split_input = input_dir / split
    split_output = str(output_dir / split)

    if not (split_input / "index.json").exists():
        logger.warning(f"No LitData index found for {split} at {split_input}")
        return

    success_marker = Path(split_output) / "_SUCCESS"
    if success_marker.exists():
        logger.info(f"[{split}] Already patched. Skipping.")
        return

    # Read from old cache
    dataset = StreamingDataset(input_dir=str(split_input), shuffle=False)
    num_samples = len(dataset)
    logger.info(f"[{split}] Patching {num_samples} samples...")

    # Phase 1: Read old samples, recompute sun vectors, write temp .npz
    tmp_dir = output_dir / f"_patch_tmp_{split}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    npz_paths = []

    for i in tqdm(range(num_samples), desc=f"Patch {split}"):
        raw = dataset[i]

        # Reconstruct torch tensors (float16 → float32 for the OLS math)
        image = torch.from_numpy(raw["image"].astype(np.float32))  # (3, H, W)
        dtm = torch.from_numpy(raw["dtm"].astype(np.float32))  # (3, H, W)
        confidence = torch.from_numpy(raw["confidence"].astype(np.float32))  # (1, H, W)

        # Recompute using single-channel DTM and the image
        dtm_1ch = dtm[:1]  # (1, H, W) — first channel of the 3ch replicated DTM
        sun_vec, intensity, ambient = estimate_sun_vector_ols(dtm_1ch, image, confidence)

        # Normalize to strict unit vector
        sun_vec = torch.nn.functional.normalize(sun_vec, p=2, dim=0)

        # Write corrected sample
        npz_path = str(tmp_dir / f"{i:08d}.npz")
        np.savez(
            npz_path,
            image=raw["image"],  # keep original float16
            dtm=raw["dtm"],  # keep original float16
            confidence=raw["confidence"],  # keep original float16
            sun_vector=sun_vec.numpy(),
            intensity=intensity.numpy(),
            ambient=ambient.numpy(),
        )
        npz_paths.append(npz_path)

    del dataset
    logger.info(f"[{split}] Phase 1 complete: {len(npz_paths)} corrected .npz files.")

    # Phase 2: Repack into LitData optimized chunks
    logger.info(f"[{split}] Phase 2: Repacking into LitData chunks at {split_output}...")
    optimize(
        fn=_repack_npz,
        inputs=npz_paths,
        output_dir=split_output,
        num_workers=min(workers, 32),
        chunk_bytes="256MB",
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    success_marker.touch()
    logger.info(f"[{split}] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch sun vectors in LitData cache")
    parser.add_argument("--input", type=str, required=True, help="Path to old litdata_cache_<hash> directory")
    parser.add_argument("--output", type=str, required=True, help="Path to write patched cache")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        patch_split(input_dir, output_dir, split, workers=args.workers)

    logger.info("All splits patched! Rename the patched folder to match your config hash.")
