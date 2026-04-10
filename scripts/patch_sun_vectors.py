import glob
import logging
from pathlib import Path

import torch
import webdataset as wds
from tqdm import tqdm

# Import your fixed math function
from depth_fm.depthfm_adapter import estimate_sun_vector_ols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def patch_split(old_dir: Path, new_dir: Path, split: str):
    """Streams an old WDS split, recalculates sun vectors, and writes to a new WDS."""
    input_pattern = str(old_dir / split / f"{split}-*.tar")
    urls = sorted(glob.glob(input_pattern))

    if not urls:
        logger.warning(f"No .tar files found for {split} at {input_pattern}")
        return

    out_split_dir = new_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_split_dir / f"{split}-%06d.tar")

    # 1. Read and decode the old WebDataset
    dataset = wds.DataPipeline(
        wds.SimpleShardList(urls),
        wds.tarfile_to_samples(),
        wds.decode("torch")
    )

    # 2. Setup the writer for the new dataset
    sink = wds.ShardWriter(out_pattern, maxsize=2e9)

    logger.info(f"Patching {split} split...")
    count = 0

    for sample in tqdm(dataset):
        # Extract the heavy tensors (already processed by GDAL/TorchGeo!)
        dtm = sample["dtm.pth"][:1]
        img = sample["image.pth"]
        mask = sample["confidence.pth"]

        # Recalculate using your current, bug-free python code
        sun_vec, intensity, ambient = estimate_sun_vector_ols(dtm, img, mask)

        # Enforce strict float32 unit vector to prevent Parquet/serialization drift
        sun_vec = torch.nn.functional.normalize(sun_vec, p=2, dim=0)

        # Overwrite the broken data in the dictionary
        sample["sun_vector.pth"] = sun_vec
        sample["intensity.pth"] = intensity
        sample["ambient.pth"] = ambient

        # Write back to the new archive
        sink.write(sample)
        count += 1

    sink.close()

    # Write the success marker so your dataloader trusts this directory
    (out_split_dir / "_SUCCESS").touch()
    logger.info(f"Successfully patched {count} samples for {split}.\n")


if __name__ == "__main__":
    # Update these paths to match your actual cache hash
    OLD_WDS_DIR = Path("/scratch/mars_hirise_dtm/wds_cache_d737fb4b0c75695d")
    NEW_WDS_DIR = Path("/scratch/mars_hirise_dtm/wds_cache_d737fb4b0c75695d_patched")

    NEW_WDS_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        patch_split(OLD_WDS_DIR, NEW_WDS_DIR, split)

    logger.info("All splits patched! You can now rename the PATCHED folder to match your config hash.")