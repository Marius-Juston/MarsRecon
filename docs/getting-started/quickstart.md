# Quickstart

A 5-minute tour: download a small slice of HiRISE data, sample patches, and run a tiny
inference.

## 1. Download a small region

```bash
# RDR (single-image) dataset — a couple of strips over Jezero
PYTHONPATH=src uv run python -m dataset.core.rdr \
    --bbox 77 17 78 19 \
    --root /scratch/mars_hirise

# DTM (stereo) dataset — same area
PYTHONPATH=src uv run python -m dataset.core.dtm \
    --bbox 77 17 78 19 \
    --root /scratch/mars_hirise_dtm
```

!!! warning "Avoid unfiltered downloads"
The full HiRISE DTM archive is >10 TB. **Always pass `--bbox`** (or `--target` for a specific
observation) when downloading.

## 2. Use the dataset from Python

```python
from dataset import MarsHiRISEDTM, HiRISEGeoSampler
from torchgeo.samplers import Units

dtm = MarsHiRISEDTM(
    root="/scratch/mars_hirise_dtm",
    include_ortho=True,
    ortho_type=["RED"],
    download=False,
)

sampler = HiRISEGeoSampler(
    dtm,
    size=0.018,            # in CRS units (~1 km @ Mars equator)
    length=None,
    units=Units.CRS,
    split_fractions=(0.8, 0.1, 0.1),
    split_method="geographic",
    split_axis="longitude",
    split="train",
)

for bbox in sampler:
    sample = dtm[bbox]
    print(sample["elevation"].shape, sample["left_red"].shape)
    break
```

## 3. Pre-convert to COG (one-time, for fast I/O)

```bash
PYTHONPATH=src uv run python -m dataset.preprocessing.cog_conversion \
    --root /scratch/mars_hirise_dtm --workers 4
```

## 4. Build LitData chunks (fastest training I/O path)

```bash
PYTHONPATH=src uv run python scripts/training/build_litdata.py \
    --config configs/train_hirise.yaml --workers 96
```

## 5. Launch a DepthFM training run

```bash
bash scripts/training/launch_train.sh --config configs/train_hirise.yaml --n_runs 1
```

See [Workflows · Training](../workflows/training.md) for the full pipeline.
