---
title: MarsRecon
hide:
  - navigation
---

# MarsRecon

**A geospatial dataset manager and deep-learning training framework for NASA's HiRISE Mars imagery.**

MarsRecon wraps the [HiRISE](https://www.uahirise.org/) Reduced Data Records (RDR) and Digital
Terrain Model (DTM) collections as [TorchGeo](https://torchgeo.readthedocs.io/) datasets, and
provides a flow-matching monocular depth pipeline (DepthFM) and a tri-modal CLIP model
(MarsCLIP) trained on top of them.

<div class="grid cards" markdown>

-   :material-database-outline: **Dataset layer**

    ---

    TorchGeo `GeoDataset` wrappers for HiRISE RDR and DTM stereo pairs, with spatiotemporal
    indexing, async PDS downloading, and radiometric calibration.

    [:octicons-arrow-right-24: Architecture](architecture/dataset-layer.md)

-   :material-waveform: **DepthFM training**

    ---

    Flow-matching monocular depth estimation adapted for Mars DTMs, on multi-GPU PyTorch
    Lightning.

    [:octicons-arrow-right-24: DepthFM pipeline](architecture/depthfm-pipeline.md)

-   :material-image-multiple-outline: **MarsCLIP**

    ---

    Tri-modal CLIP (image + elevation + text) for Mars imagery, with MAE pretraining.

    [:octicons-arrow-right-24: MarsCLIP](architecture/marsclip.md)

-   :material-rocket-launch-outline: **Get started**

    ---

    Install, run the dataset pipeline, and launch a training job.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

</div>

## Why MarsRecon?

HiRISE produces the highest-resolution images of Mars available (~25 cm/pixel RED, ~50 cm/pixel
colour), but the raw archive is awkward for deep learning: thin rotated parallelogram strips,
JP2/IMG formats, per-observation projections, and a >10 TB total volume. MarsRecon handles all
of that plumbing so you can focus on the model.

- **Strip-aware sampling** — `HiRISEGeoSampler` pre-grids valid patch centres inside actual
  strip polygons, avoiding the 60–90 % empty-pixel patches you'd get from bounding-box sampling.
- **Single CRS hub** — all observations are reprojected on-the-fly into a common
  IAU 2000 Mars geographic CRS.
- **Fast I/O** — `cog_conversion` pre-converts JP2 → Cloud-Optimized GeoTIFF; the LitData
  streaming path serves training at full GPU saturation.
- **DepthFM training** — flow-matching elevation prediction with photometric (Lunar-Lambert)
  consistency, normals, multi-scale gradients, and ordinal-ranking losses.

## Repository layout (at a glance)

```
src/
  dataset/         # MarsHiRISE / MarsHiRISEDTM / HiRISEGeoSampler
  depth_fm/        # DepthFM model, Lightning training, losses, viz
  clip/            # MarsCLIP tri-modal model and MAE pretraining
configs/           # OmegaConf YAML configs
scripts/           # entry points (training, inference, viz, ablations)
tests/             # pytest suite
docs/              # this site
```

See [Architecture · Overview](architecture/overview.md) for the full source-tree map.

## Live API reference

Every module under `src/` has an auto-generated reference page driven by
[mkdocstrings](https://mkdocstrings.github.io/) — new public symbols appear automatically on the
next build.

[:octicons-arrow-right-24: Browse the API reference](reference/)
