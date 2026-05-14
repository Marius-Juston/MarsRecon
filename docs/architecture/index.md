# Architecture

How MarsRecon is put together, from the dataset wrappers up to the training pipeline.

- [Overview](overview.md) — the system at one screen, with the full source-tree map.
- [Dataset layer](dataset-layer.md) — `MarsHiRISEBase`, RDR, DTM, the spatial index.
- [Sampler](sampler.md) — strip-aware patch sampling and geographic splits.
- [DepthFM pipeline](depthfm-pipeline.md) — flow-matching training end-to-end.
- [MarsCLIP](marsclip.md) — tri-modal contrastive model.
- [Coordinate system](coordinate-system.md) — IAU 2000 Mars geographic CRS conventions.
