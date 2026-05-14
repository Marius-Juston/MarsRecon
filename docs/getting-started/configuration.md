# Configuration

MarsRecon's training pipeline is configured via [OmegaConf](https://omegaconf.readthedocs.io/)
YAML files. The primary config is `configs/train_hirise.yaml`.

## CLI overrides

`launch_train.sh` forwards OmegaConf dotlist overrides straight through to the training entry
point:

```bash
bash scripts/training/launch_train.sh \
    --config configs/train_hirise.yaml \
    training.per_gpu_batch_size=2 \
    training.max_steps=500 \
    data.normalize_elevation=true
```

## Major config sections

| Section          | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| `data`           | Dataset roots, bbox, ortho type/scale, scaler choice (relative vs log) |
| `sampling`       | Patch size, units, split method, split axis, k-fold parameters         |
| `model`          | Backbone selection, UNet hyperparameters                               |
| `flow`           | Flow-matching noise schedule                                           |
| `losses`         | Weights for velocity / normals / gradient / photometric / ordinal      |
| `training`       | Batch size, learning rate, max steps, EMA, checkpoint policy           |
| `viz`            | Figure cadence, error-map style, debug plots                           |
| `wandb`          | W&B project, run name, tags                                            |

The complete schema is documented in the
[API reference for `depth_fm.training.train_lightning`](../reference/depth_fm/training/train_lightning.md).

## Inspecting data without training

```bash
# Render thumbnails of sampled patches
bash scripts/training/launch_train.sh \
    --config configs/train_hirise.yaml --view_thumbnails

# Full diagnostic visualization (sampler coverage, calibration, residuals)
bash scripts/training/launch_train.sh \
    --config configs/train_hirise.yaml --all_viz
```

## `PYTHONPATH`

The repo treats `src/` as the import root. Tests handle this in `tests/conftest.py`, but ad-hoc
scripts need `PYTHONPATH=src` set explicitly:

```bash
PYTHONPATH=src uv run python -m dataset.core.dtm --bbox -120 -30 150 30
```
