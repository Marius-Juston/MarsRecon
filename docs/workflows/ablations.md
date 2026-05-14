# Ablations

`scripts/training/run_ablation.py` orchestrates loss-component ablations: it sweeps a list of
loss-weight configurations, launches a training run for each, and aggregates W&B results.

## Run

```bash
PYTHONPATH=src uv run python scripts/training/run_ablation.py \
    --config configs/train_hirise.yaml \
    --sweep ablations/loss_components.yaml
```

## Sweep file shape

The sweep YAML is a list of OmegaConf override sets. For example:

```yaml
- name: no_photo
  overrides:
    losses.photometric_weight: 0.0
- name: no_normals
  overrides:
    losses.normals_weight: 0.0
- name: photo_only
  overrides:
    losses.velocity_weight: 0.0
    losses.gradient_weight: 0.0
```

Each entry becomes a separate training run, named for W&B.

## Visualizing ablation results

```bash
uv run python scripts/visualization/paper_training_dynamics.py \
    --project marsrecon-ablations \
    --output paper_figures/
```

This produces convergence plots and lunar-Lambert weight-evolution plots from the W&B history.
