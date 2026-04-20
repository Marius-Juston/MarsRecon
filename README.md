# MarsRecon

MarsRecon is a Mars remote-sensing codebase built around HiRISE imagery.

The repo currently serves two closely related purposes:
- a TorchGeo-compatible HiRISE dataset and sampling stack
- Stage A masked-autoencoder training experiments on Olympus Mons patches using SatMAE-style ViT backbones

The current mainline Stage A workflow is scratch-backed, cache-aware, and designed for iterative reconstruction experiments rather than one-off local runs.

## What Lives Here

| Path | Purpose |
| --- | --- |
| `src/dataset/mars_hirise.py` | Main HiRISE image dataset |
| `src/dataset/hirise_sampler.py` | Strip-aware geospatial sampler |
| `src/dataset/preprocessing.py` | JP2 -> COG conversion helpers |
| `src/clip/marsclip_patches.py` | Patch extraction / Stage A patch dataset logic |
| `src/clip/build_marsclip_litdata.py` | litData cache builder for Stage A |
| `src/clip/train_marsclip_satmae.py` | Main SatMAE trainer |
| `scripts/launch_satmae_olympus.sh` | Canonical Olympus Mons Stage A launcher |
| `CODEX.md` | Operational handoff for the current Stage A workflow |

## Environment

- Python `3.12`
- `uv` is the recommended environment manager
- real HiRISE data under `/scratch/mars_hirise`

Install the repo with:

```bash
uv sync --extra dev
```

## Data Layout

The HiRISE dataset expects a local mirror like:

```text
/scratch/mars_hirise/
    RDRCUMINDEX.LBL
    RDRCUMINDEX.TAB
    images/
        PSP_001430_1780_COLOR.JP2
        PSP_001430_1780_COLOR.LBL
        PSP_001430_1780_RED.JP2
        PSP_001430_1780_RED.LBL
```

Dataset-wide image normalization stats currently live at:

```text
dataset_stats/image/dataset_stats.json
```

## Scratch Layout

Generated artifacts should go to scratch, not the repo root.

Current defaults:

- raw HiRISE mirror:
  - `/scratch/mars_hirise`
- Stage A reusable assets:
  - `/scratch/marsrecon_runs/stage_a/assets`
- Stage A SatMAE runs:
  - `/scratch/marsrecon_runs/stage_a/satmae`
- dataset visualization outputs:
  - `/scratch/marsrecon_runs/dataset_viz`
- clip preview outputs:
  - `/scratch/marsrecon_runs/clip_viz`
- clip report outputs:
  - `/scratch/marsrecon_runs/clip_reports`

## Common Commands

Precompute COG sidecars for faster random access:

```bash
uv run python -m src.dataset.preprocessing --root /scratch/mars_hirise --workers 4
```

Generate a quick HiRISE coverage/sample visualization:

```bash
uv run python src/dataset/mars_hirise.py
```

Override the visualization output directory if needed:

```bash
uv run python src/dataset/mars_hirise.py --output-dir /tmp/marsrecon_demo_figures
```

Generate a patch preview/report:

```bash
uv run python src/clip/report_marsclip_patches.py
```

Launch the standard Olympus Mons SatMAE flow:

```bash
bash scripts/launch_satmae_olympus.sh
```

For serious experiment control, use the trainer directly:

```bash
.venv/bin/python src/clip/train_marsclip_satmae.py --help
```

## Testing

Run the full Python test suite:

```bash
uv run pytest tests/ -q
```

Run the Stage A tests that matter most for the current SatMAE path:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_marsclip_patches.py \
  tests/test_train_marsclip_satmae.py \
  -q
```

## Notes

- The Stage A workflow now uses scratch-backed litData caches rather than relying on slow raw patch extraction during training.
- `CODEX.md` is the best place to look for the current experiment state, run history, and operational caveats.
- The repo still contains older dataset-oriented utilities, but the most actively maintained training path is `src/clip/train_marsclip_satmae.py`.
