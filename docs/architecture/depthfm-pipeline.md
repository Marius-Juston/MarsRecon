# DepthFM training pipeline

DepthFM is a flow-matching monocular depth estimator adapted for Mars DTMs.

## End-to-end flow

```mermaid
flowchart TB
    subgraph Data
      DS["MarsHiRISEDTM"] --> SAM["HiRISEGeoSampler"]
      SAM --> AD["DepthFMHiRISEAdapterCached<br/>(scaler, stereo aug, void fill)"]
      AD --> LD["LitData StreamingDataset<br/>(or cached fallback)"]
    end

    subgraph Model
      LD --> M["MarsDepthFM<br/>(SD2.1 UNet backend)"]
      NS["Flow-matching noise<br/>(q_sample)"] --> M
      M --> V[("velocity prediction v_θ")]
    end

    subgraph Losses
      V --> L1["Velocity MSE"]
      V --> L2["Normals loss"]
      V --> L3["Multi-scale gradient"]
      V --> L4["Photoclinometric<br/>(Lunar-Lambert)"]
      V --> L5["Ordinal ranking"]
    end

    L1 & L2 & L3 & L4 & L5 --> CL["CombinedLoss"]
    CL --> OPT["Optimizer + EMA"]
    OPT --> CK1[("ckpt: best RMSE")]
    OPT --> CK2[("ckpt: best photometric")]
```

## Rendered pipeline (Graphviz)

CI renders three rich pipeline figures from
`scripts/architecture/depthfm_pipeline_diagram.py` — they embed real adapter outputs as image
nodes (GMRF fill, TIN/seam detection, sun-vector OLS, flow evolution, normals, FFT). When
present, they are written to:

- `docs/diagrams/pipeline_overview.svg`
- `docs/diagrams/adapter_detail.svg`
- `docs/diagrams/training_step.svg`

To produce them locally:

```bash
uv run python scripts/architecture/depthfm_pipeline_diagram.py
```

## Key files

| Concern                       | File                                        |
|-------------------------------|---------------------------------------------|
| Training entry point          | `src/depth_fm/training/train_lightning.py`  |
| Lightning loop / EMA / ckpt   | `src/depth_fm/training/lightning_module.py` |
| Model wrapper                 | `src/depth_fm/models/mars_depthfm.py`       |
| Flow-matching noise schedule  | `src/depth_fm/flow/noise.py`                |
| Combined loss                 | `src/depth_fm/objectives/losses.py`         |
| Metrics (RMSE, AbsRel, photo) | `src/depth_fm/objectives/metrics.py`        |
| Data adapter                  | `src/depth_fm/data/adapter.py`              |
| LitData streaming             | `src/depth_fm/data/datamodule.py`           |

## Dual-checkpoint strategy

The Lightning module tracks **two separate "best" checkpoints**:

1. Best validation **RMSE** — the standard depth metric.
2. Best **photometric consistency** — Lunar-Lambert rendering MSE against the orthoimage.

These often diverge because RMSE rewards mean correctness while photometric rewards local
surface-normal fidelity. Keeping both lets us pick the right model per downstream task.

## Round-robin figure distribution

Heavy validation viz (e.g. XGBoost failure prediction, residual analysis, Pareto plots) is
distributed round-robin across DDP ranks to avoid stalling rank 0. See
`src/depth_fm/viz/debug_viz.py`.
