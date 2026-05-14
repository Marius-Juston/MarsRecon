# Build LitData chunks

Training I/O is dominated by GDAL random reads. To saturate multiple A6000s,
MarsRecon pre-builds [LitData](https://github.com/Lightning-AI/litdata) chunks once and streams
them at training time.

## Two builders

| Script                                  | Output format                                                               |
|-----------------------------------------|-----------------------------------------------------------------------------|
| `scripts/training/build_litdata.py`     | **Processed**: includes sun-vector estimates, topographic residuals, fills. |
| `scripts/training/build_litdata_raw.py` | **Raw**: elevation + bounds + CRS only — for ablations / custom pipelines.  |

Both write to the directory specified by `data.litdata_root` in your config.

## Run

```bash
PYTHONPATH=src uv run python scripts/training/build_litdata.py \
    --config configs/train_hirise.yaml \
    --workers 96
```

`--workers` should match your CPU count. The script:

1. Instantiates the same `MarsHiRISEDTM` + `HiRISEGeoSampler` the trainer will use.
2. Iterates all patches in the chosen split.
3. Runs the same adapter (`DepthFMHiRISEAdapterCached`) — sun-vector, fills, normalization.
4. Serializes each patch to a binary chunk.

## Verifying chunks

```bash
PYTHONPATH=src uv run python -m dataset.stats.compute_stats_litdata \
    --litdata-root /scratch/litdata/hirise --split train
```

This runs a Welford accumulator over the pre-built chunks for a sanity-check of the
per-channel statistics.

See: [`dataset.stats.compute_stats_litdata`](../reference/dataset/stats/compute_stats_litdata.md).
