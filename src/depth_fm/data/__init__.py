"""Data loading and normalization for DepthFM training.

* `adapter.py`    — `DepthFMHiRISEAdapterCached` map-style wrapper over `MarsHiRISEDTM`.
* `datamodule.py` — Lightning DataModule + LitData StreamingDataset path.
* `scalers.py`    — Elevation normalization strategies (Global Fixed / Global Log / Adaptive).
"""
