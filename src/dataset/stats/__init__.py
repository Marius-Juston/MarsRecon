"""Dataset statistics (per-channel mean/std/quantiles, histograms).

Two entry points:

* `compute_stats.py`         — live GDAL reads, multi-GPU Welford accumulation.
* `compute_stats_litdata.py` — fast NumPy-only path over pre-built LitData chunks.
"""
