"""DepthFM visualization.

* `train_viz.py` — publication-quality figures (triptychs, error maps, normals).
* `debug_viz.py` — training-side analysis: imported wholesale by `train_lightning.py`
  via `from depth_fm.viz.debug_viz import *`; contains XGBoost patch-failure
  diagnostics, Pareto frontier plots, residual analysis.
"""
