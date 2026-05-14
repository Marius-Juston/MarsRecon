"""Mars DepthFM — flow-matching DTM generation from monocular Mars imagery.

Public API:

    from depth_fm import MarsDepthFM, build_model
    from depth_fm import DepthFMLightningModule

Subpackages:

* `models/`      — `MarsDepthFM` wrapper + CompVis UNet backbone + experimental variants.
* `training/`    — Lightning module and `train_lightning.py` CLI entry point.
* `data/`        — Map-style adapter, LitData streaming datamodule, normalization (scalers).
* `objectives/`  — Loss functions and evaluation metrics.
* `flow/`        — Flow-matching noise schedule.
* `viz/`         — Publication figures (`train_viz`) and training-side diagnostics (`debug_viz`).
"""

from depth_fm.models.mars_depthfm import MarsDepthFM, build_model, load_sd21_backend
from depth_fm.training.lightning_module import DepthFMLightningModule, FasterEMAWeightAveraging

__all__ = [
    "MarsDepthFM",
    "build_model",
    "load_sd21_backend",
    "DepthFMLightningModule",
    "FasterEMAWeightAveraging",
]
