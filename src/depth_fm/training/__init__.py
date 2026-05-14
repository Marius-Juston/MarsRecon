"""Training loop and Lightning module for DepthFM.

* `lightning_module.py` — `DepthFMLightningModule` (train/val/test steps, EMA, checkpoint strategy).
* `train_lightning.py`  — CLI entry point. Invoked via `torchrun -m depth_fm.training.train_lightning`.
"""

from depth_fm.training.lightning_module import DepthFMLightningModule, FasterEMAWeightAveraging

__all__ = ["DepthFMLightningModule", "FasterEMAWeightAveraging"]
