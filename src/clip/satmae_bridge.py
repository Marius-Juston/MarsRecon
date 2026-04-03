"""Compatibility bridge for using the SatMAE submodule inside MarsRecon."""

from __future__ import annotations

import importlib
import inspect
import pathlib
import sys
from types import ModuleType
from typing import Any

import numpy as np


def satmae_root() -> pathlib.Path:
    """Return the checked-out SatMAE submodule root."""
    return pathlib.Path(__file__).resolve().parents[2] / "third_party" / "SatMAE"


def ensure_satmae_on_path() -> pathlib.Path:
    """Ensure the SatMAE submodule root is importable."""
    root = satmae_root()
    if not root.exists():
        raise FileNotFoundError(
            f"SatMAE submodule not found at {root}. Initialize the submodule before training."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _install_timm_block_compat() -> None:
    """Patch newer timm Block signatures to accept SatMAE's older qk_scale arg."""
    from timm.models import vision_transformer as vit

    if "qk_scale" in inspect.signature(vit.Block).parameters:
        return
    if getattr(vit.Block, "__name__", "") == "_SatMAECompatBlock":
        return

    original_block = vit.Block

    class _SatMAECompatBlock(original_block):
        def __init__(self, *args: Any, qk_scale: float | None = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

    vit.Block = _SatMAECompatBlock


def _install_numpy_compat() -> None:
    """Restore deprecated NumPy aliases that SatMAE still references."""
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]


def load_satmae_module(module_name: str) -> ModuleType:
    """Import a module from the SatMAE submodule with local compatibility patches."""
    ensure_satmae_on_path()
    _install_numpy_compat()
    _install_timm_block_compat()
    return importlib.import_module(module_name)


def load_satmae_models_mae() -> ModuleType:
    """Load the vanilla SatMAE MAE model module."""
    return load_satmae_module("models_mae")


def load_satmae_lr_sched() -> ModuleType:
    """Load the SatMAE learning-rate schedule helper module."""
    return load_satmae_module("util.lr_sched")


def build_satmae_model(
    model_name: str | None = "mae_vit_base_patch16",
    **kwargs: Any,
):
    """Build a SatMAE model from the submodule, with optional direct ctor kwargs."""
    module = load_satmae_models_mae()
    if model_name is None:
        return module.MaskedAutoencoderViT(**kwargs)
    if not hasattr(module, model_name):
        available = sorted(name for name in dir(module) if name.startswith("mae_vit_"))
        raise ValueError(f"Unknown SatMAE model '{model_name}'. Available presets: {', '.join(available)}")
    return getattr(module, model_name)(**kwargs)
