"""Tests for Stage A MAE reconstruction preview utilities."""

from __future__ import annotations

import pathlib
import sys

import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from marsclip_mae import MarsMAEOutput
from visualize_marsclip_mae import save_mae_reconstruction_preview


def test_save_mae_reconstruction_preview_writes_png(tmp_path):
    samples = [
        {
            "image": torch.rand(3, 8, 8),
            "valid_mask": torch.ones(8, 8, dtype=torch.bool),
            "rationale_raw": "Olympus Mons patch",
            "metadata": {"obs_id": "OBS_A", "overall_valid_fraction": 1.0},
        }
    ]
    output = MarsMAEOutput(
        encoded_tokens=torch.zeros(1, 4, 8),
        pooled_embedding=torch.zeros(1, 8),
        patch_valid_fraction=torch.ones(1, 4),
        patch_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        visible_mask=torch.tensor([[True, False, True, False]], dtype=torch.bool),
        masked_valid_mask=torch.tensor([[False, True, False, True]], dtype=torch.bool),
        valid_pixel_mask=torch.ones(1, 4, 48, dtype=torch.bool),
        loss_mask=torch.ones(1, 4, 48, dtype=torch.bool),
        reconstruction=torch.rand(1, 4, 48),
        loss=torch.tensor(0.1234),
    )
    out_path = tmp_path / "mae_preview.png"

    result = save_mae_reconstruction_preview(
        samples,
        output,
        out_path,
        patch_size=4,
        max_items=1,
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
