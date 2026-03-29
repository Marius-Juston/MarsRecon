"""Tests for MarsCLIP preview utilities."""

from __future__ import annotations

import pathlib
import sys

import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from visualize_marsclip import save_sample_preview


def test_save_sample_preview_writes_png(tmp_path):
    sample = {
        "image": torch.rand(3, 8, 8),
        "valid_mask": torch.ones(8, 8, dtype=torch.bool),
        "rationale_raw": "Olympus Mons lava channels",
        "rationale_expanded": "Expanded geology context",
        "metadata": {
            "obs_id": "OBS_A",
            "overall_valid_fraction": 1.0,
        },
    }
    out_path = tmp_path / "preview.png"

    result = save_sample_preview([sample], out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
