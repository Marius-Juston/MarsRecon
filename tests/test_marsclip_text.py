"""Tests for MarsCLIP text composition and batching."""

from __future__ import annotations

import pathlib
import sys

import torch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from marsclip_text import (
    MarsCLIPBatchCollator,
    SimpleTextTokenizer,
    build_tokenizer_from_manifest,
    compose_rationale_text,
)


def _sample(raw: str, expanded: str | None = None) -> dict[str, object]:
    return {
        "image": torch.ones(3, 8, 8),
        "valid_mask": torch.ones(8, 8, dtype=torch.bool),
        "rationale_raw": raw,
        "rationale_expanded": expanded,
        "geo_features": torch.ones(8),
        "scale_features": torch.ones(8) * 2,
        "metadata": {
            "viewing_features": torch.ones(13),
            "band_presence_mask": torch.tensor([True, True, True]),
            "band_valid_fraction": torch.tensor([1.0, 1.0, 1.0]),
            "overall_valid_fraction": 1.0,
            "obs_id": "OBS",
        },
    }


def test_compose_rationale_text_modes():
    assert compose_rationale_text("raw", "expanded", text_mode="raw") == "raw"
    assert compose_rationale_text("raw", "expanded", text_mode="expanded") == "expanded"
    assert "Expanded context:" in compose_rationale_text(
        "raw",
        "expanded",
        text_mode="raw_plus_expanded",
    )
    assert compose_rationale_text("raw", None, text_mode="expanded") == "raw"


def test_tokenizer_build_and_batch_encode():
    tok = SimpleTextTokenizer.build(["Olympus Mons lava", "Karzok crater"])
    ids, mask = tok.batch_encode(["Olympus Mons lava"], max_length=8)

    assert tok.vocab_size >= 6
    assert ids.shape == (1, 8)
    assert mask.shape == (1, 8)
    assert mask[0, 0]
    assert ids[0, 0] == tok.bos_token_id


def test_build_tokenizer_from_manifest_uses_expanded_text():
    import pandas as pd

    manifest = pd.DataFrame(
        [
            {
                "rationale_desc": "raw text",
                "rationale_expanded": "expanded geology context",
            }
        ]
    )
    tok = build_tokenizer_from_manifest(manifest, text_mode="raw_plus_expanded")
    assert "expanded" in tok.vocab


def test_collator_batches_tensors_and_tokenizes_text():
    tokenizer = SimpleTextTokenizer.build(
        [
            compose_rationale_text("raw a", "expanded a"),
            compose_rationale_text("raw b", None),
        ]
    )
    collator = MarsCLIPBatchCollator(tokenizer, max_length=12)
    batch = collator([_sample("raw a", "expanded a"), _sample("raw b", None)])

    assert batch["image"].shape == (2, 3, 8, 8)
    assert batch["valid_mask"].shape == (2, 8, 8)
    assert batch["input_ids"].shape == (2, 12)
    assert batch["attention_mask"].shape == (2, 12)
    assert batch["geo_features"].shape == (2, 8)
    assert batch["scale_features"].shape == (2, 8)
    assert batch["viewing_features"].shape == (2, 13)
    assert batch["quality_features"].shape == (2, 7)
    assert "Expanded context:" in batch["text"][0]
    assert batch["text"][1] == "raw b"


def test_compose_rationale_text_raises_for_unknown_mode():
    import pytest
    with pytest.raises(ValueError, match="Unsupported text_mode"):
        compose_rationale_text("raw text", None, text_mode="invalid_mode")
