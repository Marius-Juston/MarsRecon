"""Tests for offline per-patch JSONL augment loader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from stage_b.patch_text_augment import (
    load_patch_text_augment_jsonl,
    merge_patch_text_augment_into_patch_records,
)


def test_load_patch_text_augment_jsonl_roundtrip() -> None:
    lines = [
        json.dumps({"patch_id": "p1", "augment": "dune field"}),
        json.dumps({"patch_id": "p2", "augment": "crater rim"}),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        path = Path(fh.name)
    try:
        m = load_patch_text_augment_jsonl(path)
        assert m["p1"] == "dune field"
        assert m["p2"] == "crater rim"
    finally:
        path.unlink(missing_ok=True)


def test_merge_into_patch_records_updates_rationale_column() -> None:
    pr = pd.DataFrame(
        {
            "patch_id": ["a", "b"],
            "rationale_raw": ["x", "y"],
        }
    )
    m = {"a": "foo"}
    out = merge_patch_text_augment_into_patch_records(pr, m, sep=" | ")
    assert out["rationale_raw"].tolist() == ["x | foo", "y"]
