"""Tests for offline per-patch JSONL augment loader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from stage_b.patch_text_augment import load_patch_text_augment_jsonl


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
