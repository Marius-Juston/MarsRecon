"""Offline per-patch text augmentations for MarsCLIP (human- or Cursor-authored).

Format: JSON Lines (``.jsonl``), one object per line::

    {\"patch_id\": \"<id matching patch_records.patch_id>\", \"augment\": \"short phrase\"}

The trainer concatenates ``augment`` to the observation rationale for that patch
(with a configurable separator). No LLM runs at train time."""

from __future__ import annotations

import json
import pathlib
from typing import Any


def load_patch_text_augment_jsonl(path: pathlib.Path | str) -> dict[str, str]:
    """Load a patch_id -> augment string map from JSONL.

    Each line must be a JSON object with ``patch_id`` and ``augment`` keys.
    Later lines override earlier ones for the same ``patch_id``.
    """
    p = pathlib.Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"patch text augment file not found: {p}")
    out: dict[str, str] = {}
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSON: {e}") from e
            pid = str(row.get("patch_id", "")).strip()
            aug = str(row.get("augment", "")).strip()
            if not pid:
                raise ValueError(f"{p}:{lineno}: missing patch_id")
            if aug:
                out[pid] = aug
    return out
