#!/usr/bin/env python3
"""Sample patch_ids from Olympus patch_records + split manifest; write JSONL for --patch-text-augment-jsonl.

Example::

  PYTHONPATH=src .venv/bin/python scripts/generate_sample_patch_text_augment_jsonl.py \\
    --patch-records-pkl /scratch/.../olympus_full_patch_records.pkl \\
    --split-manifest /scratch/.../olympus_full_splits.csv \\
    --split train --seed 42 --max-patches 384 \\
    --out assets/offline_augment/sample_olympus_384_patch_augments_v1.jsonl

Augment strings are short, generic geomorphology-style hints for integration testing;
replace with Cursor-authored content for real runs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from stage_b.align_text_mae_embeddings import attach_split_manifest, load_split_manifest

AUGMENT_ROTATION: tuple[str, ...] = (
    "Olympus basalt plains; HiRISE COLOR",
    "dust mantle; aeolian streaks nearby",
    "lava flow margins; wrinkle ridge context",
    "crater rim ejecta; bright bedrock",
 "mass-wasting scarp; talus slopes",
    "yardangs; wind erosion signal",
    "caldera-adjacent fractured plains",
    "sinuous channel wall; layered terrain",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-records-pkl", type=pathlib.Path, required=True)
    parser.add_argument("--split-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument("--max-patches", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    pr: pd.DataFrame = pd.read_pickle(args.patch_records_pkl)
    splits = load_split_manifest(args.split_manifest)
    merged = attach_split_manifest(pr, splits)
    sub = merged.loc[merged["stage_b_split"].astype(str).str.lower() == args.split.lower()].copy()
    sub = sub.sample(n=min(args.max_patches, len(sub)), random_state=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(sub.itertuples(index=False)):
            pid = str(row.patch_id)
            aug = AUGMENT_ROTATION[i % len(AUGMENT_ROTATION)]
            fh.write(json.dumps({"patch_id": pid, "augment": aug}, ensure_ascii=False) + "\n")
    print(f"wrote {len(sub)} lines -> {args.out}")


if __name__ == "__main__":
    main()
