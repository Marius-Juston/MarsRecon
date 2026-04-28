# Offline per-patch text augment (`sample_olympus_384_patch_augments_v1.jsonl`)

- **384 lines**, `train`-split patch IDs sampled with `--seed 42` from Olympus manifests.
- Augment tails are generic geomorphology-style phrases for **pipeline testing** only; swap for Cursor-authored Mars-specific wording for real experiments.
- Regenerate:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_sample_patch_text_augment_jsonl.py \
  --patch-records-pkl "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_patch_records.pkl" \
  --split-manifest "/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1/olympus_full_splits.csv" \
  --split train --seed 42 --max-patches 384 \
  --out assets/offline_augment/sample_olympus_384_patch_augments_v1.jsonl
```

Pass to training:

```bash
--patch-text-augment-jsonl "/home/USER/Documents/MarsRecon/assets/offline_augment/sample_olympus_384_patch_augments_v1.jsonl"
```

Only patches listed in this file receive an augmented string; overlap with `--max-patches` caps determines how many augmented rows participate per run.
