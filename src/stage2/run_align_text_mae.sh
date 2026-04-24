#!/usr/bin/env bash
set -euo pipefail

# Stage-2 text <-> MAE embedding alignment launcher.
# Usage:
#   src/stage2/run_align_text_mae.sh /path/to/checkpoint.pt [extra args...]
# Example:
#   src/stage2/run_align_text_mae.sh /scratch/marsrecon_runs/stage_a/satmae/checkpoint.pt --epochs 10

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <mae-checkpoint-path> [extra args...]"
  exit 1
fi

MAE_CHECKPOINT="$1"
shift

uv run python "src/stage2/align_text_mae_embeddings.py" \
  --mae-checkpoint "${MAE_CHECKPOINT}" \
  --root "/scratch/mars_hirise" \
  --bbox -136 12 -124 24 \
  --patch-size-deg 0.005 \
  --image-size 64 \
  --patch-size-px 16 \
  --batch-size 16 \
  --epochs 5 \
  --embed-dim 256 \
  --out-dir "stage2_align_runs" \
  "$@"
