#!/usr/bin/env bash
set -euo pipefail

# Evaluate Stage-2 text <-> MAE alignment checkpoint.
# Usage:
#   src/stage2/run_evaluate_text_mae.sh /path/to/text_mae_alignment.pt [extra args...]
# Example:
#   src/stage2/run_evaluate_text_mae.sh stage2_align_runs/text_mae_alignment.pt --max-patches 2048

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <alignment-checkpoint-path> [extra args...]"
  exit 1
fi

ALIGN_CHECKPOINT="$1"
shift

python "src/stage2/evaluate_text_mae_alignment.py" \
  --alignment-checkpoint "${ALIGN_CHECKPOINT}" \
  --root "/scratch/mars_hirise" \
  --bbox -136 12 -124 24 \
  --patch-size-deg 0.005 \
  --image-size 64 \
  --patch-size-px 16 \
  --batch-size 32 \
  --max-patches 1024 \
  "$@"
