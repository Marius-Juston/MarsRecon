#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Mars DepthFM Training — Lightning on 4x A6000 + 128 cores
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# 1. Define Defaults
CONFIG="configs/train_hirise.yaml"
N_RUNS="1"
SEED="42"
EXTRA_ARGS=()

# 2. Parse arguments flexibly
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --n_runs)
      N_RUNS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      # Collect any unknown flags (e.g., --analyze_masks_only, --view_thumbnails)
      # or OmegaConf overrides (e.g., training.batch_size=4)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

echo "═══════════════════════════════════════════════════════════════"
echo "  Mars DepthFM Lightning Training"
echo "  Config:   ${CONFIG}"
echo "  Runs:     ${N_RUNS}"
echo "  Seed:     ${SEED}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  Extras:   ${EXTRA_ARGS[*]}"
fi
echo "═══════════════════════════════════════════════════════════════"

export PYTHONPATH=src
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export TQDM_MININTERVAL=1

# Increase NCCL timeout to surface real deadlocks instead of silent hangs
export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Reduce memory fragmentation on large-VRAM cards (A6000 = 48 GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Execute Python using torchrun
uv run torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m depth_fm.train_lightning \
    --config "${CONFIG}" \
    --n_runs "${N_RUNS}" \
    --seed "${SEED}" \
    ${EXTRA_ARGS[@]:+"${EXTRA_ARGS[@]}"}

echo "Training complete. Check outputs/ for figures and metrics."