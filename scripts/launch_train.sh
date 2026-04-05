#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Mars DepthFM Training — Lightning on 4x A6000 + 128 cores
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONFIG="${1:-configs/train_hirise.yaml}"
N_RUNS="${2:-1}"
SEED="${3:-42}"

echo "═══════════════════════════════════════════════════════════════"
echo "  Mars DepthFM Lightning Training"
echo "  Config:   ${CONFIG}"
echo "  Runs:     ${N_RUNS}"
echo "  Seed:     ${SEED}"
echo "═══════════════════════════════════════════════════════════════"

export PYTHONPATH=src
export OMP_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
# Reduce memory fragmentation on large-VRAM cards (A6000 = 48 GB)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python -m depth_fm.train_lightning \
    --config "${CONFIG}" \
    --n_runs "${N_RUNS}" \
    --seed "${SEED}" \
    "${@:4}"

echo "Training complete. Check outputs/ for figures and metrics."
