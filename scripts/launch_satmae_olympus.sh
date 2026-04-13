#!/usr/bin/env bash
set -euo pipefail

# Scratch-backed Stage A SatMAE launcher for Olympus Mons.
#
# This script does two things:
#   1. Builds reusable split/patch-record assets if they do not already exist.
#   2. Launches a single-GPU SatMAE training run with organized scratch outputs.
#
# Notes:
# - The current SatMAE trainer in this branch is single-GPU. Set CUDA_VISIBLE_DEVICES
#   to the specific GPU you want to use, e.g. CUDA_VISIBLE_DEVICES=2.
# - Once a proper DDP path is added, this script can be adapted to torchrun.

ROOT="${ROOT:-/scratch/mars_hirise}"
OUT_ROOT="${OUT_ROOT:-/scratch/marsrecon_runs/stage_a/satmae}"
ASSET_ROOT="${ASSET_ROOT:-/scratch/marsrecon_runs/stage_a/assets/olympus_color_only_v1}"
NORMALIZATION_PATH="${NORMALIZATION_PATH:-dataset_stats/image/dataset_stats.json}"
CACHE_ROOT="${CACHE_ROOT:-${ASSET_ROOT}/patch_cache_norm_v1}"
LITDATA_ROOT="${LITDATA_ROOT:-${ASSET_ROOT}/litdata_cache_v1}"

RUN_NAME="${RUN_NAME:-olympus_satmae_vit_base_run1}"
MODEL="${MODEL:-mae_vit_base_patch16}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-24}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
CACHE_WORKERS="${CACHE_WORKERS:-0}"
CACHE_PREFETCH_FACTOR="${CACHE_PREFETCH_FACTOR:-2}"
CACHE_PROGRESS_INTERVAL="${CACHE_PROGRESS_INTERVAL:-250}"
FILTER_INVALID_PATCHES="${FILTER_INVALID_PATCHES:-1}"
DOMINANT_OBS_ONLY="${DOMINANT_OBS_ONLY:-1}"
CACHE_INCLUDE_TEST="${CACHE_INCLUDE_TEST:-0}"
USE_LITDATA="${USE_LITDATA:-1}"
LITDATA_WORKERS="${LITDATA_WORKERS:-24}"
LITDATA_CHUNK_BYTES="${LITDATA_CHUNK_BYTES:-256MB}"
PROGRESS_LOG_INTERVAL="${PROGRESS_LOG_INTERVAL:-1}"
MAX_PATCHES="${MAX_PATCHES:-}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-16}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-MarsRecon}"
WANDB_ENTITY="${WANDB_ENTITY:-akshayn3-auvsl}"
WANDB_DIR="${WANDB_DIR:-}"

PATCH_SIZE_DEG="${PATCH_SIZE_DEG:-0.005}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
PATCH_SIZE_PX="${PATCH_SIZE_PX:-16}"
MASK_RATIO="${MASK_RATIO:-0.75}"
BLR="${BLR:-1.5e-4}"
MIN_LR="${MIN_LR:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"

MANIFEST_PATH="${MANIFEST_PATH:-${ASSET_ROOT}/olympus_full_splits.csv}"
PATCH_RECORDS_PATH="${PATCH_RECORDS_PATH:-${ASSET_ROOT}/olympus_full_patch_records.pkl}"

BBOX_LON_MIN="${BBOX_LON_MIN:--136}"
BBOX_LAT_MIN="${BBOX_LAT_MIN:-12}"
BBOX_LON_MAX="${BBOX_LON_MAX:--124}"
BBOX_LAT_MAX="${BBOX_LAT_MAX:-24}"

mkdir -p "${ASSET_ROOT}" "${OUT_ROOT}"

if [[ ! -f "${MANIFEST_PATH}" || ! -f "${PATCH_RECORDS_PATH}" ]]; then
  echo "[satmae] Building Olympus Mons split manifest and patch-record cache..."
  BUILD_CMD=(
    .venv/bin/python src/clip/build_marsclip_split_manifest.py
    --root "${ROOT}"
    --bbox "${BBOX_LON_MIN}" "${BBOX_LAT_MIN}" "${BBOX_LON_MAX}" "${BBOX_LAT_MAX}"
    --patch-size-deg "${PATCH_SIZE_DEG}"
    --image-size "${IMAGE_SIZE}"
    --color-only
    --out-manifest "${MANIFEST_PATH}"
    --out-patch-records "${PATCH_RECORDS_PATH}"
  )
  if [[ -n "${MAX_PATCHES}" ]]; then
    BUILD_CMD+=(--max-patches "${MAX_PATCHES}")
  fi
  "${BUILD_CMD[@]}"
fi

resolve_litdata_cache_dir() {
  PYTHONPATH=src .venv/bin/python - <<PY
import json
from pathlib import Path

root = Path(${LITDATA_ROOT@Q})
required_splits = ["train", "val", "test"] if ${CACHE_INCLUDE_TEST@Q} == "1" else ["train", "val"]
expected = {
    "root": ${ROOT@Q},
    "bbox": [float(${BBOX_LON_MIN}), float(${BBOX_LAT_MIN}), float(${BBOX_LON_MAX}), float(${BBOX_LAT_MAX})],
    "patch_size_deg": float(${PATCH_SIZE_DEG}),
    "image_size": int(${IMAGE_SIZE}),
    "split_manifest": ${MANIFEST_PATH@Q},
    "patch_records_path": ${PATCH_RECORDS_PATH@Q},
    "color_only": True,
    "dataset_normalize": True,
    "dataset_normalization_path": ${NORMALIZATION_PATH@Q},
    "filter_invalid_patches": ${FILTER_INVALID_PATCHES@Q} == "1",
    "dominant_obs_only": ${DOMINANT_OBS_ONLY@Q} == "1",
    "required_splits": required_splits,
}

for summary_path in sorted(root.glob("*/litdata_summary.json")):
    data = json.loads(summary_path.read_text())
    candidate = {key: data.get(key) for key in expected}
    if candidate != expected:
        continue
    parent = summary_path.parent
    if all((parent / split / "_SUCCESS").exists() for split in required_splits):
        print(parent)
        raise SystemExit(0)

raise SystemExit(0)
PY
}

if [[ "${USE_LITDATA}" == "1" ]]; then
  LITDATA_CACHE_DIR="$(resolve_litdata_cache_dir)"
  if [[ "${CACHE_INCLUDE_TEST}" == "1" ]]; then
    CACHE_READY=1
    [[ -f "${LITDATA_CACHE_DIR}/train/_SUCCESS" ]] || CACHE_READY=0
    [[ -f "${LITDATA_CACHE_DIR}/val/_SUCCESS" ]] || CACHE_READY=0
    [[ -f "${LITDATA_CACHE_DIR}/test/_SUCCESS" ]] || CACHE_READY=0
  else
    CACHE_READY=1
    [[ -f "${LITDATA_CACHE_DIR}/train/_SUCCESS" ]] || CACHE_READY=0
    [[ -f "${LITDATA_CACHE_DIR}/val/_SUCCESS" ]] || CACHE_READY=0
  fi
else
  if [[ "${CACHE_INCLUDE_TEST}" == "1" ]]; then
    CACHE_READY=1
    [[ -f "${CACHE_ROOT}/train/_SUCCESS" ]] || CACHE_READY=0
    [[ -f "${CACHE_ROOT}/val/_SUCCESS" ]] || CACHE_READY=0
    [[ -f "${CACHE_ROOT}/test/_SUCCESS" ]] || CACHE_READY=0
  else
    CACHE_READY=1
    [[ -f "${CACHE_ROOT}/train/_SUCCESS" ]] || CACHE_READY=0
    [[ -f "${CACHE_ROOT}/val/_SUCCESS" ]] || CACHE_READY=0
  fi
fi

if [[ "${CACHE_READY}" != "1" ]]; then
  if [[ "${USE_LITDATA}" == "1" ]]; then
    echo "[satmae] Building LitData-backed Mars patch splits..."
    CACHE_CMD=(
      .venv/bin/python src/clip/build_marsclip_litdata.py
      --root "${ROOT}"
      --bbox "${BBOX_LON_MIN}" "${BBOX_LAT_MIN}" "${BBOX_LON_MAX}" "${BBOX_LAT_MAX}"
      --patch-size-deg "${PATCH_SIZE_DEG}"
      --image-size "${IMAGE_SIZE}"
      --split-manifest "${MANIFEST_PATH}"
      --patch-records-path "${PATCH_RECORDS_PATH}"
      --out-root "${LITDATA_ROOT}"
      --split-mode holdout
      --color-only
      --dataset-normalize
      --dataset-normalization-path "${NORMALIZATION_PATH}"
      --workers "${LITDATA_WORKERS}"
      --chunk-bytes "${LITDATA_CHUNK_BYTES}"
      --progress-interval "${CACHE_PROGRESS_INTERVAL}"
    )
  else
    echo "[satmae] Building cache-backed Mars patch splits..."
    CACHE_CMD=(
      .venv/bin/python src/clip/build_marsclip_cache.py
      --root "${ROOT}"
      --bbox "${BBOX_LON_MIN}" "${BBOX_LAT_MIN}" "${BBOX_LON_MAX}" "${BBOX_LAT_MAX}"
      --patch-size-deg "${PATCH_SIZE_DEG}"
      --image-size "${IMAGE_SIZE}"
      --split-manifest "${MANIFEST_PATH}"
      --out-root "${CACHE_ROOT}"
      --split-mode holdout
      --color-only
      --dataset-normalize
      --dataset-normalization-path "${NORMALIZATION_PATH}"
      --num-workers "${CACHE_WORKERS}"
      --prefetch-factor "${CACHE_PREFETCH_FACTOR}"
      --progress-interval "${CACHE_PROGRESS_INTERVAL}"
      --patch-records-path "${PATCH_RECORDS_PATH}"
    )
  fi
  if [[ "${FILTER_INVALID_PATCHES}" == "1" ]]; then
    CACHE_CMD+=(--filter-invalid-patches)
  else
    CACHE_CMD+=(--no-filter-invalid-patches)
  fi
  if [[ "${DOMINANT_OBS_ONLY}" == "1" ]]; then
    CACHE_CMD+=(--dominant-obs-only)
  else
    CACHE_CMD+=(--no-dominant-obs-only)
  fi
  if [[ "${CACHE_INCLUDE_TEST}" == "1" ]]; then
    CACHE_CMD+=(--required-splits train val test)
  else
    CACHE_CMD+=(--required-splits train val)
  fi
  if [[ -n "${MAX_PATCHES}" ]]; then
    CACHE_CMD+=(--max-patches "${MAX_PATCHES}")
  fi
  "${CACHE_CMD[@]}"
  if [[ "${USE_LITDATA}" == "1" ]]; then
    LITDATA_CACHE_DIR="$(resolve_litdata_cache_dir)"
    if [[ -z "${LITDATA_CACHE_DIR}" ]]; then
      echo "[satmae] ERROR: LitData cache build completed but no matching cache directory was found."
      exit 1
    fi
  fi
fi

echo "[satmae] Launching Mars SatMAE run..."
echo "[satmae] ROOT=${ROOT}"
echo "[satmae] OUT_ROOT=${OUT_ROOT}"
echo "[satmae] RUN_NAME=${RUN_NAME}"
echo "[satmae] MANIFEST_PATH=${MANIFEST_PATH}"
echo "[satmae] PATCH_RECORDS_PATH=${PATCH_RECORDS_PATH}"
echo "[satmae] CACHE_ROOT=${CACHE_ROOT}"
echo "[satmae] LITDATA_ROOT=${LITDATA_ROOT}"
echo "[satmae] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[satmae] NUM_WORKERS=${NUM_WORKERS}"
echo "[satmae] PREFETCH_FACTOR=${PREFETCH_FACTOR}"
echo "[satmae] CACHE_WORKERS=${CACHE_WORKERS}"
echo "[satmae] LITDATA_WORKERS=${LITDATA_WORKERS}"
echo "[satmae] FILTER_INVALID_PATCHES=${FILTER_INVALID_PATCHES}"
echo "[satmae] DOMINANT_OBS_ONLY=${DOMINANT_OBS_ONLY}"
echo "[satmae] CACHE_INCLUDE_TEST=${CACHE_INCLUDE_TEST}"
echo "[satmae] USE_LITDATA=${USE_LITDATA}"
if [[ "${USE_LITDATA}" == "1" ]]; then
  echo "[satmae] LITDATA_CACHE_DIR=${LITDATA_CACHE_DIR}"
fi
echo "[satmae] WANDB_MODE=${WANDB_MODE}"
echo "[satmae] WANDB_ENTITY=${WANDB_ENTITY:-<unset>}"

TRAIN_CMD=(
  .venv/bin/python src/clip/train_marsclip_satmae.py
  --root "${ROOT}"
  --bbox "${BBOX_LON_MIN}" "${BBOX_LAT_MIN}" "${BBOX_LON_MAX}" "${BBOX_LAT_MAX}"
  --patch-size-deg "${PATCH_SIZE_DEG}"
  --image-size "${IMAGE_SIZE}"
  --patch-size-px "${PATCH_SIZE_PX}"
  --split-manifest "${MANIFEST_PATH}"
  --patch-records-path "${PATCH_RECORDS_PATH}"
  --split-mode holdout
  --color-only
  --dataset-normalize
  --dataset-normalization-path "${NORMALIZATION_PATH}"
  --model "${MODEL}"
  --norm-pix-loss
  --mask-ratio "${MASK_RATIO}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --blr "${BLR}"
  --min-lr "${MIN_LR}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --weight-decay "${WEIGHT_DECAY}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --persistent-workers
  --pin-memory
  --checkpoint-every 1
  --val-max-batches "${VAL_MAX_BATCHES}"
  --progress-log-interval "${PROGRESS_LOG_INTERVAL}"
  --wandb-mode "${WANDB_MODE}"
  --wandb-project "${WANDB_PROJECT}"
  --save-reconstructions
  --out-root "${OUT_ROOT}"
  --run-name "${RUN_NAME}"
  --device cuda
)

if [[ "${USE_LITDATA}" == "1" ]]; then
  TRAIN_CMD+=(--litdata-root "${LITDATA_CACHE_DIR}")
else
  TRAIN_CMD+=(--cached-data-root "${CACHE_ROOT}")
fi

if [[ "${FILTER_INVALID_PATCHES}" == "1" ]]; then
  TRAIN_CMD+=(--filter-invalid-patches)
else
  TRAIN_CMD+=(--no-filter-invalid-patches)
fi

if [[ "${DOMINANT_OBS_ONLY}" == "1" ]]; then
  TRAIN_CMD+=(--dominant-obs-only)
else
  TRAIN_CMD+=(--no-dominant-obs-only)
fi

if [[ -n "${WANDB_DIR}" ]]; then
  TRAIN_CMD+=(--wandb-dir "${WANDB_DIR}")
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  TRAIN_CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ -n "${MAX_PATCHES}" ]]; then
  TRAIN_CMD+=(--max-patches "${MAX_PATCHES}")
fi

"${TRAIN_CMD[@]}"
