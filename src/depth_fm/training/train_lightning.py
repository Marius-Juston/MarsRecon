"""
Mars DepthFM Training — Lightning-based with multi-run evaluation.

Features:
- Uses MarsHiRISEDTM + HiRISEGeoSampler with train/val/test splits
- LitData StreamingDataset for hyper-optimized I/O (primary)
- DepthFMHiRISEAdapterCached fallback for first-time runs
- Multiple independent training runs for statistical significance
- Publication-quality figures with error bars
- Patch-level error analysis identifying failure modes
- All metrics logged to wandb with proper grouping

Hardware target: 2×128-core Ryzen, 4×A6000, 1 TB RAM, NVMe storage.

Usage:
    # Single run
    python src/train_lightning.py --config configs/train_hirise.yaml

    # Multi-run for error bars (3 seeds)
    python src/train_lightning.py --config configs/train_hirise.yaml --n_runs 3

    # K-fold cross-validation
    python src/train_lightning.py --config configs/train_hirise.yaml --n_folds 5
"""

import argparse
import concurrent.futures
import json
import logging
import math
import os
from copy import deepcopy
from typing import Callable, Optional, Any

from cache import litdata_cache_key, litdata_cache_root, run_id_hash

import cuml
import matplotlib.patches as mpatches
import xgboost as xgb
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from depth_fm.data.datamodule import _build_litdata_loaders
from depth_fm.objectives.losses import PhotoclinometricLoss, AbsoluteDepthLoss, LaplacianLoss, \
    OrdinalRankingLoss

# ---------------------------------------------------------------------------
# GLOBAL GDAL/IO OPTIMIZATIONS (For 256-Core / NVMe setups)
# ---------------------------------------------------------------------------
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "500000000"
os.environ["GDAL_CACHEMAX"] = "10%"
os.environ["GDAL_MAX_DATASET_POOL_SIZE"] = "1024"

import lightning as L
import torch.fft
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from depth_fm.data.adapter import (
    DepthFMHiRISEAdapterCached, fill_voids_gmrf, estimate_sun_vector_irls, SeamResult, detect_seam_artifact)
from depth_fm.training.lightning_module import DepthFMLightningModule, FasterEMAWeightAveraging
from depth_fm.viz.train_viz import (
    plot_convergence_curves,
    plot_metric_distributions,
    plot_multi_run_summary_table,
    set_neurips_style, plot_pareto_frontier,
)

import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation as R

from depth_fm.data.adapter import compute_topographic_residual
import torch.distributed as dist
import re
from depth_fm.viz.train_viz import plot_timestep_ablation
import pandas as pd
from tqdm import tqdm
from torch.func import vmap, grad, hessian
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import geoopt
from torch import nn
from depth_fm.viz.debug_viz import *

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Hardware-aware constants for 2×128-core Ryzen / 4×A6000 / 1 TB RAM
# ---------------------------------------------------------------------------
_TOTAL_CORES = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 256
_NUM_GPUS_DEFAULT = torch.cuda.device_count() if torch.cuda.is_available() else 1

# Workers per GPU: leave headroom for the main process and OS.
# 256 cores / 4 GPUs = 64 per GPU; cap at 48 to avoid memory pressure from
# too many rasterio/GDAL file handles.
_WORKERS_PER_GPU = min(24, max(4, (_TOTAL_CORES - 16) // max(_NUM_GPUS_DEFAULT, 1)))
_VAL_WORKERS = min(4, _WORKERS_PER_GPU)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def configure_worker_logger(worker_id):
    """Forces the spawned PyTorch worker to actually print INFO logs."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker {worker_id}] %(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


get_litdata_cache_key = litdata_cache_key  # backwards-compat alias


def _build_cached_loaders(config, split_seed: int = 42, parallel: bool = True) -> dict:
    """Fallback: build DataLoaders from DepthFMHiRISEAdapterCached (live GDAL reads)."""
    from dataset.sampling.sampler import HiRISEGeoSampler
    from dataset.core.dtm import MarsHiRISEDTM
    from torchgeo.samplers import Units

    hc = config.data.hirise
    sc = config.data.sampler
    tc = config.training

    bbox_tuple = tuple(hc.bbox) if hc.get("bbox") else None
    ortho_type = hc.get("ortho_type", "RED")
    if isinstance(ortho_type, str):
        ortho_type = [ortho_type]

    base_dataset = MarsHiRISEDTM(
        root=hc.root,
        include_ortho=hc.get("include_ortho", True),
        ortho_type=ortho_type,
        ortho_scale=hc.get("ortho_scale"),
        download=hc.get("download", False),
        bbox=bbox_tuple,
        reuse_cache=hc.get("reuse_cache", True),
        target=hc.get("target"),
        return_meta=True,
    )
    base_dataset._raw_index = None

    split_fractions = tuple(config.data.get("split_fractions", [0.8, 0.1, 0.1]))
    split_method = config.data.get("split_method", "geographic")
    split_axis = config.data.get("split_axis", "longitude")
    n_folds = config.data.get("n_folds")
    fold_idx = config.data.get("fold_idx", 0)

    common_sampler_kwargs = dict(
        size=sc.get("size", 0.009),
        units=Units.CRS,
        split_fractions=split_fractions,
        split_method=split_method,
        split_axis=split_axis,
        n_folds=n_folds,
        fold_idx=fold_idx,
        seed=split_seed,
        reuse_cache=True,
        center_mode=sc.get("center_mode", "simple")
    )

    resolution = config.data.get("resolution", 512)
    clip = config.data.get("clip", False)
    dtm_norm = config.data.get("dtm_normalization", "relative")
    stats_path = config.data.get("stats_path")
    num_workers = tc.get("num_workers", _WORKERS_PER_GPU)

    def _build_split_loader(split: str):
        is_train = split == "train"
        sampler = HiRISEGeoSampler(
            base_dataset,
            split=split,
            length=sc.get("length") if is_train else None,
            replacement=is_train,
            **common_sampler_kwargs,
        )
        adapter = DepthFMHiRISEAdapterCached(
            base_dataset=base_dataset,
            sampler=sampler,
            resolution=resolution,
            dtm_normalization=dtm_norm,
            random_flip=is_train,
            brightness_jitter=config.data.get("brightness_jitter", 0.1) if is_train else 0.0,
            stats_path=stats_path,
            clip=clip
        )
        loader = DataLoader(
            adapter,
            batch_size=tc.per_gpu_batch_size,
            shuffle=is_train,
            num_workers=num_workers if is_train else _VAL_WORKERS,
            pin_memory=tc.pin_memory,
            prefetch_factor=tc.get("prefetch_factor", 4) if is_train else 2,
            drop_last=is_train,
            persistent_workers=is_train,
            worker_init_fn=configure_worker_logger,
        )
        logger.info(
            "Cached DataLoader [%s]: %d samples, batch_size=%d, workers=%d",
            split,
            len(adapter),
            tc.per_gpu_batch_size,
            num_workers if is_train else _VAL_WORKERS,
        )
        return split, loader

    loaders = {}
    splits = ("train", "val", "test")

    if parallel:
        logger.info("Initializing cached dataloaders in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(splits)) as executor:
            future_to_split = {executor.submit(_build_split_loader, s): s for s in splits}
            for future in concurrent.futures.as_completed(future_to_split):
                split_name = future_to_split[future]
                try:
                    _, loader = future.result()
                    loaders[split_name] = loader
                except Exception as exc:
                    logger.error(f"Failed to build DataLoader for '{split_name}': {exc}")
                    raise
    else:
        for split in splits:
            _, loader = _build_split_loader(split)
            loaders[split] = loader

    return loaders


def build_dataloaders(config, split_seed: int = 42, parallel: bool = True) -> dict:
    """Build train/val/test DataLoaders.

    Priority:
        1. LitData StreamingDataset (pre-optimized binary chunks, fastest)
        2. DepthFMHiRISEAdapterCached (live GDAL reads, slower but always works)
    """

    # ── Try LitData first ──
    try:
        loaders = _build_litdata_loaders(config, split_seed=split_seed)
        logger.info("Using LitData StreamingDataset for maximum I/O throughput.")
        return loaders
    except FileNotFoundError as e:
        logger.exception("LitData not available (%s). Falling back to cached GDAL loader.")
    except Exception as e:
        logger.exception("LitData failed unexpectedly (%s). Falling back to cached GDAL loader.")

    # ── Fallback to cached adapter ──
    return _build_cached_loaders(config, split_seed=split_seed, parallel=parallel)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------


def run_test_only(
        config,
        ckpt_path: str,
        test_tag: str = "",
        seed: int = 42,
) -> dict:
    """Load a checkpoint and run only the test loop.

    Used by the ablation orchestrator to evaluate each variant from both the
    `best-rmse` and `best-photo` checkpoints, tagging the per-patch output
    files so downstream paired statistics can read both.
    """
    if config.training.get("num_gpus", 1) and "WORLD_SIZE" not in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    L.seed_everything(seed, workers=True)

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(config, split_seed=seed, parallel=config.data.get("parallel_load", False))

    module = DepthFMLightningModule(config)
    module._test_tag = test_tag

    _prec = config.training.mixed_precision
    if _prec == "bf16":
        precision = "bf16-mixed"
    elif _prec in ("f16", "fp16"):
        precision = "16-mixed"
    else:
        precision = "32-true"

    using_litdata = loaders["train"].__class__.__name__ == "StreamingDataLoader"

    csv_logger = CSVLogger(save_dir=str(output_dir), name=f"csv_test_{test_tag}" if test_tag else "csv_test")

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp" if config.training.num_gpus > 1 else "auto",
        precision=precision,
        logger=[csv_logger],
        enable_progress_bar=True,
        default_root_dir=str(output_dir),
        use_distributed_sampler=not using_litdata,
    )

    logger.info("*** TEST-ONLY mode: loading %s (tag=%r) ***", ckpt_path, test_tag)
    trainer.test(module, loaders["test"], ckpt_path=ckpt_path, weights_only=False)

    return {"output_dir": output_dir, "test_tag": test_tag}


def run_single_training(
        config,
        run_idx: int = 0,
        seed: int = 42,
        output_dir: str | Path = "outputs",
) -> dict:
    if config.training.get("num_gpus", 1) and "WORLD_SIZE" not in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        logger.warning(
            "You NEED to ensure that the DDP is already initialized for StreamingDataLoader to generate the correct "
            "dataloader length. Otherwise it will do it's dataset length based on purely the batch size rather than "
            "with the world size as well")

    if config.training.get("num_gpus", 1) and "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "You need to have DDP's WORLD_SIZE environemnt variable intialized for proper StreamingDataLoader to work run with `torchrun`")

    """Execute a single training run and return test metrics."""
    L.seed_everything(seed, workers=True)

    # Directory isolation for multi-run / K-fold
    n_folds = config.data.get("n_folds")
    fold_idx = config.data.get("fold_idx", 0)

    if n_folds is not None:
        output_dir = Path(output_dir) / f"fold_{fold_idx}" / f"run_{run_idx}"
    else:
        output_dir = Path(output_dir) / f"run_{run_idx}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip completed runs
    summary_path = output_dir / "test_summary.json"
    if summary_path.exists():
        logger.info(f"Run {run_idx} at {output_dir} is already complete. Skipping.")
        with open(summary_path, "r") as f:
            test_summary = json.load(f)

        # Load the saved dataframe so downstream plotting doesn't crash

        test_df_path = output_dir / "test_results.csv"
        test_df = pd.read_csv(test_df_path) if test_df_path.exists() else None

        # Load the timestep ablation if it exists
        ablation_path = output_dir / "timestep_ablation.json"
        timestep_ablation = {}
        if ablation_path.exists():
            with open(ablation_path, "r") as f:
                timestep_ablation = json.load(f)

        return {
            "test_summary": test_summary,
            "test_df": test_df,
            "timestep_ablation": timestep_ablation,
            "output_dir": output_dir,
            "skipped": True,
            "val_history": [],  # Empty list to prevent KeyError in multi-run convergence plots
            "test_aggregator": None  # Object is not in memory; requires a guard in the plotting function
        }

    # Build data
    # FIXME
    # MAJOR CRITICAL CONCERN YOU NEED TO ENSURE THAT DISTRIBUTED HAS ALREADY STARTED WITH THE PROPER SETUP OTHERWISE StreamingDataLoader WILL NOT BE USING THE CORRECT SIZES AS IT WILL NOT BE CALCULATING THE LENGTHS CORRECTLY!!!!!
    loaders = build_dataloaders(config, split_seed=seed, parallel=config.data.get("parallel_load", False))

    # Build model
    module = DepthFMLightningModule(config)

    # torch.compile
    cache_path = Path(config.training.get("cache_dir", ".torch_compile_cache")) / "mega_cache.pt"
    if config.model.get("torch_compile", False):
        if cache_path.exists():
            logger.info("Loading torch.compile Mega-Cache artifacts from %s", cache_path)
            try:
                with open(cache_path, "rb") as f:
                    torch.compiler.load_cache_artifacts(f.read())
            except Exception as e:
                logger.warning("Failed to load Mega-Cache artifacts: %s", e)

        compile_mode = config.model.get("torch_compile_mode", "default")
        full_graph = config.model.get("full_graph", False)
        module.model.backbone = torch.compile(module.model.backbone, mode=compile_mode, fullgraph=full_graph)
        logger.info("torch.compile enabled on UNet backbone (mode=%s)", compile_mode)

    # Callbacks — dual checkpoint strategy
    # 1. Best by standard RMSE (traditional depth estimation metric)
    best_checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="depthfm-best-rmse-{step}-{val/rmse_mean:.4f}",
        monitor="val/rmse_mean",
        mode="min",
        save_top_k=3,
        save_last=False,
    )
    # 2. Best by photometric consistency (resolution-independent quality)
    #    This may select a different model than RMSE when GT is low-resolution,
    #    since photo_consistency measures whether the predicted terrain
    #    correctly reproduces the observed shading in the high-res orthoimage.
    best_photo_checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="depthfm-best-photo-{step}-{val/photo_consistency_mean:.4f}",
        monitor="val/photo_consistency_mean",
        mode="max",
        save_top_k=2,
        save_last=False,
    )
    recovery_checkpoint = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="last",
        every_n_train_steps=config.training.save_every_steps,
        save_top_k=1,
    )
    callbacks = [LearningRateMonitor(logging_interval="step"), best_checkpoint, best_photo_checkpoint,
                 recovery_checkpoint]

    if config.training.get("use_ema", True):
        logger.info("EMA is ENABLED.")
        callbacks.append(FasterEMAWeightAveraging(
            decay=config.training.get("ema_decay", False),
            device=config.training.get("ema_device", None)
        ))
    else:
        logger.info("EMA is DISABLED.")

    if config.training.get("early_stopping_patience"):
        callbacks.append(
            EarlyStopping(monitor="val/rmse_mean", patience=config.training.early_stopping_patience, mode="min")
        )

    # Logger — W&B for dashboards plus a local CSV logger so the ablation
    # orchestrator's watchdog has a network-independent source of intermediate
    # metrics (val/rmse_mean, val/photo_consistency_mean).
    trainer_loggers: list = []
    if config.training.logger == "wandb":
        # Honor externally-supplied WANDB_RUN_ID / WANDB_RESUME so the ablation
        # orchestrator (or any wrapper) can resume the same W&B run across
        # checkpoint-aware restarts. Falls through to fresh runs if unset.
        _wandb_kwargs = {}
        _wandb_id = os.environ.get("WANDB_RUN_ID")
        if _wandb_id:
            _wandb_kwargs["id"] = _wandb_id
            _wandb_kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")
        wandb_logger = WandbLogger(
            project=config.training.project_name,
            name=f"{config.training.run_name}_run{run_idx}",
            save_dir=str(output_dir),
            group=config.training.run_name,
            tags=["mars", "depthfm", "flow-matching"],
            **_wandb_kwargs,
        )
        trainer_loggers.append(wandb_logger)
    csv_logger = CSVLogger(save_dir=str(output_dir), name="csv")
    trainer_loggers.append(csv_logger)

    # Precision
    _prec = config.training.mixed_precision
    if _prec == "bf16":
        precision = "bf16-mixed"
    elif _prec in ("f16", "fp16"):
        precision = "16-mixed"
    else:
        precision = "32-true"

    using_litdata = loaders["train"].__class__.__name__ == "StreamingDataLoader"

    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices='auto',
        strategy="ddp" if config.training.num_gpus > 1 else "auto",
        precision=precision,
        callbacks=callbacks,
        logger=trainer_loggers if trainer_loggers else False,
        val_check_interval=config.training.val_every_steps,
        log_every_n_steps=config.training.log_every_steps,
        gradient_clip_val=config.training.max_grad_norm,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        enable_progress_bar=True,
        default_root_dir=str(output_dir),
        use_distributed_sampler=not using_litdata,
    )

    # Train (with resumption)
    last_ckpt_path = output_dir / "checkpoints" / "last.ckpt"

    if config.training.enable:
        if last_ckpt_path.exists():
            logger.info(f"*** Resuming run {run_idx} from {last_ckpt_path} ***")
            trainer.fit(module, loaders["train"], loaders["val"], ckpt_path=str(last_ckpt_path), weights_only=False)
        else:
            logger.info(f"*** Starting fresh training for run {run_idx} ***")
            trainer.fit(module, loaders["train"], loaders["val"])

    best_choice = config.training.get("test_choice", "rmse")

    # Test — use best RMSE checkpoint as primary
    best_path_rmse = best_checkpoint.best_model_path
    best_photo_path = best_photo_checkpoint.best_model_path

    # Log both checkpoint paths for comparison
    if trainer.is_global_zero:
        logger.info("Best RMSE checkpoint: %s", best_path_rmse or "N/A")
        logger.info("Best photo checkpoint: %s", best_photo_path or "N/A")

    if best_choice == "rmse":
        best_path = best_path_rmse
    else:
        best_path = best_photo_path

    # Fallback if training was skipped (enable=False) and best_path is empty in memory
    if not best_path:
        ckpt_dir = output_dir / "checkpoints"
        best_ckpts = list(ckpt_dir.glob(f"depthfm-best-{best_choice}-*.ckpt"))

        if best_ckpts:

            # The float metric is right before the .ckpt extension
            def extract_rmse(path):
                match = re.search(r"([0-9]+\.[0-9]+)\.ckpt$", path.name)
                return float(match.group(1)) if match else float('inf')

            # Grab the checkpoint with the lowest RMSE value
            best_path = str(min(best_ckpts, key=extract_rmse))
            logger.info(f"*** Found best {best_choice.upper()} checkpoint via glob: {best_path} ***")
        else:
            logger.warning("*** No best checkpoint found via glob, falling back to last.ckpt ***")
            best_path = str(last_ckpt_path)
    else:
        logger.info(f"*** Testing with best {best_choice.upper()} checkpoint from memory: {best_path} ***")

    # Test
    trainer.test(
        module,
        loaders["test"],
        ckpt_path=best_path,
        weights_only=False
    )

    module.to(trainer.strategy.root_device)

    # Timestep ablation
    logger.info("Running timestep ablation...")

    timestep_results = module.run_timestep_ablation(
        loaders["test"],
        step_counts=[1, 2, 4, 8, 10, 20],
        max_batches=config.training.get("ablation_max_batches"),
    )

    if trainer.is_global_zero:
        with open(output_dir / "timestep_ablation.json", "w") as f:
            json.dump({str(k): v for k, v in timestep_results.items()}, f, indent=2)

        set_neurips_style()
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_timestep_ablation(
            step_counts=sorted(timestep_results.keys()),
            metrics_per_step=timestep_results,
            primary_metric="rmse",
            secondary_metrics=["delta_1", "normal_angular_error"],
            title="Mars DTM: inference quality vs Euler steps",
            save_path=fig_dir / "timestep_ablation_rmse.pdf",
        )
        plt.close(fig)
        fig = plot_timestep_ablation(
            step_counts=sorted(timestep_results.keys()),
            metrics_per_step=timestep_results,
            primary_metric="photo_consistency",
            secondary_metrics=["delta_1", "normal_angular_error"],
            title="Mars DTM: inference quality vs Euler steps",
            save_path=fig_dir / "timestep_ablation_photo.pdf",
        )
        plt.close(fig)

        fig = plot_pareto_frontier(
            metrics_per_step=timestep_results,
            primary_metric="rmse",
            save_path=fig_dir / "pareto_frontier_rmse.pdf",
        )
        plt.close(fig)

    # Collect results
    test_summary = module._test_aggregator.summary()
    test_df = module._test_aggregator.per_sample_dataframe()

    if trainer.is_global_zero:
        if config.model.get("torch_compile", False):
            logger.info("Extracting torch.compile Mega-Cache artifacts...")
            artifacts = torch.compiler.save_cache_artifacts()
            if artifacts is not None:
                artifact_bytes, cache_info = artifacts
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(artifact_bytes)
                logger.info("Mega-Cache saved to %s. Info: %s", cache_path, cache_info)

        test_df.to_csv(output_dir / "test_results.csv", index=False)
        with open(output_dir / "test_summary.json", "w") as f:
            json.dump(test_summary, f, indent=2)

    return {
        "test_summary": test_summary,
        "val_history": deepcopy(module.val_history),
        "test_aggregator": module._test_aggregator,
        "test_df": test_df,
        "timestep_ablation": timestep_results,
        "output_dir": output_dir,
        "best_rmse_ckpt": best_path,
        "best_photo_ckpt": best_photo_path or "",
    }


# ---------------------------------------------------------------------------
# Multi-run with statistical evaluation
# ---------------------------------------------------------------------------


def run_multi_seed_experiment(config, n_runs: int = 3, base_seed: int = 42):
    """Run training multiple times with different seeds for error bars."""
    output_root = Path(config.training.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    all_results = []
    seeds = [base_seed + i * 1000 for i in range(n_runs)]

    for run_idx, seed in enumerate(seeds):
        logger.info("=" * 60)
        logger.info("  RUN %d / %d  (seed=%d)", run_idx + 1, n_runs, seed)
        logger.info("=" * 60)
        result = run_single_training(config, run_idx=run_idx, seed=seed, output_dir=output_root)
        all_results.append(result)

    is_global_zero = int(os.environ.get("GLOBAL_RANK", os.environ.get("RANK", 0))) == 0

    if is_global_zero:
        set_neurips_style()
        logger.info("Generating publication figures...")

        fig = plot_convergence_curves(
            [r["val_history"] for r in all_results],
            metric_key="val/rmse",
            title="Validation RMSE convergence",
            save_path=fig_dir / "convergence_rmse.pdf",
        )
        plt.close(fig)

        fig = plot_convergence_curves(
            [r["val_history"] for r in all_results],
            metric_key="val/delta_1",
            title="Validation δ₁ convergence",
            save_path=fig_dir / "convergence_delta1.pdf",
        )
        plt.close(fig)

        fig = plot_multi_run_summary_table(
            [r["test_summary"] for r in all_results],
            metrics_to_show=["rmse", "abs_rel", "si_log", "delta_1", "normal_angular_error", "photo_consistency"],
            title=f"Test results ({n_runs} runs)",
            save_path=fig_dir / "multi_run_summary.pdf",
        )
        plt.close(fig)

        best_run = min(all_results, key=lambda r: r["test_summary"].get("rmse", {}).get("mean", 1e9))
        fig = plot_metric_distributions(
            best_run["test_df"],
            metrics_to_plot=["rmse", "abs_rel", "si_log", "delta_1", "normal_angular_error", "photo_consistency"],
            title="Test metric distributions (best run)",
            save_path=fig_dir / "metric_distributions.pdf",
        )
        plt.close(fig)

        _generate_patch_analysis(best_run, fig_dir, config)

        if all("timestep_ablation" in r for r in all_results):

            first_ablation = all_results[0]["timestep_ablation"]
            step_counts = sorted(first_ablation.keys())
            averaged_ablation = {}
            for s in step_counts:
                merged = {}
                for metric_name in first_ablation[s]:
                    run_means = [r["timestep_ablation"][s][metric_name]["mean"] for r in all_results if
                                 s in r["timestep_ablation"]]
                    merged[metric_name] = {"mean": float(np.mean(run_means)), "std": float(np.std(run_means))}
                averaged_ablation[s] = merged

            fig = plot_timestep_ablation(
                step_counts=step_counts,
                metrics_per_step=averaged_ablation,
                primary_metric="rmse",
                secondary_metrics=["delta_1", "normal_angular_error"],
                title=f"Inference quality vs Euler steps ({n_runs}-run avg)",
                save_path=fig_dir / "timestep_ablation_averaged_rmse.pdf",
            )
            plt.close(fig)
            fig = plot_timestep_ablation(
                step_counts=step_counts,
                metrics_per_step=averaged_ablation,
                primary_metric="photo_consistency",
                secondary_metrics=["delta_1", "normal_angular_error"],
                title="Mars DTM: inference quality vs Euler steps",
                save_path=fig_dir / "timestep_ablation_averaged_photo.pdf",
            )
            plt.close(fig)

        _print_final_summary(all_results, output_root)
        logger.info("All figures saved to %s", fig_dir)


def _generate_patch_analysis(best_run: dict, fig_dir: Path, config):
    """Generate detailed per-patch analysis from the best run."""
    aggregator = best_run.get("test_aggregator")
    if aggregator is None:
        logger.warning("Test aggregator not found in memory (run was skipped). Skipping patch analysis plotting.")
        return

    worst = aggregator.worst_k("rmse", k=5)
    best = aggregator.best_k("rmse", k=5)
    logger.info("Worst 5 test patches: %s", worst)
    logger.info("Best 5 test patches: %s", best)

    analysis = {
        "worst_5_rmse": [{"tile_id": t, "rmse": float(v)} for t, v in worst],
        "best_5_rmse": [{"tile_id": t, "rmse": float(v)} for t, v in best],
        "summary": best_run["test_summary"],
    }
    with open(fig_dir / "patch_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    df = best_run["test_df"]

    set_neurips_style()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["rmse"], bins=30, color=sns.color_palette("flare")[2], kde=True, ax=ax, alpha=0.6)
    for tile_id, rmse_val in worst[:3]:
        ax.axvline(rmse_val, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.text(rmse_val, ax.get_ylim()[1] * 0.9, tile_id, rotation=45, fontsize=7, color="red")
    ax.set_xlabel("RMSE (m)")
    ax.set_ylabel("Count")
    ax.set_title("Test RMSE distribution with worst patches")
    save_fig(fig, fig_dir / "rmse_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    if "slope_rmse" in df.columns and "rmse" in df.columns:
        ax.scatter(df["slope_rmse"], df["rmse"], c=df["normal_angular_error"], cmap="flare", s=20, alpha=0.6)
        ax.set_xlabel("Slope RMSE (°)")
        ax.set_ylabel("Elevation RMSE (m)")
        ax.set_title("Error vs terrain complexity")
        fig.colorbar(ax.collections[0], label="Normal error (°)")
        save_fig(fig, fig_dir / "error_vs_complexity.pdf", bbox_inches="tight")
    plt.close(fig)


def _print_final_summary(all_results: list[dict], output_dir: Path):
    """Print and save the final multi-run summary."""

    metrics_of_interest = [
        "rmse", "abs_rel", "si_log", "delta_1",
        "normal_angular_error", "slope_rmse",
        "photo_consistency", "psd_ratio",
    ]
    rows = []
    for i, r in enumerate(all_results):
        row = {"run": i}
        for m in metrics_of_interest:
            if m in r["test_summary"]:
                row[m] = r["test_summary"][m]["mean"]
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_lines = ["=" * 70, "FINAL RESULTS (mean ± std across runs)", "=" * 70]
    for m in metrics_of_interest:
        if m in df.columns:
            summary_lines.append(f"  {m:<30s}  {df[m].mean():.4f} ± {df[m].std():.4f}")
    summary_lines.append("=" * 70)

    for line in summary_lines:
        logger.info(line)

    with open(output_dir / "final_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    df.to_csv(output_dir / "all_runs_metrics.csv", index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def dataload_switch_test(config, args):
    loaders = build_dataloaders(config, split_seed=args.seed, parallel=config.data.get("parallel_load", False))

    for i in range(3):
        logger.info(f"Loading data for run {i}")
        for split, loader in loaders.items():
            logger.info(f"Starting dataloader for split {split}")
            for b in tqdm(loader):
                pass


hash_config = run_id_hash  # backwards-compat alias — use run_id_hash(config) directly


def main():
    import warnings

    warnings.filterwarnings("ignore", message=r".*isinstance(treespec, LeafSpec).*")
    warnings.filterwarnings("ignore", message=r"Found \d+ module")
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    parser = argparse.ArgumentParser(description="Train Mars DepthFM")
    parser.add_argument("--config", type=str, default="configs/train_hirise.yaml")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of training runs for error bars")
    parser.add_argument("--n_folds", type=int, default=None, help="K-fold CV (overrides config)")
    parser.add_argument("--fold_idx", type=int, default=0, help="Which fold to use as test (0 to n_folds-1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze_topography", action="store_true")
    parser.add_argument("--analyze_masks", action="store_true")
    parser.add_argument("--view_thumbnails", action="store_true")
    parser.add_argument("--view_loss_physics", action="store_true")
    parser.add_argument("--view_loss_components", action="store_true")
    parser.add_argument("--view_invalid_fill", action="store_true")
    parser.add_argument("--view_seam_artifacts", action="store_true")
    parser.add_argument("--view_tin_artifacts", action="store_true")
    parser.add_argument("--view_solar_distribution", action="store_true")
    parser.add_argument("--view_augmentations", action="store_true")
    parser.add_argument("--view_extras", action="store_true")
    parser.add_argument("--view_lambert_ablation", action="store_true")
    parser.add_argument("--all_viz", action="store_true")
    # Test-only mode for the ablation orchestrator: load a checkpoint, run
    # the test loop, and write per-patch metrics tagged for downstream paired
    # statistics. Skips training entirely.
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training; run only the test loop on the given checkpoint")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path for --test_only mode (must be set when --test_only is used)")
    parser.add_argument("--test_tag", type=str, default="",
                        help="Suffix tag for test output files (e.g. 'rmse_ckpt', 'photo_ckpt')")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    if args.n_folds is not None:
        config.data.n_folds = args.n_folds
        config.data.fold_idx = args.fold_idx

    is_global_zero = int(os.environ.get("GLOBAL_RANK", os.environ.get("RANK", 0))) == 0
    config_hash = hash_config(config)

    config.training.output_dir = Path(config.training.output_dir) / config.model.get("model_type",
                                                                                     "depthfm") / config_hash

    logger.info("Saving the data to the output directory: %s", config.training.output_dir)

    # Log hardware info
    if is_global_zero:
        logger.info("Hardware: %d CPU cores detected, %d GPUs, workers/GPU=%d", _TOTAL_CORES, _NUM_GPUS_DEFAULT,
                    _WORKERS_PER_GPU)

    all_viz = args.all_viz
    # Inspection modes
    inspection = (all_viz or
                  args.analyze_masks or
                  args.view_thumbnails or
                  args.analyze_topography or
                  args.view_loss_physics or
                  args.view_loss_components or
                  args.view_invalid_fill or
                  args.view_tin_artifacts or
                  args.view_solar_distribution or
                  args.view_seam_artifacts or
                  args.view_augmentations or
                  args.view_extras or
                  args.view_lambert_ablation)
    if inspection:
        if is_global_zero:
            logger.info("Executing isolated data inspection routine...")
            L.seed_everything(args.seed, workers=True)
            # For the visualisation we do not want to use the distributed setup, just GPU 0, as such to trick the
            # creation for the dataloaders we want to have it look at the full dataset; however, it splits per
            # rank using the WORLD_SIZE environment variable
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]
                logger.warning(
                    "Because we are running in a distributed environment and we are planning to run visualisation, we disable the other GPUs")

            loaders = build_dataloaders(config, split_seed=args.seed, parallel=config.data.get("parallel_load", False))
            output_path = Path(config.training.output_dir) / "inspection"

            if args.analyze_topography:
                compute_topography_statistics(loaders["test"], split_name="Test Set")
            if args.analyze_masks:
                for split_name, loader in loaders.items():
                    compute_mask_statistics(loader, split_name=f"{split_name.capitalize()} Set")
            if all_viz or args.view_thumbnails:
                generate_thumbnail_grids(loaders["train"], output_dir=output_path, num_samples=8)
            if all_viz or args.view_loss_physics:
                visualize_loss_physics(loaders["train"], output_dir=output_path, num_samples=8)
            if all_viz or args.view_loss_components:
                visualize_loss_components(loaders["train"], output_dir=output_path, num_samples=8)
            if all_viz or args.view_invalid_fill:
                visualize_invalid_fill(loaders["train"], output_dir=output_path, num_samples=16)
            if all_viz or args.view_seam_artifacts:
                visualize_seam_artifacts(loaders["train"], output_dir=output_path, num_samples=200)
            if all_viz or args.view_tin_artifacts:
                visualize_tin_artifacts(loaders["train"], output_dir=output_path, num_samples=8)
            if all_viz or args.view_solar_distribution:
                visualize_solar_distribution(loaders["train"], output_dir=output_path)
            if all_viz or args.view_augmentations:
                visualize_random_flips_and_rotations(loaders["train"], output_dir=output_path, num_samples=2)
            if all_viz or args.view_extras:
                visualize_huber_loss(loaders["train"], output_dir=output_path, num_samples=6)
                visualize_laplacian_loss(loaders["train"], output_dir=output_path, num_samples=6)
                visualize_ordinal_ranking(loaders["train"], output_dir=output_path, num_samples=6)
            if all_viz or args.view_lambert_ablation:
                loss_fn = PhotoclinometricLoss().to("cuda")

                plot_global_umap_invariance(loaders["train"], loss_fn=loss_fn, output_dir=output_path)
                plot_statistically_significant_landscape(loaders["train"], loss_fn=loss_fn, output_dir=output_path)
                prove_and_visualize_local_convexity(loaders["train"], loss_fn=loss_fn, output_dir=output_path)
                # plot_umap_invariance(loaders["train"], loss_fn=loss_fn, output_dir=output_path)
                plot_component_ablation(loaders["train"], loss_fn=loss_fn, output_dir=output_path)
                plot_qualitative_physics_errors(loaders["train"], loss_fn=loss_fn, output_dir=output_path)
                # plot_radial_sun_sweep(loaders["train"], loss_fn=loss_fn, output_dir=output_path)

            logger.info("Data inspection complete. Exiting without training.")
        return

    # Test-only mode (ablation orchestrator)
    if args.test_only:
        if not args.ckpt:
            raise ValueError("--test_only requires --ckpt <path>")
        run_test_only(config, ckpt_path=args.ckpt, test_tag=args.test_tag, seed=args.seed)
        return

    # Training
    if args.n_runs > 1:
        run_multi_seed_experiment(config, n_runs=args.n_runs, base_seed=args.seed)
    else:
        result = run_single_training(config, run_idx=0, seed=args.seed, output_dir=config.training.output_dir)

        if is_global_zero:
            fig_dir = Path(config.training.output_dir) / "run_0" / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            set_neurips_style()
            fig = plot_metric_distributions(result["test_df"], save_path=fig_dir / "metrics.pdf")
            plt.close(fig)
            _generate_patch_analysis(result, fig_dir, config)
            _print_final_summary([result], Path(config.training.output_dir))


if __name__ == "__main__":
    main()
