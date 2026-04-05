"""
Mars DepthFM Training — Lightning-based with multi-run evaluation.

Features:
- Uses MarsHiRISEDTM + HiRISEGeoSampler with train/val/test splits
- Multiple independent training runs for statistical significance
- Publication-quality figures with error bars
- Patch-level error analysis identifying failure modes
- All metrics logged to wandb with proper grouping

Usage:
    # Single run
    python src/train_lightning.py --config configs/train_hirise.yaml

    # Multi-run for error bars (3 seeds)
    python src/train_lightning.py --config configs/train_hirise.yaml --n_runs 3

    # K-fold cross-validation
    python src/train_lightning.py --config configs/train_hirise.yaml --n_folds 5
"""

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from depth_fm.depthfm_adapter import DepthFMHiRISEAdapter
from depth_fm.lightning_module import DepthFMLightningModule, EMACallback
from depth_fm.visualization import (
    set_neurips_style,
    plot_metric_distributions,
    plot_convergence_curves,
    plot_multi_run_summary_table,
)

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

def build_dataloaders(config, split_seed: int = 42):
    """Build train/val/test DataLoaders from MarsHiRISEDTM."""
    from dataset.mars_hirise_dtm import MarsHiRISEDTM
    from dataset.hirise_sampler import HiRISEGeoSampler
    from torchgeo.samplers import Units

    hc = config.data.hirise
    sc = config.data.sampler

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
    )

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
    )

    resolution = config.data.get("resolution", 512)
    dtm_norm = config.data.get("dtm_normalization", "relative")
    stats_path = config.data.get("stats_path")

    loaders = {}

    for split in ("train", "val", "test"):
        is_train = (split == "train")

        sampler = HiRISEGeoSampler(
            base_dataset,
            split=split,
            length=sc.get("length") if is_train else None,
            replacement=is_train,
            **common_sampler_kwargs,
        )

        adapter = DepthFMHiRISEAdapter(
            base_dataset=base_dataset,
            sampler=sampler,
            resolution=resolution,
            dtm_normalization=dtm_norm,
            random_flip=is_train,
            brightness_jitter=config.data.get("brightness_jitter", 0.1) if is_train else 0.0,
            stats_path=stats_path,
        )

        loaders[split] = DataLoader(
            adapter,
            batch_size=config.training.per_gpu_batch_size,
            shuffle=is_train,
            num_workers=config.training.num_workers if is_train else 4,
            pin_memory=config.training.pin_memory,
            prefetch_factor=config.training.get("prefetch_factor", 4) if is_train else 2,
            drop_last=is_train,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )

        logger.info(
            "DataLoader [%s]: %d samples, batch_size=%d",
            split, len(adapter), config.training.per_gpu_batch_size,
        )

    return loaders


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_single_training(
        config,
        run_idx: int = 0,
        seed: int = 42,
        output_dir: str | Path = "outputs",
) -> dict:
    """Execute a single training run and return test metrics.

    Returns:
        dict with keys: "test_summary", "val_history", "test_aggregator"
    """
    L.seed_everything(seed, workers=True)

    output_dir = Path(output_dir) / f"run_{run_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build data
    loaders = build_dataloaders(config, split_seed=seed)

    # Build model
    module = DepthFMLightningModule(config)

    # Move VAE to appropriate device
    module.model.vae = module.model.vae.to("cuda" if torch.cuda.is_available() else "cpu")

    cache_path = Path(config.training.get("cache_dir", ".torch_compile_cache")) / "mega_cache.pt"

    # torch.compile backbone for training speed (max-autotune triggers kernel auto-tuning)
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

    # Callbacks
    callbacks = [
        EMACallback(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="depthfm-{step}-{val/rmse_mean:.4f}",
            monitor="val/rmse_mean",
            mode="min",
            save_top_k=3,
            every_n_train_steps=config.training.save_every_steps,
        ),
    ]

    if config.training.get("early_stopping_patience"):
        callbacks.append(
            EarlyStopping(
                monitor="val/rmse_mean",
                patience=config.training.early_stopping_patience,
                mode="min",
            )
        )

    # Logger
    wandb_logger = None
    if config.training.logger == "wandb":
        wandb_logger = WandbLogger(
            project=config.training.project_name,
            name=f"{config.training.run_name}_run{run_idx}",
            save_dir=str(output_dir),
            group=config.training.run_name,
            tags=["mars", "depthfm", "flow-matching"],
        )

    # Trainer
    _prec = config.training.mixed_precision
    if _prec == "bf16":
        precision = "bf16-mixed"
    elif _prec in ("f16", "fp16"):
        precision = "16-mixed"
    else:
        precision = "32-true"

    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.training.num_gpus,
        strategy="ddp" if config.training.num_gpus > 1 else "auto",
        precision=precision,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=config.training.val_every_steps,
        log_every_n_steps=config.training.log_every_steps,
        gradient_clip_val=config.training.max_grad_norm,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        enable_progress_bar=True,
        default_root_dir=str(output_dir),
    )

    # Train
    trainer.fit(module, loaders["train"], loaders["val"])

    # Test (with EMA weights via callback)
    trainer.test(module, loaders["test"])

    # Timestep ablation: evaluate at [1, 2, 4, 8, 10, 20] Euler steps
    logger.info("Running timestep ablation...")
    if module._ema_initialised:
        module.load_ema_weights()
    timestep_results = module.run_timestep_ablation(
        loaders["test"],
        step_counts=[1, 2, 4, 8, 10, 20],
        max_batches=config.training.get("ablation_max_batches"),
    )
    if module._ema_initialised:
        module.restore_training_weights()

    # Save timestep ablation results
    with open(output_dir / "timestep_ablation.json", "w") as f:
        json.dump(
            {str(k): v for k, v in timestep_results.items()},
            f, indent=2,
        )

    # Generate timestep ablation figure
    from depth_fm.visualization import plot_timestep_ablation
    set_neurips_style()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_timestep_ablation(
        step_counts=sorted(timestep_results.keys()),
        metrics_per_step=timestep_results,
        primary_metric="rmse",
        secondary_metrics=["delta_1", "normal_angular_error"],
        title="Mars DTM: inference quality vs Euler steps",
        save_path=fig_dir / "timestep_ablation.pdf",
    )
    plt.close(fig)

    # Collect results
    test_summary = module._test_aggregator.summary()
    test_df = module._test_aggregator.per_sample_dataframe()

    if config.model.get("torch_compile", False):
        logger.info("Extracting torch.compile Mega-Cache artifacts...")
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is not None:
            artifact_bytes, cache_info = artifacts
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(artifact_bytes)
            logger.info("Mega-Cache saved successfully to %s. Info: %s", cache_path, cache_info)
        else:
            logger.info("No compiler artifacts found to save.")

    # Save per-sample test results
    test_df.to_csv(output_dir / "test_results.csv", index=False)

    # Save test summary
    with open(output_dir / "test_summary.json", "w") as f:
        json.dump(test_summary, f, indent=2)

    return {
        "test_summary": test_summary,
        "val_history": deepcopy(module.val_history),
        "test_aggregator": module._test_aggregator,
        "test_df": test_df,
        "timestep_ablation": timestep_results,
        "output_dir": output_dir,
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

        result = run_single_training(
            config, run_idx=run_idx, seed=seed, output_dir=output_root,
        )
        all_results.append(result)

    # ── Generate publication figures ──
    set_neurips_style()

    logger.info("Generating publication figures...")

    # 1. Convergence curves with error bands
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

    # 2. Multi-run summary bar chart with error bars
    fig = plot_multi_run_summary_table(
        [r["test_summary"] for r in all_results],
        metrics_to_show=["rmse", "abs_rel", "delta_1", "normal_angular_error", "slope_rmse"],
        title=f"Test results ({n_runs} runs)",
        save_path=fig_dir / "multi_run_summary.pdf",
    )
    plt.close(fig)

    # 3. Metric distributions from the best run
    best_run = min(all_results, key=lambda r: r["test_summary"].get("rmse", {}).get("mean", 1e9))
    best_df = best_run["test_df"]

    fig = plot_metric_distributions(
        best_df,
        metrics_to_plot=["rmse", "abs_rel", "delta_1", "normal_angular_error"],
        title="Test metric distributions (best run)",
        save_path=fig_dir / "metric_distributions.pdf",
    )
    plt.close(fig)

    # 4. Worst and best patches from the best run
    _generate_patch_analysis(best_run, fig_dir, config)

    # 5. Timestep ablation (averaged across runs if available)
    if all("timestep_ablation" in r for r in all_results):
        from depth_fm.visualization import plot_timestep_ablation

        # Average metrics across runs for each step count
        first_ablation = all_results[0]["timestep_ablation"]
        step_counts = sorted(first_ablation.keys())

        averaged_ablation = {}
        for s in step_counts:
            merged = {}
            for metric_name in first_ablation[s]:
                run_means = [
                    r["timestep_ablation"][s][metric_name]["mean"]
                    for r in all_results if s in r["timestep_ablation"]
                ]
                merged[metric_name] = {
                    "mean": float(np.mean(run_means)),
                    "std": float(np.std(run_means)),
                }
            averaged_ablation[s] = merged

        fig = plot_timestep_ablation(
            step_counts=step_counts,
            metrics_per_step=averaged_ablation,
            primary_metric="rmse",
            secondary_metrics=["delta_1", "normal_angular_error"],
            title=f"Inference quality vs Euler steps ({n_runs}-run avg)",
            save_path=fig_dir / "timestep_ablation_averaged.pdf",
        )
        plt.close(fig)

    # 6. Final summary to console and file
    _print_final_summary(all_results, output_root)

    logger.info("All figures saved to %s", fig_dir)


def _generate_patch_analysis(best_run: dict, fig_dir: Path, config):
    """Generate detailed per-patch analysis from the best run."""
    aggregator = best_run["test_aggregator"]

    worst = aggregator.worst_k("rmse", k=5)
    best = aggregator.best_k("rmse", k=5)

    logger.info("Worst 5 test patches: %s", worst)
    logger.info("Best 5 test patches: %s", best)

    # Save to JSON for later analysis
    analysis = {
        "worst_5_rmse": [{"tile_id": t, "rmse": float(v)} for t, v in worst],
        "best_5_rmse": [{"tile_id": t, "rmse": float(v)} for t, v in best],
        "summary": best_run["test_summary"],
    }
    with open(fig_dir / "patch_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Per-metric error distribution with patch identification
    df = best_run["test_df"]
    import seaborn as sns

    set_neurips_style()

    # RMSE distribution with worst patches annotated
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["rmse"], bins=30, color=sns.color_palette("flare")[2],
                 kde=True, ax=ax, alpha=0.6)
    for tile_id, rmse_val in worst[:3]:
        ax.axvline(rmse_val, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.text(rmse_val, ax.get_ylim()[1] * 0.9, tile_id,
                rotation=45, fontsize=7, color="red")
    ax.set_xlabel("RMSE (m)")
    ax.set_ylabel("Count")
    ax.set_title("Test RMSE distribution with worst patches")
    fig.savefig(fig_dir / "rmse_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # Error vs slope complexity scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    if "slope_rmse" in df.columns and "rmse" in df.columns:
        ax.scatter(df["slope_rmse"], df["rmse"],
                   c=df["normal_angular_error"], cmap="flare",
                   s=20, alpha=0.6)
        ax.set_xlabel("Slope RMSE (°)")
        ax.set_ylabel("Elevation RMSE (m)")
        ax.set_title("Error vs terrain complexity")
        cbar = fig.colorbar(ax.collections[0], label="Normal error (°)")
        fig.savefig(fig_dir / "error_vs_complexity.pdf", bbox_inches="tight")
    plt.close(fig)


def _print_final_summary(all_results: list[dict], output_dir: Path):
    """Print and save the final multi-run summary."""
    import pandas as pd

    metrics_of_interest = ["rmse", "abs_rel", "delta_1", "delta_2",
                           "normal_angular_error", "slope_rmse"]

    rows = []
    for i, r in enumerate(all_results):
        row = {"run": i}
        for m in metrics_of_interest:
            if m in r["test_summary"]:
                row[m] = r["test_summary"][m]["mean"]
        rows.append(row)

    df = pd.DataFrame(rows)

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("FINAL RESULTS (mean ± std across runs)")
    summary_lines.append("=" * 70)
    for m in metrics_of_interest:
        if m in df.columns:
            mean = df[m].mean()
            std = df[m].std()
            summary_lines.append(f"  {m:<30s}  {mean:.4f} ± {std:.4f}")
    summary_lines.append("=" * 70)

    for line in summary_lines:
        logger.info(line)

    # Save to file
    with open(output_dir / "final_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    df.to_csv(output_dir / "all_runs_metrics.csv", index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import warnings
    warnings.filterwarnings("ignore", message=r".*isinstance(treespec, LeafSpec).*")
    warnings.filterwarnings("ignore", message=r"Found \d+ module")
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    parser = argparse.ArgumentParser(description="Train Mars DepthFM")
    parser.add_argument("--config", type=str, default="configs/train_hirise.yaml")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of training runs for error bars")
    parser.add_argument("--n_folds", type=int, default=None,
                        help="K-fold CV (overrides config)")
    parser.add_argument("--fold_idx", type=int, default=0,
                        help="Which fold to use as test (0 to n_folds-1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))

    # Override K-fold from CLI
    if args.n_folds is not None:
        config.data.n_folds = args.n_folds
        config.data.fold_idx = args.fold_idx

    if args.n_runs > 1:
        run_multi_seed_experiment(config, n_runs=args.n_runs, base_seed=args.seed)
    else:
        result = run_single_training(config, run_idx=0, seed=args.seed,
                                     output_dir=config.training.output_dir)
        # Generate single-run figures
        fig_dir = Path(config.training.output_dir) / "run_0" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        set_neurips_style()
        df = result["test_df"]
        fig = plot_metric_distributions(df, save_path=fig_dir / "metrics.pdf")
        plt.close(fig)

        _generate_patch_analysis(result, fig_dir, config)
        _print_final_summary([result], Path(config.training.output_dir))


if __name__ == "__main__":
    main()
