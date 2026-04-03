"""Training utilities and CLI for paired multiscale multimodal alignment."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import tempfile
import time
import traceback
from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset

from clip.marsclip_alignment import PairedMultiscaleAlignmentModel
from clip.marsclip_paired_multiscale import (
    MarsCLIPPairedBatchCollator,
    MarsCLIPPairedCropDataset,
    load_paired_crop_records,
    save_paired_crop_records,
)
from clip.marsclip_patches import DEFAULT_PATCH_VALID_FRACTION, load_patch_records, save_patch_records
from clip.marsclip_splits import build_dataset_subsets, load_patch_split_manifest
from clip.marsclip_text import SimpleTextTokenizer, compose_rationale_text
from clip.train_marsclip_mae import (
    build_mae_model_from_config,
    build_scheduler,
    count_trainable_parameters,
    current_learning_rate,
    infer_mae_config_from_state_dict,
    load_mae_checkpoint,
    resolve_map_location,
    save_training_history,
    save_training_progress,
)

_MPL_CACHE = pathlib.Path(tempfile.gettempdir()) / "marsclip_artifacts" / "support" / "mpl_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib as mpl

mpl.use("Agg")
from matplotlib import pyplot as plt


def _resolve_device(
    device: str | torch.device | None,
    model: torch.nn.Module,
) -> torch.device:
    if isinstance(device, str) and device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device is None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    return torch.device(device)


def _move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _texts_from_base_paired_dataset(
    dataset: Any,
    indices: Sequence[int],
    *,
    text_mode: str,
) -> list[str]:
    rows = dataset.paired_records.iloc[list(indices)]
    obs_lookup = dataset.observation_metadata
    texts: list[str] = []
    for _, row in rows.iterrows():
        obs_row = obs_lookup.loc[str(row["dominant_obs_id"])]
        rationale_expanded = obs_row.get("rationale_expanded")
        if pd.isna(rationale_expanded):
            rationale_expanded = None
        texts.append(
            compose_rationale_text(
                str(obs_row["rationale_desc"]),
                None if rationale_expanded is None else str(rationale_expanded),
                text_mode=text_mode,
            )
        )
    return texts


def collect_alignment_texts(
    dataset: Dataset | Sequence[dict[str, Any]],
    *,
    text_mode: str = "raw_plus_expanded",
) -> list[str]:
    """Collect training texts for tokenizer fitting without forcing image loads when possible."""
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "paired_records"):
        return _texts_from_base_paired_dataset(
            dataset.dataset,
            dataset.indices,
            text_mode=text_mode,
        )
    if hasattr(dataset, "paired_records") and hasattr(dataset, "observation_metadata"):
        indices = range(len(dataset.paired_records))
        return _texts_from_base_paired_dataset(dataset, indices, text_mode=text_mode)

    texts: list[str] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        texts.append(
            compose_rationale_text(
                sample["rationale_raw"],
                sample.get("rationale_expanded"),
                text_mode=text_mode,
            )
        )
    return texts


def build_alignment_tokenizer(
    dataset: Dataset | Sequence[dict[str, Any]],
    *,
    text_mode: str = "raw_plus_expanded",
    min_freq: int = 1,
) -> SimpleTextTokenizer:
    """Fit a lightweight tokenizer on the paired multiscale train split."""
    texts = collect_alignment_texts(dataset, text_mode=text_mode)
    if not texts:
        raise ValueError("dataset must yield at least one text example.")
    return SimpleTextTokenizer.build(texts, min_freq=min_freq)


def build_alignment_dataloader(
    dataset: Dataset,
    *,
    tokenizer: SimpleTextTokenizer,
    batch_size: int = 4,
    shuffle: bool = False,
    text_mode: str = "raw_plus_expanded",
    max_length: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
) -> DataLoader:
    """Construct a paired multiscale dataloader with shared text batching."""
    collator = MarsCLIPPairedBatchCollator(
        tokenizer,
        text_mode=text_mode,
        max_length=max_length,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "collate_fn": collator,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(**loader_kwargs)


def build_alignment_subsets(
    dataset: Dataset,
    *,
    split_manifest: pathlib.Path | str | pd.DataFrame,
    mode: str = "holdout",
    fold_index: int = 0,
) -> tuple[Subset, Subset, Subset]:
    """Create train/val/test paired multiscale subsets from a split manifest."""
    if not hasattr(dataset, "paired_records"):
        raise ValueError("dataset must expose paired_records for split-manifest alignment.")
    manifest = (
        split_manifest.copy()
        if isinstance(split_manifest, pd.DataFrame)
        else load_patch_split_manifest(split_manifest)
    )
    available_patch_ids = set(dataset.paired_records["patch_id"].astype(str))
    manifest = manifest.loc[
        manifest["patch_id"].astype(str).isin(available_patch_ids)
    ].copy()
    if manifest.empty:
        raise ValueError(
            "No split-manifest patch ids remain after intersecting with paired multiscale records."
        )
    return build_dataset_subsets(
        dataset,
        dataset.paired_records,
        manifest,
        mode=mode,
        fold_index=fold_index,
    )


def load_alignment_model_from_stage_a_checkpoint(
    checkpoint_path: pathlib.Path | str,
    *,
    vocab_size: int,
    map_location: str | torch.device = "cpu",
    freeze_visual_backbone: bool = False,
    alignment_config: dict[str, Any] | None = None,
) -> tuple[PairedMultiscaleAlignmentModel, dict[str, Any]]:
    """Instantiate the paired multiscale alignment model from a trained Stage A MAE checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=resolve_map_location(map_location))
    stage_a_config = dict(checkpoint.get("config", {}))
    stage_a_config.update(
        infer_mae_config_from_state_dict(checkpoint.get("model_state", {}), stage_a_config)
    )
    stage_a_backbone = build_mae_model_from_config(stage_a_config)
    stage_a_state = load_mae_checkpoint(
        checkpoint_path,
        stage_a_backbone,
        map_location=map_location,
    )
    model = PairedMultiscaleAlignmentModel(
        visual_backbone=stage_a_backbone,
        vocab_size=int(vocab_size),
        **dict(alignment_config or {}),
    )
    if freeze_visual_backbone:
        for parameter in model.visual_backbone.stage_a_backbone.parameters():
            parameter.requires_grad = False
    return model, {
        "stage_a_step": int(stage_a_state["step"]),
        "stage_a_history": list(stage_a_state["history"]),
        "stage_a_config": stage_a_config,
        "alignment_config": dict(alignment_config or {}),
        "freeze_visual_backbone": bool(freeze_visual_backbone),
    }


def build_alignment_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
) -> Optimizer:
    """Create a simple optimizer for the multimodal alignment stage."""
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("model must expose at least one trainable parameter.")
    name = optimizer_name.lower().strip()
    if name == "adamw":
        return AdamW(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    if name == "adam":
        return Adam(params, lr=float(learning_rate), weight_decay=float(weight_decay))
    raise ValueError("optimizer_name must be either 'adamw' or 'adam'.")


def _retrieval_direction_metrics(logits: torch.Tensor) -> tuple[float, float, float]:
    """Return top-1, mean-rank, and MRR for one retrieval direction."""
    targets = torch.arange(logits.shape[0], device=logits.device)
    top1 = float((logits.argmax(dim=1) == targets).float().mean().detach().cpu())
    ordering = torch.argsort(logits, dim=1, descending=True)
    ranks = (
        torch.argmax(
            (ordering == targets.unsqueeze(1)).to(torch.int64),
            dim=1,
        )
        + 1
    ).to(torch.float32)
    mean_rank = float(ranks.mean().detach().cpu())
    mrr = float((1.0 / ranks).mean().detach().cpu())
    return top1, mean_rank, mrr


def contrastive_retrieval_metrics(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float]:
    """Compute interpretable retrieval metrics for one normalized embedding pair."""
    logits = torch.matmul(a, b.T)
    forward_top1, forward_mean_rank, forward_mrr = _retrieval_direction_metrics(logits)
    backward_top1, backward_mean_rank, backward_mrr = _retrieval_direction_metrics(logits.T)
    return {
        f"{prefix}_top1_forward": forward_top1,
        f"{prefix}_top1_backward": backward_top1,
        f"{prefix}_top1_mean": 0.5 * (forward_top1 + backward_top1),
        f"{prefix}_mean_rank_forward": forward_mean_rank,
        f"{prefix}_mean_rank_backward": backward_mean_rank,
        f"{prefix}_mean_rank_mean": 0.5 * (forward_mean_rank + backward_mean_rank),
        f"{prefix}_mrr_forward": forward_mrr,
        f"{prefix}_mrr_backward": backward_mrr,
        f"{prefix}_mrr_mean": 0.5 * (forward_mrr + backward_mrr),
    }


def summarize_alignment_output_metrics(
    output: Any,
    *,
    batch_size: int,
) -> dict[str, float]:
    """Summarize loss and retrieval metrics from one alignment forward pass."""
    metrics = {
        "loss": float(output.loss.detach().cpu()),
        "local_context_loss": float(output.losses["local_context"].detach().cpu()),
        "global_context_loss": float(output.losses["global_context"].detach().cpu()),
        "cross_scale_loss": float(output.losses["cross_scale"].detach().cpu()),
        "batch_size": float(batch_size),
    }
    metrics.update(
        contrastive_retrieval_metrics(
            output.local_image_embedding,
            output.context_embedding,
            prefix="local_context",
        )
    )
    metrics.update(
        contrastive_retrieval_metrics(
            output.global_image_embedding,
            output.context_embedding,
            prefix="global_context",
        )
    )
    metrics.update(
        contrastive_retrieval_metrics(
            output.local_image_embedding,
            output.global_image_embedding,
            prefix="cross_scale",
        )
    )
    random_baseline_loss = 3.0 * math.log(max(int(batch_size), 2))
    metrics["random_baseline_loss"] = float(random_baseline_loss)
    metrics["loss_to_random_ratio"] = float(metrics["loss"] / random_baseline_loss)
    return metrics


def save_alignment_loss_curve(
    history: list[dict[str, float]],
    out_path: pathlib.Path | str,
    *,
    val_history: list[dict[str, float]] | None = None,
) -> pathlib.Path:
    """Save a paired multiscale alignment loss curve PNG."""
    if not history:
        raise ValueError("history must not be empty.")

    steps = [entry["step"] for entry in history]
    losses = [entry["loss"] for entry in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses, color="#1b9e77", linewidth=2, label="train")
    if val_history:
        ax.plot(
            [entry["step"] for entry in val_history],
            [entry["loss"] for entry in val_history],
            color="#d95f02",
            linewidth=2,
            label="val",
        )
        ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("Alignment loss")
    ax.set_yscale("log")
    ax.set_title("MarsCLIP Paired Multiscale Alignment Loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def train_alignment_batch(
    model: PairedMultiscaleAlignmentModel,
    batch: dict[str, Any],
    optimizer: Optimizer,
    *,
    device: str | torch.device | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """Run one multimodal alignment optimization step and return scalar metrics."""
    resolved_device = _resolve_device(device, model)
    model.to(resolved_device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = _move_batch_to_device(batch, resolved_device)
    output = model(batch)
    output.loss.backward()
    if max_grad_norm is not None:
        clip_grad_norm_(model.parameters(), float(max_grad_norm))
    optimizer.step()
    return summarize_alignment_output_metrics(
        output,
        batch_size=int(batch["local_image"].shape[0]),
    )


def train_alignment_epoch(
    model: PairedMultiscaleAlignmentModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    *,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """Average alignment metrics over one training epoch or bounded batch slice."""
    totals = {
        "loss": 0.0,
        "local_context_loss": 0.0,
        "global_context_loss": 0.0,
        "cross_scale_loss": 0.0,
        "local_context_top1_mean": 0.0,
        "global_context_top1_mean": 0.0,
        "cross_scale_top1_mean": 0.0,
        "local_context_mean_rank_mean": 0.0,
        "global_context_mean_rank_mean": 0.0,
        "cross_scale_mean_rank_mean": 0.0,
        "loss_to_random_ratio": 0.0,
    }
    num_batches = 0
    num_items = 0.0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        metrics = train_alignment_batch(
            model,
            batch,
            optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        batch_weight = float(metrics["batch_size"])
        for key in totals:
            totals[key] += float(metrics[key]) * batch_weight
        num_batches += 1
        num_items += batch_weight
    if num_batches == 0:
        raise ValueError("Training dataloader produced no batches.")
    return {key: value / float(num_items) for key, value in totals.items()} | {
        "num_batches": float(num_batches)
    }


def _progress_timestamp() -> str:
    """Return a compact local timestamp for progress artifacts."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


@torch.no_grad()
def evaluate_alignment_dataloader(
    model: PairedMultiscaleAlignmentModel,
    dataloader: DataLoader,
    *,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Average multimodal alignment losses over a dataloader."""
    resolved_device = _resolve_device(device, model)
    model.to(resolved_device)
    was_training = model.training
    model.eval()
    totals = {
        "loss": 0.0,
        "local_context_loss": 0.0,
        "global_context_loss": 0.0,
        "cross_scale_loss": 0.0,
        "local_context_top1_mean": 0.0,
        "global_context_top1_mean": 0.0,
        "cross_scale_top1_mean": 0.0,
        "local_context_mean_rank_mean": 0.0,
        "global_context_mean_rank_mean": 0.0,
        "cross_scale_mean_rank_mean": 0.0,
        "loss_to_random_ratio": 0.0,
    }
    num_batches = 0
    num_items = 0.0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = _move_batch_to_device(batch, resolved_device)
        output = model(batch)
        metrics = summarize_alignment_output_metrics(
            output,
            batch_size=int(batch["local_image"].shape[0]),
        )
        batch_weight = float(metrics["batch_size"])
        for key in totals:
            totals[key] += float(metrics[key]) * batch_weight
        num_batches += 1
        num_items += batch_weight
    if was_training:
        model.train()
    if num_batches == 0:
        raise ValueError("Evaluation dataloader produced no batches.")
    return {key: value / float(num_items) for key, value in totals.items()} | {
        "num_batches": float(num_batches)
    }


def save_alignment_checkpoint(
    path: pathlib.Path | str,
    model: PairedMultiscaleAlignmentModel,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    *,
    step: int,
    history: list[dict[str, float]] | None = None,
    config: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Persist a minimal paired multiscale alignment checkpoint."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "history": list(history or []),
            "config": dict(config or {}),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        },
        out,
    )
    return out


def load_alignment_checkpoint(
    path: pathlib.Path | str,
    model: PairedMultiscaleAlignmentModel,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a paired multiscale alignment checkpoint into a model and optional optimizer."""
    checkpoint = torch.load(path, map_location=resolve_map_location(map_location))
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return {
        "step": int(checkpoint.get("step", 0)),
        "history": list(checkpoint.get("history", [])),
        "config": dict(checkpoint.get("config", {})),
    }


def save_alignment_summary(
    summary: dict[str, Any],
    out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Persist an alignment training/eval summary as JSON."""
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return out


def run_alignment_training(
    model: PairedMultiscaleAlignmentModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    *,
    out_dir: pathlib.Path | str,
    num_steps: int,
    device: str | torch.device | None = None,
    resume_state: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    checkpoint_every: int = 0,
    val_dataloader: DataLoader | None = None,
    val_every: int = 0,
    val_max_batches: int | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """Run a small paired multiscale alignment training session with checkpoints and validation."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    progress_path = out_path / "progress.json"

    history = list((resume_state or {}).get("history", []))
    start_step = int((resume_state or {}).get("step", 0))
    checkpoint_paths: list[str] = []
    val_history: list[dict[str, float]] = []
    best_metric_name = "val_loss" if val_dataloader is not None else "train_loss"
    best_loss = float("inf")
    best_step = start_step
    best_checkpoint_path = out_path / "best_checkpoint.pt"

    if history and val_dataloader is None:
        best_entry = min(history, key=lambda item: float(item["loss"]))
        best_loss = float(best_entry["loss"])
        best_step = int(best_entry["step"])

    save_training_progress(
        {
            "status": "running",
            "phase": "training",
            "start_step": start_step,
            "current_step": start_step,
            "target_step": start_step + num_steps,
            "best_loss": None if best_loss == float("inf") else best_loss,
            "best_step": best_step,
            "latest_loss": None,
            "latest_lr": current_learning_rate(optimizer),
            "latest_cross_scale_top1": None,
            "latest_local_context_top1": None,
            "latest_global_context_top1": None,
            "latest_loss_to_random_ratio": None,
            "val_best_loss": None,
            "out_dir": str(out_path),
        },
        progress_path,
    )

    iterator = iter(dataloader)
    resolved_device = _resolve_device(device, model)
    for offset in range(num_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        step = start_step + offset + 1
        metrics = train_alignment_batch(
            model,
            batch,
            optimizer,
            device=resolved_device,
            max_grad_norm=max_grad_norm,
        )
        if scheduler is not None:
            scheduler.step()
        metrics["step"] = float(step)
        metrics["lr"] = current_learning_rate(optimizer)
        history.append(metrics)

        save_training_progress(
            {
                "status": "running",
                "phase": "training",
                "start_step": start_step,
                "current_step": step,
                "target_step": start_step + num_steps,
                "best_loss": None if best_loss == float("inf") else best_loss,
                "best_step": best_step,
                "latest_loss": float(metrics["loss"]),
                "latest_lr": float(metrics["lr"]),
                "latest_cross_scale_top1": float(metrics["cross_scale_top1_mean"]),
                "latest_local_context_top1": float(metrics["local_context_top1_mean"]),
                "latest_global_context_top1": float(metrics["global_context_top1_mean"]),
                "latest_loss_to_random_ratio": float(metrics["loss_to_random_ratio"]),
                "val_best_loss": min((float(item["loss"]) for item in val_history), default=None),
                "out_dir": str(out_path),
            },
            progress_path,
        )

        if val_dataloader is None and float(metrics["loss"]) <= best_loss:
            best_loss = float(metrics["loss"])
            best_step = step
            save_alignment_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                step=step,
                history=history,
                config=config,
            )

        should_run_val = val_dataloader is not None and (
            (val_every > 0 and step % val_every == 0)
            or (val_every <= 0 and checkpoint_every > 0 and step % checkpoint_every == 0)
        )
        if should_run_val:
            save_training_progress(
                {
                    "status": "running",
                    "phase": "validating",
                    "start_step": start_step,
                    "current_step": step,
                    "target_step": start_step + num_steps,
                    "best_loss": None if best_loss == float("inf") else best_loss,
                    "best_step": best_step,
                    "latest_loss": float(metrics["loss"]),
                    "latest_lr": float(metrics["lr"]),
                    "latest_cross_scale_top1": float(metrics["cross_scale_top1_mean"]),
                    "latest_local_context_top1": float(metrics["local_context_top1_mean"]),
                    "latest_global_context_top1": float(metrics["global_context_top1_mean"]),
                    "latest_loss_to_random_ratio": float(metrics["loss_to_random_ratio"]),
                    "val_best_loss": min((float(item["loss"]) for item in val_history), default=None),
                    "out_dir": str(out_path),
                },
                progress_path,
            )
            val_metrics = evaluate_alignment_dataloader(
                model,
                val_dataloader,
                device=resolved_device,
                max_batches=val_max_batches,
            )
            val_entry = dict(val_metrics)
            val_entry["step"] = float(step)
            val_history.append(val_entry)
            if float(val_entry["loss"]) <= best_loss:
                best_loss = float(val_entry["loss"])
                best_step = step
                save_alignment_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    history=history,
                    config=config,
                )

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            checkpoint_path = save_alignment_checkpoint(
                out_path / f"checkpoint_step_{step:06d}.pt",
                model,
                optimizer,
                scheduler,
                step=step,
                history=history,
                config=config,
            )
            checkpoint_paths.append(str(checkpoint_path))

    final_checkpoint = save_alignment_checkpoint(
        out_path / "checkpoint.pt",
        model,
        optimizer,
        scheduler,
        step=start_step + num_steps,
        history=history,
        config=config,
    )
    history_path = save_training_history(history, out_path / "history.json")
    val_history_path = (
        save_training_history(val_history, out_path / "val_history.json")
        if val_history
        else None
    )
    loss_curve_path = save_alignment_loss_curve(
        history,
        out_path / "loss_curve.png",
        val_history=val_history,
    )
    summary = {
        "checkpoint": str(final_checkpoint),
        "best_checkpoint": str(best_checkpoint_path) if best_loss != float("inf") else None,
        "history_path": str(history_path),
        "val_history_path": str(val_history_path) if val_history_path is not None else None,
        "loss_curve": str(loss_curve_path),
        "num_steps": int(num_steps),
        "start_step": int(start_step),
        "final_step": int(start_step + num_steps),
        "best_metric_name": best_metric_name,
        "best_loss": None if best_loss == float("inf") else float(best_loss),
        "best_step": int(best_step),
        "final_loss": float(history[-1]["loss"]) if history else None,
        "final_loss_to_random_ratio": float(history[-1]["loss_to_random_ratio"]) if history else None,
        "final_cross_scale_top1": float(history[-1]["cross_scale_top1_mean"]) if history else None,
        "final_local_context_top1": float(history[-1]["local_context_top1_mean"]) if history else None,
        "final_global_context_top1": float(history[-1]["global_context_top1_mean"]) if history else None,
        "val_final_loss": float(val_history[-1]["loss"]) if val_history else None,
        "val_best_loss": min((float(item["loss"]) for item in val_history), default=None),
        "val_final_cross_scale_top1": float(val_history[-1]["cross_scale_top1_mean"]) if val_history else None,
        "val_final_local_context_top1": float(val_history[-1]["local_context_top1_mean"]) if val_history else None,
        "val_final_global_context_top1": float(val_history[-1]["global_context_top1_mean"]) if val_history else None,
        "checkpoint_paths": checkpoint_paths,
        "config": dict(config or {}),
        "summary_created_at": _progress_timestamp(),
    }
    summary_path = save_alignment_summary(summary, out_path / "summary.json")
    save_training_progress(
        {
            "status": "completed",
            "phase": "complete",
            "start_step": start_step,
            "current_step": start_step + num_steps,
            "target_step": start_step + num_steps,
            "best_loss": None if best_loss == float("inf") else best_loss,
            "best_step": best_step,
            "latest_loss": history[-1]["loss"] if history else None,
            "latest_lr": history[-1]["lr"] if history else current_learning_rate(optimizer),
            "latest_cross_scale_top1": history[-1]["cross_scale_top1_mean"] if history else None,
            "latest_local_context_top1": history[-1]["local_context_top1_mean"] if history else None,
            "latest_global_context_top1": history[-1]["global_context_top1_mean"] if history else None,
            "latest_loss_to_random_ratio": history[-1]["loss_to_random_ratio"] if history else None,
            "val_best_loss": min((float(item["loss"]) for item in val_history), default=None),
            "out_dir": str(out_path),
            "summary_path": str(summary_path),
            "best_checkpoint": str(best_checkpoint_path) if best_loss != float("inf") else None,
        },
        progress_path,
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a paired multiscale MarsCLIP alignment training smoke loop."
    )
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("marsclip_alignment_train"))
    parser.add_argument(
        "--stage-a-checkpoint",
        type=pathlib.Path,
        required=True,
        help="Selected Stage A MAE checkpoint reused as the visual backbone.",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-136.0, 12.0, -124.0, 24.0),
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--global-scale-factor", type=float, default=3.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--split-manifest", type=pathlib.Path, default=None)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="holdout",
        choices=("holdout", "kfold"),
    )
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=("adamw", "adam"),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=("none", "cosine", "onecycle"),
    )
    parser.add_argument("--scheduler-total-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--text-mode", type=str, default="raw_plus_expanded")
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--tokenizer-min-freq", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--text-hidden-dim", type=int, default=256)
    parser.add_argument("--text-depth", type=int, default=2)
    parser.add_argument("--text-heads", type=int, default=8)
    parser.add_argument("--location-hidden-dim", type=int, default=256)
    parser.add_argument("--location-frequencies", type=int, default=4)
    parser.add_argument("--geometry-hidden-dim", type=int, default=256)
    parser.add_argument("--geometry-fourier-dim", type=int, default=32)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--loss-weight-local-context", type=float, default=1.0)
    parser.add_argument("--loss-weight-global-context", type=float, default=1.0)
    parser.add_argument("--loss-weight-cross-scale", type=float, default=1.0)
    parser.add_argument("--freeze-visual-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=DEFAULT_PATCH_VALID_FRACTION,
    )
    parser.add_argument(
        "--color-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict paired crops to observations with full COLOR coverage.",
    )
    parser.add_argument("--patch-records-path", type=pathlib.Path, default=None)
    parser.add_argument("--paired-records-path", type=pathlib.Path, default=None)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    startup_progress_path = out_dir / "startup_progress.json"
    progress_path = out_dir / "progress.json"
    error_path = out_dir / "error_traceback.txt"
    save_training_progress(
        {
            "status": "running",
            "phase": "startup",
            "message": "Initializing paired multiscale alignment run.",
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    patch_records = None
    if args.patch_records_path is not None and args.patch_records_path.exists():
        print(f"[startup] loading cached patch records from {args.patch_records_path}", flush=True)
        patch_records = load_patch_records(args.patch_records_path)

    paired_records = None
    if args.paired_records_path is not None and args.paired_records_path.exists():
        print(f"[startup] loading cached paired records from {args.paired_records_path}", flush=True)
        paired_records = load_paired_crop_records(args.paired_records_path)

    print("[startup] building paired multiscale dataset", flush=True)
    dataset = MarsCLIPPairedCropDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=args.patch_size_deg,
        global_scale_factor=args.global_scale_factor,
        image_size=args.image_size,
        min_valid_fraction=args.min_valid_fraction,
        max_patches=args.max_patches,
        color_only=args.color_only,
        patch_records=patch_records,
        paired_records=paired_records,
    )
    if args.patch_records_path is not None and patch_records is None:
        save_patch_records(dataset.patch_records, args.patch_records_path)
    if args.paired_records_path is not None and paired_records is None:
        save_paired_crop_records(dataset.paired_records, args.paired_records_path)
    save_training_progress(
        {
            "status": "running",
            "phase": "dataset_ready",
            "message": "Paired multiscale dataset constructed.",
            "num_dataset_samples": len(dataset),
            "patch_records_path": str(args.patch_records_path) if args.patch_records_path is not None else None,
            "paired_records_path": str(args.paired_records_path) if args.paired_records_path is not None else None,
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    if args.split_manifest is not None:
        print("[startup] loading split manifest", flush=True)
        train_dataset, val_dataset, test_dataset = build_alignment_subsets(
            dataset,
            split_manifest=args.split_manifest,
            mode=args.split_mode,
            fold_index=args.fold_index,
        )
        split_summary_path = str(args.split_manifest)
    else:
        train_dataset = dataset
        val_dataset = None
        test_dataset = None
        split_summary_path = None
    save_training_progress(
        {
            "status": "running",
            "phase": "splits_ready",
            "message": "Train/val/test subsets ready.",
            "train_count": len(train_dataset),
            "val_count": len(val_dataset) if val_dataset is not None else None,
            "test_count": len(test_dataset) if test_dataset is not None else None,
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    tokenizer = build_alignment_tokenizer(
        train_dataset,
        text_mode=args.text_mode,
        min_freq=args.tokenizer_min_freq,
    )
    train_dataloader = build_alignment_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        text_mode=args.text_mode,
        max_length=args.text_max_length,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    val_dataloader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_dataloader = build_alignment_dataloader(
            val_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            shuffle=False,
            text_mode=args.text_mode,
            max_length=args.text_max_length,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
    save_training_progress(
        {
            "status": "running",
            "phase": "dataloaders_ready",
            "message": "Alignment dataloaders initialized.",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    alignment_config = {
        "embed_dim": args.embed_dim,
        "text_max_length": args.text_max_length,
        "text_hidden_dim": args.text_hidden_dim,
        "text_depth": args.text_depth,
        "text_heads": args.text_heads,
        "location_hidden_dim": args.location_hidden_dim,
        "location_frequencies": args.location_frequencies,
        "geometry_hidden_dim": args.geometry_hidden_dim,
        "geometry_fourier_dim": args.geometry_fourier_dim,
        "fusion_heads": args.fusion_heads,
        "loss_weight_local_context": args.loss_weight_local_context,
        "loss_weight_global_context": args.loss_weight_global_context,
        "loss_weight_cross_scale": args.loss_weight_cross_scale,
    }
    model, stage_a_state = load_alignment_model_from_stage_a_checkpoint(
        args.stage_a_checkpoint,
        vocab_size=tokenizer.vocab_size,
        map_location="cpu",
        freeze_visual_backbone=args.freeze_visual_backbone,
        alignment_config=alignment_config,
    )
    trainable_parameters, total_parameters = count_trainable_parameters(model)
    optimizer = build_alignment_optimizer(
        model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler_total_steps = args.scheduler_total_steps or args.num_steps
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        total_steps=int(scheduler_total_steps),
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    resume_state = None
    if args.resume_checkpoint is not None:
        resume_state = load_alignment_checkpoint(
            args.resume_checkpoint,
            model,
            optimizer,
            scheduler,
            map_location="cpu",
        )
        resume_state["checkpoint_path"] = str(args.resume_checkpoint)

    config = {
        "stage_a_checkpoint": str(args.stage_a_checkpoint),
        "stage_a_step": stage_a_state["stage_a_step"],
        "freeze_visual_backbone": args.freeze_visual_backbone,
        "bbox": list(args.bbox),
        "patch_size_deg": args.patch_size_deg,
        "global_scale_factor": args.global_scale_factor,
        "image_size": args.image_size,
        "max_patches": args.max_patches,
        "color_only": args.color_only,
        "min_valid_fraction": args.min_valid_fraction,
        "split_mode": args.split_mode if args.split_manifest is not None else None,
        "fold_index": args.fold_index,
        "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
        "split_summary_path": split_summary_path,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset) if val_dataset is not None else None,
        "test_count": len(test_dataset) if test_dataset is not None else None,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
        "num_steps": args.num_steps,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "scheduler_total_steps": int(scheduler_total_steps),
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "text_mode": args.text_mode,
        "text_max_length": args.text_max_length,
        "tokenizer_min_freq": args.tokenizer_min_freq,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "alignment_config": alignment_config,
        "checkpoint_every": args.checkpoint_every,
        "val_every": args.val_every,
        "val_max_batches": args.val_max_batches,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "patch_records_path": str(args.patch_records_path) if args.patch_records_path is not None else None,
        "paired_records_path": str(args.paired_records_path) if args.paired_records_path is not None else None,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }
    save_training_progress(
        {
            "status": "running",
            "phase": "training_ready",
            "message": "Alignment model, optimizer, and scheduler initialized.",
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    try:
        summary = run_alignment_training(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            out_dir=out_dir,
            num_steps=args.num_steps,
            device=args.device,
            resume_state=resume_state,
            config=config,
            checkpoint_every=args.checkpoint_every,
            val_dataloader=val_dataloader,
            val_every=args.val_every,
            val_max_batches=args.val_max_batches,
            max_grad_norm=args.max_grad_norm,
        )
    except Exception as exc:
        error_path.write_text(traceback.format_exc())
        save_training_progress(
            {
                "status": "failed",
                "phase": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback_path": str(error_path),
                "out_dir": str(out_dir),
            },
            progress_path,
        )
        raise

    print(f"Saved alignment checkpoint to {summary['checkpoint']}")
    print(f"Saved alignment history to {summary['history_path']}")
    print(f"Saved alignment loss curve to {summary['loss_curve']}")
    print(f"Saved alignment summary to {summary['summary_path']}")


if __name__ == "__main__":
    main()
