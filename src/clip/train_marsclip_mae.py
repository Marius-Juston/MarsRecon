"""Minimal Stage A MAE training utilities and CLI for MarsCLIP."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import tempfile
import time
import traceback
from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any

_MPL_CACHE = pathlib.Path(tempfile.gettempdir()) / "marsclip_artifacts" / "support" / "mpl_cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib as mpl
import torch
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset

mpl.use("Agg")
from matplotlib import pyplot as plt

from clip.marsclip_mae import MarsMAEOutput, MarsMaskedAutoencoder, collate_patch_samples_for_mae
from clip.marsclip_patches import (
    DEFAULT_PATCH_VALID_FRACTION,
    MarsCLIPPatchDataset,
    load_patch_records,
)
from clip.marsclip_splits import build_dataset_subsets, load_patch_split_manifest
from clip.visualize_marsclip_mae import save_mae_reconstruction_preview

MAE_MODEL_CONFIG_DEFAULTS: dict[str, Any] = {
    "image_size": 64,
    "patch_size_px": 16,
    "in_channels": 3,
    "encoder_dim": 64,
    "encoder_depth": 2,
    "encoder_heads": 4,
    "decoder_dim": 32,
    "decoder_depth": 1,
    "decoder_heads": 4,
    "min_valid_fraction": DEFAULT_PATCH_VALID_FRACTION,
    "normalize_inputs": False,
    "normalize_targets": False,
    "input_mean": None,
    "input_std": None,
}

MAE_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "mars_small": dict(MAE_MODEL_CONFIG_DEFAULTS),
    # Standard ViT-B/16-style MAE dimensions used by the broader MAE literature.
    "vit_base": {
        **dict(MAE_MODEL_CONFIG_DEFAULTS),
        "encoder_dim": 768,
        "encoder_depth": 12,
        "encoder_heads": 12,
        "decoder_dim": 512,
        "decoder_depth": 8,
        "decoder_heads": 16,
        "min_valid_fraction": DEFAULT_PATCH_VALID_FRACTION,
    },
}


@dataclass
class WandbLogger:
    """Small wrapper around an optional W&B run."""

    run: Any
    module: Any
    mode: str
    project: str
    run_name: str | None
    log_dir: str

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        self.run.log(dict(metrics), step=int(step))

    def log_image(self, key: str, path: pathlib.Path | str, *, step: int, caption: str | None = None) -> None:
        image = self.module.Image(str(path), caption=caption)
        self.run.log({key: image}, step=int(step))

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if summary:
            self.run.summary.update(dict(summary))
        self.run.finish()

    @property
    def run_id(self) -> str | None:
        return getattr(self.run, "id", None)

    @property
    def run_dir(self) -> str | None:
        return getattr(self.run, "dir", None)


def resolve_mae_model_config(
    config: dict[str, Any] | None = None,
    *,
    preset: str | None = None,
) -> dict[str, Any]:
    """Resolve checkpoint config values into MAE constructor kwargs."""
    preset_name = (preset or (config or {}).get("model_preset") or "mars_small").strip()
    if preset_name not in MAE_MODEL_PRESETS:
        available = ", ".join(sorted(MAE_MODEL_PRESETS))
        raise ValueError(f"Unknown MAE model preset '{preset_name}'. Available presets: {available}")

    preset_defaults = dict(MAE_MODEL_PRESETS[preset_name])
    source = dict(config or {})

    def _get(key: str, default_key: str | None = None) -> Any:
        value = source.get(key)
        if value is None:
            lookup_key = default_key or key
            return preset_defaults[lookup_key]
        return value

    return {
        "image_size": int(_get("image_size")),
        "patch_size": int(_get("patch_size_px", "patch_size_px") if source.get("patch_size") is None else _get("patch_size")),
        "in_channels": int(_get("in_channels")),
        "encoder_dim": int(_get("encoder_dim")),
        "encoder_depth": int(_get("encoder_depth")),
        "encoder_heads": int(_get("encoder_heads")),
        "decoder_dim": int(_get("decoder_dim")),
        "decoder_depth": int(_get("decoder_depth")),
        "decoder_heads": int(_get("decoder_heads")),
        "min_valid_fraction": float(_get("min_valid_fraction")),
        "normalize_inputs": bool(_get("normalize_inputs")),
        "normalize_targets": bool(_get("normalize_targets")),
        "input_mean": source.get("input_mean", preset_defaults.get("input_mean")),
        "input_std": source.get("input_std", preset_defaults.get("input_std")),
    }


def build_mae_model_from_config(config: dict[str, Any] | None = None) -> MarsMaskedAutoencoder:
    """Instantiate a Stage A MAE from a saved checkpoint config."""
    return MarsMaskedAutoencoder(**resolve_mae_model_config(config))


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts for a model."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def compute_valid_pixel_channel_stats(
    dataset: Dataset | list[dict[str, Any]],
    *,
    max_samples: int | None = None,
) -> tuple[list[float], list[float]]:
    """Compute per-channel mean/std using only pixels marked valid."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")

    sample_count = len(dataset)
    if sample_count == 0:
        raise ValueError("dataset must not be empty.")
    limit = sample_count if max_samples is None else min(sample_count, max_samples)

    channel_sum: torch.Tensor | None = None
    channel_sq_sum: torch.Tensor | None = None
    channel_count: torch.Tensor | None = None

    for idx in range(limit):
        sample = dataset[idx]
        image = torch.as_tensor(sample["image"], dtype=torch.float64)
        valid_mask = torch.as_tensor(sample["valid_mask"], dtype=torch.bool)
        if image.ndim != 3:
            raise ValueError("dataset samples must provide image tensors shaped (C, H, W).")
        if valid_mask.shape != image.shape[-2:]:
            raise ValueError("dataset sample valid_mask must match image spatial dimensions.")
        expanded_valid = valid_mask.unsqueeze(0).expand(image.shape[0], -1, -1)
        masked = torch.where(expanded_valid, image, torch.zeros_like(image))
        valid_counts = expanded_valid.to(torch.float64).sum(dim=(1, 2))

        if channel_sum is None:
            channels = image.shape[0]
            channel_sum = torch.zeros(channels, dtype=torch.float64)
            channel_sq_sum = torch.zeros(channels, dtype=torch.float64)
            channel_count = torch.zeros(channels, dtype=torch.float64)

        channel_sum += masked.sum(dim=(1, 2))
        channel_sq_sum += masked.pow(2).sum(dim=(1, 2))
        channel_count += valid_counts

    assert channel_sum is not None and channel_sq_sum is not None and channel_count is not None
    if not bool((channel_count > 0).all()):
        raise ValueError("All channels must have at least one valid pixel to compute normalization stats.")

    mean = channel_sum / channel_count
    variance = (channel_sq_sum / channel_count) - mean.pow(2)
    std = variance.clamp_min(1e-12).sqrt()
    return mean.to(torch.float32).tolist(), std.to(torch.float32).tolist()


def build_optimizer(
    model: torch.nn.Module,
    *,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
) -> Optimizer:
    """Build the requested optimizer for Stage A training."""
    normalized_name = optimizer_name.lower().strip()
    if normalized_name == "adamw":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if normalized_name == "adam":
        return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError("optimizer_name must be one of: adamw, adam.")


def build_scheduler(
    optimizer: Optimizer,
    *,
    scheduler_name: str = "none",
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
) -> LRScheduler | None:
    """Build an optional per-step learning-rate scheduler."""
    normalized_name = scheduler_name.lower().strip()
    if normalized_name == "none":
        return None
    if total_steps <= 0:
        raise ValueError("total_steps must be positive when a scheduler is enabled.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps.")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError("min_lr_ratio must satisfy 0 <= value <= 1.")

    if normalized_name == "cosine":
        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step + 1) / float(warmup_steps), 1e-8)
            if total_steps == warmup_steps:
                return min_lr_ratio
            progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

        return LambdaLR(optimizer, lr_lambda=_lr_lambda)

    if normalized_name == "onecycle":
        pct_start = float(warmup_steps) / float(total_steps) if warmup_steps > 0 else 0.1
        pct_start = min(max(pct_start, 0.01), 0.99)
        max_lrs = [group["lr"] for group in optimizer.param_groups]
        return OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

    raise ValueError("scheduler_name must be one of: none, cosine, onecycle.")


def current_learning_rate(optimizer: Optimizer) -> float:
    """Return the learning rate from the first optimizer param group."""
    return float(optimizer.param_groups[0]["lr"])


def init_wandb_logger(
    *,
    mode: str = "disabled",
    project: str = "marsclip-stagea",
    run_name: str | None = None,
    out_dir: pathlib.Path | str = pathlib.Path("../../tests"),
    log_dir: pathlib.Path | str | None = None,
    config: dict[str, Any] | None = None,
) -> WandbLogger | None:
    """Initialize an optional W&B run.

    The repo defaults to disabled logging so tests and simple runs do not
    require the dependency. Users can opt into `offline` mode for local-only
    dashboards or `online` when they want to sync to W&B.
    """
    normalized_mode = mode.lower().strip()
    if normalized_mode == "disabled":
        return None
    if normalized_mode not in {"offline", "online"}:
        raise ValueError("W&B mode must be one of: disabled, offline, online.")
    try:
        wandb = importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wandb is not installed. Install it before using --wandb-mode offline or online."
        ) from exc

    resolved_log_dir = pathlib.Path(log_dir) if log_dir is not None else pathlib.Path(out_dir) / "wandb"
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        name=run_name,
        mode=normalized_mode,
        dir=str(resolved_log_dir),
        config=dict(config or {}),
        reinit="finish_previous",
    )
    return WandbLogger(
        run=run,
        module=wandb,
        mode=normalized_mode,
        project=project,
        run_name=run_name,
        log_dir=str(resolved_log_dir),
    )


def build_mae_dataloader(
    dataset: Dataset | list[dict[str, Any]],
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
) -> DataLoader:
    """Build a DataLoader that emits Stage A MAE-ready patch batches."""
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_patch_samples_for_mae,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _resolve_device(device: str | torch.device | None, model: torch.nn.Module) -> torch.device:
    if isinstance(device, str) and device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device is None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    return torch.device(device)


def resolve_map_location(map_location: str | torch.device = "cpu") -> str | torch.device:
    """Normalize torch.load map_location values, including the repo's 'auto' alias."""
    if isinstance(map_location, str) and map_location.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return map_location


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _output_to_cpu(output: MarsMAEOutput) -> MarsMAEOutput:
    return MarsMAEOutput(
        encoded_tokens=output.encoded_tokens.detach().cpu(),
        pooled_embedding=output.pooled_embedding.detach().cpu(),
        patch_valid_fraction=output.patch_valid_fraction.detach().cpu(),
        patch_valid_mask=output.patch_valid_mask.detach().cpu(),
        visible_mask=output.visible_mask.detach().cpu(),
        masked_valid_mask=output.masked_valid_mask.detach().cpu(),
        valid_pixel_mask=output.valid_pixel_mask.detach().cpu(),
        loss_mask=output.loss_mask.detach().cpu(),
        reconstruction=output.reconstruction.detach().cpu(),
        loss=output.loss.detach().cpu(),
    )


def train_mae_batch(
    model: MarsMaskedAutoencoder,
    batch: dict[str, Any],
    optimizer: Optimizer,
    *,
    mask_ratio: float = 0.75,
    device: str | torch.device | None = None,
    step: int | None = None,
) -> tuple[dict[str, float], MarsMAEOutput]:
    """Run one MAE optimization step on a collated batch."""
    resolved_device = _resolve_device(device, model)
    model.to(resolved_device)
    model.train()

    batch = _move_batch_to_device(batch, resolved_device)
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_patch_batch(batch, mask_ratio=mask_ratio)
    loss = output.loss
    if not torch.isfinite(loss):
        raise ValueError("Encountered non-finite MAE loss.")
    loss.backward()
    optimizer.step()

    entry = {
        "loss": float(loss.detach().cpu()),
        "visible_patch_fraction": float(output.visible_mask.float().mean().detach().cpu()),
        "masked_valid_patch_fraction": float(output.masked_valid_mask.float().mean().detach().cpu()),
        "loss_pixel_fraction": float(output.loss_mask.float().mean().detach().cpu()),
    }
    if step is not None:
        entry["step"] = float(step)
    return entry, _output_to_cpu(output)


def train_mae_steps(
    model: MarsMaskedAutoencoder,
    dataloader: DataLoader,
    optimizer: Optimizer,
    *,
    num_steps: int,
    mask_ratio: float = 0.75,
    device: str | torch.device | None = None,
    start_step: int = 0,
    step_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    """Run a fixed number of MAE optimization steps, cycling the dataloader if needed."""
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")

    resolved_device = _resolve_device(device, model)
    model.to(resolved_device)
    model.train()

    history: list[dict[str, float]] = []
    iterator: Iterable[Any] | Any = iter(dataloader)
    for step_idx in range(num_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        step = start_step + step_idx + 1
        entry, _ = train_mae_batch(
            model,
            batch,
            optimizer,
            mask_ratio=mask_ratio,
            device=resolved_device,
            step=step,
        )
        history.append(entry)
        if step_callback is not None:
            step_callback(step, entry)
    return history


def evaluate_mae_dataloader(
    model: MarsMaskedAutoencoder,
    dataloader: DataLoader,
    *,
    mask_ratio: float = 0.75,
    device: str | torch.device | None = None,
    max_batches: int | None = None,
    batch_callback: Callable[[int, int | None], None] | None = None,
) -> dict[str, float]:
    """Evaluate MAE reconstruction loss on a validation/test dataloader."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    resolved_device = _resolve_device(device, model)
    model.to(resolved_device)
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_visible = 0.0
    total_masked = 0.0
    total_loss_pixels = 0.0
    num_batches = 0
    total_batches: int | None
    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None
    if max_batches is not None:
        total_batches = min(total_batches, max_batches) if total_batches is not None else max_batches

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if batch_callback is not None:
                batch_callback(batch_idx + 1, total_batches)
            batch = _move_batch_to_device(batch, resolved_device)
            output = model.forward_patch_batch(batch, mask_ratio=mask_ratio)
            total_loss += float(output.loss.detach().cpu())
            total_visible += float(output.visible_mask.float().mean().detach().cpu())
            total_masked += float(output.masked_valid_mask.float().mean().detach().cpu())
            total_loss_pixels += float(output.loss_mask.float().mean().detach().cpu())
            num_batches += 1

    if was_training:
        model.train()
    if num_batches == 0:
        raise ValueError("Validation/test dataloader produced no batches.")

    return {
        "loss": total_loss / float(num_batches),
        "visible_patch_fraction": total_visible / float(num_batches),
        "masked_valid_patch_fraction": total_masked / float(num_batches),
        "loss_pixel_fraction": total_loss_pixels / float(num_batches),
        "num_batches": float(num_batches),
    }


def save_mae_checkpoint(
    path: pathlib.Path | str,
    model: MarsMaskedAutoencoder,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None = None,
    *,
    step: int,
    history: list[dict[str, float]],
    config: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Save a minimal MAE training checkpoint."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "history": history,
            "config": config or {},
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        },
        out,
    )
    return out


def load_mae_checkpoint(
    path: pathlib.Path | str,
    model: MarsMaskedAutoencoder,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a MAE checkpoint into a model and optional optimizer."""
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


def save_training_history(
    history: list[dict[str, float]],
    out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Persist MAE loss history as JSON."""
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(history, indent=2))
    return out


def select_best_history_entry(history: list[dict[str, float]]) -> dict[str, float]:
    """Return the minimum-loss training record."""
    if not history:
        raise ValueError("history must not be empty.")
    return min(history, key=lambda item: float(item["loss"]))


def save_training_summary(
    summary: dict[str, Any],
    out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Persist the MAE training summary as JSON."""
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return out


def _progress_timestamp() -> str:
    """Return a compact local timestamp for training progress artifacts."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_training_progress(
    progress: dict[str, Any],
    out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Persist an incremental MAE training progress snapshot as JSON."""
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(progress)
    payload["updated_at"] = _progress_timestamp()
    out.write_text(json.dumps(payload, indent=2))
    return out


def load_normalization_stats(path: pathlib.Path | str) -> tuple[list[float], list[float], dict[str, Any]]:
    """Load cached normalization statistics from JSON."""
    source = pathlib.Path(path)
    payload = json.loads(source.read_text())
    mean = [float(value) for value in payload["mean"]]
    std = [float(value) for value in payload["std"]]
    return mean, std, payload


def save_normalization_stats(
    path: pathlib.Path | str,
    *,
    mean: list[float],
    std: list[float],
    metadata: dict[str, Any] | None = None,
) -> pathlib.Path:
    """Persist normalization statistics for reuse across comparable runs."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }
    if metadata:
        payload.update(dict(metadata))
    out.write_text(json.dumps(payload, indent=2))
    return out


def save_loss_curve(
    history: list[dict[str, float]],
    out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Save a simple MAE loss curve PNG using a log-scaled y-axis."""
    if not history:
        raise ValueError("history must not be empty.")

    steps = [entry["step"] for entry in history]
    losses = [entry["loss"] for entry in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses, color="#d95f02", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("MAE loss")
    ax.set_yscale("log")
    ax.set_title("MarsCLIP Stage A training loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def generate_mae_preview(
    model: MarsMaskedAutoencoder,
    samples: list[dict[str, Any]],
    out_path: pathlib.Path | str,
    *,
    mask_ratio: float,
    device: str | torch.device | None = None,
) -> pathlib.Path:
    """Generate a masked/reconstructed preview from a small sample set."""
    if not samples:
        raise ValueError("samples must not be empty.")

    resolved_device = _resolve_device(device, model)
    was_training = model.training
    batch = collate_patch_samples_for_mae(samples)
    batch = _move_batch_to_device(batch, resolved_device)
    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        output = model.forward_patch_batch(
            batch,
            mask_ratio=mask_ratio,
            generator=torch.Generator().manual_seed(0),
        )
    cpu_output = _output_to_cpu(output)
    if was_training:
        model.train()

    return save_mae_reconstruction_preview(
        samples,
        cpu_output,
        out_path,
        patch_size=model.patch_size,
        max_items=len(samples),
    )


def run_mae_training(
    model: MarsMaskedAutoencoder,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    *,
    out_dir: pathlib.Path | str,
    num_steps: int,
    mask_ratio: float,
    preview_samples: list[dict[str, Any]] | None = None,
    checkpoint_every: int = 0,
    preview_every: int = 0,
    device: str | torch.device | None = None,
    resume_state: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    wandb_logger: WandbLogger | None = None,
    val_dataloader: DataLoader | None = None,
    val_every: int = 0,
    val_max_batches: int | None = None,
) -> dict[str, Any]:
    """Run a small Stage A training session with periodic artifacts."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    history = list((resume_state or {}).get("history", []))
    start_step = int((resume_state or {}).get("step", 0))
    checkpoint_paths: list[str] = []
    preview_paths: list[str] = []
    progress_path = out_path / "progress.json"

    best_entry = select_best_history_entry(history) if history and val_dataloader is None else None
    best_loss = float(best_entry["loss"]) if best_entry is not None else float("inf")
    best_step = int(best_entry["step"]) if best_entry is not None else start_step
    best_metric_name = "val_loss" if val_dataloader is not None else "train_loss"
    best_checkpoint_path = out_path / "best_checkpoint.pt"
    val_history: list[dict[str, float]] = []

    save_training_progress(
        {
            "status": "running",
            "phase": "starting",
            "start_step": start_step,
            "current_step": start_step,
            "target_step": start_step + num_steps,
            "best_loss": None,
            "best_step": start_step,
            "latest_loss": None,
            "latest_lr": current_learning_rate(optimizer),
            "val_best_loss": None,
            "out_dir": str(out_path),
        },
        progress_path,
    )

    start_preview = None
    if preview_samples:
        start_preview = generate_mae_preview(
            model,
            preview_samples,
            out_path / f"preview_step_{start_step:06d}.png",
            mask_ratio=mask_ratio,
            device=device,
        )
        preview_paths.append(str(start_preview))
        if wandb_logger is not None:
            wandb_logger.log_image(
                "train/preview_start",
                start_preview,
                step=start_step,
                caption=f"Stage A preview at step {start_step}",
            )
        save_training_progress(
            {
                "status": "running",
                "phase": "preview_start",
                "start_step": start_step,
                "current_step": start_step,
                "target_step": start_step + num_steps,
                "best_loss": best_loss if best_loss != float("inf") else None,
                "best_step": best_step,
                "latest_loss": None,
                "latest_lr": current_learning_rate(optimizer),
                "val_best_loss": None,
                "out_dir": str(out_path),
                "latest_preview": str(start_preview),
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
        entry, _ = train_mae_batch(
            model,
            batch,
            optimizer,
            mask_ratio=mask_ratio,
            device=resolved_device,
            step=step,
        )
        if scheduler is not None:
            scheduler.step()
        entry["lr"] = current_learning_rate(optimizer)
        history.append(entry)
        if wandb_logger is not None:
            wandb_logger.log_metrics(
                {
                    "train/loss": float(entry["loss"]),
                    "train/lr": float(entry["lr"]),
                    "train/visible_patch_fraction": float(entry["visible_patch_fraction"]),
                    "train/masked_valid_patch_fraction": float(entry["masked_valid_patch_fraction"]),
                    "train/loss_pixel_fraction": float(entry["loss_pixel_fraction"]),
                },
                step=step,
            )
        current_val_best = min((float(item["loss"]) for item in val_history), default=None)
        save_training_progress(
            {
                "status": "running",
                "phase": "training",
                "start_step": start_step,
                "current_step": step,
                "target_step": start_step + num_steps,
                "best_loss": best_loss if best_loss != float("inf") else None,
                "best_step": best_step,
                "latest_loss": float(entry["loss"]),
                "latest_lr": float(entry["lr"]),
                "val_best_loss": current_val_best,
                "out_dir": str(out_path),
            },
            progress_path,
        )

        if val_dataloader is None and float(entry["loss"]) <= best_loss:
            best_loss = float(entry["loss"])
            best_step = step
            save_mae_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                step=step,
                history=history,
                config=config,
            )
            if wandb_logger is not None:
                wandb_logger.log_metrics(
                    {
                        "train/best_loss": float(best_loss),
                        "train/best_step": float(best_step),
                        "train/best_lr": float(entry["lr"]),
                    },
                    step=step,
                )

        should_run_val = val_dataloader is not None and (
            (val_every > 0 and step % val_every == 0)
            or (val_every <= 0 and checkpoint_every > 0 and step % checkpoint_every == 0)
        )
        if should_run_val:
            try:
                expected_val_batches = len(val_dataloader)
            except TypeError:
                expected_val_batches = None
            if val_max_batches is not None:
                expected_val_batches = (
                    min(expected_val_batches, val_max_batches)
                    if expected_val_batches is not None
                    else val_max_batches
                )
            save_training_progress(
                {
                    "status": "running",
                    "phase": "validating",
                    "start_step": start_step,
                    "current_step": step,
                    "target_step": start_step + num_steps,
                    "best_loss": best_loss if best_loss != float("inf") else None,
                    "best_step": best_step,
                    "latest_loss": float(entry["loss"]),
                    "latest_lr": float(entry["lr"]),
                    "val_progress_batches_completed": 0,
                    "val_progress_batches_total": expected_val_batches,
                    "val_best_loss": min((float(item["loss"]) for item in val_history), default=None),
                    "out_dir": str(out_path),
                },
                progress_path,
            )

            def _on_val_batch(batch_number: int, total_batches: int | None) -> None:
                if batch_number == 1 or batch_number % 5 == 0 or batch_number == total_batches:
                    save_training_progress(
                        {
                            "status": "running",
                            "phase": "validating",
                            "start_step": start_step,
                            "current_step": step,
                            "target_step": start_step + num_steps,
                            "best_loss": best_loss if best_loss != float("inf") else None,
                            "best_step": best_step,
                            "latest_loss": float(entry["loss"]),
                            "latest_lr": float(entry["lr"]),
                            "val_progress_batches_completed": batch_number,
                            "val_progress_batches_total": total_batches,
                            "val_best_loss": min((float(item["loss"]) for item in val_history), default=None),
                            "out_dir": str(out_path),
                        },
                        progress_path,
                    )

            val_entry = evaluate_mae_dataloader(
                model,
                val_dataloader,
                mask_ratio=mask_ratio,
                device=resolved_device,
                max_batches=val_max_batches,
                batch_callback=_on_val_batch,
            )
            val_entry["step"] = float(step)
            val_history.append(val_entry)
            if wandb_logger is not None:
                wandb_logger.log_metrics(
                    {
                        "val/loss": float(val_entry["loss"]),
                        "val/visible_patch_fraction": float(val_entry["visible_patch_fraction"]),
                        "val/masked_valid_patch_fraction": float(val_entry["masked_valid_patch_fraction"]),
                        "val/loss_pixel_fraction": float(val_entry["loss_pixel_fraction"]),
                    },
                    step=step,
                )
            save_training_progress(
                {
                    "status": "running",
                    "phase": "validating",
                    "start_step": start_step,
                    "current_step": step,
                    "target_step": start_step + num_steps,
                    "best_loss": best_loss if best_loss != float("inf") else None,
                    "best_step": best_step,
                    "latest_loss": float(entry["loss"]),
                    "latest_lr": float(entry["lr"]),
                    "val_latest_loss": float(val_entry["loss"]),
                    "val_best_loss": min(float(item["loss"]) for item in val_history),
                    "out_dir": str(out_path),
                },
                progress_path,
            )
            if float(val_entry["loss"]) <= best_loss:
                best_loss = float(val_entry["loss"])
                best_step = step
                save_mae_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    history=history,
                    config=config,
                )
                if wandb_logger is not None:
                    wandb_logger.log_metrics(
                        {
                            "val/best_loss": float(best_loss),
                            "val/best_step": float(best_step),
                        },
                        step=step,
                    )

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt = save_mae_checkpoint(
                out_path / f"checkpoint_step_{step:06d}.pt",
                model,
                optimizer,
                scheduler,
                step=step,
                history=history,
                config=config,
            )
            checkpoint_paths.append(str(ckpt))

        if preview_samples and preview_every > 0 and step % preview_every == 0:
            preview = generate_mae_preview(
                model,
                preview_samples,
                out_path / f"preview_step_{step:06d}.png",
                mask_ratio=mask_ratio,
                device=resolved_device,
            )
            preview_paths.append(str(preview))
            if wandb_logger is not None:
                wandb_logger.log_image(
                    "train/preview_periodic",
                    preview,
                    step=step,
                    caption=f"Stage A preview at step {step}",
                )

    final_step = start_step + num_steps
    final_preview = None
    if preview_samples:
        final_preview = generate_mae_preview(
            model,
            preview_samples,
            out_path / "preview_final.png",
            mask_ratio=mask_ratio,
            device=resolved_device,
        )
        preview_paths.append(str(final_preview))
        if wandb_logger is not None:
            wandb_logger.log_image(
                "train/preview_final",
                final_preview,
                step=final_step,
                caption=f"Stage A preview at step {final_step}",
            )

    checkpoint_path = save_mae_checkpoint(
        out_path / "checkpoint.pt",
        model,
        optimizer,
        scheduler,
        step=final_step,
        history=history,
        config=config,
    )
    final_val_entry = None
    if val_dataloader is not None and (not val_history or int(val_history[-1]["step"]) != final_step):
        final_val_entry = evaluate_mae_dataloader(
            model,
            val_dataloader,
            mask_ratio=mask_ratio,
            device=resolved_device,
            max_batches=val_max_batches,
        )
        final_val_entry["step"] = float(final_step)
        val_history.append(final_val_entry)
        if wandb_logger is not None:
            wandb_logger.log_metrics(
                {
                    "val/loss": float(final_val_entry["loss"]),
                    "val/visible_patch_fraction": float(final_val_entry["visible_patch_fraction"]),
                    "val/masked_valid_patch_fraction": float(final_val_entry["masked_valid_patch_fraction"]),
                    "val/loss_pixel_fraction": float(final_val_entry["loss_pixel_fraction"]),
                },
                step=final_step,
            )
        if float(final_val_entry["loss"]) <= best_loss:
            best_loss = float(final_val_entry["loss"])
            best_step = final_step
            save_mae_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                step=final_step,
                history=history,
                config=config,
            )

    history_path = save_training_history(history, out_path / "history.json")
    curve_path = save_loss_curve(history, out_path / "loss_curve.png")
    val_history_path = save_training_history(val_history, out_path / "val_history.json") if val_history else None

    best_metric_label = "training loss" if best_metric_name == "train_loss" else "val loss"

    summary = {
        "start_step": start_step,
        "num_new_steps": num_steps,
        "num_steps_total": len(history),
        "model_preset": (config or {}).get("model_preset"),
        "trainable_parameters": int((config or {}).get("trainable_parameters"))
        if (config or {}).get("trainable_parameters") is not None
        else None,
        "total_parameters": int((config or {}).get("total_parameters"))
        if (config or {}).get("total_parameters") is not None
        else None,
        "batch_size": int((config or {}).get("batch_size")) if (config or {}).get("batch_size") is not None else None,
        "color_only": bool((config or {}).get("color_only"))
        if (config or {}).get("color_only") is not None
        else None,
        "optimizer": (config or {}).get("optimizer"),
        "scheduler": (config or {}).get("scheduler"),
        "normalization_mode": (config or {}).get("normalization_mode"),
        "normalization_max_samples": int((config or {}).get("normalization_max_samples"))
        if (config or {}).get("normalization_max_samples") is not None
        else None,
        "normalization_stats_path": str((config or {}).get("normalization_stats_path"))
        if (config or {}).get("normalization_stats_path") is not None
        else None,
        "patch_records_path": str((config or {}).get("patch_records_path"))
        if (config or {}).get("patch_records_path") is not None
        else None,
        "normalize_inputs": bool((config or {}).get("normalize_inputs"))
        if (config or {}).get("normalize_inputs") is not None
        else None,
        "normalize_targets": bool((config or {}).get("normalize_targets"))
        if (config or {}).get("normalize_targets") is not None
        else None,
        "input_mean": list((config or {}).get("input_mean"))
        if (config or {}).get("input_mean") is not None
        else None,
        "input_std": list((config or {}).get("input_std"))
        if (config or {}).get("input_std") is not None
        else None,
        "scheduler_total_steps": int((config or {}).get("scheduler_total_steps"))
        if (config or {}).get("scheduler_total_steps") is not None
        else None,
        "warmup_steps": int((config or {}).get("warmup_steps")) if (config or {}).get("warmup_steps") is not None else None,
        "min_lr_ratio": float((config or {}).get("min_lr_ratio"))
        if (config or {}).get("min_lr_ratio") is not None
        else None,
        "split_mode": (config or {}).get("split_mode"),
        "split_manifest": str((config or {}).get("split_manifest"))
        if (config or {}).get("split_manifest") is not None
        else None,
        "train_count": int((config or {}).get("train_count")) if (config or {}).get("train_count") is not None else None,
        "val_count": int((config or {}).get("val_count")) if (config or {}).get("val_count") is not None else None,
        "test_count": int((config or {}).get("test_count")) if (config or {}).get("test_count") is not None else None,
        "split_summary_path": str((config or {}).get("split_summary_path"))
        if (config or {}).get("split_summary_path") is not None
        else None,
        "num_workers": int((config or {}).get("num_workers")) if (config or {}).get("num_workers") is not None else None,
        "pin_memory": bool((config or {}).get("pin_memory")) if (config or {}).get("pin_memory") is not None else None,
        "prefetch_factor": int((config or {}).get("prefetch_factor")) if (config or {}).get("prefetch_factor") is not None else None,
        "persistent_workers": bool((config or {}).get("persistent_workers")) if (config or {}).get("persistent_workers") is not None else None,
        "initial_loss": history[0]["loss"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
        "initial_lr": history[0]["lr"] if history and "lr" in history[0] else current_learning_rate(optimizer),
        "final_lr": history[-1]["lr"] if history and "lr" in history[-1] else current_learning_rate(optimizer),
        "best_loss": best_loss if history else None,
        "best_metric_name": best_metric_name,
        "best_step": best_step if history else None,
        "val_initial_loss": val_history[0]["loss"] if val_history else None,
        "val_final_loss": val_history[-1]["loss"] if val_history else None,
        "val_best_loss": min(float(entry["loss"]) for entry in val_history) if val_history else None,
        "val_history_path": str(val_history_path) if val_history_path is not None else None,
        "val_every": int(val_every) if val_every is not None else None,
        "val_max_batches": int(val_max_batches) if val_max_batches is not None else None,
        "resume_step": start_step if resume_state is not None else None,
        "resume_checkpoint": str((resume_state or {}).get("checkpoint_path"))
        if resume_state is not None and (resume_state or {}).get("checkpoint_path") is not None
        else None,
        "start_preview": str(start_preview) if start_preview is not None else None,
        "final_preview": str(final_preview) if final_preview is not None else None,
        "preview_paths": preview_paths,
        "checkpoint": str(checkpoint_path),
        "checkpoint_paths": checkpoint_paths,
        "best_checkpoint": str(best_checkpoint_path) if history else None,
        "best_checkpoint_rule": f"minimum observed {best_metric_label}",
        "history_path": str(history_path),
        "loss_curve": str(curve_path),
        "progress_path": str(progress_path),
        "wandb_mode": wandb_logger.mode if wandb_logger is not None else "disabled",
        "wandb_project": wandb_logger.project if wandb_logger is not None else None,
        "wandb_run_name": wandb_logger.run_name if wandb_logger is not None else None,
        "wandb_run_id": wandb_logger.run_id if wandb_logger is not None else None,
        "wandb_log_dir": wandb_logger.log_dir if wandb_logger is not None else None,
        "wandb_run_dir": wandb_logger.run_dir if wandb_logger is not None else None,
    }
    if wandb_logger is not None:
        wandb_logger.log_image(
            "train/loss_curve",
            curve_path,
            step=final_step,
            caption="MarsCLIP Stage A training loss curve",
        )
        wandb_logger.finish(summary)
    summary_path = save_training_summary(summary, out_path / "summary.json")
    save_training_progress(
        {
            "status": "completed",
            "phase": "finished",
            "start_step": start_step,
            "current_step": final_step,
            "target_step": final_step,
            "best_loss": best_loss if history else None,
            "best_step": best_step if history else None,
            "latest_loss": history[-1]["loss"] if history else None,
            "latest_lr": history[-1]["lr"] if history and "lr" in history[-1] else current_learning_rate(optimizer),
            "val_best_loss": min((float(item["loss"]) for item in val_history), default=None),
            "out_dir": str(out_path),
            "summary_path": str(summary_path),
            "best_checkpoint": str(best_checkpoint_path) if history else None,
            "final_preview": str(final_preview) if final_preview is not None else None,
        },
        progress_path,
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run a minimal Stage A MAE training loop.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("marsclip_mae_train"),
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-136.0, 12.0, -124.0, 24.0),
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--patch-size-px", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--split-manifest", type=pathlib.Path, default=None)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="holdout",
        choices=("holdout", "kfold"),
    )
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-steps", type=int, default=4)
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
        default="none",
        choices=("none", "cosine", "onecycle"),
    )
    parser.add_argument("--scheduler-total-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument(
        "--normalization-mode",
        type=str,
        default="none",
        choices=("none", "input", "input_and_target"),
    )
    parser.add_argument("--normalization-max-samples", type=int, default=None)
    parser.add_argument("--preview-items", type=int, default=2)
    parser.add_argument(
        "--model-preset",
        type=str,
        default="mars_small",
        choices=tuple(sorted(MAE_MODEL_PRESETS)),
    )
    parser.add_argument("--encoder-dim", type=int, default=None)
    parser.add_argument("--encoder-depth", type=int, default=None)
    parser.add_argument("--encoder-heads", type=int, default=None)
    parser.add_argument("--decoder-dim", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--decoder-heads", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=("disabled", "offline", "online"),
    )
    parser.add_argument("--wandb-project", type=str, default="marsclip-stagea")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-dir", type=pathlib.Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--preview-every", type=int, default=0)
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
        help="Restrict Stage A patches to observations with full COLOR coverage.",
    )
    parser.add_argument(
        "--normalization-stats-path",
        type=pathlib.Path,
        default=None,
        help="Optional JSON cache for input normalization statistics shared across comparable runs.",
    )
    parser.add_argument(
        "--patch-records-path",
        type=pathlib.Path,
        default=None,
        help="Optional cache of prebuilt patch records to skip full patch-table reconstruction at startup.",
    )
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
            "message": "Initializing Stage A training run.",
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )
    print("[startup] building patch dataset", flush=True)
    patch_records = None
    if args.patch_records_path is not None and args.patch_records_path.exists():
        print(f"[startup] loading cached patch records from {args.patch_records_path}", flush=True)
        patch_records = load_patch_records(args.patch_records_path)

    dataset = MarsCLIPPatchDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=args.patch_size_deg,
        image_size=args.image_size,
        max_patches=args.max_patches,
        min_valid_fraction=args.min_valid_fraction,
        color_only=args.color_only,
        patch_records=patch_records,
    )
    save_training_progress(
        {
            "status": "running",
            "phase": "dataset_ready",
            "message": "Patch dataset constructed.",
            "num_dataset_samples": len(dataset),
            "patch_records_path": str(args.patch_records_path) if args.patch_records_path is not None else None,
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )
    print(f"[startup] dataset ready with {len(dataset)} patches", flush=True)
    if args.split_manifest is not None:
        print("[startup] loading split manifest", flush=True)
        split_manifest = load_patch_split_manifest(args.split_manifest)
        train_dataset, val_dataset, test_dataset = build_dataset_subsets(
            dataset,
            dataset.patch_records,
            split_manifest,
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
    print(
        f"[startup] split sizes train={len(train_dataset)} val={len(val_dataset) if val_dataset is not None else 0} test={len(test_dataset) if test_dataset is not None else 0}",
        flush=True,
    )

    dataloader = build_mae_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    val_dataloader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_dataloader = build_mae_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            generator=None,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
    save_training_progress(
        {
            "status": "running",
            "phase": "dataloaders_ready",
            "message": "Dataloaders initialized.",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )
    print("[startup] dataloaders ready", flush=True)
    preview_count = min(args.preview_items, len(train_dataset))
    if preview_count > 0:
        print(f"[startup] collecting {preview_count} preview samples", flush=True)
    preview_samples = [train_dataset[i] for i in range(preview_count)]
    save_training_progress(
        {
            "status": "running",
            "phase": "preview_ready",
            "message": "Preview samples collected.",
            "preview_count": preview_count,
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    input_mean = None
    input_std = None
    normalize_inputs = args.normalization_mode in {"input", "input_and_target"}
    normalize_targets = args.normalization_mode == "input_and_target"
    if args.normalization_mode != "none":
        if args.normalization_stats_path is not None and args.normalization_stats_path.exists():
            print(
                f"[startup] loading cached normalization stats from {args.normalization_stats_path}",
                flush=True,
            )
            input_mean, input_std, _ = load_normalization_stats(args.normalization_stats_path)
        else:
            print(
                f"[startup] computing normalization stats from up to {args.normalization_max_samples} samples",
                flush=True,
            )
            input_mean, input_std = compute_valid_pixel_channel_stats(
                train_dataset,
                max_samples=args.normalization_max_samples,
            )
            if args.normalization_stats_path is not None:
                save_normalization_stats(
                    args.normalization_stats_path,
                    mean=input_mean,
                    std=input_std,
                    metadata={
                        "created_at": _progress_timestamp(),
                        "normalization_mode": args.normalization_mode,
                        "normalization_max_samples": args.normalization_max_samples,
                        "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
                        "color_only": bool(args.color_only),
                        "bbox": list(args.bbox),
                    },
                )
        save_training_progress(
            {
                "status": "running",
                "phase": "normalization_ready",
                "message": "Normalization statistics ready.",
                "normalization_mode": args.normalization_mode,
                "normalization_stats_path": str(args.normalization_stats_path)
                if args.normalization_stats_path is not None
                else None,
                "out_dir": str(out_dir),
            },
            startup_progress_path,
        )
        print("[startup] normalization stats ready", flush=True)

    model_config = resolve_mae_model_config(
        {
            "image_size": args.image_size,
            "patch_size_px": args.patch_size_px,
            "in_channels": 3,
            "encoder_dim": args.encoder_dim,
            "encoder_depth": args.encoder_depth,
            "encoder_heads": args.encoder_heads,
            "decoder_dim": args.decoder_dim,
            "decoder_depth": args.decoder_depth,
            "decoder_heads": args.decoder_heads,
            "min_valid_fraction": args.min_valid_fraction,
            "model_preset": args.model_preset,
            "normalize_inputs": normalize_inputs,
            "normalize_targets": normalize_targets,
            "input_mean": input_mean,
            "input_std": input_std,
        },
        preset=args.model_preset,
    )
    model = MarsMaskedAutoencoder(**model_config)
    trainable_parameters, total_parameters = count_trainable_parameters(model)
    optimizer = build_optimizer(
        model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    resume_state = None
    resume_config: dict[str, Any] = {}
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=resolve_map_location("cpu"))
        resume_config = dict(checkpoint.get("config", {}))

    scheduler_total_steps = args.scheduler_total_steps
    if scheduler_total_steps is None:
        scheduler_total_steps = int(resume_config.get("scheduler_total_steps", args.num_steps))
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        total_steps=int(scheduler_total_steps),
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    config = {
        "image_size": args.image_size,
        "patch_size_deg": args.patch_size_deg,
        "patch_size_px": args.patch_size_px,
        "in_channels": 3,
        "model_preset": args.model_preset,
        "encoder_dim": model_config["encoder_dim"],
        "encoder_depth": model_config["encoder_depth"],
        "encoder_heads": model_config["encoder_heads"],
        "decoder_dim": model_config["decoder_dim"],
        "decoder_depth": model_config["decoder_depth"],
        "decoder_heads": model_config["decoder_heads"],
        "min_valid_fraction": args.min_valid_fraction,
        "color_only": args.color_only,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": args.persistent_workers,
        "num_steps": args.num_steps,
        "optimizer": args.optimizer,
        "mask_ratio": args.mask_ratio,
        "normalization_mode": args.normalization_mode,
        "normalization_max_samples": args.normalization_max_samples,
        "normalization_stats_path": str(args.normalization_stats_path)
        if args.normalization_stats_path is not None
        else None,
        "patch_records_path": str(args.patch_records_path) if args.patch_records_path is not None else None,
        "normalize_inputs": normalize_inputs,
        "normalize_targets": normalize_targets,
        "input_mean": input_mean,
        "input_std": input_std,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "scheduler_total_steps": int(scheduler_total_steps),
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "checkpoint_every": args.checkpoint_every,
        "preview_every": args.preview_every,
        "bbox": list(args.bbox),
        "seed": args.seed,
        "split_mode": args.split_mode if args.split_manifest is not None else None,
        "fold_index": args.fold_index,
        "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
        "split_summary_path": split_summary_path,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset) if val_dataset is not None else None,
        "test_count": len(test_dataset) if test_dataset is not None else None,
        "wandb_mode": args.wandb_mode,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }
    wandb_logger = init_wandb_logger(
        mode=args.wandb_mode,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        out_dir=out_dir,
        log_dir=args.wandb_dir,
        config=config,
    )
    save_training_progress(
        {
            "status": "running",
            "phase": "training_ready",
            "message": "Model, optimizer, and logging initialized.",
            "out_dir": str(out_dir),
            "startup_progress_path": str(startup_progress_path),
        },
        startup_progress_path,
    )
    print("[startup] entering training loop", flush=True)
    if args.resume_checkpoint is not None:
        state = load_mae_checkpoint(args.resume_checkpoint, model, optimizer, scheduler)
        state["checkpoint_path"] = str(args.resume_checkpoint)
        resume_state = state

    try:
        summary = run_mae_training(
            model,
            dataloader,
            optimizer,
            scheduler,
            out_dir=out_dir,
            num_steps=args.num_steps,
            mask_ratio=args.mask_ratio,
            preview_samples=preview_samples,
            checkpoint_every=args.checkpoint_every,
            preview_every=args.preview_every,
            device=args.device,
            resume_state=resume_state,
            config=config,
            wandb_logger=wandb_logger,
            val_dataloader=val_dataloader,
            val_every=args.val_every,
            val_max_batches=args.val_max_batches,
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

    print(f"Saved MAE checkpoint to {summary['checkpoint']}")
    print(f"Saved MAE history to {summary['history_path']}")
    print(f"Saved MAE loss curve to {summary['loss_curve']}")
    if summary["final_preview"] is not None:
        print(f"Saved MAE preview to {summary['final_preview']}")
    if summary["wandb_mode"] != "disabled" and summary["wandb_run_dir"] is not None:
        print(f"Saved W&B offline run to {summary['wandb_run_dir']}")
    if startup_progress_path.exists():
        startup_progress_path.unlink()


if __name__ == "__main__":
    main()
