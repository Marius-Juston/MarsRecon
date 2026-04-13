"""Train the SatMAE submodule on Mars HiRISE patch data."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import pathlib
import re
import sys
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from clip.fb_mae_train_utils import (
    build_fb_mae_dataloader,
    count_trainable_parameters,
    save_training_history,
    save_training_progress,
)
from clip.marsclip_cache import CachedMarsCLIPPatchDataset
from clip.marsclip_litdata import build_marsclip_litdata_dataloader
from clip.marsclip_patches import (
    DEFAULT_PATCH_VALID_FRACTION,
    MarsCLIPPatchDataset,
    load_patch_records,
)
from clip.marsclip_splits import (
    align_manifest_to_patch_records,
    build_dataset_subsets,
    load_patch_split_manifest,
)
from clip.satmae_bridge import build_satmae_model, load_satmae_lr_sched

DEFAULT_SATMAE_OUT_ROOT = pathlib.Path("/scratch/marsrecon_runs/stage_a/satmae")


@dataclass
class WandbLogger:
    """Small wrapper around an optional W&B run."""

    run: Any
    module: Any
    mode: str
    project: str
    entity: str | None
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


def init_wandb_logger(
    *,
    mode: str = "disabled",
    project: str = "MarsRecon",
    entity: str | None = "akshayn3-auvsl",
    run_name: str | None = None,
    out_dir: pathlib.Path | str,
    log_dir: pathlib.Path | str | None = None,
    config: dict[str, Any] | None = None,
) -> WandbLogger | None:
    """Initialize an optional W&B run for SatMAE training."""
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
    init_kwargs = {
        "entity": entity,
        "project": project,
        "name": run_name,
        "mode": normalized_mode,
        "dir": str(resolved_log_dir),
        "config": dict(config or {}),
        "reinit": "finish_previous",
    }
    try:
        run = wandb.init(**init_kwargs)
    except Exception:
        if normalized_mode != "online":
            raise
        print(
            "[wandb] Online initialization failed; falling back to offline mode for this run."
        )
        init_kwargs["mode"] = "offline"
        run = wandb.init(**init_kwargs)
        normalized_mode = "offline"
    return WandbLogger(
        run=run,
        module=wandb,
        mode=normalized_mode,
        project=project,
        entity=entity,
        run_name=run_name,
        log_dir=str(resolved_log_dir),
    )


def resolve_effective_lr(
    *,
    batch_size: int,
    accum_iter: int,
    base_lr: float,
    explicit_lr: float | None,
) -> float:
    """Resolve the MAE learning rate using SatMAE's ``blr`` convention."""
    if explicit_lr is not None:
        return float(explicit_lr)
    effective_batch = int(batch_size) * int(accum_iter)
    return float(base_lr) * effective_batch / 256.0


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, str) and device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _slugify(text: str) -> str:
    """Convert a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "satmae-run"


def resolve_run_output_dir(
    *,
    out_dir: pathlib.Path | None,
    out_root: pathlib.Path,
    run_name: str | None,
    model_name: str,
) -> pathlib.Path:
    """Resolve a unique output directory for a SatMAE run.

    If ``out_dir`` is provided, it must not already contain artifacts.
    Otherwise a timestamped directory is created under ``out_root/YYYYMMDD``.
    """
    if out_dir is not None:
        resolved = out_dir.expanduser().resolve()
        if resolved.exists() and any(resolved.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty output directory: {resolved}"
            )
        return resolved

    date_stamp = time.strftime("%Y%m%d")
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = _slugify(run_name or f"satmae-{model_name}")
    parent = out_root.expanduser().resolve() / date_stamp
    candidate = parent / f"{time_stamp}_{base_name}"
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{time_stamp}_{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def _filter_batch_by_validity(
    batch: dict[str, Any],
    *,
    require_patch_valid: bool,
) -> tuple[dict[str, Any] | None, dict[str, float]]:
    """Optionally drop samples whose Mars patch validity metadata is below threshold."""
    metadata = list(batch.get("metadata", []))
    total = len(metadata)
    if not require_patch_valid or total == 0:
        return batch, {"batch_size": float(total), "kept_samples": float(total), "dropped_samples": 0.0}

    keep_indices = [idx for idx, item in enumerate(metadata) if bool(item.get("is_patch_valid", True))]
    if not keep_indices:
        return None, {"batch_size": float(total), "kept_samples": 0.0, "dropped_samples": float(total)}

    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
    filtered = dict(batch)
    filtered["image"] = batch["image"].index_select(0, keep_tensor)
    filtered["valid_mask"] = batch["valid_mask"].index_select(0, keep_tensor)
    filtered["metadata"] = [metadata[idx] for idx in keep_indices]
    return filtered, {
        "batch_size": float(total),
        "kept_samples": float(len(keep_indices)),
        "dropped_samples": float(total - len(keep_indices)),
    }


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _load_dataset_normalization_stats(
    path: pathlib.Path | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load per-channel dataset mean/std tensors for reconstruction previews."""
    if path is None:
        return None
    stats = json.loads(path.read_text())
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(1, -1, 1, 1)
    return mean, std


def _denormalize_preview_image(
    image: torch.Tensor,
    valid_mask: torch.Tensor,
    stats: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    """Convert a normalized image back to display space while keeping invalid pixels at zero."""
    if stats is not None:
        mean, std = stats
        image = image * std.to(image.device) + mean.to(image.device)
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    image = torch.where(valid_mask, image, torch.zeros_like(image))
    return image.clamp(0.0, 1.0)


@torch.no_grad()
def save_reconstruction_preview(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_ratio: float,
    out_path: pathlib.Path,
    amp_enabled: bool,
    require_patch_valid: bool,
    dataset_normalization_stats: tuple[torch.Tensor, torch.Tensor] | None,
    max_items: int = 4,
) -> pathlib.Path | None:
    """Save a lightweight reconstruction preview from the first valid batch."""
    import os
    import tempfile

    mpl_cache = pathlib.Path(tempfile.gettempdir()) / "marsrecon-mpl"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib as mpl
    import numpy as np

    mpl.use("Agg")
    from matplotlib import pyplot as plt

    batch = None
    for candidate in dataloader:
        filtered, _ = _filter_batch_by_validity(candidate, require_patch_valid=require_patch_valid)
        if filtered is not None:
            batch = filtered
            break
    if batch is None:
        return None

    batch = _move_batch_to_device(batch, device)
    images = batch["image"]
    valid_mask = batch["valid_mask"]
    metadata = list(batch.get("metadata", []))

    was_training = model.training
    model.eval()
    with autocast(device_type=device.type, enabled=amp_enabled):
        _, pred, mask = model(images, mask_ratio=mask_ratio)

    patch_size = int(model.patch_embed.patch_size[0])
    in_chans = int(getattr(model, "in_c", images.shape[1]))
    target = model.patchify(images, patch_size, in_chans)
    pred_vis = pred
    if bool(getattr(model, "norm_pix_loss", False)):
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        pred_vis = pred * (var + 1.0e-6).sqrt() + mean

    pred_img = model.unpatchify(pred_vis, patch_size, in_chans)
    mask_tokens = mask.unsqueeze(-1).repeat(1, 1, patch_size * patch_size * in_chans)
    mask_img = model.unpatchify(mask_tokens, patch_size, in_chans)
    valid_mask_4d = valid_mask.unsqueeze(1).bool()

    image_disp = _denormalize_preview_image(images, valid_mask_4d, dataset_normalization_stats)
    pred_disp = _denormalize_preview_image(pred_img, valid_mask_4d, dataset_normalization_stats)
    masked_disp = _denormalize_preview_image(images * (1.0 - mask_img), valid_mask_4d, dataset_normalization_stats)
    composite_disp = _denormalize_preview_image(
        images * (1.0 - mask_img) + pred_img * mask_img,
        valid_mask_4d,
        dataset_normalization_stats,
    )

    rows = min(int(max_items), image_disp.shape[0])
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3.2 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row_idx in range(rows):
        panels = [
            ("input", image_disp[row_idx]),
            ("masked", masked_disp[row_idx]),
            ("reconstruction", pred_disp[row_idx]),
            ("composite", composite_disp[row_idx]),
        ]
        for col_idx, (title, tensor) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(tensor.detach().cpu().permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
        if row_idx < len(metadata):
            axes[row_idx, 0].set_ylabel(
                str(metadata[row_idx].get("patch_id", f"item_{row_idx}"))[:32],
                fontsize=8,
            )

    fig.suptitle("SatMAE Mars reconstruction preview", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    if was_training:
        model.train()
    return out_path


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    epochs: int,
    mask_ratio: float,
    accum_iter: int,
    lr: float,
    min_lr: float,
    warmup_epochs: int,
    scaler: GradScaler | None,
    amp_enabled: bool,
    require_patch_valid: bool,
    progress_path: pathlib.Path | None = None,
    progress_log_interval: int = 10,
    out_dir: pathlib.Path | None = None,
    wandb_logger: WandbLogger | None = None,
) -> dict[str, float]:
    """Run one SatMAE pretraining epoch on Mars patches."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    lr_sched = load_satmae_lr_sched()
    sched_args = SimpleNamespace(lr=lr, min_lr=min_lr, warmup_epochs=warmup_epochs, epochs=epochs)

    total_loss = 0.0
    total_steps = 0
    total_kept = 0.0
    total_dropped = 0.0
    start_time = time.time()
    current_lr = float(optimizer.param_groups[0]["lr"])
    progress_interval = max(int(progress_log_interval), 1)
    num_steps_estimate = len(dataloader)

    for step, batch in enumerate(dataloader):
        progress = float(epoch) + (float(step) / max(len(dataloader), 1))
        if step % accum_iter == 0:
            current_lr = float(lr_sched.adjust_learning_rate(optimizer, progress, sched_args))

        filtered_batch, batch_stats = _filter_batch_by_validity(batch, require_patch_valid=require_patch_valid)
        total_kept += batch_stats["kept_samples"]
        total_dropped += batch_stats["dropped_samples"]
        if filtered_batch is None:
            continue

        filtered_batch = _move_batch_to_device(filtered_batch, device)
        with autocast(device_type=device.type, enabled=amp_enabled):
            loss, _, _ = model(filtered_batch["image"], mask_ratio=mask_ratio)
            loss = loss / float(accum_iter)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_iter == 0 or (step + 1) == len(dataloader):
            if scaler is not None and amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu()) * float(accum_iter)
        total_steps += 1
        should_report = ((step + 1) % progress_interval == 0 or (step + 1) == num_steps_estimate)
        global_step = epoch * max(num_steps_estimate, 1) + step + 1
        if progress_path is not None and should_report:
            save_training_progress(
                {
                    "status": "running",
                    "phase": "training",
                    "current_epoch": epoch,
                    "target_epoch": epochs,
                    "step_in_epoch": int(step + 1),
                    "steps_in_epoch": int(num_steps_estimate),
                    "latest_loss": float(loss.detach().cpu()) * float(accum_iter),
                    "running_mean_loss": total_loss / float(max(total_steps, 1)),
                    "latest_lr": current_lr,
                    "kept_samples": total_kept,
                    "dropped_samples": total_dropped,
                    "out_dir": str(out_dir) if out_dir is not None else None,
                },
                progress_path,
            )
        if wandb_logger is not None and should_report:
            wandb_logger.log_metrics(
                {
                    "train/loss": float(loss.detach().cpu()) * float(accum_iter),
                    "train/running_mean_loss": total_loss / float(max(total_steps, 1)),
                    "train/lr": current_lr,
                    "train/kept_samples": total_kept,
                    "train/dropped_samples": total_dropped,
                    "train/epoch_progress": progress,
                },
                step=global_step,
            )

    if total_steps == 0:
        raise ValueError("No valid training batches remained after patch-valid filtering.")

    return {
        "loss": total_loss / float(total_steps),
        "lr": current_lr,
        "epoch_time_sec": time.time() - start_time,
        "kept_samples": total_kept,
        "dropped_samples": total_dropped,
        "effective_batches": float(total_steps),
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    mask_ratio: float,
    amp_enabled: bool,
    require_patch_valid: bool,
    max_batches: int | None = None,
    progress_path: pathlib.Path | None = None,
    progress_log_interval: int = 10,
    current_epoch: int = 0,
    target_epochs: int = 1,
    out_dir: pathlib.Path | None = None,
    wandb_logger: WandbLogger | None = None,
    wandb_step: int | None = None,
) -> dict[str, float]:
    """Evaluate SatMAE reconstruction loss on Mars patches."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_kept = 0.0
    total_dropped = 0.0
    progress_interval = max(int(progress_log_interval), 1)
    total_batches = int(max_batches) if max_batches is not None else int(len(dataloader))

    for step, batch in enumerate(dataloader):
        if max_batches is not None and step >= max_batches:
            break
        filtered_batch, batch_stats = _filter_batch_by_validity(batch, require_patch_valid=require_patch_valid)
        total_kept += batch_stats["kept_samples"]
        total_dropped += batch_stats["dropped_samples"]
        if filtered_batch is None:
            continue

        filtered_batch = _move_batch_to_device(filtered_batch, device)
        with autocast(device_type=device.type, enabled=amp_enabled):
            loss, _, _ = model(filtered_batch["image"], mask_ratio=mask_ratio)
        total_loss += float(loss.detach().cpu())
        total_steps += 1
        should_report = ((step + 1) % progress_interval == 0 or (step + 1) == total_batches)
        if progress_path is not None and should_report:
            save_training_progress(
                {
                    "status": "running",
                    "phase": "validating",
                    "current_epoch": current_epoch,
                    "target_epoch": target_epochs,
                    "val_batches_completed": int(step + 1),
                    "val_batches_total": int(total_batches),
                    "latest_val_loss": float(loss.detach().cpu()),
                    "running_mean_val_loss": total_loss / float(max(total_steps, 1)),
                    "kept_samples": total_kept,
                    "dropped_samples": total_dropped,
                    "out_dir": str(out_dir) if out_dir is not None else None,
                },
                progress_path,
            )
        if wandb_logger is not None and should_report:
            wandb_logger.log_metrics(
                {
                    "val/running_loss": total_loss / float(max(total_steps, 1)),
                    "val/kept_samples": total_kept,
                    "val/dropped_samples": total_dropped,
                    "val/batches_completed": float(step + 1),
                    "val/batches_total": float(total_batches),
                },
                step=wandb_step if wandb_step is not None else current_epoch,
            )

    if total_steps == 0:
        raise ValueError("No valid validation batches remained after patch-valid filtering.")

    return {
        "loss": total_loss / float(total_steps),
        "num_batches": float(total_steps),
        "kept_samples": total_kept,
        "dropped_samples": total_dropped,
    }


def save_satmae_checkpoint(
    path: pathlib.Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    epoch: int,
    history: list[dict[str, Any]],
    val_history: list[dict[str, Any]],
    config: dict[str, Any],
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "history": list(history),
        "val_history": list(val_history),
        "config": dict(config),
    }
    torch.save(payload, path)
    return path


def run_training(
    *,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    out_dir: pathlib.Path,
    epochs: int,
    mask_ratio: float,
    accum_iter: int,
    lr: float,
    min_lr: float,
    warmup_epochs: int,
    amp_enabled: bool,
    checkpoint_every: int,
    val_max_batches: int | None,
    require_patch_valid: bool,
    config: dict[str, Any],
    preview_loader: torch.utils.data.DataLoader | None,
    reconstruction_dir: pathlib.Path,
    dataset_normalization_stats: tuple[torch.Tensor, torch.Tensor] | None,
    reconstruction_max_items: int,
    progress_log_interval: int = 10,
    wandb_logger: WandbLogger | None = None,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    progress_path = out_dir / "progress.json"
    history_path = out_dir / "history.json"
    val_history_path = out_dir / "val_history.json"
    checkpoints_dir = out_dir / "checkpoints"
    checkpoint_path = checkpoints_dir / "checkpoint.pt"
    best_checkpoint_path = checkpoints_dir / "best_checkpoint.pt"
    summary_path = out_dir / "summary.json"

    best_val_loss: float | None = None
    best_epoch = 0
    steps_per_epoch = max(len(train_loader), 1)

    for epoch in range(epochs):
        epoch_wandb_step = int((epoch + 1) * steps_per_epoch)
        save_training_progress(
            {
                "status": "running",
                "phase": "training",
                "current_epoch": epoch,
                "target_epoch": epochs,
                "out_dir": str(out_dir),
            },
            progress_path,
        )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
            mask_ratio=mask_ratio,
            accum_iter=accum_iter,
            lr=lr,
            min_lr=min_lr,
            warmup_epochs=warmup_epochs,
            scaler=scaler,
            amp_enabled=amp_enabled,
            require_patch_valid=require_patch_valid,
            progress_path=progress_path,
            progress_log_interval=progress_log_interval,
            out_dir=out_dir,
            wandb_logger=wandb_logger,
        )
        train_metrics["epoch"] = float(epoch + 1)
        history.append(train_metrics)
        save_training_history(history, history_path)
        if wandb_logger is not None:
            wandb_logger.log_metrics(
                {
                    "epoch/train_loss": float(train_metrics["loss"]),
                    "epoch/train_lr": float(train_metrics["lr"]),
                    "epoch/train_kept_samples": float(train_metrics["kept_samples"]),
                    "epoch/train_dropped_samples": float(train_metrics["dropped_samples"]),
                },
                step=epoch_wandb_step,
            )

        val_metrics = None
        if val_loader is not None:
            save_training_progress(
                {
                    "status": "running",
                    "phase": "validating",
                    "current_epoch": epoch + 1,
                    "target_epoch": epochs,
                    "latest_loss": train_metrics["loss"],
                    "out_dir": str(out_dir),
                },
                progress_path,
            )
            val_metrics = evaluate_epoch(
                model,
                val_loader,
                device=device,
                mask_ratio=mask_ratio,
                amp_enabled=amp_enabled,
                require_patch_valid=require_patch_valid,
                max_batches=val_max_batches,
                progress_path=progress_path,
                progress_log_interval=progress_log_interval,
                current_epoch=epoch + 1,
                target_epochs=epochs,
                out_dir=out_dir,
                wandb_logger=wandb_logger,
                wandb_step=epoch_wandb_step,
            )
            val_metrics["epoch"] = float(epoch + 1)
            val_history.append(val_metrics)
            save_training_history(val_history, val_history_path)
            if wandb_logger is not None:
                wandb_logger.log_metrics(
                    {
                        "epoch/val_loss": float(val_metrics["loss"]),
                        "epoch/val_batches": float(val_metrics["num_batches"]),
                        "epoch/val_kept_samples": float(val_metrics["kept_samples"]),
                        "epoch/val_dropped_samples": float(val_metrics["dropped_samples"]),
                    },
                    step=epoch_wandb_step,
                )
            if best_val_loss is None or val_metrics["loss"] < best_val_loss:
                best_val_loss = float(val_metrics["loss"])
                best_epoch = int(epoch + 1)
                save_satmae_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch + 1,
                    history=history,
                    val_history=val_history,
                    config=config,
                )
                if preview_loader is not None:
                    best_preview = save_reconstruction_preview(
                        model=model,
                        dataloader=preview_loader,
                        device=device,
                        mask_ratio=mask_ratio,
                        out_path=reconstruction_dir / "reconstruction_best.png",
                        amp_enabled=amp_enabled,
                        require_patch_valid=require_patch_valid,
                        dataset_normalization_stats=dataset_normalization_stats,
                        max_items=reconstruction_max_items,
                    )
                    if wandb_logger is not None and best_preview is not None:
                        wandb_logger.log_image(
                            "reconstructions/best",
                            best_preview,
                            step=epoch_wandb_step,
                            caption=f"Best reconstruction preview at epoch {epoch + 1}",
                        )

        if checkpoint_every > 0 and ((epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs):
            save_satmae_checkpoint(
                checkpoints_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                history=history,
                val_history=val_history,
                config=config,
            )

        if preview_loader is not None:
            epoch_preview = save_reconstruction_preview(
                model=model,
                dataloader=preview_loader,
                device=device,
                mask_ratio=mask_ratio,
                out_path=reconstruction_dir / f"reconstruction_epoch_{epoch + 1:04d}.png",
                amp_enabled=amp_enabled,
                require_patch_valid=require_patch_valid,
                dataset_normalization_stats=dataset_normalization_stats,
                max_items=reconstruction_max_items,
            )
            if wandb_logger is not None and epoch_preview is not None:
                wandb_logger.log_image(
                    "reconstructions/epoch",
                    epoch_preview,
                    step=epoch_wandb_step,
                    caption=f"Epoch {epoch + 1} reconstruction preview",
                )

        save_satmae_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            history=history,
            val_history=val_history,
            config=config,
        )
        save_training_progress(
            {
                "status": "running",
                "phase": "training_complete" if (epoch + 1) == epochs else "training",
                "current_epoch": epoch + 1,
                "target_epoch": epochs,
                "latest_loss": train_metrics["loss"],
                "latest_lr": train_metrics["lr"],
                "val_best_loss": best_val_loss,
                "out_dir": str(out_dir),
            },
            progress_path,
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
        "history_path": str(history_path),
        "val_history_path": str(val_history_path) if val_loader is not None else None,
        "checkpoints_dir": str(checkpoints_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "epochs": int(epochs),
        "best_metric_name": "val_loss" if val_loader is not None else "loss",
        "best_loss": best_val_loss if val_loader is not None else min(item["loss"] for item in history),
        "best_epoch": best_epoch if val_loader is not None else min(history, key=lambda item: item["loss"])["epoch"],
        "final_loss": history[-1]["loss"] if history else None,
        "val_final_loss": val_history[-1]["loss"] if val_history else None,
        "wandb_mode": wandb_logger.mode if wandb_logger is not None else "disabled",
        "wandb_project": wandb_logger.project if wandb_logger is not None else None,
        "wandb_entity": wandb_logger.entity if wandb_logger is not None else None,
        "wandb_run_name": wandb_logger.run_name if wandb_logger is not None else None,
        "wandb_run_id": wandb_logger.run_id if wandb_logger is not None else None,
        "wandb_log_dir": wandb_logger.log_dir if wandb_logger is not None else None,
        "wandb_run_dir": wandb_logger.run_dir if wandb_logger is not None else None,
        "config": config,
        "summary_created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    save_training_progress(
        {
            "status": "completed",
            "phase": "complete",
            "current_epoch": epochs,
            "target_epoch": epochs,
            "best_loss": summary["best_loss"],
            "best_epoch": summary["best_epoch"],
            "latest_loss": summary["final_loss"],
            "val_best_loss": best_val_loss,
            "summary_path": str(summary_path),
            "out_dir": str(out_dir),
        },
        progress_path,
    )
    if preview_loader is not None:
        final_preview = save_reconstruction_preview(
            model=model,
            dataloader=preview_loader,
            device=device,
            mask_ratio=mask_ratio,
            out_path=reconstruction_dir / "reconstruction_final.png",
            amp_enabled=amp_enabled,
            require_patch_valid=require_patch_valid,
            dataset_normalization_stats=dataset_normalization_stats,
            max_items=reconstruction_max_items,
        )
        if wandb_logger is not None and final_preview is not None:
            wandb_logger.log_image(
                "reconstructions/final",
                final_preview,
                step=int(epochs * steps_per_epoch),
                caption="Final reconstruction preview",
            )
    if wandb_logger is not None:
        wandb_logger.finish(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SatMAE on Mars HiRISE patches.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_SATMAE_OUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-136.0, 12.0, -124.0, 24.0),
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size-px", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--split-manifest", type=pathlib.Path, default=None)
    parser.add_argument("--split-mode", type=str, default="holdout", choices=("holdout", "kfold"))
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--color-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-valid-fraction", type=float, default=DEFAULT_PATCH_VALID_FRACTION)
    parser.add_argument("--patch-records-path", type=pathlib.Path, default=None)
    parser.add_argument(
        "--cached-data-root",
        type=pathlib.Path,
        default=None,
        help="Optional cache root containing train/val/test memmap-backed patch splits.",
    )
    parser.add_argument(
        "--litdata-root",
        type=pathlib.Path,
        default=None,
        help="Optional LitData root containing train/val/test streaming patch splits.",
    )
    parser.add_argument("--dataset-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset-normalization-path", type=pathlib.Path, default=None)
    parser.add_argument("--filter-invalid-patches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dominant-obs-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", type=str, default="mae_vit_base_patch16")
    parser.add_argument("--norm-pix-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accum-iter", type=int, default=1)
    parser.add_argument("--blr", type=float, default=1.5e-4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--val-max-batches", type=int, default=None)
    parser.add_argument("--progress-log-interval", type=int, default=10)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=("disabled", "offline", "online"),
    )
    parser.add_argument("--wandb-project", type=str, default="MarsRecon")
    parser.add_argument("--wandb-entity", type=str, default="akshayn3-auvsl")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-dir", type=pathlib.Path, default=None)
    parser.add_argument("--save-reconstructions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reconstruction-max-items", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = resolve_run_output_dir(
        out_dir=args.out_dir,
        out_root=args.out_root,
        run_name=args.run_name,
        model_name=args.model,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    startup_progress_path = out_dir / "startup_progress.json"
    progress_path = out_dir / "progress.json"
    error_path = out_dir / "error_traceback.txt"
    run_config_path = out_dir / "run_config.json"
    reconstruction_dir = out_dir / "reconstructions"
    save_training_progress(
        {
            "status": "running",
            "phase": "startup",
            "message": "Initializing SatMAE Mars pretraining run.",
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    patch_records = None
    split_summary_path = None
    if args.litdata_root is not None:
        litdata_root = args.litdata_root.expanduser().resolve()
        train_dataset = None
        val_dataset = None
        test_dataset = None
        dataset_size = 0
        for split_name in ("train", "val", "test"):
            info_path = litdata_root / split_name / "litdata_info.json"
            if info_path.exists():
                split_info = json.loads(info_path.read_text())
                dataset_size += int(split_info.get("num_samples", 0))
        split_summary_path = str(litdata_root / "litdata_summary.json")
        save_training_progress(
            {
                "status": "running",
                "phase": "dataset_ready",
                "message": "LitData-backed Mars patch datasets ready.",
                "dataset_size": dataset_size,
                "litdata_root": str(litdata_root),
                "out_dir": str(out_dir),
            },
            startup_progress_path,
        )
    elif args.cached_data_root is not None:
        cache_root = args.cached_data_root.expanduser().resolve()
        train_dataset = CachedMarsCLIPPatchDataset(cache_root / "train")
        val_dataset = CachedMarsCLIPPatchDataset(cache_root / "val")
        test_cache = cache_root / "test"
        test_dataset = CachedMarsCLIPPatchDataset(test_cache) if test_cache.joinpath("_SUCCESS").exists() else None
        dataset_size = len(train_dataset) + len(val_dataset)
        if test_dataset is not None:
            dataset_size += len(test_dataset)
        split_summary_path = str(cache_root / "cache_summary.json")
        save_training_progress(
            {
                "status": "running",
                "phase": "dataset_ready",
                "message": "Cache-backed Mars patch datasets loaded.",
                "dataset_size": dataset_size,
                "cache_root": str(cache_root),
                "out_dir": str(out_dir),
            },
            startup_progress_path,
        )
    else:
        if args.patch_records_path is not None and args.patch_records_path.exists():
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
            dataset_normalize=args.dataset_normalize,
            dataset_normalization_path=args.dataset_normalization_path,
            use_dominant_obs_only=args.dominant_obs_only,
        )
        save_training_progress(
            {
                "status": "running",
                "phase": "dataset_ready",
                "message": "Mars patch dataset constructed.",
                "dataset_size": len(dataset),
                "out_dir": str(out_dir),
            },
            startup_progress_path,
        )

        if args.split_manifest is not None:
            split_manifest = load_patch_split_manifest(args.split_manifest)
            if args.max_patches is not None:
                split_manifest = align_manifest_to_patch_records(
                    split_manifest,
                    dataset.patch_records,
                )
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

    if args.litdata_root is not None:
        litdata_root = args.litdata_root.expanduser().resolve()
        train_loader = build_marsclip_litdata_dataloader(
            litdata_root / "train",
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
            seed=args.seed,
        )
        val_loader = None
        if (litdata_root / "val" / "_SUCCESS").exists():
            val_loader = build_marsclip_litdata_dataloader(
                litdata_root / "val",
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=min(int(args.num_workers), 4),
                pin_memory=args.pin_memory,
                drop_last=False,
                seed=args.seed,
            )
        train_count = len(train_loader.dataset)
        val_count = len(val_loader.dataset) if val_loader is not None else None
        test_count = None
        if (litdata_root / "test" / "_SUCCESS").exists():
            test_count = len(
                build_marsclip_litdata_dataloader(
                    litdata_root / "test",
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=False,
                    seed=args.seed,
                ).dataset
            )
    else:
        train_loader = build_fb_mae_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            drop_last=True,
        )
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = build_fb_mae_dataloader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                drop_last=False,
            )
        train_count = len(train_dataset)
        val_count = len(val_dataset) if val_dataset is not None else None
        test_count = len(test_dataset) if test_dataset is not None else None
    save_training_progress(
        {
            "status": "running",
            "phase": "dataloaders_ready",
            "message": "SatMAE dataloaders initialized.",
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    model = build_satmae_model(
        args.model,
        img_size=args.image_size,
        patch_size=args.patch_size_px,
        in_chans=3,
        norm_pix_loss=args.norm_pix_loss,
    )
    device = _resolve_device(args.device)
    model.to(device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = GradScaler(device.type, enabled=amp_enabled)
    trainable_parameters, total_parameters = count_trainable_parameters(model)

    lr = resolve_effective_lr(
        batch_size=args.batch_size,
        accum_iter=args.accum_iter,
        base_lr=args.blr,
        explicit_lr=args.lr,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    config = {
        "model": args.model,
        "norm_pix_loss": bool(args.norm_pix_loss),
        "mask_ratio": float(args.mask_ratio),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "accum_iter": int(args.accum_iter),
        "blr": float(args.blr),
        "lr": float(lr),
        "min_lr": float(args.min_lr),
        "warmup_epochs": int(args.warmup_epochs),
        "weight_decay": float(args.weight_decay),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "bbox": list(args.bbox),
        "patch_size_deg": float(args.patch_size_deg),
        "patch_size_px": int(args.patch_size_px),
        "image_size": int(args.image_size),
        "max_patches": args.max_patches,
        "color_only": bool(args.color_only),
        "dataset_normalize": bool(args.dataset_normalize),
        "dataset_normalization_path": str(args.dataset_normalization_path)
        if args.dataset_normalization_path is not None
        else None,
        "dominant_obs_only": bool(args.dominant_obs_only),
        "split_mode": args.split_mode if args.split_manifest is not None else None,
        "fold_index": int(args.fold_index),
        "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
        "split_summary_path": split_summary_path,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "cached_data_root": str(args.cached_data_root) if args.cached_data_root is not None else None,
        "litdata_root": str(args.litdata_root) if args.litdata_root is not None else None,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": bool(args.persistent_workers),
        "progress_log_interval": int(args.progress_log_interval),
        "patch_records_path": str(args.patch_records_path) if args.patch_records_path is not None else None,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "filter_invalid_patches": bool(args.filter_invalid_patches),
        "run_name": args.run_name,
        "out_dir": str(out_dir),
        "out_root": str(args.out_root),
        "wandb_mode": args.wandb_mode,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_run_name": args.wandb_run_name,
        "wandb_dir": str(args.wandb_dir) if args.wandb_dir is not None else None,
        "save_reconstructions": bool(args.save_reconstructions),
        "reconstruction_max_items": int(args.reconstruction_max_items),
    }
    run_config_path.write_text(json.dumps(config, indent=2))
    wandb_logger = init_wandb_logger(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or args.run_name,
        out_dir=out_dir,
        log_dir=args.wandb_dir,
        config=config,
    )
    save_training_progress(
        {
            "status": "running",
            "phase": "training_ready",
            "message": "SatMAE model initialized.",
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )
    preview_loader = val_loader if val_loader is not None else train_loader
    dataset_normalization_stats = None
    if args.dataset_normalize and args.dataset_normalization_path is not None:
        dataset_normalization_stats = _load_dataset_normalization_stats(args.dataset_normalization_path)
    if not args.save_reconstructions:
        preview_loader = None

    try:
        summary = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            out_dir=out_dir,
            epochs=args.epochs,
            mask_ratio=args.mask_ratio,
            accum_iter=args.accum_iter,
            lr=lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            amp_enabled=amp_enabled,
            checkpoint_every=args.checkpoint_every,
            val_max_batches=args.val_max_batches,
            require_patch_valid=args.filter_invalid_patches,
            config=config,
            preview_loader=preview_loader,
            reconstruction_dir=reconstruction_dir,
            dataset_normalization_stats=dataset_normalization_stats,
            reconstruction_max_items=args.reconstruction_max_items,
            progress_log_interval=args.progress_log_interval,
            wandb_logger=wandb_logger,
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
        if wandb_logger is not None:
            wandb_logger.finish(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "error_traceback_path": str(error_path),
                }
            )
        raise

    print(f"Saved SatMAE checkpoint to {summary['checkpoint']}")
    if summary["wandb_mode"] != "disabled" and summary["wandb_run_dir"] is not None:
        print(f"Saved W&B run to {summary['wandb_run_dir']}")


if __name__ == "__main__":
    main()
