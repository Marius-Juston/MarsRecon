"""Epoch-based Facebook MAE-style pretraining for MarsCLIP patches."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import traceback
from typing import Any

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from torch.amp import GradScaler, autocast
from torch import nn
from torch.optim import AdamW

from clip.fb_mae import FacebookMAEOutput, build_fb_mae_model
from clip.fb_mae_train_utils import (
    build_fb_mae_dataloader,
    count_trainable_parameters,
    save_training_history,
    save_training_progress,
)
from clip.marsclip_patches import DEFAULT_PATCH_VALID_FRACTION, MarsCLIPPatchDataset, load_patch_records
from clip.marsclip_splits import build_dataset_subsets, load_patch_split_manifest


def resolve_effective_lr(
    *,
    batch_size: int,
    accum_iter: int,
    base_lr: float,
    explicit_lr: float | None,
) -> float:
    """Resolve the MAE learning rate using the upstream ``blr`` convention."""
    if explicit_lr is not None:
        return float(explicit_lr)
    effective_batch = int(batch_size) * int(accum_iter)
    return float(base_lr) * effective_batch / 256.0


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    progress: float,
    lr: float,
    min_lr: float,
    epochs: int,
    warmup_epochs: int,
) -> float:
    """Per-iteration warmup + cosine decay like the official MAE trainer."""
    if progress < warmup_epochs:
        current_lr = lr * progress / max(float(warmup_epochs), 1.0)
    else:
        cosine_progress = (progress - warmup_epochs) / max(float(epochs - warmup_epochs), 1.0)
        cosine_progress = min(max(cosine_progress, 0.0), 1.0)
        current_lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    return float(current_lr)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, str) and device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


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
) -> dict[str, float]:
    """Run one MAE pretraining epoch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_steps = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader):
        progress = float(epoch) + (float(step) / max(len(dataloader), 1))
        current_lr = adjust_learning_rate(
            optimizer,
            progress=progress,
            lr=lr,
            min_lr=min_lr,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
        )

        batch = _move_batch_to_device(batch, device)
        with autocast(device_type=device.type, enabled=amp_enabled):
            output: FacebookMAEOutput = model(
                batch["image"],
                batch["valid_mask"],
                mask_ratio=mask_ratio,
            )
            loss = output.loss / float(accum_iter)

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

        total_loss += float(output.loss.detach().cpu())
        total_steps += 1

    return {
        "loss": total_loss / max(total_steps, 1),
        "lr": current_lr,
        "epoch_time_sec": time.time() - start_time,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    mask_ratio: float,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate the MAE loss on a validation split."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for step, batch in enumerate(dataloader):
        if max_batches is not None and step >= max_batches:
            break
        batch = _move_batch_to_device(batch, device)
        with autocast(device_type=device.type, enabled=amp_enabled):
            output: FacebookMAEOutput = model(
                batch["image"],
                batch["valid_mask"],
                mask_ratio=mask_ratio,
            )
        total_loss += float(output.loss.detach().cpu())
        total_steps += 1

    return {
        "loss": total_loss / max(total_steps, 1),
        "num_batches": float(total_steps),
    }


def save_fb_mae_checkpoint(
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
    config: dict[str, Any],
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []
    progress_path = out_dir / "progress.json"
    history_path = out_dir / "history.json"
    val_history_path = out_dir / "val_history.json"
    checkpoint_path = out_dir / "checkpoint.pt"
    best_checkpoint_path = out_dir / "best_checkpoint.pt"
    summary_path = out_dir / "summary.json"

    best_val_loss: float | None = None
    best_epoch = 0

    for epoch in range(epochs):
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
        )
        train_metrics["epoch"] = float(epoch + 1)
        history.append(train_metrics)
        save_training_history(history, history_path)

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
                max_batches=val_max_batches,
            )
            val_metrics["epoch"] = float(epoch + 1)
            val_history.append(val_metrics)
            save_training_history(val_history, val_history_path)
            if best_val_loss is None or val_metrics["loss"] < best_val_loss:
                best_val_loss = float(val_metrics["loss"])
                best_epoch = int(epoch + 1)
                save_fb_mae_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch + 1,
                    history=history,
                    val_history=val_history,
                    config=config,
                )

        if checkpoint_every > 0 and ((epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs):
            save_fb_mae_checkpoint(
                out_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                history=history,
                val_history=val_history,
                config=config,
            )

        save_fb_mae_checkpoint(
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
        "epochs": int(epochs),
        "best_metric_name": "val_loss" if val_loader is not None else "loss",
        "best_loss": best_val_loss if val_loader is not None else min(item["loss"] for item in history),
        "best_epoch": best_epoch if val_loader is not None else min(history, key=lambda item: item["loss"])["epoch"],
        "final_loss": history[-1]["loss"] if history else None,
        "val_final_loss": val_history[-1]["loss"] if val_history else None,
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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Facebook MAE-style pretraining for MarsCLIP patches.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=(-136.0, 12.0, -124.0, 24.0),
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
    )
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--split-manifest", type=pathlib.Path, default=None)
    parser.add_argument("--split-mode", type=str, default="holdout", choices=("holdout", "kfold"))
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--color-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-valid-fraction", type=float, default=DEFAULT_PATCH_VALID_FRACTION)
    parser.add_argument("--patch-records-path", type=pathlib.Path, default=None)
    parser.add_argument("--dataset-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset-normalization-path", type=pathlib.Path, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="mae_vit_base_patch16",
        choices=("mae_vit_small_patch16", "mae_vit_base_patch16", "mae_vit_large_patch16"),
    )
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
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    startup_progress_path = out_dir / "startup_progress.json"
    progress_path = out_dir / "progress.json"
    error_path = out_dir / "error_traceback.txt"
    save_training_progress(
        {
            "status": "running",
            "phase": "startup",
            "message": "Initializing Facebook MAE-style Mars pretraining run.",
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    patch_records = None
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

    train_loader = build_fb_mae_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
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
        )
    save_training_progress(
        {
            "status": "running",
            "phase": "dataloaders_ready",
            "message": "Epoch dataloaders initialized.",
            "train_count": len(train_dataset),
            "val_count": len(val_dataset) if val_dataset is not None else None,
            "test_count": len(test_dataset) if test_dataset is not None else None,
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    model = build_fb_mae_model(
        args.model,
        image_size=args.image_size,
        in_chans=3,
        norm_pix_loss=args.norm_pix_loss,
        min_valid_fraction=args.min_valid_fraction,
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
        "image_size": int(args.image_size),
        "max_patches": args.max_patches,
        "color_only": bool(args.color_only),
        "dataset_normalize": bool(args.dataset_normalize),
        "dataset_normalization_path": str(args.dataset_normalization_path)
        if args.dataset_normalization_path is not None
        else None,
        "split_mode": args.split_mode if args.split_manifest is not None else None,
        "fold_index": int(args.fold_index),
        "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
        "split_summary_path": split_summary_path,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset) if val_dataset is not None else None,
        "test_count": len(test_dataset) if test_dataset is not None else None,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "prefetch_factor": args.prefetch_factor,
        "persistent_workers": bool(args.persistent_workers),
        "patch_records_path": str(args.patch_records_path) if args.patch_records_path is not None else None,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }
    save_training_progress(
        {
            "status": "running",
            "phase": "training_ready",
            "message": "Facebook MAE-style model initialized.",
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

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
            config=config,
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

    print(f"Saved Facebook MAE checkpoint to {summary['checkpoint']}")
    print(f"Saved Facebook MAE summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
