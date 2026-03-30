"""Minimal Stage A MAE training utilities and CLI for MarsCLIP."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
from collections.abc import Callable, Iterable
from typing import Any

_MPL_CACHE = pathlib.Path(tempfile.gettempdir()) / "marsrecon-mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib as mpl
import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset

mpl.use("Agg")
from matplotlib import pyplot as plt

from marsclip_mae import MarsMAEOutput, MarsMaskedAutoencoder, collate_patch_samples_for_mae
from marsclip_patches import DEFAULT_PATCH_VALID_FRACTION, MarsCLIPPatchDataset
from visualize_marsclip_mae import save_mae_reconstruction_preview

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
}


def resolve_mae_model_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve checkpoint config values into MAE constructor kwargs."""
    source = dict(config or {})
    return {
        "image_size": int(source.get("image_size", MAE_MODEL_CONFIG_DEFAULTS["image_size"])),
        "patch_size": int(
            source.get("patch_size_px", source.get("patch_size", MAE_MODEL_CONFIG_DEFAULTS["patch_size_px"]))
        ),
        "in_channels": int(source.get("in_channels", MAE_MODEL_CONFIG_DEFAULTS["in_channels"])),
        "encoder_dim": int(source.get("encoder_dim", MAE_MODEL_CONFIG_DEFAULTS["encoder_dim"])),
        "encoder_depth": int(
            source.get("encoder_depth", MAE_MODEL_CONFIG_DEFAULTS["encoder_depth"])
        ),
        "encoder_heads": int(
            source.get("encoder_heads", MAE_MODEL_CONFIG_DEFAULTS["encoder_heads"])
        ),
        "decoder_dim": int(source.get("decoder_dim", MAE_MODEL_CONFIG_DEFAULTS["decoder_dim"])),
        "decoder_depth": int(
            source.get("decoder_depth", MAE_MODEL_CONFIG_DEFAULTS["decoder_depth"])
        ),
        "decoder_heads": int(
            source.get("decoder_heads", MAE_MODEL_CONFIG_DEFAULTS["decoder_heads"])
        ),
        "min_valid_fraction": float(
            source.get("min_valid_fraction", MAE_MODEL_CONFIG_DEFAULTS["min_valid_fraction"])
        ),
    }


def build_mae_model_from_config(config: dict[str, Any] | None = None) -> MarsMaskedAutoencoder:
    """Instantiate a Stage A MAE from a saved checkpoint config."""
    return MarsMaskedAutoencoder(**resolve_mae_model_config(config))


def build_mae_dataloader(
    dataset: Dataset | list[dict[str, Any]],
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader that emits Stage A MAE-ready patch batches."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_patch_samples_for_mae,
    )


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

    entry = {"loss": float(loss.detach().cpu())}
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


def save_mae_checkpoint(
    path: pathlib.Path | str,
    model: MarsMaskedAutoencoder,
    optimizer: Optimizer | None,
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
        },
        out,
    )
    return out


def load_mae_checkpoint(
    path: pathlib.Path | str,
    model: MarsMaskedAutoencoder,
    optimizer: Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a MAE checkpoint into a model and optional optimizer."""
    checkpoint = torch.load(path, map_location=resolve_map_location(map_location))
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
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


def save_loss_curve(
    history: list[dict[str, float]],
    out_path: pathlib.Path | str,
) -> pathlib.Path:
    """Save a simple MAE loss curve PNG."""
    if not history:
        raise ValueError("history must not be empty.")

    steps = [entry["step"] for entry in history]
    losses = [entry["loss"] for entry in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses, color="#d95f02", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("MAE loss")
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
) -> dict[str, Any]:
    """Run a small Stage A training session with periodic artifacts."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    history = list((resume_state or {}).get("history", []))
    start_step = int((resume_state or {}).get("step", 0))
    checkpoint_paths: list[str] = []
    preview_paths: list[str] = []

    best_entry = select_best_history_entry(history) if history else None
    best_loss = float(best_entry["loss"]) if best_entry is not None else float("inf")
    best_step = int(best_entry["step"]) if best_entry is not None else start_step
    best_checkpoint_path = out_path / "best_checkpoint.pt"

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
        history.append(entry)

        if float(entry["loss"]) <= best_loss:
            best_loss = float(entry["loss"])
            best_step = step
            save_mae_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                step=step,
                history=history,
                config=config,
            )

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt = save_mae_checkpoint(
                out_path / f"checkpoint_step_{step:06d}.pt",
                model,
                optimizer,
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

    checkpoint_path = save_mae_checkpoint(
        out_path / "checkpoint.pt",
        model,
        optimizer,
        step=final_step,
        history=history,
        config=config,
    )
    history_path = save_training_history(history, out_path / "history.json")
    curve_path = save_loss_curve(history, out_path / "loss_curve.png")

    summary = {
        "start_step": start_step,
        "num_new_steps": num_steps,
        "num_steps_total": len(history),
        "initial_loss": history[0]["loss"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
        "best_loss": best_loss if history else None,
        "best_step": best_step if history else None,
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
        "best_checkpoint_rule": "minimum observed training loss",
        "history_path": str(history_path),
        "loss_curve": str(curve_path),
    }
    summary_path = save_training_summary(summary, out_path / "summary.json")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
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
    parser.add_argument("--max-patches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--preview-items", type=int, default=2)
    parser.add_argument("--encoder-dim", type=int, default=64)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--decoder-dim", type=int, default=32)
    parser.add_argument("--decoder-depth", type=int, default=1)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--preview-every", type=int, default=0)
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=DEFAULT_PATCH_VALID_FRACTION,
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    dataset = MarsCLIPPatchDataset(
        root=args.root,
        bbox=tuple(args.bbox),
        patch_size=args.patch_size_deg,
        image_size=args.image_size,
        max_patches=args.max_patches,
        min_valid_fraction=args.min_valid_fraction,
    )
    dataloader = build_mae_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    preview_count = min(args.preview_items, len(dataset))
    preview_samples = [dataset[i] for i in range(preview_count)]

    model = MarsMaskedAutoencoder(
        image_size=args.image_size,
        patch_size=args.patch_size_px,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        min_valid_fraction=args.min_valid_fraction,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config = {
        "image_size": args.image_size,
        "patch_size_deg": args.patch_size_deg,
        "patch_size_px": args.patch_size_px,
        "in_channels": 3,
        "encoder_dim": args.encoder_dim,
        "encoder_depth": args.encoder_depth,
        "encoder_heads": args.encoder_heads,
        "decoder_dim": args.decoder_dim,
        "decoder_depth": args.decoder_depth,
        "decoder_heads": args.decoder_heads,
        "min_valid_fraction": args.min_valid_fraction,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "mask_ratio": args.mask_ratio,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "checkpoint_every": args.checkpoint_every,
        "preview_every": args.preview_every,
        "bbox": list(args.bbox),
        "seed": args.seed,
    }
    resume_state = None
    if args.resume_checkpoint is not None:
        state = load_mae_checkpoint(args.resume_checkpoint, model, optimizer)
        state["checkpoint_path"] = str(args.resume_checkpoint)
        resume_state = state

    summary = run_mae_training(
        model,
        dataloader,
        optimizer,
        out_dir=out_dir,
        num_steps=args.num_steps,
        mask_ratio=args.mask_ratio,
        preview_samples=preview_samples,
        checkpoint_every=args.checkpoint_every,
        preview_every=args.preview_every,
        device=args.device,
        resume_state=resume_state,
        config=config,
    )

    print(f"Saved MAE checkpoint to {summary['checkpoint']}")
    print(f"Saved MAE history to {summary['history_path']}")
    print(f"Saved MAE loss curve to {summary['loss_curve']}")
    if summary["final_preview"] is not None:
        print(f"Saved MAE preview to {summary['final_preview']}")


if __name__ == "__main__":
    main()
