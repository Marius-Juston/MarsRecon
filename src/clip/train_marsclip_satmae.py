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

print("[satmae][debug] module import/start: train_marsclip_satmae.py", flush=True)

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from torch import nn

# AMP imports can differ across torch versions.
# Prefer torch.amp (newer), fall back to torch.cuda.amp (older).
try:  # pragma: no cover
    from torch.amp import GradScaler, autocast  # type: ignore
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

from torch.optim import AdamW

# NEW: minimal recipe-alignment transforms
# NOTE: Avoid importing torchvision here; this environment's torchvision import
# chain pulls in transformers/onnx and can crash due to torch API mismatches.
# We implement the minimal crop/resize/flip ops with torch-only code below.


from clip.fb_mae_train_utils import (
    build_fb_mae_dataloader,
    count_trainable_parameters,
    save_training_history,
    save_training_progress,
)
from clip.fb_mae import (
    DEFAULT_PATCH_VALID_FRACTION,
    compute_patch_valid_fraction,
    normalize_patch_targets,
    patchify_valid_mask,
)
from clip.marsclip_cache import CachedMarsCLIPPatchDataset
from clip.marsclip_litdata import build_marsclip_litdata_dataloader

# NOTE: Importing marsclip_patches triggers torchgeo -> torchvision in some envs.
# We only need marsclip_patches for the raw dataset path (not for litData/cache).
# So we import it lazily inside that codepath.

from clip.marsclip_splits import (
    align_manifest_to_patch_records,
    build_dataset_subsets,
    load_patch_split_manifest,
)
from clip.satmae_bridge import build_satmae_model, load_satmae_lr_sched

DEFAULT_SATMAE_OUT_ROOT = pathlib.Path("/scratch/marsrecon_runs/stage_a/satmae")


# -----------------------------
# Pretrained initialization
# -----------------------------

def _safe_torch_load_checkpoint(path: pathlib.Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location=map_location)
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _extract_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("state_dict", "model", "model_state", "module"):
        value = ckpt.get(key)
        if isinstance(value, dict) and value and all(isinstance(k, str) for k in value.keys()):
            return value  # type: ignore[return-value]
    # If it already looks like a state dict.
    if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt  # type: ignore[return-value]
    return {}


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p) :]
        out[new_k] = v
    return out


def _interpolate_pos_embed_if_needed(
    pos_embed: torch.Tensor,
    *,
    target_num_patches: int,
    num_extra_tokens: int = 1,
) -> torch.Tensor:
    """Interpolate ViT absolute position embeddings to a new patch grid size."""
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        return pos_embed

    target_tokens = num_extra_tokens + int(target_num_patches)
    if pos_embed.shape[1] == target_tokens:
        return pos_embed

    extra = pos_embed[:, :num_extra_tokens]
    pos_tokens = pos_embed[:, num_extra_tokens:]
    src_n = pos_tokens.shape[1]
    src_size = int(math.sqrt(src_n))
    tgt_size = int(math.sqrt(int(target_num_patches)))
    if src_size * src_size != src_n or tgt_size * tgt_size != int(target_num_patches):
        # Non-square grid; best effort is to skip.
        return pos_embed

    pos_tokens = pos_tokens.reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2)
    pos_tokens = nn.functional.interpolate(pos_tokens, size=(tgt_size, tgt_size), mode="bicubic", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, tgt_size * tgt_size, -1)
    return torch.cat((extra, pos_tokens), dim=1)


def init_satmae_from_checkpoint(
    model: nn.Module,
    checkpoint_path: pathlib.Path,
    *,
    mode: str = "encoder",
    pos_embed: str = "interp",
    verbose: bool = False,
) -> dict[str, Any]:
    """Initialize SatMAE weights from a checkpoint.

    Args:
        mode: "encoder" (default) loads only ViT encoder blocks + norms.
        pos_embed: "interp" | "keep" | "skip".
    """
    checkpoint_path = checkpoint_path.expanduser().resolve()
    ckpt = _safe_torch_load_checkpoint(checkpoint_path)
    state = _extract_state_dict(ckpt)
    state = _strip_prefix(state, ("module.", "model.", "encoder.", "backbone."))

    if not state:
        raise ValueError(f"No state dict found in checkpoint: {checkpoint_path}")

    current = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}

    def _accept(k: str) -> bool:
        if mode == "all":
            return True
        if mode == "encoder":
            return k.startswith("blocks.") or k in {"norm.weight", "norm.bias"}
        raise ValueError("--init-mode must be one of: encoder, all")

    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        if not _accept(k):
            skipped[k] = "filtered_by_mode"
            continue
        if k not in current:
            skipped[k] = "missing_in_target"
            continue
        if current[k].shape != v.shape:
            skipped[k] = f"shape_mismatch {tuple(v.shape)} != {tuple(current[k].shape)}"
            continue
        filtered[k] = v

    # Handle pos_embed separately.
    if "pos_embed" in state and "pos_embed" in current:
        if pos_embed == "skip":
            skipped["pos_embed"] = "skipped_by_flag"
        elif pos_embed == "keep":
            if state["pos_embed"].shape == current["pos_embed"].shape:
                filtered["pos_embed"] = state["pos_embed"]
            else:
                skipped["pos_embed"] = (
                    f"shape_mismatch {tuple(state['pos_embed'].shape)} != {tuple(current['pos_embed'].shape)}"
                )
        elif pos_embed == "interp":
            try:
                num_patches = int(getattr(model.patch_embed, "num_patches"))
                num_extra = int(current["pos_embed"].shape[1] - num_patches)
                interpolated = _interpolate_pos_embed_if_needed(
                    state["pos_embed"],
                    target_num_patches=num_patches,
                    num_extra_tokens=num_extra,
                )
                if interpolated.shape == current["pos_embed"].shape:
                    filtered["pos_embed"] = interpolated
                else:
                    skipped["pos_embed"] = (
                        f"interp_shape_mismatch {tuple(interpolated.shape)} != {tuple(current['pos_embed'].shape)}"
                    )
            except Exception as exc:  # pragma: no cover
                skipped["pos_embed"] = f"interp_failed: {type(exc).__name__}: {exc}"
        else:
            raise ValueError("--init-pos-embed must be one of: interp, keep, skip")

    # Never force-load patch embedding when patch size differs.
    for k in ("patch_embed.proj.weight", "patch_embed.proj.bias"):
        if k in filtered:
            del filtered[k]
            skipped[k] = "always_skipped"

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    report = {
        "checkpoint": str(checkpoint_path),
        "mode": mode,
        "pos_embed": pos_embed,
        "loaded": len(filtered),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
    }
    if verbose:
        print("[init] Loaded checkpoint:", checkpoint_path)
        print(f"[init] mode={mode} pos_embed={pos_embed}")
        print(f"[init] loaded_keys={len(filtered)} missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if skipped:
            print(f"[init] skipped_keys={len(skipped)}")
    return report


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
    min_valid_fraction: float = DEFAULT_PATCH_VALID_FRACTION,
) -> tuple[dict[str, Any] | None, dict[str, float]]:
    """Optionally drop samples whose Mars patch validity metadata is below threshold."""
    metadata = list(batch.get("metadata", []))
    total = len(metadata)
    if not require_patch_valid or total == 0:
        return batch, {"batch_size": float(total), "kept_samples": float(total), "dropped_samples": 0.0}

    if torch.is_tensor(batch.get("image")) and torch.is_tensor(batch.get("valid_mask")):
        refined_valid_mask = _refine_valid_mask_from_image(batch["image"], batch["valid_mask"])
        keep_indices: list[int] = []
        updated_metadata: list[dict[str, Any]] = []
        for idx, item in enumerate(metadata):
            item_dict = dict(item)
            overall_valid_fraction = float(refined_valid_mask[idx].float().mean().item())
            threshold = float(item_dict.get("min_valid_fraction", min_valid_fraction))
            is_patch_valid = overall_valid_fraction >= threshold
            item_dict["overall_valid_fraction"] = overall_valid_fraction
            item_dict["is_patch_valid"] = bool(is_patch_valid)
            item_dict["min_valid_fraction"] = threshold
            updated_metadata.append(item_dict)
            if is_patch_valid:
                keep_indices.append(idx)
        metadata = updated_metadata
    else:
        refined_valid_mask = batch.get("valid_mask")
        keep_indices = [idx for idx, item in enumerate(metadata) if bool(item.get("is_patch_valid", True))]

    if not keep_indices:
        return None, {"batch_size": float(total), "kept_samples": 0.0, "dropped_samples": float(total)}

    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
    filtered = dict(batch)
    filtered["image"] = batch["image"].index_select(0, keep_tensor)
    if torch.is_tensor(refined_valid_mask):
        filtered["valid_mask"] = refined_valid_mask.index_select(0, keep_tensor)
    else:
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


def _prefer_valid_random_masking(
    tokens: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    mask_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MAE-style masking that prefers valid patches over nodata patches."""
    if not (0.0 <= mask_ratio < 1.0):
        raise ValueError("mask_ratio must satisfy 0 <= value < 1.")
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape (B, L, D).")
    if patch_valid_mask.shape[:2] != tokens.shape[:2]:
        raise ValueError("patch_valid_mask must align with token batch/length dimensions.")

    batch, num_patches, dim = tokens.shape
    len_keep = max(1, int(num_patches * (1.0 - mask_ratio)))

    noise = torch.rand(batch, num_patches, device=tokens.device)
    # Push invalid patches to the end of the keep ordering whenever possible.
    noise = noise + (~patch_valid_mask).to(dtype=noise.dtype) * 2.0

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))

    mask = torch.ones((batch, num_patches), device=tokens.device)
    mask[:, :len_keep] = 0.0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


def _compute_token_validity_weights(
    patch_valid_fraction: torch.Tensor,
    *,
    min_valid_fraction: float,
    enabled: bool,
    exponent: float,
) -> torch.Tensor:
    """Convert per-token valid fractions into loss weights.

    When disabled, this reproduces the existing hard-threshold behavior:
    valid tokens receive weight 1 and invalid tokens receive weight 0.

    When enabled, valid tokens are weighted continuously by
    ``patch_valid_fraction ** exponent`` while tokens below the threshold still
    receive weight 0.
    """
    if not (0.0 <= min_valid_fraction <= 1.0):
        raise ValueError("min_valid_fraction must satisfy 0 <= value <= 1.")
    if exponent <= 0.0:
        raise ValueError("exponent must be > 0.")

    patch_valid_fraction = patch_valid_fraction.clamp(0.0, 1.0)
    valid_tokens = patch_valid_fraction >= float(min_valid_fraction)
    if not enabled:
        return valid_tokens.to(dtype=patch_valid_fraction.dtype)

    weights = patch_valid_fraction.pow(float(exponent))
    return weights * valid_tokens.to(dtype=patch_valid_fraction.dtype)


def _satmae_forward_with_valid_mask(
    model: nn.Module,
    imgs: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mask_ratio: float,
    min_valid_fraction: float = DEFAULT_PATCH_VALID_FRACTION,
    token_validity_weighting: bool = False,
    token_validity_weight_exponent: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run SatMAE while excluding invalid pixels/patches from masking and loss."""
    patch_size = int(model.patch_embed.patch_size[0])
    in_chans = int(getattr(model, "in_c", imgs.shape[1]))

    x = model.patch_embed(imgs)
    x = x + model.pos_embed[:, 1:, :]

    patch_valid_fraction = compute_patch_valid_fraction(valid_mask, patch_size).to(device=x.device)
    patch_valid_mask = patch_valid_fraction >= float(min_valid_fraction)
    x_masked, mask, ids_restore = _prefer_valid_random_masking(x, patch_valid_mask, mask_ratio)

    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
    latent = torch.cat((cls_tokens, x_masked), dim=1)

    for blk in model.blocks:
        latent = blk(latent)
    latent = model.norm(latent)

    pred = model.forward_decoder(latent, ids_restore)

    target = model.patchify(imgs, patch_size, in_chans)
    valid_pixel_mask = patchify_valid_mask(valid_mask, patch_size, channels=in_chans).to(device=target.device)
    if bool(getattr(model, "norm_pix_loss", False)):
        target = normalize_patch_targets(target, valid_pixel_mask)

    valid = valid_pixel_mask.to(dtype=pred.dtype)
    valid_counts = valid.sum(dim=-1).clamp_min(1.0)
    per_patch_loss = ((pred - target).pow(2) * valid).sum(dim=-1) / valid_counts

    token_validity_weights = _compute_token_validity_weights(
        patch_valid_fraction.to(dtype=per_patch_loss.dtype),
        min_valid_fraction=float(min_valid_fraction),
        enabled=bool(token_validity_weighting),
        exponent=float(token_validity_weight_exponent),
    ).to(device=per_patch_loss.device)
    masked_token_weights = mask.to(dtype=per_patch_loss.dtype) * token_validity_weights
    if torch.any(masked_token_weights > 0):
        loss = (per_patch_loss * masked_token_weights).sum() / masked_token_weights.sum()
    else:
        loss = per_patch_loss.sum() * 0.0

    return loss, pred, mask, patch_valid_mask


def _apply_spectral_dropout(
    images: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    dropout_prob: float,
    max_channels: int,
) -> torch.Tensor:
    """Randomly zero valid pixels in a subset of channels during training."""
    if dropout_prob <= 0.0 or max_channels <= 0:
        return images
    if images.ndim != 4 or valid_mask.ndim != 3:
        return images

    out = images.clone()
    valid_mask_4d = valid_mask.unsqueeze(1).to(dtype=torch.bool)
    batch, channels, _, _ = out.shape
    max_drop = min(int(max_channels), max(channels - 1, 1))

    for batch_idx in range(batch):
        if float(torch.rand((), device=out.device).item()) >= float(dropout_prob):
            continue
        num_drop = int(torch.randint(1, max_drop + 1, (1,), device=out.device).item())
        drop_indices = torch.randperm(channels, device=out.device)[:num_drop]
        for channel_idx in drop_indices.tolist():
            out[batch_idx, channel_idx] = torch.where(
                valid_mask_4d[batch_idx, 0],
                torch.zeros_like(out[batch_idx, channel_idx]),
                out[batch_idx, channel_idx],
            )
    return out


def _refine_valid_mask_from_image(
    images: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Tighten a cached valid mask using image support heuristics.

    - Treat non-zero support by absolute magnitude so z-scored negative values remain valid.
    - For 3-channel Mars COLOR imagery, suppress pixels supported by only a single band.
    """
    if images.ndim != 4 or valid_mask.ndim != 3:
        return valid_mask

    refined = valid_mask.to(dtype=torch.bool)
    pixel_active = images.abs() > float(eps)
    refined = refined & pixel_active.any(dim=1)
    if images.shape[1] >= 3:
        refined = refined & (pixel_active.sum(dim=1) >= 2)
    return refined


def _forward_mars_satmae_batch(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    mask_ratio: float,
    valid_mask_aware: bool,
    min_valid_fraction: float,
    token_validity_weighting: bool,
    token_validity_weight_exponent: float,
    spectral_dropout_prob: float,
    spectral_dropout_max_channels: int,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward one batch through SatMAE with Mars-specific nodata handling."""
    images = batch["image"]
    valid_mask = _refine_valid_mask_from_image(images, batch["valid_mask"])

    if training and spectral_dropout_prob > 0.0:
        images = _apply_spectral_dropout(
            images,
            valid_mask,
            dropout_prob=float(spectral_dropout_prob),
            max_channels=int(spectral_dropout_max_channels),
        )

    if valid_mask_aware:
        loss, pred, mask, _ = _satmae_forward_with_valid_mask(
            model,
            images,
            valid_mask,
            mask_ratio=mask_ratio,
            min_valid_fraction=min_valid_fraction,
            token_validity_weighting=token_validity_weighting,
            token_validity_weight_exponent=token_validity_weight_exponent,
        )
        return loss, pred, mask

    return model(images, mask_ratio=mask_ratio)


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


def _stretch_preview_rgb(
    image: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    reference_image: torch.Tensor | None = None,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply a per-channel percentile stretch for visualization.

    This mirrors the thumbnail visualization approach in `src/dataset/validate_sampling.py`.

    Args:
        image: Tensor shaped (B, C, H, W) in display space (typically [0,1]).
        valid_mask: Bool tensor shaped (B, 1, H, W) or (B, H, W) where True means valid.
    """
    if image.ndim != 4:
        raise ValueError("Expected image tensor with shape (B, C, H, W).")
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    if valid_mask.ndim != 4:
        raise ValueError("Expected valid_mask tensor with shape (B, 1, H, W) or (B, H, W).")
    if reference_image is not None and reference_image.shape != image.shape:
        raise ValueError("reference_image must match image shape when provided.")

    # Force invalid pixels to 0 so they don't influence percentiles.
    mask = valid_mask.to(dtype=torch.bool)
    masked = torch.where(mask, image, torch.zeros_like(image))

    out = masked.clone()
    batch, channels, _, _ = out.shape
    # Compute percentiles per (B, C) on CPU for stability and to avoid GPU sync overhead.
    out_cpu = out.detach().cpu()
    ref_cpu = reference_image.detach().cpu() if reference_image is not None else out_cpu
    mask_cpu = mask.detach().cpu()

    for b in range(batch):
        for c in range(channels):
            values = ref_cpu[b, c][mask_cpu[b, 0]].flatten()
            if values.numel() < 10:
                continue
            p_low = torch.quantile(values, low_pct / 100.0)
            p_high = torch.quantile(values, high_pct / 100.0)
            denom = float((p_high - p_low).item())
            if denom <= eps:
                continue
            channel = out_cpu[b, c]
            stretched = torch.clamp((channel - p_low) / (p_high - p_low), 0.0, 1.0)
            stretched[~mask_cpu[b, 0]] = 0.0
            out_cpu[b, c] = stretched

    return out_cpu.to(device=image.device, dtype=image.dtype)


def _prepare_preview_display(
    image: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Convert Mars multispectral tensors into a more interpretable display view.

    HiRISE color products are stored as (NIR, RED, BLUE-GREEN). Rendering those
    bands directly as RGB is valid false color, but it can make early MAE
    reconstructions look misleadingly magenta. For training-time inspection we
    therefore allow a structural display mode that repeats the RED channel as
    grayscale.
    """
    if image.ndim != 4:
        raise ValueError("Expected image tensor with shape (B, C, H, W).")

    channels = image.shape[1]
    if channels == 1:
        return image.repeat(1, 3, 1, 1)
    if channels < 3:
        base = image[:, :1].repeat(1, 3, 1, 1)
        return base

    if mode == "false_color":
        return image[:, :3]
    if mode == "approx_natural":
        red = image[:, 1:2]
        blue_green = image[:, 2:3]
        return torch.cat([red, blue_green, blue_green], dim=1)
    if mode == "red_grayscale":
        red = image[:, 1:2]
        return red.repeat(1, 3, 1, 1)

    raise ValueError(f"Unsupported preview display mode: {mode}")


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
    valid_mask_aware: bool,
    min_valid_fraction: float,
    preview_display_mode: str = "red_grayscale",
    token_validity_weighting: bool = False,
    token_validity_weight_exponent: float = 2.0,
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
        filtered, _ = _filter_batch_by_validity(
            candidate,
            require_patch_valid=require_patch_valid,
            min_valid_fraction=min_valid_fraction,
        )
        if filtered is not None:
            batch = filtered
            break
    if batch is None:
        return None

    batch = _move_batch_to_device(batch, device)
    images = batch["image"]
    valid_mask = _refine_valid_mask_from_image(images, batch["valid_mask"])
    metadata = list(batch.get("metadata", []))

    was_training = model.training
    model.eval()
    with autocast(device_type=device.type, enabled=amp_enabled):
        _, pred, mask = _forward_mars_satmae_batch(
            model,
            batch,
            mask_ratio=mask_ratio,
            valid_mask_aware=valid_mask_aware,
            min_valid_fraction=min_valid_fraction,
            token_validity_weighting=token_validity_weighting,
            token_validity_weight_exponent=token_validity_weight_exponent,
            spectral_dropout_prob=0.0,
            spectral_dropout_max_channels=0,
            training=False,
        )

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

    image_disp_raw = _denormalize_preview_image(images, valid_mask_4d, dataset_normalization_stats)
    pred_disp_raw = _denormalize_preview_image(pred_img, valid_mask_4d, dataset_normalization_stats)
    masked_disp_raw = _denormalize_preview_image(
        images * (1.0 - mask_img), valid_mask_4d, dataset_normalization_stats
    )
    composite_disp_raw = _denormalize_preview_image(
        images * (1.0 - mask_img) + pred_img * mask_img,
        valid_mask_4d,
        dataset_normalization_stats,
    )

    image_disp_view = _prepare_preview_display(image_disp_raw, mode=preview_display_mode)
    pred_disp_view = _prepare_preview_display(pred_disp_raw, mode=preview_display_mode)
    masked_disp_view = _prepare_preview_display(masked_disp_raw, mode=preview_display_mode)
    composite_disp_view = _prepare_preview_display(
        composite_disp_raw,
        mode=preview_display_mode,
    )

    # Stretch all panels using the original input's per-channel statistics so
    # hue differences reflect the model output instead of per-panel renormalization.
    image_disp = _stretch_preview_rgb(
        image_disp_view,
        valid_mask_4d,
        reference_image=image_disp_view,
    )
    pred_disp = _stretch_preview_rgb(
        pred_disp_view,
        valid_mask_4d,
        reference_image=image_disp_view,
    )
    masked_disp = _stretch_preview_rgb(
        masked_disp_view,
        valid_mask_4d,
        reference_image=image_disp_view,
    )
    composite_disp = _stretch_preview_rgb(
        composite_disp_view,
        valid_mask_4d,
        reference_image=image_disp_view,
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

    display_titles = {
        "false_color": "false colour (NIR->R, RED->G, BG->B)",
        "approx_natural": "approx natural colour (RED, BG, BG)",
        "red_grayscale": "RED-channel grayscale",
    }
    fig.suptitle(
        f"SatMAE Mars reconstruction preview — {display_titles.get(preview_display_mode, preview_display_mode)}",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    if was_training:
        model.train()
    return out_path


def _preview_variant_path(
    out_path: pathlib.Path,
    *,
    mode: str,
    primary_mode: str,
) -> pathlib.Path:
    """Return a stable filename for a preview display variant."""
    if mode == primary_mode:
        return out_path
    return out_path.with_name(f"{out_path.stem}_{mode}{out_path.suffix}")


def save_reconstruction_preview_variants(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_ratio: float,
    out_path: pathlib.Path,
    amp_enabled: bool,
    require_patch_valid: bool,
    dataset_normalization_stats: tuple[torch.Tensor, torch.Tensor] | None,
    valid_mask_aware: bool,
    min_valid_fraction: float,
    preview_display_mode: str = "red_grayscale",
    preview_secondary_display_mode: str | None = "false_color",
    token_validity_weighting: bool = False,
    token_validity_weight_exponent: float = 2.0,
    max_items: int = 4,
) -> dict[str, pathlib.Path]:
    """Save one or more preview display variants for the same reconstruction batch."""
    modes = [preview_display_mode]
    if preview_secondary_display_mode is not None and preview_secondary_display_mode not in modes:
        modes.append(preview_secondary_display_mode)

    saved: dict[str, pathlib.Path] = {}
    for mode in modes:
        variant_path = _preview_variant_path(
            out_path,
            mode=mode,
            primary_mode=preview_display_mode,
        )
        written = save_reconstruction_preview(
            model=model,
            dataloader=dataloader,
            device=device,
            mask_ratio=mask_ratio,
            out_path=variant_path,
            amp_enabled=amp_enabled,
            require_patch_valid=require_patch_valid,
            dataset_normalization_stats=dataset_normalization_stats,
            valid_mask_aware=valid_mask_aware,
            min_valid_fraction=min_valid_fraction,
            preview_display_mode=mode,
            token_validity_weighting=token_validity_weighting,
            token_validity_weight_exponent=token_validity_weight_exponent,
            max_items=max_items,
        )
        if written is not None:
            saved[mode] = written
    return saved


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
    valid_mask_aware: bool,
    min_valid_fraction: float,
    token_validity_weighting: bool,
    token_validity_weight_exponent: float,
    spectral_dropout_prob: float,
    spectral_dropout_max_channels: int,
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

        filtered_batch, batch_stats = _filter_batch_by_validity(
            batch,
            require_patch_valid=require_patch_valid,
            min_valid_fraction=min_valid_fraction,
        )
        total_kept += batch_stats["kept_samples"]
        total_dropped += batch_stats["dropped_samples"]
        if filtered_batch is None:
            continue

        filtered_batch = _move_batch_to_device(filtered_batch, device)
        with autocast(device_type=device.type, enabled=amp_enabled):
            loss, _, _ = _forward_mars_satmae_batch(
                model,
                filtered_batch,
                mask_ratio=mask_ratio,
                valid_mask_aware=valid_mask_aware,
                min_valid_fraction=min_valid_fraction,
                token_validity_weighting=token_validity_weighting,
                token_validity_weight_exponent=token_validity_weight_exponent,
                spectral_dropout_prob=spectral_dropout_prob,
                spectral_dropout_max_channels=spectral_dropout_max_channels,
                training=True,
            )
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
    valid_mask_aware: bool,
    min_valid_fraction: float,
    token_validity_weighting: bool,
    token_validity_weight_exponent: float,
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
        filtered_batch, batch_stats = _filter_batch_by_validity(
            batch,
            require_patch_valid=require_patch_valid,
            min_valid_fraction=min_valid_fraction,
        )
        total_kept += batch_stats["kept_samples"]
        total_dropped += batch_stats["dropped_samples"]
        if filtered_batch is None:
            continue

        filtered_batch = _move_batch_to_device(filtered_batch, device)
        with autocast(device_type=device.type, enabled=amp_enabled):
            loss, _, _ = _forward_mars_satmae_batch(
                model,
                filtered_batch,
                mask_ratio=mask_ratio,
                valid_mask_aware=valid_mask_aware,
                min_valid_fraction=min_valid_fraction,
                token_validity_weighting=token_validity_weighting,
                token_validity_weight_exponent=token_validity_weight_exponent,
                spectral_dropout_prob=0.0,
                spectral_dropout_max_channels=0,
                training=False,
            )
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
    valid_mask_aware: bool = True,
    min_valid_fraction: float = DEFAULT_PATCH_VALID_FRACTION,
    token_validity_weighting: bool = False,
    token_validity_weight_exponent: float = 2.0,
    spectral_dropout_prob: float = 0.0,
    spectral_dropout_max_channels: int = 1,
    config: dict[str, Any],
    preview_loader: torch.utils.data.DataLoader | None,
    reconstruction_dir: pathlib.Path,
    dataset_normalization_stats: tuple[torch.Tensor, torch.Tensor] | None,
    preview_display_mode: str = "red_grayscale",
    reconstruction_max_items: int,
    preview_secondary_display_mode: str | None = "false_color",
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
            valid_mask_aware=valid_mask_aware,
            min_valid_fraction=min_valid_fraction,
            token_validity_weighting=token_validity_weighting,
            token_validity_weight_exponent=token_validity_weight_exponent,
            spectral_dropout_prob=spectral_dropout_prob,
            spectral_dropout_max_channels=spectral_dropout_max_channels,
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
                valid_mask_aware=valid_mask_aware,
                min_valid_fraction=min_valid_fraction,
                token_validity_weighting=token_validity_weighting,
                token_validity_weight_exponent=token_validity_weight_exponent,
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
                    best_previews = save_reconstruction_preview_variants(
                        model=model,
                        dataloader=preview_loader,
                        device=device,
                        mask_ratio=mask_ratio,
                        out_path=reconstruction_dir / "reconstruction_best.png",
                        amp_enabled=amp_enabled,
                        require_patch_valid=require_patch_valid,
                        dataset_normalization_stats=dataset_normalization_stats,
                        valid_mask_aware=valid_mask_aware,
                        min_valid_fraction=min_valid_fraction,
                        preview_display_mode=preview_display_mode,
                        preview_secondary_display_mode=preview_secondary_display_mode,
                        token_validity_weighting=token_validity_weighting,
                        token_validity_weight_exponent=token_validity_weight_exponent,
                        max_items=reconstruction_max_items,
                    )
                    if wandb_logger is not None:
                        for mode, best_preview in best_previews.items():
                            key = "reconstructions/best"
                            if mode != preview_display_mode:
                                key = f"{key}_{mode}"
                            wandb_logger.log_image(
                                key,
                                best_preview,
                                step=epoch_wandb_step,
                                caption=f"Best reconstruction preview at epoch {epoch + 1} ({mode})",
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
            epoch_previews = save_reconstruction_preview_variants(
                model=model,
                dataloader=preview_loader,
                device=device,
                mask_ratio=mask_ratio,
                out_path=reconstruction_dir / f"reconstruction_epoch_{epoch + 1:04d}.png",
                amp_enabled=amp_enabled,
                require_patch_valid=require_patch_valid,
                dataset_normalization_stats=dataset_normalization_stats,
                valid_mask_aware=valid_mask_aware,
                min_valid_fraction=min_valid_fraction,
                preview_display_mode=preview_display_mode,
                preview_secondary_display_mode=preview_secondary_display_mode,
                token_validity_weighting=token_validity_weighting,
                token_validity_weight_exponent=token_validity_weight_exponent,
                max_items=reconstruction_max_items,
            )
            if wandb_logger is not None:
                for mode, epoch_preview in epoch_previews.items():
                    key = "reconstructions/epoch"
                    if mode != preview_display_mode:
                        key = f"{key}_{mode}"
                    wandb_logger.log_image(
                        key,
                        epoch_preview,
                        step=epoch_wandb_step,
                        caption=f"Epoch {epoch + 1} reconstruction preview ({mode})",
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
        final_previews = save_reconstruction_preview_variants(
            model=model,
            dataloader=preview_loader,
            device=device,
            mask_ratio=mask_ratio,
            out_path=reconstruction_dir / "reconstruction_final.png",
            amp_enabled=amp_enabled,
            require_patch_valid=require_patch_valid,
            dataset_normalization_stats=dataset_normalization_stats,
            valid_mask_aware=valid_mask_aware,
            min_valid_fraction=min_valid_fraction,
            preview_display_mode=preview_display_mode,
            preview_secondary_display_mode=preview_secondary_display_mode,
            token_validity_weighting=token_validity_weighting,
            token_validity_weight_exponent=token_validity_weight_exponent,
            max_items=reconstruction_max_items,
        )
        if wandb_logger is not None:
            for mode, final_preview in final_previews.items():
                key = "reconstructions/final"
                if mode != preview_display_mode:
                    key = f"{key}_{mode}"
                wandb_logger.log_image(
                    key,
                    final_preview,
                    step=int(epochs * steps_per_epoch),
                    caption=f"Final reconstruction preview ({mode})",
                )
    if wandb_logger is not None:
        wandb_logger.finish(summary)
    return summary


# -----------------------------
# Recipe-alignment transforms
# -----------------------------

def _resized_crop_torch(
    img: torch.Tensor,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    out_size: int,
    mode: str,
) -> torch.Tensor:
    """Torch-only resized crop for CHW (float) or HW (uint8/bool) tensors."""
    if img.ndim == 2:
        img_c = img.unsqueeze(0)
    elif img.ndim == 3:
        img_c = img
    else:
        return img

    cropped = img_c[:, top : top + height, left : left + width]
    cropped = cropped.unsqueeze(0)
    resized = nn.functional.interpolate(
        cropped,
        size=(int(out_size), int(out_size)),
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    resized = resized.squeeze(0)
    return resized[0] if img.ndim == 2 else resized


def _get_random_resized_crop_params(
    h: int,
    w: int,
    *,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    num_tries: int = 10,
) -> tuple[int, int, int, int]:
    """Approximate torchvision RandomResizedCrop.get_params without torchvision."""
    area = float(h * w)
    log_ratio_min = math.log(float(ratio[0]))
    log_ratio_max = math.log(float(ratio[1]))

    for _ in range(int(num_tries)):
        target_area = area * float(torch.empty(1).uniform_(float(scale[0]), float(scale[1])).item())
        aspect = math.exp(float(torch.empty(1).uniform_(log_ratio_min, log_ratio_max).item()))

        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))

        if 0 < crop_w <= w and 0 < crop_h <= h:
            top = int(torch.randint(0, h - crop_h + 1, (1,)).item())
            left = int(torch.randint(0, w - crop_w + 1, (1,)).item())
            return top, left, crop_h, crop_w

    # Fallback: center crop with clipped aspect.
    in_ratio = w / h
    if in_ratio < float(ratio[0]):
        crop_w = w
        crop_h = int(round(crop_w / float(ratio[0])))
    elif in_ratio > float(ratio[1]):
        crop_h = h
        crop_w = int(round(crop_h * float(ratio[1])))
    else:
        crop_h = h
        crop_w = w

    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return int(top), int(left), int(crop_h), int(crop_w)


def _apply_spatial_transform_to_sample(
    sample: dict[str, Any],
    *,
    out_size: int,
    train: bool,
) -> dict[str, Any]:
    """Apply train-only spatial augs while preserving valid_mask semantics.

    - Applies the same crop/resize/flip to `image` (C,H,W) and `valid_mask` (H,W).
    - Uses nearest-neighbor when resizing `valid_mask`.

    Implemented without torchvision to avoid import-time dependency issues.
    """
    image = sample.get("image")
    valid_mask = sample.get("valid_mask")
    if not torch.is_tensor(image) or not torch.is_tensor(valid_mask):
        return sample

    if image.ndim != 3:
        return sample
    if valid_mask.ndim != 2:
        if valid_mask.ndim == 3 and valid_mask.shape[0] == 1:
            valid_mask = valid_mask.squeeze(0)
        else:
            return sample

    if train:
        _, h, w = image.shape
        top, left, ch, cw = _get_random_resized_crop_params(
            h,
            w,
            scale=(0.2, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
        )
        image = _resized_crop_torch(
            image,
            top=top,
            left=left,
            height=ch,
            width=cw,
            out_size=out_size,
            mode="bilinear",
        )
        vm = valid_mask.to(dtype=torch.float32)
        vm = _resized_crop_torch(
            vm,
            top=top,
            left=left,
            height=ch,
            width=cw,
            out_size=out_size,
            mode="nearest",
        )
        valid_mask = vm >= 0.5

        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
            valid_mask = torch.flip(valid_mask, dims=(-1,))

    out = dict(sample)
    out["image"] = image
    out["valid_mask"] = valid_mask
    return out


def main() -> None:
    print("[satmae][debug] main() entered", flush=True)
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
    # Default copied from clip.marsclip_patches.DEFAULT_PATCH_VALID_FRACTION to avoid
    # importing marsclip_patches at module import time (torchgeo/torchvision dependency).
    parser.add_argument("--min-valid-fraction", type=float, default=0.25)
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
    parser.add_argument("--valid-mask-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-min-valid-fraction", type=float, default=DEFAULT_PATCH_VALID_FRACTION)
    parser.add_argument("--token-validity-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--token-validity-weight-exponent", type=float, default=2.0)
    parser.add_argument("--spectral-dropout-prob", type=float, default=0.0)
    parser.add_argument("--spectral-dropout-max-channels", type=int, default=1)

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
    parser.add_argument(
        "--preview-display-mode",
        type=str,
        default="red_grayscale",
        choices=("red_grayscale", "false_color", "approx_natural"),
    )
    parser.add_argument(
        "--preview-secondary-display-mode",
        type=str,
        default="false_color",
        choices=("none", "red_grayscale", "false_color", "approx_natural"),
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    # Pretrained init plumbing.
    parser.add_argument("--init-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--init-mode", type=str, default="encoder", choices=("encoder", "all"))
    parser.add_argument("--init-pos-embed", type=str, default="interp", choices=("interp", "keep", "skip"))
    parser.add_argument("--init-verbose", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    print(
        f"[satmae][debug] args parsed: run_name={getattr(args, 'run_name', None)} out_root={getattr(args, 'out_root', None)}",
        flush=True,
    )

    # Debug marker right before heavy setup.
    print("[satmae][debug] pre-setup: about to build model/dataloaders", flush=True)

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

    try:
        # -----------------
        # Dataset / splits
        # -----------------
        patch_records = None
        split_summary_path = None

        # Lazily import raw patch dataset only if that codepath is needed.
        MarsCLIPPatchDataset = None
        load_patch_records = None

        def _lazy_import_raw_dataset() -> None:
            nonlocal MarsCLIPPatchDataset, load_patch_records
            if MarsCLIPPatchDataset is not None:
                return
            from clip.marsclip_patches import MarsCLIPPatchDataset as _MarsCLIPPatchDataset
            from clip.marsclip_patches import load_patch_records as _load_patch_records

            MarsCLIPPatchDataset = _MarsCLIPPatchDataset
            load_patch_records = _load_patch_records

        train_dataset = None
        val_dataset = None
        test_dataset = None

        if args.litdata_root is not None:
            litdata_root = args.litdata_root.expanduser().resolve()
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
            _lazy_import_raw_dataset()
            if args.patch_records_path is not None and args.patch_records_path.exists():
                assert load_patch_records is not None
                patch_records = load_patch_records(args.patch_records_path)

            assert MarsCLIPPatchDataset is not None
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

        # -----------------
        # Transforms
        # -----------------
        train_transform = lambda sample: _apply_spatial_transform_to_sample(sample, out_size=int(args.image_size), train=True)
        eval_transform = lambda sample: _apply_spatial_transform_to_sample(sample, out_size=int(args.image_size), train=False)
        print(f"[data] train_transform = recipe_align_rsz_crop+hflip -> {args.image_size}")
        print(f"[data] val_transform = deterministic -> {args.image_size}")

        def _wrap_dataset_with_transform(ds: Any, *, transform: Any) -> Any:
            if ds is None:
                return None

            class _Wrapped(torch.utils.data.Dataset):
                def __init__(self, inner: Any, t: Any):
                    self.inner = inner
                    self.t = t

                def __len__(self) -> int:
                    return len(self.inner)

                def __getitem__(self, idx: int) -> Any:
                    item = self.inner[idx]
                    return self.t(item)

            return _Wrapped(ds, transform)

        # Apply transforms for cache/raw datasets (LitData has its own pipeline).
        train_dataset_xform = train_dataset
        val_dataset_xform = val_dataset
        test_dataset_xform = test_dataset
        if args.litdata_root is None:
            train_dataset_xform = _wrap_dataset_with_transform(train_dataset, transform=train_transform)
            val_dataset_xform = _wrap_dataset_with_transform(val_dataset, transform=eval_transform)
            test_dataset_xform = _wrap_dataset_with_transform(test_dataset, transform=eval_transform)

        # -----------------
        # Dataloaders
        # -----------------
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
            assert train_dataset_xform is not None
            train_loader = build_fb_mae_dataloader(
                train_dataset_xform,
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
            if val_dataset_xform is not None and len(val_dataset_xform) > 0:
                val_loader = build_fb_mae_dataloader(
                    val_dataset_xform,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    prefetch_factor=args.prefetch_factor,
                    persistent_workers=args.persistent_workers,
                    drop_last=False,
                )
            train_count = len(train_dataset_xform)
            val_count = len(val_dataset_xform) if val_dataset_xform is not None else None
            test_count = len(test_dataset_xform) if test_dataset_xform is not None else None

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

        # -----------------
        # Model
        # -----------------
        in_chans = 3
        model = build_satmae_model(
            args.model,
            img_size=args.image_size,
            patch_size=args.patch_size_px,
            in_chans=in_chans,
            norm_pix_loss=args.norm_pix_loss,
        )

        if args.init_checkpoint is not None:
            init_report = init_satmae_from_checkpoint(
                model,
                args.init_checkpoint,
                mode=args.init_mode,
                pos_embed=args.init_pos_embed,
                verbose=bool(args.init_verbose),
            )
            (out_dir / "init_report.json").write_text(json.dumps(init_report, indent=2))

        device = _resolve_device(args.device)
        model.to(device)

        amp_enabled = bool(args.amp and device.type == "cuda")
        # GradScaler signatures differ across torch versions. Try common forms.
        scaler: GradScaler | None
        try:
            scaler = GradScaler(device.type, enabled=amp_enabled)
        except TypeError:  # pragma: no cover
            scaler = GradScaler(enabled=amp_enabled)

        trainable_parameters, total_parameters = count_trainable_parameters(model)

        lr = resolve_effective_lr(
            batch_size=args.batch_size,
            accum_iter=args.accum_iter,
            base_lr=args.blr,
            explicit_lr=args.lr,
        )
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

        # -----------------
        # Run config + W&B
        # -----------------
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
            "valid_mask_aware": bool(args.valid_mask_aware),
            "token_min_valid_fraction": float(args.token_min_valid_fraction),
            "token_validity_weighting": bool(args.token_validity_weighting),
            "token_validity_weight_exponent": float(args.token_validity_weight_exponent),
            "spectral_dropout_prob": float(args.spectral_dropout_prob),
            "spectral_dropout_max_channels": int(args.spectral_dropout_max_channels),
            "val_max_batches": int(args.val_max_batches) if args.val_max_batches is not None else None,
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
            "preview_display_mode": args.preview_display_mode,
            "preview_secondary_display_mode": None
            if args.preview_secondary_display_mode == "none"
            else args.preview_secondary_display_mode,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "init_mode": args.init_mode,
            "init_pos_embed": args.init_pos_embed,
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
            valid_mask_aware=args.valid_mask_aware,
            min_valid_fraction=args.token_min_valid_fraction,
            token_validity_weighting=args.token_validity_weighting,
            token_validity_weight_exponent=args.token_validity_weight_exponent,
            spectral_dropout_prob=args.spectral_dropout_prob,
            spectral_dropout_max_channels=args.spectral_dropout_max_channels,
            config=config,
            preview_loader=preview_loader,
            reconstruction_dir=reconstruction_dir,
            dataset_normalization_stats=dataset_normalization_stats,
            preview_display_mode=args.preview_display_mode,
            preview_secondary_display_mode=None
            if args.preview_secondary_display_mode == "none"
            else args.preview_secondary_display_mode,
            reconstruction_max_items=args.reconstruction_max_items,
            progress_log_interval=args.progress_log_interval,
            wandb_logger=wandb_logger,
        )

        print(f"Saved SatMAE checkpoint to {summary['checkpoint']}")
        if summary.get("wandb_mode") != "disabled" and summary.get("wandb_run_dir") is not None:
            print(f"Saved W&B run to {summary['wandb_run_dir']}")

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


if __name__ == "__main__":
    main()
