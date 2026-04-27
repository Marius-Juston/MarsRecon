"""Train a Stage-B aligner between Mars text and MAE image embeddings.

This script:
1) loads Mars patch samples from ``clip.marsclip_patches``,
2) loads a SatMAE encoder using ``clip.satmae_bridge``,
3) encodes rationale text with ``stage_b.T5_encoder``,
4) learns projection heads with CLIP-style contrastive loss.

The trained aligner can then be reused for downstream retrieval, clustering,
and lightweight classifier heads.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import pathlib
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clip.fb_mae_train_utils import (
    count_trainable_parameters,
    save_training_history,
    save_training_progress,
)
from clip.marsclip_patches import MarsCLIPPatchDataset, load_patch_records
from clip.satmae_bridge import build_satmae_model
from stage_b.T5_encoder import T5Encoder

DEFAULT_STAGE_B_OUT_ROOT = pathlib.Path("/scratch/marsrecon_runs/stage_b/text_mae_align")
WANDB_STEP_METRIC = "trainer/global_step"


class RunInterruptedError(RuntimeError):
    """Raised when the process receives a termination signal."""


class TerminationMonitor:
    """Capture SIGINT/SIGTERM and convert them into controlled shutdowns."""

    def __init__(self) -> None:
        self._requested = False
        self._signal_name: str | None = None
        self._handlers: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: Any) -> None:
        self._requested = True
        self._signal_name = signal.Signals(signum).name
        print(f"[align] received {self._signal_name}; finishing current step then shutting down.", flush=True)

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._handlers.items():
            signal.signal(signum, handler)
        self._handlers.clear()

    def raise_if_requested(self) -> None:
        if self._requested:
            signal_name = self._signal_name or "SIGTERM"
            raise RunInterruptedError(f"Received termination signal: {signal_name}")

    @property
    def signal_name(self) -> str | None:
        return self._signal_name


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _slugify(text: str) -> str:
    """Convert a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "stage-b-align-run"


def resolve_run_output_dir(
    *,
    out_dir: pathlib.Path | None,
    out_root: pathlib.Path,
    run_name: str | None,
    model_name: str,
) -> pathlib.Path:
    """Resolve a unique output directory for a Stage-B alignment run."""
    if out_dir is not None:
        resolved = out_dir.expanduser().resolve()
        if resolved.exists() and any(resolved.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {resolved}")
        return resolved

    date_stamp = time.strftime("%Y%m%d")
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = _slugify(run_name or f"text-mae-align-{model_name}")
    parent = out_root.expanduser().resolve() / date_stamp
    candidate = parent / f"{time_stamp}_{base_name}"
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{time_stamp}_{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def _extract_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint  # type: ignore[return-value]
    raise ValueError("Checkpoint does not contain a recognized model state dict.")


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return str(value)


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    progress: float,
    lr: float,
    min_lr: float,
    warmup_epochs: int,
    epochs: int,
) -> float:
    """Apply per-step warmup + cosine decay, matching the Stage-A schedule shape."""
    if progress < float(warmup_epochs):
        current_lr = float(lr) * progress / max(float(warmup_epochs), 1.0)
    else:
        cosine_progress = (progress - float(warmup_epochs)) / max(float(epochs - warmup_epochs), 1.0)
        cosine_progress = min(max(cosine_progress, 0.0), 1.0)
        current_lr = float(min_lr) + (float(lr) - float(min_lr)) * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = float(current_lr)
    return float(current_lr)


def _limit_patch_records(
    patch_records: pd.DataFrame,
    *,
    max_patches: int | None,
    seed: int = 0,
) -> pd.DataFrame:
    if max_patches is None or max_patches >= len(patch_records):
        return patch_records.reset_index(drop=True)
    generator = torch.Generator().manual_seed(int(seed))
    selected = torch.randperm(len(patch_records), generator=generator).tolist()[: int(max_patches)]
    return patch_records.iloc[selected].reset_index(drop=True)


def load_split_manifest(path: pathlib.Path | str) -> pd.DataFrame:
    """Load the Stage-A split manifest and normalize it for Stage-B reuse."""
    manifest = pd.read_csv(pathlib.Path(path))
    if "patch_id" not in manifest.columns:
        raise ValueError("split manifest must contain a 'patch_id' column.")

    normalized = manifest.copy()
    normalized["patch_id"] = normalized["patch_id"].astype(str)
    holdout = normalized.get("holdout_split", pd.Series(index=normalized.index, dtype="object"))
    holdout = holdout.fillna("").astype(str).str.strip().str.lower()
    is_test = normalized.get("is_test", pd.Series(False, index=normalized.index))
    is_test = is_test.fillna(False)
    if is_test.dtype != bool:
        is_test = is_test.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})

    stage_b_split = pd.Series("train", index=normalized.index, dtype="object")
    stage_b_split.loc[holdout == "val"] = "val"
    stage_b_split.loc[holdout == "test"] = "test"
    stage_b_split.loc[is_test.astype(bool)] = "test"
    normalized["stage_b_split"] = stage_b_split
    return normalized[["patch_id", "stage_b_split", *[col for col in ("holdout_split", "fold", "is_test") if col in normalized.columns]]]


def attach_split_manifest(
    patch_records: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Attach normalized split labels to a patch-record table."""
    merged = patch_records.merge(
        split_manifest,
        on="patch_id",
        how="left",
        validate="one_to_one",
    )
    if "stage_b_split" not in merged.columns:
        raise ValueError("split manifest merge did not produce a 'stage_b_split' column.")
    missing = merged["stage_b_split"].isna()
    if bool(missing.any()):
        sample_ids = merged.loc[missing, "patch_id"].astype(str).head(5).tolist()
        raise ValueError(
            f"split manifest is missing {int(missing.sum())} patch ids from patch records; "
            f"examples: {sample_ids}"
        )
    merged["stage_b_split"] = merged["stage_b_split"].astype(str)
    return merged.reset_index(drop=True)


def filter_patch_records_by_split(
    patch_records: pd.DataFrame,
    *,
    split_name: str,
) -> pd.DataFrame:
    """Return only patch records from a named split."""
    if "stage_b_split" not in patch_records.columns:
        raise ValueError("patch_records must include 'stage_b_split' before filtering by split.")
    split = str(split_name).strip().lower()
    filtered = patch_records.loc[patch_records["stage_b_split"].astype(str).str.lower() == split].copy()
    if filtered.empty:
        raise ValueError(f"No patch records found for split '{split_name}'.")
    return filtered.reset_index(drop=True)


def select_balanced_patch_records(
    patch_records: pd.DataFrame,
    *,
    max_patches: int | None,
    label_column: str = "rationale_raw",
    seed: int = 0,
) -> pd.DataFrame:
    """Downsample patch records while keeping label coverage as even as possible."""
    if max_patches is None or max_patches >= len(patch_records):
        return patch_records.reset_index(drop=True)
    if label_column not in patch_records.columns:
        return _limit_patch_records(patch_records, max_patches=max_patches, seed=seed)

    grouped = [
        group.sample(frac=1.0, random_state=int(seed) + idx).reset_index(drop=True)
        for idx, (_, group) in enumerate(patch_records.groupby(label_column, sort=False))
    ]
    if not grouped:
        return patch_records.reset_index(drop=True)

    label_count = len(grouped)
    base_quota = int(max_patches) // label_count
    extras = int(max_patches) % label_count
    selected_parts: list[pd.DataFrame] = []
    remainder_parts: list[pd.DataFrame] = []

    for idx, group in enumerate(grouped):
        take = min(len(group), base_quota + (1 if idx < extras else 0))
        if take > 0:
            selected_parts.append(group.iloc[:take].copy())
        if take < len(group):
            remainder_parts.append(group.iloc[take:].copy())

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else patch_records.iloc[:0].copy()
    if len(selected) < int(max_patches) and remainder_parts:
        remaining = pd.concat(remainder_parts, ignore_index=True)
        fill = remaining.sample(
            n=min(int(max_patches) - len(selected), len(remaining)),
            random_state=int(seed) + 10_000,
            replace=False,
        )
        selected = pd.concat([selected, fill], ignore_index=True)

    selected = selected.sample(frac=1.0, random_state=int(seed) + 20_000).reset_index(drop=True)
    return selected.iloc[: int(max_patches)].reset_index(drop=True)


def _ranks_from_similarity(similarity: torch.Tensor, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 1-indexed positive ranks for rows and columns."""
    text_ids = {text: idx for idx, text in enumerate(sorted(set(texts)))}
    encoded = torch.tensor([text_ids[text] for text in texts], device=similarity.device)
    positive_mask = encoded.unsqueeze(1) == encoded.unsqueeze(0)

    row_order = torch.argsort(similarity, dim=1, descending=True)
    row_positive = torch.gather(positive_mask, dim=1, index=row_order)
    row_rank = row_positive.to(torch.int64).argmax(dim=1) + 1

    col_order = torch.argsort(similarity, dim=0, descending=True)
    col_positive = torch.gather(positive_mask, dim=0, index=col_order)
    col_rank = col_positive.to(torch.int64).argmax(dim=0) + 1
    return row_rank, col_rank


def _recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= int(k)).float().mean().item())


def _mean_reciprocal_rank(ranks: torch.Tensor) -> float:
    return float((1.0 / ranks.to(torch.float32)).mean().item())


def compute_retrieval_metrics(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    texts: list[str],
) -> dict[str, float]:
    similarity = image_embeddings @ text_embeddings.T
    img_to_txt_rank, txt_to_img_rank = _ranks_from_similarity(similarity, texts)
    return {
        "num_samples": float(similarity.shape[0]),
        "num_unique_texts": float(len(set(texts))),
        "image_to_text_r1": _recall_at_k(img_to_txt_rank, 1),
        "image_to_text_r5": _recall_at_k(img_to_txt_rank, 5),
        "image_to_text_r10": _recall_at_k(img_to_txt_rank, 10),
        "image_to_text_mrr": _mean_reciprocal_rank(img_to_txt_rank),
        "image_to_text_median_rank": float(torch.median(img_to_txt_rank.to(torch.float32)).item()),
        "text_to_image_r1": _recall_at_k(txt_to_img_rank, 1),
        "text_to_image_r5": _recall_at_k(txt_to_img_rank, 5),
        "text_to_image_r10": _recall_at_k(txt_to_img_rank, 10),
        "text_to_image_mrr": _mean_reciprocal_rank(txt_to_img_rank),
        "text_to_image_median_rank": float(torch.median(txt_to_img_rank.to(torch.float32)).item()),
    }


def compute_alignment_score(metrics: dict[str, float]) -> float:
    """Scalar retrieval score used for best-checkpoint selection."""
    return 0.5 * (
        float(metrics["image_to_text_mrr"]) + float(metrics["text_to_image_mrr"])
    )


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
        payload = dict(metrics)
        payload.setdefault(WANDB_STEP_METRIC, float(step))
        self.run.log(payload, step=int(step))

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

    def update_config(self, config: dict[str, Any]) -> None:
        if hasattr(self.run, "config"):
            self.run.config.update(dict(config), allow_val_change=True)


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
    """Initialize an optional W&B run for Stage-B alignment."""
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
        print("[wandb] Online initialization failed; falling back to offline mode for this run.")
        init_kwargs["mode"] = "offline"
        run = wandb.init(**init_kwargs)
        normalized_mode = "offline"

    if hasattr(run, "define_metric"):
        run.define_metric(WANDB_STEP_METRIC)
        run.define_metric("*", step_metric=WANDB_STEP_METRIC)

    return WandbLogger(
        run=run,
        module=wandb,
        mode=normalized_mode,
        project=project,
        entity=entity,
        run_name=run_name,
        log_dir=str(resolved_log_dir),
    )


def _default_run_name(*, mae_checkpoint: pathlib.Path) -> str:
    resolved_checkpoint = mae_checkpoint.expanduser().resolve()
    if len(resolved_checkpoint.parents) >= 2:
        return f"text-mae-align-stage-b-{resolved_checkpoint.parents[1].name}"
    return f"text-mae-align-stage-b-{resolved_checkpoint.stem}"


def load_satmae_encoder(
    *,
    model_name: str,
    checkpoint_path: pathlib.Path,
    image_size: int,
    patch_size_px: int,
    in_chans: int = 3,
    freeze: bool = True,
) -> nn.Module:
    """Load a SatMAE model and return it ready for feature extraction."""
    model = build_satmae_model(
        model_name,
        img_size=image_size,
        patch_size=patch_size_px,
        in_chans=in_chans,
        norm_pix_loss=True,
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = _extract_model_state(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[align] warning: missing MAE keys: {len(missing)}")
    if unexpected:
        print(f"[align] warning: unexpected MAE keys: {len(unexpected)}")

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.eval()
    return model


IMAGE_POOL_CHOICES = ("cls", "mean_patch", "cls_plus_mean")


def image_pool_output_dim(pool: str, base_dim: int) -> int:
    """Effective image-feature dimensionality given a SatMAE pooling strategy."""
    if pool not in IMAGE_POOL_CHOICES:
        raise ValueError(f"Unknown image_pool '{pool}'. Expected one of {IMAGE_POOL_CHOICES}.")
    return int(base_dim) * (2 if pool == "cls_plus_mean" else 1)


@torch.no_grad()
def encode_image_with_satmae_encoder(
    model: nn.Module,
    images: torch.Tensor,
    *,
    pool: str = "cls",
) -> torch.Tensor:
    """Encode images with the SatMAE encoder using the requested token pooling."""
    if pool not in IMAGE_POOL_CHOICES:
        raise ValueError(f"Unknown image_pool '{pool}'. Expected one of {IMAGE_POOL_CHOICES}.")
    tokens = model.patch_embed(images)
    tokens = tokens + model.pos_embed[:, 1:, :]
    cls = model.cls_token + model.pos_embed[:, :1, :]
    cls = cls.expand(images.shape[0], -1, -1)
    hidden = torch.cat((cls, tokens), dim=1)
    for block in model.blocks:
        hidden = block(hidden)
    hidden = model.norm(hidden)
    cls_out = hidden[:, 0]
    if pool == "cls":
        return cls_out
    patch_mean = hidden[:, 1:].mean(dim=1)
    if pool == "mean_patch":
        return patch_mean
    return torch.cat([cls_out, patch_mean], dim=1)


def collate_patch_text(samples: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(sample.get("rationale_raw", "")) for sample in samples]
    metadata = [dict(sample.get("metadata", {})) for sample in samples]
    batch: dict[str, Any] = {"text": texts, "metadata": metadata}
    if "image" in samples[0]:
        batch["image"] = torch.stack([sample["image"] for sample in samples], dim=0)
    if "image_features" in samples[0]:
        batch["image_features"] = torch.stack([sample["image_features"] for sample in samples], dim=0)
    return batch


def build_alignment_dataloader(
    dataset: Dataset | list[dict[str, Any]],
    *,
    batch_size: int = 16,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader for Stage-B alignment batches."""
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_patch_text,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def build_text_labels(texts: list[str], device: torch.device) -> torch.Tensor:
    """Map a batch of rationale strings to dense integer labels."""
    label_lookup: dict[str, int] = {}
    label_ids: list[int] = []
    for text in texts:
        key = str(text)
        if key not in label_lookup:
            label_lookup[key] = len(label_lookup)
        label_ids.append(label_lookup[key])
    return torch.tensor(label_ids, device=device, dtype=torch.long)


def _masked_smoothed_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy that ignores masked logits in both the softmax and the
    smoothing distribution.

    ``F.cross_entropy(label_smoothing=...)`` averages the smoothed term over
    *all* classes, so combining it with ``-inf`` masked logits produces ``inf``.
    This helper masks invalid classes out of the softmax denominator and the
    smoothing average, keeping label smoothing well-defined when same-text
    duplicates are removed from the negative set.
    """
    smoothing = float(label_smoothing)
    if mask is None:
        return F.cross_entropy(logits, target, label_smoothing=smoothing)
    masked_logits = logits.masked_fill(mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=1)
    n = logits.shape[0]
    nll = -log_probs[torch.arange(n, device=logits.device), target]
    if smoothing <= 0.0:
        return nll.mean()
    valid = ~mask
    num_valid = valid.sum(dim=1).clamp(min=1).to(log_probs.dtype)
    safe_log_probs = log_probs.masked_fill(mask, 0.0)
    smoothed = -(safe_log_probs * valid.to(log_probs.dtype)).sum(dim=1) / num_valid
    return ((1.0 - smoothing) * nll + smoothing * smoothed).mean()


def symmetric_contrastive_loss(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    text_labels: torch.Tensor | None = None,
    logit_scale_max: float = 100.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Symmetric InfoNCE with optional false-negative masking and label smoothing.

    The Olympus rationale vocabulary is small (~244 unique strings) and highly
    imbalanced, so every contrastive batch contains many same-text samples.
    When ``text_labels`` is provided we mask non-diagonal same-text entries out
    of the softmax denominator, preventing those duplicates from being treated
    as negatives. ``label_smoothing`` softens the diagonal target distribution
    while respecting the mask so it stays finite.
    """
    scale = logit_scale.exp().clamp(max=float(logit_scale_max))
    logits = torch.matmul(image_embeddings, text_embeddings.T) * scale
    n = image_embeddings.shape[0]
    target = torch.arange(n, device=image_embeddings.device)
    mask: torch.Tensor | None = None
    if text_labels is not None:
        same_text = text_labels.unsqueeze(0) == text_labels.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=image_embeddings.device)
        false_negative_mask = same_text & ~eye
        if bool(false_negative_mask.any()):
            mask = false_negative_mask
    forward_loss = _masked_smoothed_cross_entropy(
        logits, target, mask=mask, label_smoothing=label_smoothing
    )
    reverse_loss = _masked_smoothed_cross_entropy(
        logits.T, target, mask=mask.T if mask is not None else None, label_smoothing=label_smoothing
    )
    return 0.5 * (forward_loss + reverse_loss)


def prototype_classification_loss(
    image_embeddings: torch.Tensor,
    prototype_embeddings: torch.Tensor,
    class_ids: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    logit_scale_max: float = 100.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy of images against the full bank of class-prototype text embeddings.

    With only ~244 unique rationales in Stage B, treating each unique text as
    a class prototype removes batch-size sensitivity: every image is scored
    against all 244 candidate texts, which is a much stronger learning signal
    than a 256-way contrastive subset.
    """
    scale = logit_scale.exp().clamp(max=float(logit_scale_max))
    logits = image_embeddings @ prototype_embeddings.T * scale
    return F.cross_entropy(logits, class_ids, label_smoothing=float(label_smoothing))


def build_projector(
    in_dim: int,
    embed_dim: int,
    *,
    kind: str = "linear",
    hidden_dim: int = 768,
    depth: int = 2,
    dropout: float = 0.0,
) -> nn.Module:
    """Construct an image/text projection head.

    ``kind="linear"`` reproduces the original B0 baseline. ``kind="mlp"`` adds
    capacity via an LayerNorm/GELU MLP with ``depth`` total ``Linear`` layers.
    """
    if kind == "linear":
        return nn.Linear(int(in_dim), int(embed_dim))
    if kind != "mlp":
        raise ValueError(f"Unknown projector kind '{kind}'. Expected 'linear' or 'mlp'.")
    if int(depth) < 1:
        raise ValueError("projector depth must be >= 1.")
    layers: list[nn.Module] = []
    prev = int(in_dim)
    hidden = int(hidden_dim)
    for _ in range(int(depth) - 1):
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.LayerNorm(hidden))
        layers.append(nn.GELU())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        prev = hidden
    layers.append(nn.Linear(prev, int(embed_dim)))
    return nn.Sequential(*layers)


class ClassBalancedSampler(Sampler[int]):
    """Class-stratified sampler that interleaves samples across class buckets.

    Each output cycle visits every class still holding remaining samples, so
    contiguous windows of length ``num_classes`` contain at most one sample per
    class. With ``batch_size >= num_classes`` every batch covers every class,
    eliminating the false-negative problem at the source.
    """

    def __init__(
        self,
        class_ids: list[int],
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if not class_ids:
            raise ValueError("ClassBalancedSampler requires at least one class id.")
        groups: dict[int, list[int]] = {}
        for index, class_id in enumerate(class_ids):
            groups.setdefault(int(class_id), []).append(int(index))
        self._groups: list[list[int]] = list(groups.values())
        self._total: int = sum(len(group) for group in self._groups)
        self.generator = generator

    def __iter__(self):
        rng = self.generator if self.generator is not None else torch.Generator()
        shuffled: list[list[int]] = []
        for group in self._groups:
            permutation = torch.randperm(len(group), generator=rng).tolist()
            shuffled.append([group[index] for index in permutation])
        cursors = [0] * len(shuffled)
        out: list[int] = []
        while len(out) < self._total:
            class_order = torch.randperm(len(shuffled), generator=rng).tolist()
            for class_index in class_order:
                if cursors[class_index] < len(shuffled[class_index]):
                    out.append(shuffled[class_index][cursors[class_index]])
                    cursors[class_index] += 1
                if len(out) >= self._total:
                    break
        return iter(out)

    def __len__(self) -> int:
        return self._total


class ParameterEMA:
    """Exponential moving average over an ``nn.Module``'s parameters and buffers.

    The aligner is small enough to keep a full shadow copy on-device. Floating
    point tensors are EMA-tracked; integer/bool buffers (e.g. counts) are
    copied verbatim so things like ``num_batches_tracked`` keep working.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError("EMA decay must lie in the open interval (0, 1).")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, tensor in model.state_dict().items():
            shadow = self.shadow[name]
            if tensor.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(tensor.detach().to(shadow.dtype), alpha=1.0 - self.decay)
            else:
                shadow.copy_(tensor.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup, strict=True)


@dataclass(eq=False)
class AlignmentModel(nn.Module):
    """Projection heads that map image/text into a shared embedding space."""

    image_projector: nn.Module
    text_projector: nn.Module
    logit_scale: nn.Parameter

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        embed_dim: int,
        *,
        projector_type: str = "linear",
        projector_hidden_dim: int = 768,
        projector_depth: int = 2,
        projector_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_projector = build_projector(
            image_dim,
            embed_dim,
            kind=projector_type,
            hidden_dim=projector_hidden_dim,
            depth=projector_depth,
            dropout=projector_dropout,
        )
        self.text_projector = build_projector(
            text_dim,
            embed_dim,
            kind=projector_type,
            hidden_dim=projector_hidden_dim,
            depth=projector_depth,
            dropout=projector_dropout,
        )
        self.logit_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1 / 0.07)))))
        self.aligner_config: dict[str, Any] = {
            "image_dim": int(image_dim),
            "text_dim": int(text_dim),
            "embed_dim": int(embed_dim),
            "projector_type": str(projector_type),
            "projector_hidden_dim": int(projector_hidden_dim),
            "projector_depth": int(projector_depth),
            "projector_dropout": float(projector_dropout),
        }

    def project_text(self, text_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projector(text_features), dim=1)

    def project_image(self, image_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_projector(image_features), dim=1)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.project_image(image_features), self.project_text(text_features)


class FrozenTextEmbeddingCache:
    """Cache frozen text embeddings so repeated rationales aren't re-encoded."""

    def __init__(self, *, text_encoder: T5Encoder, device: torch.device, batch_size: int = 128) -> None:
        self.text_encoder = text_encoder
        self.device = device
        self.batch_size = max(int(batch_size), 1)
        self._cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _unique_ordered(texts: list[str]) -> list[str]:
        return list(dict.fromkeys(str(text) for text in texts))

    def __len__(self) -> int:
        return len(self._cache)

    @torch.no_grad()
    def _encode_missing(self, texts: list[str]) -> None:
        missing = [text for text in self._unique_ordered(texts) if text not in self._cache]
        if not missing:
            return
        embeddings, _ = self.text_encoder(missing, device=self.device)
        for text, embedding in zip(missing, embeddings.detach().cpu(), strict=False):
            self._cache[text] = embedding

    @torch.no_grad()
    def warmup(
        self,
        texts: list[str],
        *,
        progress_path: pathlib.Path | None = None,
        out_dir: pathlib.Path | None = None,
        progress_log_interval: int = 10,
    ) -> int:
        unique_texts = self._unique_ordered(texts)
        total = len(unique_texts)
        if total == 0:
            return 0

        progress_interval = max(int(progress_log_interval), 1)
        for start in range(0, total, self.batch_size):
            chunk = unique_texts[start : start + self.batch_size]
            self._encode_missing(chunk)
            completed = min(start + len(chunk), total)
            chunk_index = start // self.batch_size + 1
            should_report = chunk_index % progress_interval == 0 or completed == total
            if should_report:
                payload = {
                    "status": "running",
                    "phase": "text_cache_warmup",
                    "cached_text_embeddings": len(self._cache),
                    "target_text_embeddings": total,
                    "out_dir": str(out_dir) if out_dir is not None else None,
                }
                if progress_path is not None:
                    save_training_progress(payload, progress_path)
                print(
                    f"[align] text-cache {completed}/{total} unique rationales encoded",
                    flush=True,
                )
        return total

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        self._encode_missing(texts)
        return torch.stack([self._cache[str(text)] for text in texts], dim=0).to(
            device=self.device,
            non_blocking=True,
        )


class FrozenImageFeatureDataset(Dataset):
    """Dataset wrapper backed by cached frozen image features."""

    def __init__(
        self,
        *,
        image_features: torch.Tensor,
        texts: list[str],
        metadata_rows: list[dict[str, Any]],
    ) -> None:
        if image_features.ndim != 2:
            raise ValueError("image_features must have shape (N, D).")
        if image_features.shape[0] != len(texts) or image_features.shape[0] != len(metadata_rows):
            raise ValueError("image_features, texts, and metadata_rows must have matching lengths.")
        self.image_features = image_features.contiguous()
        self.texts = list(texts)
        self.metadata_rows = [dict(row) for row in metadata_rows]

    def __len__(self) -> int:
        return int(self.image_features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image_features": self.image_features[index],
            "rationale_raw": self.texts[index],
            "metadata": dict(self.metadata_rows[index]),
        }


@torch.no_grad()
def build_frozen_image_feature_dataset(
    dataset: Dataset | list[dict[str, Any]],
    *,
    mae_encoder: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    progress_path: pathlib.Path | None = None,
    out_dir: pathlib.Path | None = None,
    progress_log_interval: int = 10,
    phase: str = "image_cache_warmup",
    image_pool: str = "cls",
) -> FrozenImageFeatureDataset:
    """Precompute frozen MAE image features once and train on the cached vectors."""
    dataloader = build_alignment_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    non_blocking = bool(pin_memory and device.type == "cuda")
    total_steps = max(len(dataloader), 1)
    feature_batches: list[torch.Tensor] = []
    text_rows: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    start_time = time.time()
    report_interval = max(int(progress_log_interval), 1)

    for step, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device, non_blocking=non_blocking)
        image_features = encode_image_with_satmae_encoder(mae_encoder, images, pool=image_pool)
        feature_batches.append(image_features.detach().cpu())
        text_rows.extend(batch["text"])
        metadata_rows.extend(batch["metadata"])

        should_report = (step % report_interval == 0) or (step == total_steps)
        if should_report:
            payload = {
                "status": "running",
                "phase": phase,
                "cached_samples": len(text_rows),
                "target_samples": len(dataset),
                "cache_steps": int(step),
                "cache_total_steps": int(total_steps),
                "cache_elapsed_sec": time.time() - start_time,
                "out_dir": str(out_dir) if out_dir is not None else None,
            }
            if progress_path is not None:
                save_training_progress(payload, progress_path)
            print(
                f"[align] {phase} samples={len(text_rows)}/{len(dataset)} "
                f"steps={step}/{total_steps}",
                flush=True,
            )

    return FrozenImageFeatureDataset(
        image_features=torch.cat(feature_batches, dim=0),
        texts=text_rows,
        metadata_rows=metadata_rows,
    )


@torch.no_grad()
def build_alignment_embeddings(
    *,
    dataloader: DataLoader,
    mae_encoder: nn.Module,
    text_encoder: T5Encoder,
    aligner: nn.Module,
    device: torch.device,
    text_cache: FrozenTextEmbeddingCache | None = None,
    non_blocking: bool = False,
    image_pool: str = "cls",
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[str]]:
    """Build aligned image/text embeddings for retrieval evaluation."""
    image_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    text_rows: list[str] = []

    for batch in dataloader:
        texts = batch["text"]
        if "image_features" in batch:
            image_features = batch["image_features"].to(device, non_blocking=non_blocking)
        else:
            images = batch["image"].to(device, non_blocking=non_blocking)
            image_features = encode_image_with_satmae_encoder(mae_encoder, images, pool=image_pool)

        if text_cache is not None:
            text_features = text_cache.encode(texts)
        else:
            text_features, _ = text_encoder(texts, device=device)

        image_emb, text_emb = aligner(image_features, text_features)
        image_embeddings.append(image_emb.detach().cpu())
        text_embeddings.append(text_emb.detach().cpu())
        metadata_rows.extend(batch["metadata"])
        text_rows.extend(texts)

    return (
        torch.cat(image_embeddings, dim=0),
        torch.cat(text_embeddings, dim=0),
        metadata_rows,
        text_rows,
    )


@torch.no_grad()
def evaluate_alignment_retrieval(
    *,
    dataloader: DataLoader,
    mae_encoder: nn.Module,
    text_encoder: T5Encoder,
    aligner: nn.Module,
    device: torch.device,
    text_cache: FrozenTextEmbeddingCache | None = None,
    non_blocking: bool = False,
    image_pool: str = "cls",
) -> dict[str, float]:
    """Evaluate retrieval quality for the current alignment model."""
    image_emb, text_emb, _, text_rows = build_alignment_embeddings(
        dataloader=dataloader,
        mae_encoder=mae_encoder,
        text_encoder=text_encoder,
        aligner=aligner,
        device=device,
        text_cache=text_cache,
        non_blocking=non_blocking,
        image_pool=image_pool,
    )
    return compute_retrieval_metrics(image_emb, text_emb, text_rows)


def _extract_cache_candidate_texts(dataset: MarsCLIPPatchDataset) -> list[str]:
    patch_records = getattr(dataset, "patch_records", None)
    if patch_records is None or "rationale_raw" not in patch_records.columns:
        return []
    return patch_records["rationale_raw"].fillna("").astype(str).tolist()


def _save_alignment_checkpoint(
    path: pathlib.Path,
    *,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    aligner: AlignmentModel,
    text_encoder: T5Encoder,
    mae_encoder: nn.Module,
    args: argparse.Namespace,
    image_dim: int,
    text_dim: int,
    wandb_logger: WandbLogger | None,
    epoch: int,
    ema: ParameterEMA | None = None,
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "config": config,
            "history": history,
            "aligner_state": aligner.state_dict(),
            "aligner_config": dict(getattr(aligner, "aligner_config", {})),
            "ema_state": ema.state_dict() if ema is not None else None,
            "ema_decay": float(ema.decay) if ema is not None else None,
            "text_encoder_state": text_encoder.state_dict() if args.train_text_encoder else None,
            "mae_encoder_state": mae_encoder.state_dict() if not args.freeze_mae else None,
            "text_model_name": args.text_model,
            "mae_model_name": args.mae_model,
            "mae_checkpoint": str(args.mae_checkpoint),
            "image_dim": image_dim,
            "text_dim": text_dim,
            "embed_dim": int(args.embed_dim),
            "image_pool": str(getattr(args, "image_pool", "cls")),
            "loss_type": str(getattr(args, "loss_type", "infonce")),
            "projector_type": str(getattr(args, "projector_type", "linear")),
            "wandb_mode": wandb_logger.mode if wandb_logger is not None else "disabled",
            "wandb_project": wandb_logger.project if wandb_logger is not None else None,
            "wandb_entity": wandb_logger.entity if wandb_logger is not None else None,
            "wandb_run_name": wandb_logger.run_name if wandb_logger is not None else None,
            "wandb_run_id": wandb_logger.run_id if wandb_logger is not None else None,
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Align Mars text with MAE image embeddings for Stage B.")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--bbox", type=float, nargs=4, default=(-136.0, 12.0, -124.0, 24.0))
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size-px", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--patch-records-path", type=pathlib.Path, default=None)
    parser.add_argument("--split-manifest", type=pathlib.Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--mae-model", type=str, default="mae_vit_base_patch16")
    parser.add_argument("--mae-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--freeze-mae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-model", type=str, default="google-t5/t5-base")
    parser.add_argument("--text-max-length", type=int, default=128)
    parser.add_argument("--train-text-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-text-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-cache-batch-size", type=int, default=128)
    parser.add_argument("--cache-image-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-cache-batch-size", type=int, default=64)
    parser.add_argument("--val-max-patches", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--false-negative-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logit-scale-max", type=float, default=100.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--projector-type", type=str, default="linear", choices=("linear", "mlp"))
    parser.add_argument("--projector-hidden-dim", type=int, default=768)
    parser.add_argument("--projector-depth", type=int, default=2)
    parser.add_argument("--projector-dropout", type=float, default=0.0)
    parser.add_argument("--image-pool", type=str, default="cls", choices=IMAGE_POOL_CHOICES)
    parser.add_argument("--loss-type", type=str, default="infonce", choices=("infonce", "prototype"))
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--progress-log-interval", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_STAGE_B_OUT_ROOT)
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=("disabled", "offline", "online"))
    parser.add_argument("--wandb-project", type=str, default="MarsRecon")
    parser.add_argument("--wandb-entity", type=str, default="akshayn3-auvsl")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    run_name = args.run_name or _default_run_name(mae_checkpoint=args.mae_checkpoint)
    out_dir = resolve_run_output_dir(
        out_dir=args.out_dir,
        out_root=args.out_root,
        run_name=run_name,
        model_name=args.mae_model,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    startup_progress_path = out_dir / "startup_progress.json"
    progress_path = out_dir / "progress.json"
    error_path = out_dir / "error_traceback.txt"
    history_path = out_dir / "history.json"
    summary_path = out_dir / "summary.json"
    run_config_path = out_dir / "run_config.json"
    checkpoints_dir = out_dir / "checkpoints"
    checkpoint_path = checkpoints_dir / "checkpoint.pt"
    best_checkpoint_path = checkpoints_dir / "best_checkpoint.pt"
    progress_interval = max(int(args.progress_log_interval), 1)
    non_blocking = bool(args.pin_memory and device.type == "cuda")
    wandb_logger: WandbLogger | None = None
    termination_monitor = TerminationMonitor()
    termination_monitor.install()
    history: list[dict[str, Any]] = []
    aligner: AlignmentModel | None = None
    text_encoder: T5Encoder | None = None
    mae_encoder: nn.Module | None = None
    config: dict[str, Any] | None = None
    image_dim: int | None = None
    text_dim: int | None = None
    ema: ParameterEMA | None = None

    save_training_progress(
        {
            "status": "running",
            "phase": "startup",
            "message": "Initializing Stage-B text/MAE alignment run.",
            "out_dir": str(out_dir),
        },
        startup_progress_path,
    )

    base_wandb_config = _json_ready(
        {
            **vars(args),
            "device": str(device),
            "run_name": run_name,
            "out_dir": str(out_dir),
            "out_root": str(args.out_root),
            "checkpoints_dir": str(checkpoints_dir),
            "history_path": str(history_path),
            "progress_path": str(progress_path),
            "startup_progress_path": str(startup_progress_path),
            "summary_path": str(summary_path),
            "mae_checkpoint": str(args.mae_checkpoint.expanduser().resolve()),
            "patch_records_path": str(args.patch_records_path) if args.patch_records_path is not None else None,
            "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
        }
    )
    wandb_run_name = args.wandb_run_name or run_name
    wandb_logger = init_wandb_logger(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=wandb_run_name,
        out_dir=out_dir,
        log_dir=args.wandb_dir,
        config=base_wandb_config,
    )

    try:
        termination_monitor.raise_if_requested()
        patch_records = None
        if args.patch_records_path is not None and args.patch_records_path.exists():
            patch_records = load_patch_records(args.patch_records_path)
        if args.split_manifest is not None:
            if patch_records is None:
                raise ValueError("--split-manifest requires --patch-records-path so patch ids match the split file.")
            patch_records = attach_split_manifest(
                patch_records,
                load_split_manifest(args.split_manifest),
            )

        train_patch_records = patch_records
        val_patch_records = None
        if patch_records is not None and "stage_b_split" in patch_records.columns:
            train_patch_records = filter_patch_records_by_split(patch_records, split_name="train")
            val_patch_records = filter_patch_records_by_split(patch_records, split_name="val")

        if train_patch_records is not None:
            train_patch_records = _limit_patch_records(
                train_patch_records,
                max_patches=args.max_patches,
                seed=args.seed,
            )
        train_dataset_max_patches = None if train_patch_records is not None else args.max_patches

        if val_patch_records is not None:
            val_patch_records = select_balanced_patch_records(
                val_patch_records,
                max_patches=args.val_max_patches,
                seed=args.seed,
            )

        dataset = MarsCLIPPatchDataset(
            root=args.root,
            bbox=tuple(args.bbox),
            patch_size=args.patch_size_deg,
            image_size=args.image_size,
            max_patches=train_dataset_max_patches,
            color_only=True,
            patch_records=train_patch_records,
        )
        val_dataset = None
        if val_patch_records is not None:
            val_dataset = MarsCLIPPatchDataset(
                geo_dataset=dataset.geo_dataset,
                patch_size=args.patch_size_deg,
                image_size=args.image_size,
                color_only=True,
                observation_metadata=dataset.observation_metadata.reset_index(drop=True),
                patch_records=val_patch_records,
            )
        save_training_progress(
            {
                "status": "running",
                "phase": "dataset_ready",
                "message": "Stage-B alignment dataset ready.",
                "dataset_size": len(dataset),
                "train_dataset_size": len(dataset),
                "val_dataset_size": len(val_dataset) if val_dataset is not None else 0,
                "out_dir": str(out_dir),
            },
            startup_progress_path,
        )

        mae_encoder = load_satmae_encoder(
            model_name=args.mae_model,
            checkpoint_path=args.mae_checkpoint,
            image_size=args.image_size,
            patch_size_px=args.patch_size_px,
            freeze=args.freeze_mae,
        ).to(device)
        text_encoder = T5Encoder(
            model_name=args.text_model,
            max_length=args.text_max_length,
            trainable=args.train_text_encoder,
        ).to(device)

        text_cache = None
        if args.cache_text_embeddings and not args.train_text_encoder:
            text_cache = FrozenTextEmbeddingCache(
                text_encoder=text_encoder,
                device=device,
                batch_size=args.text_cache_batch_size,
            )
            candidate_texts = _extract_cache_candidate_texts(dataset)
            if val_dataset is not None:
                candidate_texts.extend(_extract_cache_candidate_texts(val_dataset))
            if candidate_texts:
                cached_count = text_cache.warmup(
                    candidate_texts,
                    progress_path=startup_progress_path,
                    out_dir=out_dir,
                    progress_log_interval=max(progress_interval // 2, 1),
                )
                save_training_progress(
                    {
                        "status": "running",
                        "phase": "text_cache_ready",
                        "cached_text_embeddings": cached_count,
                        "out_dir": str(out_dir),
                    },
                    startup_progress_path,
                )

        image_pool = str(args.image_pool)
        train_source_dataset: Dataset | list[dict[str, Any]] = dataset
        val_source_dataset: Dataset | list[dict[str, Any]] | None = val_dataset
        if args.cache_image_embeddings and args.freeze_mae:
            train_source_dataset = build_frozen_image_feature_dataset(
                dataset,
                mae_encoder=mae_encoder,
                device=device,
                batch_size=args.image_cache_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                progress_path=startup_progress_path,
                out_dir=out_dir,
                progress_log_interval=max(progress_interval // 2, 1),
                phase="train_image_cache_warmup",
                image_pool=image_pool,
            )
            if val_dataset is not None:
                val_source_dataset = build_frozen_image_feature_dataset(
                    val_dataset,
                    mae_encoder=mae_encoder,
                    device=device,
                    batch_size=args.image_cache_batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    prefetch_factor=args.prefetch_factor,
                    persistent_workers=args.persistent_workers,
                    progress_path=startup_progress_path,
                    out_dir=out_dir,
                    progress_log_interval=max(progress_interval // 2, 1),
                    phase="val_image_cache_warmup",
                    image_pool=image_pool,
                )
            save_training_progress(
                {
                    "status": "running",
                    "phase": "image_cache_ready",
                    "cached_train_image_embeddings": len(train_source_dataset),
                    "cached_val_image_embeddings": len(val_source_dataset) if val_source_dataset is not None else 0,
                    "out_dir": str(out_dir),
                },
                startup_progress_path,
            )

        using_cached_train_features = isinstance(train_source_dataset, FrozenImageFeatureDataset)
        using_cached_val_features = isinstance(val_source_dataset, FrozenImageFeatureDataset)
        train_loader_num_workers = 0 if using_cached_train_features else int(args.num_workers)
        train_loader_pin_memory = False if using_cached_train_features else bool(args.pin_memory)
        train_loader_prefetch_factor = None if using_cached_train_features else args.prefetch_factor
        train_loader_persistent_workers = False if using_cached_train_features else bool(args.persistent_workers)

        train_texts_for_labels: list[str]
        if isinstance(train_source_dataset, FrozenImageFeatureDataset):
            train_texts_for_labels = list(train_source_dataset.texts)
        else:
            train_texts_for_labels = list(
                dataset.patch_records["rationale_raw"].fillna("").astype(str).tolist()
            )
        global_text_label_map: dict[str, int] = {}
        for text in train_texts_for_labels:
            key = str(text)
            if key not in global_text_label_map:
                global_text_label_map[key] = len(global_text_label_map)
        unique_texts_in_order: list[str] = sorted(
            global_text_label_map, key=lambda value: global_text_label_map[value]
        )
        global_class_ids: list[int] = [
            global_text_label_map[str(text)] for text in train_texts_for_labels
        ]

        sampler: Sampler[int] | None = None
        loader_shuffle = True
        if bool(args.balanced_sampler):
            sampler = ClassBalancedSampler(
                global_class_ids,
                generator=torch.Generator().manual_seed(int(args.seed)),
            )
            loader_shuffle = False

        if sampler is not None:
            loader_kwargs: dict[str, Any] = {
                "batch_size": args.batch_size,
                "sampler": sampler,
                "collate_fn": collate_patch_text,
                "num_workers": train_loader_num_workers,
                "pin_memory": train_loader_pin_memory,
                "drop_last": False,
            }
            if train_loader_num_workers > 0:
                loader_kwargs["persistent_workers"] = train_loader_persistent_workers
                if train_loader_prefetch_factor is not None:
                    loader_kwargs["prefetch_factor"] = train_loader_prefetch_factor
            dataloader = DataLoader(train_source_dataset, **loader_kwargs)
        else:
            dataloader = build_alignment_dataloader(
                train_source_dataset,
                batch_size=args.batch_size,
                shuffle=loader_shuffle,
                generator=torch.Generator().manual_seed(args.seed),
                num_workers=train_loader_num_workers,
                pin_memory=train_loader_pin_memory,
                prefetch_factor=train_loader_prefetch_factor,
                persistent_workers=train_loader_persistent_workers,
                drop_last=False,
            )
        val_dataloader = None
        if val_source_dataset is not None and args.val_every > 0:
            val_loader_num_workers = 0 if using_cached_val_features else int(args.num_workers)
            val_loader_pin_memory = False if using_cached_val_features else bool(args.pin_memory)
            val_loader_prefetch_factor = None if using_cached_val_features else args.prefetch_factor
            val_loader_persistent_workers = False if using_cached_val_features else bool(args.persistent_workers)
            val_dataloader = build_alignment_dataloader(
                val_source_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=val_loader_num_workers,
                pin_memory=val_loader_pin_memory,
                prefetch_factor=val_loader_prefetch_factor,
                persistent_workers=val_loader_persistent_workers,
                drop_last=False,
            )
        save_training_progress(
            {
                "status": "running",
                "phase": "dataloaders_ready",
                "message": "Stage-B alignment dataloader initialized.",
                "steps_per_epoch": len(dataloader),
                "val_steps": len(val_dataloader) if val_dataloader is not None else 0,
                "num_workers": int(train_loader_num_workers),
                "pin_memory": bool(train_loader_pin_memory),
                "prefetch_factor": train_loader_prefetch_factor,
                "persistent_workers": bool(train_loader_persistent_workers),
                "cached_train_loader": bool(using_cached_train_features),
                "cached_val_loader": bool(using_cached_val_features),
                "out_dir": str(out_dir),
            },
            startup_progress_path,
        )

        with torch.no_grad():
            warmup_batch = next(iter(dataloader))
            if "image_features" in warmup_batch:
                image_dim = int(warmup_batch["image_features"].shape[-1])
            else:
                warmup_images = warmup_batch["image"].to(device, non_blocking=non_blocking)
                image_dim = int(
                    encode_image_with_satmae_encoder(
                        mae_encoder, warmup_images, pool=image_pool
                    ).shape[-1]
                )
            if text_cache is not None:
                warmup_text_features = text_cache.encode(warmup_batch["text"])
            else:
                warmup_text_features, _ = text_encoder(warmup_batch["text"], device=device)
            text_dim = int(warmup_text_features.shape[-1])

        aligner = AlignmentModel(
            image_dim=image_dim,
            text_dim=text_dim,
            embed_dim=args.embed_dim,
            projector_type=str(args.projector_type),
            projector_hidden_dim=int(args.projector_hidden_dim),
            projector_depth=int(args.projector_depth),
            projector_dropout=float(args.projector_dropout),
        ).to(device)

        if float(args.ema_decay) > 0.0:
            ema = ParameterEMA(aligner, decay=float(args.ema_decay))

        loss_type = str(args.loss_type)
        prototype_text_features: torch.Tensor | None = None
        global_class_id_tensor: torch.Tensor | None = None
        if loss_type == "prototype":
            if text_cache is None:
                raise ValueError("--loss-type prototype requires --cache-text-embeddings.")
            prototype_text_features = text_cache.encode(unique_texts_in_order).detach()
            global_class_id_tensor = torch.tensor(
                global_class_ids, device=device, dtype=torch.long
            )

        parameters: list[nn.Parameter] = list(aligner.parameters())
        if args.train_text_encoder:
            parameters += [parameter for parameter in text_encoder.parameters() if parameter.requires_grad]
        if not args.freeze_mae:
            parameters += [parameter for parameter in mae_encoder.parameters() if parameter.requires_grad]

        optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        trainable_parameters, total_parameters = count_trainable_parameters(aligner)
        if args.train_text_encoder:
            trainable_parameters += sum(parameter.numel() for parameter in text_encoder.parameters() if parameter.requires_grad)
            total_parameters += sum(parameter.numel() for parameter in text_encoder.parameters())
        else:
            total_parameters += sum(parameter.numel() for parameter in text_encoder.parameters())
        if not args.freeze_mae:
            trainable_parameters += sum(parameter.numel() for parameter in mae_encoder.parameters() if parameter.requires_grad)
        total_parameters += sum(parameter.numel() for parameter in mae_encoder.parameters())

        steps_per_epoch = max(len(dataloader), 1)
        val_steps = len(val_dataloader) if val_dataloader is not None else 0
        config = _json_ready(
            {
                **vars(args),
                "device": str(device),
                "run_name": run_name,
                "out_dir": str(out_dir),
                "out_root": str(args.out_root),
                "checkpoints_dir": str(checkpoints_dir),
                "history_path": str(history_path),
                "progress_path": str(progress_path),
                "startup_progress_path": str(startup_progress_path),
                "summary_path": str(summary_path),
                "mae_checkpoint": str(args.mae_checkpoint.expanduser().resolve()),
                "patch_records_path": str(args.patch_records_path) if args.patch_records_path is not None else None,
                "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
                "wandb_run_name": wandb_run_name,
                "dataset_size": len(dataset),
                "train_dataset_size": len(dataset),
                "val_dataset_size": len(val_dataset) if val_dataset is not None else 0,
                "steps_per_epoch": steps_per_epoch,
                "val_steps": val_steps,
                "image_dim": image_dim,
                "text_dim": text_dim,
                "cached_text_embeddings": len(text_cache) if text_cache is not None else None,
                "cached_image_embeddings": bool(args.cache_image_embeddings and args.freeze_mae),
                "train_unique_texts": int(dataset.patch_records["rationale_raw"].astype(str).nunique()),
                "val_unique_texts": (
                    int(val_dataset.patch_records["rationale_raw"].astype(str).nunique())
                    if val_dataset is not None
                    else 0
                ),
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            }
        )
        run_config_path.write_text(json.dumps(config, indent=2))
        if wandb_logger is not None:
            wandb_logger.update_config(config)

        save_training_progress(
            {
                "status": "running",
                "phase": "training_ready",
                "message": "Stage-B alignment model initialized.",
                "out_dir": str(out_dir),
            },
            startup_progress_path,
        )

        global_step = 0
        lowest_train_loss: float | None = None
        lowest_train_loss_epoch = 0
        best_checkpoint_metric_name = "val/alignment_score" if val_dataloader is not None else "loss"
        best_checkpoint_metric_value: float | None = None
        best_checkpoint_epoch = 0

        amp_enabled = bool(args.use_amp and device.type == "cuda")
        amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
        amp_needs_scaler = amp_enabled and amp_dtype == torch.float16
        amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_needs_scaler)
        logit_scale_max_log = math.log(float(args.logit_scale_max))

        for epoch in range(args.epochs):
            termination_monitor.raise_if_requested()
            aligner.train()
            if args.train_text_encoder:
                text_encoder.train()
            else:
                text_encoder.eval()
            if args.freeze_mae:
                mae_encoder.eval()
            else:
                mae_encoder.train()

            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_steps = 0
            current_lr = float(optimizer.param_groups[0]["lr"])
            save_training_progress(
                {
                    "status": "running",
                    "phase": "training",
                    "current_epoch": int(epoch),
                    "target_epoch": int(args.epochs),
                    "out_dir": str(out_dir),
                },
                progress_path,
            )

            for step, batch in enumerate(dataloader):
                termination_monitor.raise_if_requested()
                step_in_epoch = step + 1
                texts = batch["text"]
                progress = float(epoch) + (float(step) / max(float(steps_per_epoch), 1.0))
                current_lr = adjust_learning_rate(
                    optimizer,
                    progress=progress,
                    lr=args.learning_rate,
                    min_lr=args.min_lr,
                    warmup_epochs=args.warmup_epochs,
                    epochs=args.epochs,
                )

                optimizer.zero_grad(set_to_none=True)

                if "image_features" in batch:
                    image_features = batch["image_features"].to(device, non_blocking=non_blocking)
                else:
                    images = batch["image"].to(device, non_blocking=non_blocking)
                    if args.freeze_mae:
                        with torch.no_grad():
                            image_features = encode_image_with_satmae_encoder(mae_encoder, images)
                    else:
                        image_features = encode_image_with_satmae_encoder(mae_encoder, images)

                batch_class_ids = torch.tensor(
                    [global_text_label_map[str(text)] for text in texts],
                    device=device,
                    dtype=torch.long,
                )

                if loss_type == "prototype":
                    text_features = None
                elif args.train_text_encoder:
                    text_features, _ = text_encoder(texts, device=device)
                elif text_cache is not None:
                    text_features = text_cache.encode(texts)
                else:
                    with torch.no_grad():
                        text_features, _ = text_encoder(texts, device=device)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    image_embeddings = aligner.project_image(image_features)
                    if loss_type == "prototype":
                        prototype_embeddings = aligner.project_text(prototype_text_features)
                        loss = prototype_classification_loss(
                            image_embeddings,
                            prototype_embeddings,
                            batch_class_ids,
                            aligner.logit_scale,
                            logit_scale_max=float(args.logit_scale_max),
                            label_smoothing=float(args.label_smoothing),
                        )
                    else:
                        text_embeddings = aligner.project_text(text_features)
                        loss = symmetric_contrastive_loss(
                            image_embeddings,
                            text_embeddings,
                            aligner.logit_scale,
                            text_labels=batch_class_ids if args.false_negative_mask else None,
                            logit_scale_max=float(args.logit_scale_max),
                            label_smoothing=float(args.label_smoothing),
                        )

                if amp_needs_scaler:
                    amp_scaler.scale(loss).backward()
                    if float(args.grad_clip_norm) > 0.0:
                        amp_scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(parameters, float(args.grad_clip_norm))
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    loss.backward()
                    if float(args.grad_clip_norm) > 0.0:
                        nn.utils.clip_grad_norm_(parameters, float(args.grad_clip_norm))
                    optimizer.step()

                with torch.no_grad():
                    aligner.logit_scale.clamp_(max=logit_scale_max_log)

                if ema is not None:
                    ema.update(aligner)

                global_step += 1
                epoch_steps += 1
                loss_value = float(loss.detach().cpu())
                epoch_loss += loss_value
                should_report = (step_in_epoch % progress_interval == 0 or step_in_epoch == steps_per_epoch)
                if should_report:
                    running_mean_loss = epoch_loss / max(epoch_steps, 1)
                    elapsed = time.time() - epoch_start
                    progress_payload = {
                        "status": "running",
                        "phase": "training",
                        "current_epoch": int(epoch + 1),
                        "target_epoch": int(args.epochs),
                        "step_in_epoch": int(step_in_epoch),
                        "steps_in_epoch": int(steps_per_epoch),
                        "global_step": int(global_step),
                        "latest_loss": loss_value,
                        "running_mean_loss": running_mean_loss,
                        "latest_lr": current_lr,
                        "epoch_elapsed_sec": elapsed,
                        "cached_text_embeddings": len(text_cache) if text_cache is not None else None,
                        "out_dir": str(out_dir),
                    }
                    save_training_progress(progress_payload, progress_path)
                    print(
                        f"[align] epoch={epoch + 1}/{args.epochs} "
                        f"step={step_in_epoch}/{steps_per_epoch} "
                        f"loss={loss_value:.6f} "
                        f"mean={running_mean_loss:.6f} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    if wandb_logger is not None:
                        wandb_logger.log_metrics(
                            {
                                "train/loss": loss_value,
                                "train/running_mean_loss": running_mean_loss,
                                "train/logit_scale": float(aligner.logit_scale.exp().detach().cpu().item()),
                                "train/lr": current_lr,
                                "train/step_in_epoch": float(step_in_epoch),
                                "train/steps_in_epoch": float(steps_per_epoch),
                                "train/global_step": float(global_step),
                                "train/epoch_progress": float(epoch)
                                + (float(step_in_epoch) / max(float(steps_per_epoch), 1.0)),
                                "train/cached_text_embeddings": float(len(text_cache)) if text_cache is not None else 0.0,
                            },
                            step=global_step,
                        )

            avg_loss = epoch_loss / max(epoch_steps, 1)
            epoch_duration = time.time() - epoch_start
            val_metrics: dict[str, float] = {}
            if val_dataloader is not None and (((epoch + 1) % max(args.val_every, 1)) == 0 or ((epoch + 1) == args.epochs)):
                aligner.eval()
                text_encoder.eval()
                mae_encoder.eval()
                val_start = time.time()
                ema_backup: dict[str, torch.Tensor] | None = None
                if ema is not None:
                    ema_backup = ema.apply_to(aligner)
                try:
                    raw_val_metrics = evaluate_alignment_retrieval(
                        dataloader=val_dataloader,
                        mae_encoder=mae_encoder,
                        text_encoder=text_encoder,
                        aligner=aligner,
                        device=device,
                        text_cache=text_cache if not args.train_text_encoder else None,
                        non_blocking=non_blocking,
                        image_pool=image_pool,
                    )
                finally:
                    if ema is not None and ema_backup is not None:
                        ema.restore(aligner, ema_backup)
                val_duration = time.time() - val_start
                val_metrics = {f"val/{key}": float(value) for key, value in raw_val_metrics.items()}
                val_metrics["val/alignment_score"] = compute_alignment_score(raw_val_metrics)
                val_metrics["val/duration_sec"] = float(val_duration)
                print(
                    f"[align] epoch={epoch + 1}/{args.epochs} "
                    f"val score={val_metrics['val/alignment_score']:.6f} "
                    f"i2t_r10={val_metrics['val/image_to_text_r10']:.4f} "
                    f"t2i_r10={val_metrics['val/text_to_image_r10']:.4f} "
                    f"duration={val_duration:.1f}s",
                    flush=True,
                )
                if wandb_logger is not None:
                    wandb_logger.log_metrics(val_metrics, step=global_step)

            epoch_record = {
                "epoch": float(epoch + 1),
                "loss": avg_loss,
                "steps": float(epoch_steps),
                "duration_sec": epoch_duration,
            }
            epoch_record.update(val_metrics)
            history.append(epoch_record)
            save_training_history(history, history_path)
            print(
                f"[align] epoch={epoch + 1}/{args.epochs} "
                f"complete loss={avg_loss:.6f} duration={epoch_duration:.1f}s",
                flush=True,
            )
            if wandb_logger is not None:
                wandb_logger.log_metrics(
                    {
                        "epoch/train_loss": avg_loss,
                        "epoch/train_steps": float(epoch_steps),
                        "epoch/steps_in_epoch": float(steps_per_epoch),
                        "epoch/global_step_end": float(global_step),
                        "epoch/duration_sec": epoch_duration,
                        **val_metrics,
                    },
                    step=global_step,
                )

            _save_alignment_checkpoint(
                checkpoint_path,
                config=config,
                history=history,
                aligner=aligner,
                text_encoder=text_encoder,
                mae_encoder=mae_encoder,
                args=args,
                image_dim=image_dim,
                text_dim=text_dim,
                wandb_logger=wandb_logger,
                epoch=epoch + 1,
                ema=ema,
            )
            if lowest_train_loss is None or avg_loss < lowest_train_loss:
                lowest_train_loss = avg_loss
                lowest_train_loss_epoch = epoch + 1

            current_checkpoint_metric_value: float | None = None
            if val_metrics:
                current_checkpoint_metric_value = float(val_metrics["val/alignment_score"])
            elif val_dataloader is None:
                current_checkpoint_metric_value = float(avg_loss)

            should_update_best_checkpoint = False
            if current_checkpoint_metric_value is not None:
                if val_dataloader is None:
                    should_update_best_checkpoint = (
                        best_checkpoint_metric_value is None
                        or current_checkpoint_metric_value < best_checkpoint_metric_value
                    )
                else:
                    should_update_best_checkpoint = (
                        best_checkpoint_metric_value is None
                        or current_checkpoint_metric_value > best_checkpoint_metric_value
                    )

            if should_update_best_checkpoint:
                best_checkpoint_metric_value = current_checkpoint_metric_value
                best_checkpoint_epoch = epoch + 1
                best_ema_backup: dict[str, torch.Tensor] | None = None
                if ema is not None:
                    best_ema_backup = ema.apply_to(aligner)
                try:
                    _save_alignment_checkpoint(
                        best_checkpoint_path,
                        config=config,
                        history=history,
                        aligner=aligner,
                        text_encoder=text_encoder,
                        mae_encoder=mae_encoder,
                        args=args,
                        image_dim=image_dim,
                        text_dim=text_dim,
                        wandb_logger=wandb_logger,
                        epoch=epoch + 1,
                        ema=ema,
                    )
                finally:
                    if ema is not None and best_ema_backup is not None:
                        ema.restore(aligner, best_ema_backup)
            if args.checkpoint_every > 0 and (((epoch + 1) % args.checkpoint_every == 0) or ((epoch + 1) == args.epochs)):
                _save_alignment_checkpoint(
                    checkpoints_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                    config=config,
                    history=history,
                    aligner=aligner,
                    text_encoder=text_encoder,
                    mae_encoder=mae_encoder,
                    args=args,
                    image_dim=image_dim,
                    text_dim=text_dim,
                    wandb_logger=wandb_logger,
                    epoch=epoch + 1,
                    ema=ema,
                )

            save_training_progress(
                {
                    "status": "running",
                    "phase": "training_complete" if (epoch + 1) == args.epochs else "training",
                    "current_epoch": int(epoch + 1),
                    "target_epoch": int(args.epochs),
                    "latest_loss": avg_loss,
                    "latest_lr": current_lr,
                    "best_loss": lowest_train_loss,
                    "best_epoch": int(lowest_train_loss_epoch),
                    "best_checkpoint_metric_name": best_checkpoint_metric_name,
                    "best_checkpoint_metric_value": best_checkpoint_metric_value,
                    "best_checkpoint_epoch": int(best_checkpoint_epoch),
                    **val_metrics,
                    "out_dir": str(out_dir),
                },
                progress_path,
            )

        best_epoch_record = min(history, key=lambda item: float(item["loss"]))
        summary = _json_ready(
            {
                "checkpoint": str(checkpoint_path),
                "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
                "history_path": str(history_path),
                "progress_path": str(progress_path),
                "startup_progress_path": str(startup_progress_path),
                "summary_path": str(summary_path),
                "run_config_path": str(run_config_path),
                "checkpoints_dir": str(checkpoints_dir),
                "epochs": int(args.epochs),
                "steps_per_epoch": steps_per_epoch,
                "val_steps": val_steps,
                "dataset_size": len(dataset),
                "train_dataset_size": len(dataset),
                "val_dataset_size": len(val_dataset) if val_dataset is not None else 0,
                "best_metric_name": best_checkpoint_metric_name,
                "best_metric_value": best_checkpoint_metric_value,
                "best_checkpoint_epoch": int(best_checkpoint_epoch) if best_checkpoint_epoch > 0 else None,
                "best_loss": float(best_epoch_record["loss"]),
                "best_epoch": int(best_epoch_record["epoch"]),
                "lowest_train_loss": float(best_epoch_record["loss"]),
                "lowest_train_loss_epoch": int(best_epoch_record["epoch"]),
                "final_loss": float(history[-1]["loss"]) if history else None,
                "final_val_alignment_score": history[-1].get("val/alignment_score") if history else None,
                "cached_text_embeddings": len(text_cache) if text_cache is not None else None,
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
        )
        summary_path.write_text(json.dumps(summary, indent=2))
        save_training_progress(
            {
                "status": "completed",
                "phase": "complete",
                "current_epoch": int(args.epochs),
                "target_epoch": int(args.epochs),
                "best_loss": summary["best_loss"],
                "best_epoch": summary["best_epoch"],
                "best_checkpoint_metric_name": summary["best_metric_name"],
                "best_checkpoint_metric_value": summary["best_metric_value"],
                "best_checkpoint_epoch": summary["best_checkpoint_epoch"],
                "latest_loss": summary["final_loss"],
                "summary_path": str(summary_path),
                "out_dir": str(out_dir),
            },
            progress_path,
        )
        if wandb_logger is not None:
            wandb_logger.finish(summary)
        print(f"[align] saved checkpoint: {checkpoint_path}")
        if summary.get("wandb_mode") != "disabled" and summary.get("wandb_run_dir") is not None:
            print(f"[align] saved W&B run: {summary['wandb_run_dir']}")

    except BaseException as exc:
        error_path.write_text(traceback.format_exc())
        interrupted = isinstance(exc, (KeyboardInterrupt, RunInterruptedError))
        status = "aborted" if interrupted else "failed"
        if (
            aligner is not None
            and text_encoder is not None
            and mae_encoder is not None
            and config is not None
            and image_dim is not None
            and text_dim is not None
        ):
            try:
                _save_alignment_checkpoint(
                    checkpoints_dir / "interrupted_checkpoint.pt",
                    config=config,
                    history=history,
                    aligner=aligner,
                    text_encoder=text_encoder,
                    mae_encoder=mae_encoder,
                    args=args,
                    image_dim=image_dim,
                    text_dim=text_dim,
                    wandb_logger=wandb_logger,
                    epoch=int(history[-1]["epoch"]) if history else 0,
                    ema=ema,
                )
            except Exception:
                pass
        failure_payload = {
            "status": status,
            "phase": "aborted" if interrupted else "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "signal": termination_monitor.signal_name,
            "history_path": str(history_path),
            "checkpoint": str(checkpoint_path),
            "interrupted_checkpoint": str(checkpoints_dir / "interrupted_checkpoint.pt"),
            "out_dir": str(out_dir),
        }
        save_training_progress(failure_payload, progress_path)
        save_training_progress(failure_payload, startup_progress_path)
        if wandb_logger is not None:
            wandb_logger.finish(failure_payload)
        raise
    finally:
        termination_monitor.restore()


if __name__ == "__main__":
    main()
