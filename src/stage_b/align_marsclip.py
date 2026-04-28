"""B1a-geo MarsCLIP alignment trainer.

Adds a geo-context head to the Stage B aligner so the model is trained with
two contrastive objectives:

    L = w_text * InfoNCE(image, text) + w_geo * InfoNCE(image, geo)

The geo head is a small MLP over per-patch ``geo_features`` /
``scale_features`` / ``viewing_features`` already emitted by
``clip.marsclip_patches``. Both losses share the same false-negative mask
based on the rationale class id, so same-text duplicates are not treated as
negatives in either objective.

This trainer reuses the B0+ machinery (caches, EMA, balanced sampler,
mask-aware label smoothing, scratch-run layout, W&B logging, graceful
shutdown) by importing helpers from
``stage_b.align_text_mae_embeddings``. With ``--geo-loss-weight 0`` it
reproduces the B0+ image-text training path on the same data.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import traceback
from dataclasses import dataclass
from functools import partial
from typing import Any

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
from stage_b.patch_text_augment import load_patch_text_augment_jsonl
from stage_b.geo_encoders import (
    GEO_ENCODER_CHOICES,
    LocationEncoder,
    build_location_encoder,
)
from stage_b.T5_encoder import T5Encoder
from stage_b.align_text_mae_embeddings import (
    IMAGE_POOL_CHOICES,
    ClassBalancedSampler,
    FrozenTextEmbeddingCache,
    ParameterEMA,
    RunInterruptedError,
    TerminationMonitor,
    WandbLogger,
    _default_run_name,
    _extract_cache_candidate_texts,
    _json_ready,
    _limit_patch_records,
    _mean_reciprocal_rank,
    _recall_at_k,
    _resolve_device,
    adjust_learning_rate,
    attach_split_manifest,
    build_projector,
    compute_alignment_score,
    compute_retrieval_metrics,
    encode_image_with_satmae_encoder,
    filter_patch_records_by_split,
    init_wandb_logger,
    load_satmae_encoder,
    load_split_manifest,
    resolve_run_output_dir,
    select_balanced_patch_records,
    symmetric_contrastive_loss,
)

DEFAULT_MARSCLIP_OUT_ROOT = pathlib.Path("/scratch/marsrecon_runs/stage_b/marsclip_align")
WANDB_STEP_METRIC = "trainer/global_step"

GEO_CONTEXT_CHOICES = (
    "none",
    "latlon_view",
    "latlon_view_scale",
    "coords_only",
    "coords_view",
    "coords_view_scale",
)
COORDS_CONTEXTS: frozenset[str] = frozenset({"coords_only", "coords_view", "coords_view_scale"})

PAIRED_VIEWS_CHOICES = ("none", "local_global")


def make_local_view(images: torch.Tensor, *, crop_fraction: float) -> torch.Tensor:
    """Deterministic centered crop + bilinear resize back to the original shape.

    Used for the B1a-pairs CACo-style local view: the global view stays at the
    full 256-px patch and the local view is the inner ``crop_fraction``
    centered crop resized back to 256 px so the same SatMAE encoder can be
    reused without changing input resolution.
    """
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W)")
    if not (0.0 < float(crop_fraction) < 1.0):
        raise ValueError("local_crop_fraction must be in (0, 1)")
    _, _, h, w = images.shape
    crop_h = max(1, int(round(h * float(crop_fraction))))
    crop_w = max(1, int(round(w * float(crop_fraction))))
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    cropped = images[:, :, y0 : y0 + crop_h, x0 : x0 + crop_w]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


def geo_input_dim(geo_context: str) -> int:
    """Number of raw float features fed into the GeoEncoder for ``geo_context``."""
    if geo_context == "none":
        return 0
    if geo_context == "latlon_view":
        return 8 + 13
    if geo_context == "latlon_view_scale":
        return 8 + 8 + 13
    if geo_context == "coords_only":
        return 2
    if geo_context == "coords_view":
        return 2 + 13
    if geo_context == "coords_view_scale":
        return 2 + 8 + 13
    raise ValueError(
        f"Unknown geo_context '{geo_context}'. Expected one of {GEO_CONTEXT_CHOICES}."
    )


def geo_coords_dim(geo_context: str) -> int:
    """Number of leading columns of ``geo_input`` that hold raw ``(lat, lon)``."""
    return 2 if geo_context in COORDS_CONTEXTS else 0


def assemble_geo_input(sample: dict[str, Any], geo_context: str) -> torch.Tensor:
    """Concatenate the requested geo/scale/viewing features for one sample.

    Legacy ``latlon_*`` contexts feed cyclic ``geo_features`` straight into the
    geo head. The newer ``coords_*`` contexts emit ``(lat_deg, lon_deg)`` as
    the first two columns so a downstream positional encoder (RFF / SH) can
    handle the high-frequency lat/lon → embedding mapping without spectral
    bias from a vanilla MLP. Auxiliary scale / viewing features are appended
    after the coords for the ``coords_view*`` variants and concatenated post-PE
    by ``LocationEncoder``.
    """
    if geo_context == "none":
        return torch.empty(0, dtype=torch.float32)
    metadata = sample.get("metadata", {})
    viewing = torch.as_tensor(metadata["viewing_features"], dtype=torch.float32).flatten()
    if geo_context == "latlon_view":
        geo = torch.as_tensor(sample["geo_features"], dtype=torch.float32).flatten()
        return torch.cat([geo, viewing], dim=0)
    if geo_context == "latlon_view_scale":
        geo = torch.as_tensor(sample["geo_features"], dtype=torch.float32).flatten()
        scale = torch.as_tensor(sample["scale_features"], dtype=torch.float32).flatten()
        return torch.cat([geo, scale, viewing], dim=0)
    if geo_context in COORDS_CONTEXTS:
        location = sample.get("location")
        if location is None:
            raise KeyError(
                "coords_* geo contexts require sample['location'] = (lon, lat) from "
                "MarsCLIPPatchDataset."
            )
        loc = torch.as_tensor(location, dtype=torch.float32).flatten()
        coords = torch.stack([loc[1], loc[0]], dim=0)  # MarsCLIPPatchDataset emits (lon, lat); encoders expect (lat, lon)
        if geo_context == "coords_only":
            return coords
        if geo_context == "coords_view":
            return torch.cat([coords, viewing], dim=0)
        if geo_context == "coords_view_scale":
            scale = torch.as_tensor(sample["scale_features"], dtype=torch.float32).flatten()
            return torch.cat([coords, scale, viewing], dim=0)
    raise ValueError(
        f"Unknown geo_context '{geo_context}'. Expected one of {GEO_CONTEXT_CHOICES}."
    )


def collate_geo_warmup(samples: list[dict[str, Any]], *, geo_context: str) -> dict[str, Any]:
    """Collate raw patch samples for the cache-warmup loader (image + geo + text)."""
    images = torch.stack([sample["image"] for sample in samples], dim=0)
    geo_inputs = torch.stack(
        [assemble_geo_input(sample, geo_context) for sample in samples], dim=0
    )
    texts = [str(sample.get("rationale_raw", "")) for sample in samples]
    metadata = [dict(sample.get("metadata", {})) for sample in samples]
    return {
        "image": images,
        "geo_input": geo_inputs,
        "text": texts,
        "metadata": metadata,
    }


def collate_marsclip_train(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate cached B1a-geo / B1a-pairs training batches."""
    texts = [str(sample.get("rationale_raw", "")) for sample in samples]
    metadata = [dict(sample.get("metadata", {})) for sample in samples]
    batch: dict[str, Any] = {"text": texts, "metadata": metadata}
    if "image" in samples[0]:
        batch["image"] = torch.stack([sample["image"] for sample in samples], dim=0)
    if "image_features" in samples[0]:
        batch["image_features"] = torch.stack(
            [sample["image_features"] for sample in samples], dim=0
        )
    if "local_image_features" in samples[0]:
        batch["local_image_features"] = torch.stack(
            [sample["local_image_features"] for sample in samples], dim=0
        )
    if "geo_input" in samples[0]:
        batch["geo_input"] = torch.stack([sample["geo_input"] for sample in samples], dim=0)
    return batch


class GeoEncoder(nn.Module):
    """LayerNorm + MLP encoder over flat geo/scale/viewing features.

    Used as the back-compat path for ``latlon_*`` contexts where the entire
    ``geo_input`` tensor is fed into a single MLP. The ``coords_*`` contexts
    use :class:`LocationEncoder` from ``stage_b.geo_encoders`` instead.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        *,
        hidden_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("GeoEncoder requires a positive in_dim.")
        if int(depth) < 1:
            raise ValueError("GeoEncoder depth must be >= 1.")
        layers: list[nn.Module] = [nn.LayerNorm(int(in_dim))]
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
        self.net = nn.Sequential(*layers)

    def forward(self, geo_input: torch.Tensor) -> torch.Tensor:
        return self.net(geo_input)


@dataclass(eq=False)
class MarsCLIPAlignmentModel(nn.Module):
    """Image / text / geo projection heads for B1a-geo alignment."""

    image_projector: nn.Module
    text_projector: nn.Module
    geo_encoder: nn.Module
    logit_scale: nn.Parameter

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        geo_dim: int,
        embed_dim: int,
        *,
        projector_type: str = "linear",
        projector_hidden_dim: int = 768,
        projector_depth: int = 2,
        projector_dropout: float = 0.0,
        geo_encoder_type: str = "mlp",
        geo_coords_dim: int = 0,
        geo_hidden_dim: int = 256,
        geo_depth: int = 2,
        geo_dropout: float = 0.0,
        geo_rff_sigmas: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0),
        geo_rff_encoded_size: int = 128,
        geo_siren_w0: float = 1.0,
        geo_siren_w0_initial: float = 30.0,
        geo_sh_legendre_polys: int = 10,
        text_temperature_init: float = 0.07,
        geo_temperature_init: float = 0.07,
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
        self.geo_encoder_type = str(geo_encoder_type)
        self.geo_coords_dim = int(geo_coords_dim)
        self.geo_aux_dim = max(int(geo_dim) - int(geo_coords_dim), 0)
        if self.geo_coords_dim > 0:
            self.geo_encoder: nn.Module = build_location_encoder(
                encoder_type=str(geo_encoder_type),
                embed_dim=int(embed_dim),
                aux_dim=int(self.geo_aux_dim),
                hidden_dim=int(geo_hidden_dim),
                depth=int(geo_depth),
                dropout=float(geo_dropout),
                rff_sigmas=tuple(float(s) for s in geo_rff_sigmas),
                rff_encoded_size=int(geo_rff_encoded_size),
                siren_w0=float(geo_siren_w0),
                siren_w0_initial=float(geo_siren_w0_initial),
                sh_legendre_polys=int(geo_sh_legendre_polys),
            )
        else:
            self.geo_encoder = GeoEncoder(
                int(geo_dim) if int(geo_dim) > 0 else 1,
                int(embed_dim),
                hidden_dim=int(geo_hidden_dim),
                depth=int(geo_depth),
                dropout=float(geo_dropout),
            )
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / float(text_temperature_init)), dtype=torch.float32)
        )
        self.geo_logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / float(geo_temperature_init)), dtype=torch.float32)
        )
        self.aligner_config: dict[str, Any] = {
            "image_dim": int(image_dim),
            "text_dim": int(text_dim),
            "geo_dim": int(geo_dim),
            "geo_coords_dim": int(self.geo_coords_dim),
            "geo_aux_dim": int(self.geo_aux_dim),
            "embed_dim": int(embed_dim),
            "projector_type": str(projector_type),
            "projector_hidden_dim": int(projector_hidden_dim),
            "projector_depth": int(projector_depth),
            "projector_dropout": float(projector_dropout),
            "geo_encoder_type": str(geo_encoder_type),
            "geo_hidden_dim": int(geo_hidden_dim),
            "geo_depth": int(geo_depth),
            "geo_dropout": float(geo_dropout),
            "geo_rff_sigmas": [float(s) for s in geo_rff_sigmas],
            "geo_rff_encoded_size": int(geo_rff_encoded_size),
            "geo_siren_w0": float(geo_siren_w0),
            "geo_siren_w0_initial": float(geo_siren_w0_initial),
            "geo_sh_legendre_polys": int(geo_sh_legendre_polys),
            "text_temperature_init": float(text_temperature_init),
            "geo_temperature_init": float(geo_temperature_init),
        }

    def project_image(self, image_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_projector(image_features), dim=1)

    def project_text(self, text_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projector(text_features), dim=1)

    def project_geo(self, geo_input: torch.Tensor) -> torch.Tensor:
        if self.geo_coords_dim > 0 and isinstance(self.geo_encoder, LocationEncoder):
            coords = geo_input[:, : self.geo_coords_dim]
            aux = geo_input[:, self.geo_coords_dim :] if self.geo_aux_dim > 0 else None
            return F.normalize(self.geo_encoder(coords, aux), dim=1)
        return F.normalize(self.geo_encoder(geo_input), dim=1)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        geo_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        image_emb = self.project_image(image_features)
        text_emb = self.project_text(text_features)
        geo_emb = self.project_geo(geo_input) if geo_input is not None else None
        return image_emb, text_emb, geo_emb


class GeoAwareCachedDataset(Dataset):
    """Cached frozen image features paired with assembled geo input vectors.

    Optionally also stores ``local_image_features`` for the B1a-pairs view
    (deterministic centered crop of the patch resized back to 256 px and run
    through the same frozen SatMAE encoder).
    """

    def __init__(
        self,
        *,
        image_features: torch.Tensor,
        geo_inputs: torch.Tensor,
        texts: list[str],
        metadata_rows: list[dict[str, Any]],
        local_image_features: torch.Tensor | None = None,
    ) -> None:
        if image_features.ndim != 2:
            raise ValueError("image_features must have shape (N, D).")
        if geo_inputs.ndim != 2:
            raise ValueError("geo_inputs must have shape (N, G).")
        if not (image_features.shape[0] == geo_inputs.shape[0] == len(texts) == len(metadata_rows)):
            raise ValueError(
                "image_features, geo_inputs, texts, and metadata_rows must agree in length."
            )
        if local_image_features is not None:
            if local_image_features.ndim != 2:
                raise ValueError("local_image_features must have shape (N, D).")
            if local_image_features.shape != image_features.shape:
                raise ValueError(
                    "local_image_features must match image_features shape."
                )
        self.image_features = image_features.contiguous()
        self.geo_inputs = geo_inputs.contiguous()
        self.texts = list(texts)
        self.metadata_rows = [dict(row) for row in metadata_rows]
        self.local_image_features = (
            local_image_features.contiguous() if local_image_features is not None else None
        )

    def __len__(self) -> int:
        return int(self.image_features.shape[0])

    @property
    def geo_dim(self) -> int:
        return int(self.geo_inputs.shape[-1])

    @property
    def has_local_features(self) -> bool:
        return self.local_image_features is not None

    def __getitem__(self, index: int) -> dict[str, Any]:
        out = {
            "image_features": self.image_features[index],
            "geo_input": self.geo_inputs[index],
            "rationale_raw": self.texts[index],
            "metadata": dict(self.metadata_rows[index]),
        }
        if self.local_image_features is not None:
            out["local_image_features"] = self.local_image_features[index]
        return out


@torch.no_grad()
def build_geo_aware_cached_dataset(
    dataset: Dataset | list[dict[str, Any]],
    *,
    mae_encoder: nn.Module,
    device: torch.device,
    geo_context: str,
    image_pool: str = "cls",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    progress_path: pathlib.Path | None = None,
    out_dir: pathlib.Path | None = None,
    progress_log_interval: int = 10,
    phase: str = "image_geo_cache_warmup",
    paired_views: str = "none",
    local_crop_fraction: float = 0.5,
) -> GeoAwareCachedDataset:
    """Precompute frozen MAE image features and stack the geo input tensors.

    When ``paired_views == "local_global"`` the cache also stores a parallel
    ``local_image_features`` tensor obtained by deterministically center-cropping
    each patch by ``local_crop_fraction`` and resizing back to the patch
    resolution before a second SatMAE forward pass.
    """
    if paired_views not in PAIRED_VIEWS_CHOICES:
        raise ValueError(
            f"paired_views={paired_views!r} must be one of {PAIRED_VIEWS_CHOICES}"
        )
    collate = partial(collate_geo_warmup, geo_context=geo_context)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(**loader_kwargs)

    non_blocking = bool(pin_memory and device.type == "cuda")
    total_steps = max(len(dataloader), 1)
    feature_batches: list[torch.Tensor] = []
    local_batches: list[torch.Tensor] = []
    geo_batches: list[torch.Tensor] = []
    text_rows: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    start_time = time.time()
    report_interval = max(int(progress_log_interval), 1)
    cache_local = paired_views == "local_global"

    for step, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device, non_blocking=non_blocking)
        image_features = encode_image_with_satmae_encoder(mae_encoder, images, pool=image_pool)
        feature_batches.append(image_features.detach().cpu())
        if cache_local:
            local_images = make_local_view(images, crop_fraction=local_crop_fraction)
            local_features = encode_image_with_satmae_encoder(
                mae_encoder, local_images, pool=image_pool
            )
            local_batches.append(local_features.detach().cpu())
        geo_batches.append(batch["geo_input"].detach().cpu())
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
                "paired_views": str(paired_views),
                "local_crop_fraction": float(local_crop_fraction) if cache_local else None,
                "out_dir": str(out_dir) if out_dir is not None else None,
            }
            if progress_path is not None:
                save_training_progress(payload, progress_path)
            print(
                f"[align-marsclip] {phase} samples={len(text_rows)}/{len(dataset)} "
                f"steps={step}/{total_steps}"
                + (" (+local)" if cache_local else ""),
                flush=True,
            )

    image_features = torch.cat(feature_batches, dim=0)
    geo_inputs = torch.cat(geo_batches, dim=0)
    local_image_features = torch.cat(local_batches, dim=0) if cache_local else None
    return GeoAwareCachedDataset(
        image_features=image_features,
        geo_inputs=geo_inputs,
        texts=text_rows,
        metadata_rows=metadata_rows,
        local_image_features=local_image_features,
    )


def _diagonal_ranks(similarity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1-indexed positive ranks for diagonal-positive 1-to-1 retrieval."""
    n = similarity.shape[0]
    targets = torch.arange(n, device=similarity.device)
    row_order = torch.argsort(similarity, dim=1, descending=True)
    row_rank = (row_order == targets.unsqueeze(1)).to(torch.int64).argmax(dim=1) + 1
    col_order = torch.argsort(similarity, dim=0, descending=True)
    col_rank = (col_order == targets.unsqueeze(0)).to(torch.int64).argmax(dim=0) + 1
    return row_rank, col_rank


@torch.no_grad()
def compute_geocell_image_to_geo_metrics(
    image_emb: torch.Tensor,
    geo_emb: torch.Tensor,
    lat_lon_deg: torch.Tensor,
    *,
    cell_degs: tuple[float, ...],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Fraction of image queries whose top-k *geo* neighbors share a coarse lon/lat cell with the query patch.

    ``lat_lon_deg`` has shape ``(N, 2)`` with columns ``[lat, lon]`` in degrees, aligned with rows of
    ``image_emb``. This is a softer diagnostic than diagonal R@k when many patches sit on the same grid.
    """
    if image_emb.shape[0] != lat_lon_deg.shape[0] or lat_lon_deg.ndim != 2 or lat_lon_deg.shape[1] != 2:
        raise ValueError("lat_lon_deg must be (N, 2) [lat, lon] matching image_emb rows.")
    similarity = image_emb @ geo_emb.T
    kmax = min(max(ks), similarity.shape[1])
    if kmax < 1:
        return {}
    _, top_idx = similarity.topk(kmax, dim=1, largest=True)
    lat_lon = lat_lon_deg.to(device=similarity.device, dtype=torch.float32)
    out: dict[str, float] = {}
    for cell_deg in cell_degs:
        if cell_deg <= 0.0:
            continue
        tgt_cell = torch.stack(
            [torch.floor(lat_lon[:, 0] / cell_deg), torch.floor(lat_lon[:, 1] / cell_deg)], dim=1
        )
        pred_cell = tgt_cell[top_idx]
        match = (pred_cell == tgt_cell.unsqueeze(1)).all(dim=-1)
        for kk in ks:
            use = min(kk, kmax)
            hit = match[:, :use].any(dim=-1).float().mean().item()
            key_deg = str(float(cell_deg)).replace(".", "p")
            out[f"image_to_geo_geocell_{key_deg}deg_top{use}_any_neighbor"] = float(hit)
    return out


def compute_pairwise_retrieval_metrics(
    a_embeddings: torch.Tensor,
    b_embeddings: torch.Tensor,
    *,
    a_to_b_prefix: str,
    b_to_a_prefix: str,
) -> dict[str, float]:
    """Diagonal-positive retrieval metrics for two paired embedding sets."""
    similarity = a_embeddings @ b_embeddings.T
    a_to_b, b_to_a = _diagonal_ranks(similarity)
    return {
        "num_samples": float(similarity.shape[0]),
        f"{a_to_b_prefix}_r1": _recall_at_k(a_to_b, 1),
        f"{a_to_b_prefix}_r5": _recall_at_k(a_to_b, 5),
        f"{a_to_b_prefix}_r10": _recall_at_k(a_to_b, 10),
        f"{a_to_b_prefix}_mrr": _mean_reciprocal_rank(a_to_b),
        f"{a_to_b_prefix}_median_rank": float(torch.median(a_to_b.to(torch.float32)).item()),
        f"{b_to_a_prefix}_r1": _recall_at_k(b_to_a, 1),
        f"{b_to_a_prefix}_r5": _recall_at_k(b_to_a, 5),
        f"{b_to_a_prefix}_r10": _recall_at_k(b_to_a, 10),
        f"{b_to_a_prefix}_mrr": _mean_reciprocal_rank(b_to_a),
        f"{b_to_a_prefix}_median_rank": float(torch.median(b_to_a.to(torch.float32)).item()),
    }


@torch.no_grad()
def build_marsclip_embeddings(
    *,
    dataloader: DataLoader,
    mae_encoder: nn.Module,
    text_encoder: T5Encoder,
    aligner: MarsCLIPAlignmentModel,
    device: torch.device,
    text_cache: FrozenTextEmbeddingCache | None = None,
    non_blocking: bool = False,
    image_pool: str = "cls",
    geo_context: str = "latlon_view_scale",
    paired_views: str = "none",
    local_crop_fraction: float = 0.5,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    list[str],
    torch.Tensor | None,
]:
    """Build aligned image / text / (optional) geo / (optional) local embeddings.

    Returns ``(..., centroid_lat_lon)`` where the last tensor is optional ``(N, 2)`` with
    columns ``[lat_deg, lon_deg]`` when batch metadata includes ``centroid_lat`` /
    ``centroid_lon`` (MarsCLIP patch samples).
    """
    image_chunks: list[torch.Tensor] = []
    text_chunks: list[torch.Tensor] = []
    geo_chunks: list[torch.Tensor] = []
    local_chunks: list[torch.Tensor] = []
    latlon_chunks: list[torch.Tensor] = []
    collect_latlon: bool | None = None
    text_rows: list[str] = []
    use_local = paired_views == "local_global"

    for batch in dataloader:
        texts = batch["text"]
        if "image_features" in batch:
            image_features = batch["image_features"].to(device, non_blocking=non_blocking)
        else:
            images = batch["image"].to(device, non_blocking=non_blocking)
            image_features = encode_image_with_satmae_encoder(
                mae_encoder, images, pool=image_pool
            )

        if text_cache is not None:
            text_features = text_cache.encode(texts)
        else:
            text_features, _ = text_encoder(texts, device=device)

        image_emb = aligner.project_image(image_features)
        text_emb = aligner.project_text(text_features)
        image_chunks.append(image_emb.detach().cpu())
        text_chunks.append(text_emb.detach().cpu())

        if use_local:
            if "local_image_features" in batch:
                local_features = batch["local_image_features"].to(
                    device, non_blocking=non_blocking
                )
            else:
                if "image" not in batch:
                    raise KeyError(
                        "paired_views='local_global' requires either cached "
                        "local_image_features or raw 'image' tensors in the batch."
                    )
                images = batch["image"].to(device, non_blocking=non_blocking)
                local_images = make_local_view(images, crop_fraction=local_crop_fraction)
                local_features = encode_image_with_satmae_encoder(
                    mae_encoder, local_images, pool=image_pool
                )
            local_emb = aligner.project_image(local_features)
            local_chunks.append(local_emb.detach().cpu())

        if geo_context != "none":
            if "geo_input" in batch:
                geo_input = batch["geo_input"].to(device, non_blocking=non_blocking)
            else:
                geo_input = torch.stack(
                    [
                        assemble_geo_input(
                            {
                                "geo_features": meta.get("geo_features"),
                                "scale_features": meta.get("scale_features"),
                                "metadata": meta,
                            },
                            geo_context,
                        )
                        for meta in batch["metadata"]
                    ],
                    dim=0,
                ).to(device, non_blocking=non_blocking)
            geo_emb = aligner.project_geo(geo_input)
            geo_chunks.append(geo_emb.detach().cpu())

        text_rows.extend(texts)

        md = batch.get("metadata")
        if isinstance(md, list) and md:
            sample0 = md[0]
            if isinstance(sample0, dict) and "centroid_lat" in sample0 and "centroid_lon" in sample0:
                if collect_latlon is False:
                    raise ValueError(
                        "Mixed batches: centroid_lat/lon must be present in all batches or none."
                    )
                collect_latlon = True
                lat_np = torch.tensor(
                    [float(m["centroid_lat"]) for m in md], dtype=torch.float32
                )
                lon_np = torch.tensor(
                    [float(m["centroid_lon"]) for m in md], dtype=torch.float32
                )
                latlon_chunks.append(torch.stack([lat_np, lon_np], dim=1))
            else:
                if collect_latlon is True:
                    raise ValueError(
                        "Mixed batches: centroid_lat/lon must be present in all batches or none."
                    )
                collect_latlon = False

    image_emb_all = torch.cat(image_chunks, dim=0)
    text_emb_all = torch.cat(text_chunks, dim=0)
    geo_emb_all = torch.cat(geo_chunks, dim=0) if geo_chunks else None
    local_emb_all = torch.cat(local_chunks, dim=0) if local_chunks else None
    centroid_lat_lon = torch.cat(latlon_chunks, dim=0) if collect_latlon is True else None
    return image_emb_all, text_emb_all, geo_emb_all, local_emb_all, text_rows, centroid_lat_lon


@torch.no_grad()
def evaluate_marsclip_retrieval(
    *,
    dataloader: DataLoader,
    mae_encoder: nn.Module,
    text_encoder: T5Encoder,
    aligner: MarsCLIPAlignmentModel,
    device: torch.device,
    text_cache: FrozenTextEmbeddingCache | None = None,
    non_blocking: bool = False,
    image_pool: str = "cls",
    geo_context: str = "latlon_view_scale",
    paired_views: str = "none",
    local_crop_fraction: float = 0.5,
    geocell_degs: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Combined image↔text + (optional) image↔geo + (optional) local↔global metrics."""
    image_emb, text_emb, geo_emb, local_emb, text_rows, centroid_lat_lon = build_marsclip_embeddings(
        dataloader=dataloader,
        mae_encoder=mae_encoder,
        text_encoder=text_encoder,
        aligner=aligner,
        device=device,
        text_cache=text_cache,
        non_blocking=non_blocking,
        image_pool=image_pool,
        geo_context=geo_context,
        paired_views=paired_views,
        local_crop_fraction=local_crop_fraction,
    )
    metrics = compute_retrieval_metrics(image_emb, text_emb, text_rows)
    if geo_emb is not None:
        geo_metrics = compute_pairwise_retrieval_metrics(
            image_emb,
            geo_emb,
            a_to_b_prefix="image_to_geo",
            b_to_a_prefix="geo_to_image",
        )
        for key, value in geo_metrics.items():
            if key == "num_samples":
                continue
            metrics[key] = value
        if (
            centroid_lat_lon is not None
            and geocell_degs is not None
            and len(geocell_degs) > 0
            and centroid_lat_lon.shape[0] == image_emb.shape[0]
        ):
            gc = compute_geocell_image_to_geo_metrics(
                image_emb,
                geo_emb,
                centroid_lat_lon,
                cell_degs=tuple(geocell_degs),
                ks=(1, 5, 10),
            )
            metrics.update(gc)
    if local_emb is not None:
        lg_metrics = compute_pairwise_retrieval_metrics(
            local_emb,
            image_emb,
            a_to_b_prefix="local_to_global",
            b_to_a_prefix="global_to_local",
        )
        for key, value in lg_metrics.items():
            if key == "num_samples":
                continue
            metrics[key] = value
    return metrics


def _save_marsclip_checkpoint(
    path: pathlib.Path,
    *,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    aligner: MarsCLIPAlignmentModel,
    text_encoder: T5Encoder,
    mae_encoder: nn.Module,
    args: argparse.Namespace,
    image_dim: int,
    text_dim: int,
    geo_dim: int,
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
            "geo_dim": geo_dim,
            "embed_dim": int(args.embed_dim),
            "image_pool": str(getattr(args, "image_pool", "cls")),
            "geo_context": str(getattr(args, "geo_context", "latlon_view_scale")),
            "geo_encoder_type": str(getattr(args, "geo_encoder_type", "mlp")),
            "geo_loss_weight": float(getattr(args, "geo_loss_weight", 0.0)),
            "text_loss_weight": float(getattr(args, "text_loss_weight", 1.0)),
            "geo_warmup_epochs": int(getattr(args, "geo_warmup_epochs", 0)),
            "geo_temperature_init": float(getattr(args, "geo_temperature_init", 0.07)),
            "text_temperature_init": float(getattr(args, "text_temperature_init", 0.07)),
            "geo_logit_scale_max": float(getattr(args, "geo_logit_scale_max", 100.0)),
            "paired_views": str(getattr(args, "paired_views", "none")),
            "local_crop_fraction": float(getattr(args, "local_crop_fraction", 0.5)),
            "lg_loss_weight": float(getattr(args, "lg_loss_weight", 0.0)),
            "loss_type": "infonce",
            "projector_type": str(getattr(args, "projector_type", "mlp")),
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
    parser = argparse.ArgumentParser(
        description="Train a MarsCLIP B1a-geo aligner (image+text+geo contrastive)."
    )
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("/scratch/mars_hirise"))
    parser.add_argument("--bbox", type=float, nargs=4, default=(-136.0, 12.0, -124.0, 24.0))
    parser.add_argument("--patch-size-deg", type=float, default=0.005)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size-px", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--patch-records-path", type=pathlib.Path, default=None)
    parser.add_argument("--split-manifest", type=pathlib.Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
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
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--projector-type", type=str, default="mlp", choices=("linear", "mlp"))
    parser.add_argument("--projector-hidden-dim", type=int, default=768)
    parser.add_argument("--projector-depth", type=int, default=2)
    parser.add_argument("--projector-dropout", type=float, default=0.0)
    parser.add_argument("--image-pool", type=str, default="cls", choices=IMAGE_POOL_CHOICES)
    parser.add_argument("--geo-context", type=str, default="latlon_view_scale", choices=GEO_CONTEXT_CHOICES)
    parser.add_argument("--geo-encoder-type", type=str, default="mlp", choices=GEO_ENCODER_CHOICES)
    parser.add_argument("--geo-hidden-dim", type=int, default=256)
    parser.add_argument("--geo-depth", type=int, default=2)
    parser.add_argument("--geo-dropout", type=float, default=0.0)
    parser.add_argument(
        "--geo-rff-sigmas",
        type=float,
        nargs="+",
        default=(1.0, 4.0, 16.0, 64.0),
        help="Hierarchical RFF sigmas (cycles per degree); GeoCLIP-style.",
    )
    parser.add_argument("--geo-rff-encoded-size", type=int, default=128)
    parser.add_argument("--geo-siren-w0", type=float, default=1.0)
    parser.add_argument("--geo-siren-w0-initial", type=float, default=30.0)
    parser.add_argument("--geo-sh-legendre-polys", type=int, default=10)
    parser.add_argument("--geo-temperature-init", type=float, default=0.07)
    parser.add_argument("--geo-logit-scale-max", type=float, default=100.0)
    parser.add_argument(
        "--geo-warmup-epochs",
        type=int,
        default=0,
        help="Number of opening epochs to train with text-loss disabled (geo-only warmup).",
    )
    parser.add_argument("--text-temperature-init", type=float, default=0.07)
    parser.add_argument("--geo-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--paired-views",
        type=str,
        default="none",
        choices=PAIRED_VIEWS_CHOICES,
        help="If 'local_global', cache a deterministic centered local crop "
        "(re-encoded by SatMAE) and add a CACo-style local↔global InfoNCE term.",
    )
    parser.add_argument("--local-crop-fraction", type=float, default=0.5)
    parser.add_argument(
        "--lg-loss-weight",
        type=float,
        default=0.0,
        help="CACo-style local↔global InfoNCE weight; defaults to 0 so existing "
        "b1a-geo recipes keep working. Set to 0.5 alongside --paired-views local_global "
        "for the canonical b1a-pairs recipe.",
    )
    parser.add_argument("--balanced-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--progress-log-interval", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None)
    parser.add_argument("--out-root", type=pathlib.Path, default=DEFAULT_MARSCLIP_OUT_ROOT)
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=("disabled", "offline", "online"))
    parser.add_argument("--wandb-project", type=str, default="MarsRecon")
    parser.add_argument("--wandb-entity", type=str, default="akshayn3-auvsl")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--patch-text-augment-jsonl",
        type=pathlib.Path,
        default=None,
        help="Optional JSONL map {patch_id, augment} merged into per-patch rationale_raw.",
    )
    parser.add_argument(
        "--geocell-deg",
        type=float,
        nargs="*",
        default=argparse.SUPPRESS,
        help="Grid size in degrees for geocell diagnostics on image→geo (default: 0.1 0.5). "
        "Pass an empty list after the flag to disable.",
    )
    args = parser.parse_args()

    if hasattr(args, "geocell_deg"):
        effective_geocell_degs: tuple[float, ...] = (
            tuple(float(x) for x in args.geocell_deg) if args.geocell_deg else ()
        )
    else:
        effective_geocell_degs = (0.1, 0.5)

    torch.manual_seed(args.seed)

    if args.geo_context == "none" and float(args.geo_loss_weight) > 0.0:
        raise ValueError("--geo-context none requires --geo-loss-weight 0.")
    if args.paired_views == "none" and float(args.lg_loss_weight) > 0.0:
        raise ValueError("--paired-views none requires --lg-loss-weight 0.")
    if args.paired_views == "local_global" and not (0.0 < float(args.local_crop_fraction) < 1.0):
        raise ValueError("--local-crop-fraction must be in (0, 1) when --paired-views local_global.")
    if args.paired_views == "local_global" and float(args.lg_loss_weight) <= 0.0:
        print(
            "[align-marsclip] warning: --paired-views local_global with --lg-loss-weight=0; "
            "the local view will be cached and reported in val metrics but not used for training.",
            flush=True,
        )

    device = _resolve_device(args.device)
    run_name = args.run_name or _default_run_name(mae_checkpoint=args.mae_checkpoint).replace(
        "text-mae-align", "marsclip-b1a-geo"
    )
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
    aligner: MarsCLIPAlignmentModel | None = None
    text_encoder: T5Encoder | None = None
    mae_encoder: nn.Module | None = None
    config: dict[str, Any] | None = None
    image_dim: int | None = None
    text_dim: int | None = None
    geo_dim: int = geo_input_dim(args.geo_context)
    ema: ParameterEMA | None = None

    save_training_progress(
        {
            "status": "running",
            "phase": "startup",
            "message": "Initializing MarsCLIP B1a-geo alignment run.",
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
            "patch_records_path": str(args.patch_records_path)
            if args.patch_records_path is not None
            else None,
            "split_manifest": str(args.split_manifest) if args.split_manifest is not None else None,
            "geo_dim": int(geo_dim),
            "trainer": "align_marsclip",
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
                raise ValueError(
                    "--split-manifest requires --patch-records-path so patch ids match the split file."
                )
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

        patch_text_augment_by_id: dict[str, str] | None = None
        if args.patch_text_augment_jsonl is not None:
            patch_text_augment_by_id = load_patch_text_augment_jsonl(args.patch_text_augment_jsonl)

        dataset = MarsCLIPPatchDataset(
            root=args.root,
            bbox=tuple(args.bbox),
            patch_size=args.patch_size_deg,
            image_size=args.image_size,
            max_patches=train_dataset_max_patches,
            color_only=True,
            patch_records=train_patch_records,
            patch_text_augment_by_id=patch_text_augment_by_id,
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
                patch_text_augment_by_id=patch_text_augment_by_id,
            )
        save_training_progress(
            {
                "status": "running",
                "phase": "dataset_ready",
                "message": "MarsCLIP B1a-geo dataset ready.",
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

        if not (args.cache_image_embeddings and args.freeze_mae):
            raise ValueError(
                "align_marsclip currently requires --cache-image-embeddings and --freeze-mae; "
                "uncached/unfrozen training is not supported in B1a-geo Phase 1."
            )

        image_pool = str(args.image_pool)
        train_source_dataset = build_geo_aware_cached_dataset(
            dataset,
            mae_encoder=mae_encoder,
            device=device,
            geo_context=args.geo_context,
            image_pool=image_pool,
            batch_size=args.image_cache_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            progress_path=startup_progress_path,
            out_dir=out_dir,
            progress_log_interval=max(progress_interval // 2, 1),
            phase="train_image_geo_cache_warmup",
            paired_views=str(args.paired_views),
            local_crop_fraction=float(args.local_crop_fraction),
        )
        val_source_dataset: GeoAwareCachedDataset | None = None
        if val_dataset is not None:
            val_source_dataset = build_geo_aware_cached_dataset(
                val_dataset,
                mae_encoder=mae_encoder,
                device=device,
                geo_context=args.geo_context,
                image_pool=image_pool,
                batch_size=args.image_cache_batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                progress_path=startup_progress_path,
                out_dir=out_dir,
                progress_log_interval=max(progress_interval // 2, 1),
                phase="val_image_geo_cache_warmup",
                paired_views=str(args.paired_views),
                local_crop_fraction=float(args.local_crop_fraction),
            )
        save_training_progress(
            {
                "status": "running",
                "phase": "image_geo_cache_ready",
                "cached_train_samples": len(train_source_dataset),
                "cached_val_samples": len(val_source_dataset) if val_source_dataset is not None else 0,
                "out_dir": str(out_dir),
            },
            startup_progress_path,
        )

        if int(geo_dim) > 0 and train_source_dataset.geo_dim != int(geo_dim):
            raise ValueError(
                f"Cached geo dim {train_source_dataset.geo_dim} disagrees with expected {geo_dim}."
            )

        train_texts_for_labels: list[str] = list(train_source_dataset.texts)
        global_text_label_map: dict[str, int] = {}
        for text in train_texts_for_labels:
            key = str(text)
            if key not in global_text_label_map:
                global_text_label_map[key] = len(global_text_label_map)
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

        loader_kwargs: dict[str, Any] = {
            "batch_size": args.batch_size,
            "collate_fn": collate_marsclip_train,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        }
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        else:
            loader_kwargs["shuffle"] = loader_shuffle
            loader_kwargs["generator"] = torch.Generator().manual_seed(args.seed)
        dataloader = DataLoader(train_source_dataset, **loader_kwargs)

        val_dataloader = None
        if val_source_dataset is not None and args.val_every > 0:
            val_dataloader = DataLoader(
                val_source_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                collate_fn=collate_marsclip_train,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )
        save_training_progress(
            {
                "status": "running",
                "phase": "dataloaders_ready",
                "message": "MarsCLIP B1a-geo dataloader initialized.",
                "steps_per_epoch": len(dataloader),
                "val_steps": len(val_dataloader) if val_dataloader is not None else 0,
                "balanced_sampler": bool(sampler is not None),
                "out_dir": str(out_dir),
            },
            startup_progress_path,
        )

        warmup_batch = next(iter(dataloader))
        image_dim = int(warmup_batch["image_features"].shape[-1])
        if text_cache is not None:
            warmup_text_features = text_cache.encode(warmup_batch["text"])
        else:
            warmup_text_features, _ = text_encoder(warmup_batch["text"], device=device)
        text_dim = int(warmup_text_features.shape[-1])

        coords_dim = geo_coords_dim(str(args.geo_context))
        if coords_dim > 0 and str(args.geo_encoder_type) == "mlp":
            print(
                "[align-marsclip] note: --geo-context coords_* with --geo-encoder-type mlp "
                "skips the positional encoder; consider rff_siren or sh_siren for geo retrieval.",
                flush=True,
            )
        if coords_dim == 0 and str(args.geo_encoder_type) != "mlp":
            raise ValueError(
                "--geo-encoder-type rff_/sh_ requires --geo-context coords_only / coords_view / "
                "coords_view_scale (raw lat/lon as the first two columns of geo_input)."
            )
        aligner = MarsCLIPAlignmentModel(
            image_dim=image_dim,
            text_dim=text_dim,
            geo_dim=int(geo_dim) if int(geo_dim) > 0 else 1,
            embed_dim=args.embed_dim,
            projector_type=str(args.projector_type),
            projector_hidden_dim=int(args.projector_hidden_dim),
            projector_depth=int(args.projector_depth),
            projector_dropout=float(args.projector_dropout),
            geo_encoder_type=str(args.geo_encoder_type),
            geo_coords_dim=int(coords_dim),
            geo_hidden_dim=int(args.geo_hidden_dim),
            geo_depth=int(args.geo_depth),
            geo_dropout=float(args.geo_dropout),
            geo_rff_sigmas=tuple(float(s) for s in args.geo_rff_sigmas),
            geo_rff_encoded_size=int(args.geo_rff_encoded_size),
            geo_siren_w0=float(args.geo_siren_w0),
            geo_siren_w0_initial=float(args.geo_siren_w0_initial),
            geo_sh_legendre_polys=int(args.geo_sh_legendre_polys),
            text_temperature_init=float(args.text_temperature_init),
            geo_temperature_init=float(args.geo_temperature_init),
        ).to(device)

        if float(args.ema_decay) > 0.0:
            ema = ParameterEMA(aligner, decay=float(args.ema_decay))

        parameters: list[nn.Parameter] = list(aligner.parameters())
        if args.train_text_encoder:
            parameters += [p for p in text_encoder.parameters() if p.requires_grad]
        if not args.freeze_mae:
            parameters += [p for p in mae_encoder.parameters() if p.requires_grad]

        optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        trainable_parameters, total_parameters = count_trainable_parameters(aligner)
        if args.train_text_encoder:
            trainable_parameters += sum(
                p.numel() for p in text_encoder.parameters() if p.requires_grad
            )
            total_parameters += sum(p.numel() for p in text_encoder.parameters())
        else:
            total_parameters += sum(p.numel() for p in text_encoder.parameters())
        if not args.freeze_mae:
            trainable_parameters += sum(
                p.numel() for p in mae_encoder.parameters() if p.requires_grad
            )
        total_parameters += sum(p.numel() for p in mae_encoder.parameters())

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
                "patch_records_path": str(args.patch_records_path)
                if args.patch_records_path is not None
                else None,
                "split_manifest": str(args.split_manifest)
                if args.split_manifest is not None
                else None,
                "wandb_run_name": wandb_run_name,
                "dataset_size": len(dataset),
                "train_dataset_size": len(dataset),
                "val_dataset_size": len(val_dataset) if val_dataset is not None else 0,
                "steps_per_epoch": steps_per_epoch,
                "val_steps": val_steps,
                "image_dim": image_dim,
                "text_dim": text_dim,
                "geo_dim": int(geo_dim),
                "trainer": "align_marsclip",
                "cached_text_embeddings": len(text_cache) if text_cache is not None else None,
                "cached_image_embeddings": True,
                "train_unique_texts": int(
                    dataset.patch_records["rationale_raw"].astype(str).nunique()
                ),
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
                "message": "MarsCLIP B1a-geo aligner initialized.",
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
        geo_logit_scale_max_log = math.log(float(args.geo_logit_scale_max))

        base_text_loss_weight = float(args.text_loss_weight)
        base_geo_loss_weight = float(args.geo_loss_weight) if args.geo_context != "none" else 0.0
        base_lg_loss_weight = (
            float(args.lg_loss_weight) if args.paired_views == "local_global" else 0.0
        )
        geo_warmup_epochs = max(int(args.geo_warmup_epochs), 0)

        for epoch in range(args.epochs):
            termination_monitor.raise_if_requested()
            aligner.train()
            text_encoder.eval()
            mae_encoder.eval()

            in_geo_warmup = epoch < geo_warmup_epochs and base_geo_loss_weight > 0.0
            text_loss_weight = 0.0 if in_geo_warmup else base_text_loss_weight
            geo_loss_weight = base_geo_loss_weight
            lg_loss_weight = base_lg_loss_weight
            if in_geo_warmup:
                print(
                    f"[align-marsclip] epoch={epoch + 1}/{args.epochs} "
                    f"phase=geo_warmup (text_loss_weight=0)",
                    flush=True,
                )

            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_text_loss = 0.0
            epoch_geo_loss = 0.0
            epoch_lg_loss = 0.0
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

                image_features = batch["image_features"].to(device, non_blocking=non_blocking)
                geo_input = (
                    batch["geo_input"].to(device, non_blocking=non_blocking)
                    if "geo_input" in batch and geo_loss_weight > 0.0
                    else None
                )
                local_image_features = (
                    batch["local_image_features"].to(device, non_blocking=non_blocking)
                    if "local_image_features" in batch and lg_loss_weight > 0.0
                    else None
                )
                if text_cache is not None:
                    text_features = text_cache.encode(texts)
                else:
                    with torch.no_grad():
                        text_features, _ = text_encoder(texts, device=device)

                batch_class_ids = torch.tensor(
                    [global_text_label_map[str(text)] for text in texts],
                    device=device,
                    dtype=torch.long,
                )

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    image_emb = aligner.project_image(image_features)
                    text_emb = aligner.project_text(text_features)
                    text_loss = symmetric_contrastive_loss(
                        image_emb,
                        text_emb,
                        aligner.logit_scale,
                        text_labels=batch_class_ids if args.false_negative_mask else None,
                        logit_scale_max=float(args.logit_scale_max),
                        label_smoothing=float(args.label_smoothing),
                    )
                    if geo_input is not None:
                        geo_emb = aligner.project_geo(geo_input)
                        geo_loss = symmetric_contrastive_loss(
                            image_emb,
                            geo_emb,
                            aligner.geo_logit_scale,
                            text_labels=None,
                            logit_scale_max=float(args.geo_logit_scale_max),
                            label_smoothing=float(args.label_smoothing),
                        )
                    else:
                        geo_loss = torch.zeros((), device=device, dtype=text_loss.dtype)
                    if local_image_features is not None:
                        local_emb = aligner.project_image(local_image_features)
                        lg_loss = symmetric_contrastive_loss(
                            local_emb,
                            image_emb,
                            aligner.logit_scale,
                            text_labels=batch_class_ids if args.false_negative_mask else None,
                            logit_scale_max=float(args.logit_scale_max),
                            label_smoothing=float(args.label_smoothing),
                        )
                    else:
                        lg_loss = torch.zeros((), device=device, dtype=text_loss.dtype)
                    loss = (
                        text_loss_weight * text_loss
                        + geo_loss_weight * geo_loss
                        + lg_loss_weight * lg_loss
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
                    aligner.geo_logit_scale.clamp_(max=geo_logit_scale_max_log)

                if ema is not None:
                    ema.update(aligner)

                global_step += 1
                epoch_steps += 1
                loss_value = float(loss.detach().cpu())
                text_loss_value = float(text_loss.detach().cpu())
                geo_loss_value = float(geo_loss.detach().cpu()) if geo_input is not None else 0.0
                lg_loss_value = (
                    float(lg_loss.detach().cpu()) if local_image_features is not None else 0.0
                )
                epoch_loss += loss_value
                epoch_text_loss += text_loss_value
                epoch_geo_loss += geo_loss_value
                epoch_lg_loss += lg_loss_value
                should_report = (
                    step_in_epoch % progress_interval == 0 or step_in_epoch == steps_per_epoch
                )
                if should_report:
                    running_mean_loss = epoch_loss / max(epoch_steps, 1)
                    running_mean_text = epoch_text_loss / max(epoch_steps, 1)
                    running_mean_geo = epoch_geo_loss / max(epoch_steps, 1)
                    running_mean_lg = epoch_lg_loss / max(epoch_steps, 1)
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
                        "latest_text_loss": text_loss_value,
                        "latest_geo_loss": geo_loss_value,
                        "latest_lg_loss": lg_loss_value,
                        "running_mean_loss": running_mean_loss,
                        "running_mean_text_loss": running_mean_text,
                        "running_mean_geo_loss": running_mean_geo,
                        "running_mean_lg_loss": running_mean_lg,
                        "latest_lr": current_lr,
                        "epoch_elapsed_sec": elapsed,
                        "out_dir": str(out_dir),
                    }
                    save_training_progress(progress_payload, progress_path)
                    print(
                        f"[align-marsclip] epoch={epoch + 1}/{args.epochs} "
                        f"step={step_in_epoch}/{steps_per_epoch} "
                        f"loss={loss_value:.6f} (text={text_loss_value:.4f} geo={geo_loss_value:.4f} "
                        f"lg={lg_loss_value:.4f}) mean={running_mean_loss:.6f} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    if wandb_logger is not None:
                        wandb_logger.log_metrics(
                            {
                                "train/loss": loss_value,
                                "train/text_loss": text_loss_value,
                                "train/geo_loss": geo_loss_value,
                                "train/lg_loss": lg_loss_value,
                                "train/running_mean_loss": running_mean_loss,
                                "train/running_mean_text_loss": running_mean_text,
                                "train/running_mean_geo_loss": running_mean_geo,
                                "train/running_mean_lg_loss": running_mean_lg,
                                "train/logit_scale": float(
                                    aligner.logit_scale.exp().detach().cpu().item()
                                ),
                                "train/geo_logit_scale": float(
                                    aligner.geo_logit_scale.exp().detach().cpu().item()
                                ),
                                "train/text_loss_weight": float(text_loss_weight),
                                "train/geo_loss_weight": float(geo_loss_weight),
                                "train/lg_loss_weight": float(lg_loss_weight),
                                "train/lr": current_lr,
                                "train/step_in_epoch": float(step_in_epoch),
                                "train/steps_in_epoch": float(steps_per_epoch),
                                "train/global_step": float(global_step),
                                "train/epoch_progress": float(epoch)
                                + (float(step_in_epoch) / max(float(steps_per_epoch), 1.0)),
                            },
                            step=global_step,
                        )

            avg_loss = epoch_loss / max(epoch_steps, 1)
            avg_text_loss = epoch_text_loss / max(epoch_steps, 1)
            avg_geo_loss = epoch_geo_loss / max(epoch_steps, 1)
            avg_lg_loss = epoch_lg_loss / max(epoch_steps, 1)
            epoch_duration = time.time() - epoch_start
            val_metrics: dict[str, float] = {}
            if val_dataloader is not None and (
                ((epoch + 1) % max(args.val_every, 1)) == 0 or ((epoch + 1) == args.epochs)
            ):
                aligner.eval()
                val_start = time.time()
                ema_backup: dict[str, torch.Tensor] | None = None
                if ema is not None:
                    ema_backup = ema.apply_to(aligner)
                try:
                    raw_val_metrics = evaluate_marsclip_retrieval(
                        dataloader=val_dataloader,
                        mae_encoder=mae_encoder,
                        text_encoder=text_encoder,
                        aligner=aligner,
                        device=device,
                        text_cache=text_cache if not args.train_text_encoder else None,
                        non_blocking=non_blocking,
                        image_pool=image_pool,
                        geo_context=str(args.geo_context),
                        paired_views=str(args.paired_views),
                        local_crop_fraction=float(args.local_crop_fraction),
                        geocell_degs=effective_geocell_degs
                        if str(args.geo_context) != "none"
                        else None,
                    )
                finally:
                    if ema is not None and ema_backup is not None:
                        ema.restore(aligner, ema_backup)
                val_duration = time.time() - val_start
                val_metrics = {f"val/{key}": float(value) for key, value in raw_val_metrics.items()}
                val_metrics["val/alignment_score"] = compute_alignment_score(raw_val_metrics)
                val_metrics["val/duration_sec"] = float(val_duration)
                geo_score_msg = (
                    f" i2g_r10={val_metrics.get('val/image_to_geo_r10', float('nan')):.4f}"
                    if "val/image_to_geo_r10" in val_metrics
                    else ""
                )
                lg_score_msg = (
                    f" l2g_r10={val_metrics.get('val/local_to_global_r10', float('nan')):.4f}"
                    if "val/local_to_global_r10" in val_metrics
                    else ""
                )
                print(
                    f"[align-marsclip] epoch={epoch + 1}/{args.epochs} "
                    f"val score={val_metrics['val/alignment_score']:.6f} "
                    f"i2t_r10={val_metrics['val/image_to_text_r10']:.4f} "
                    f"t2i_r10={val_metrics['val/text_to_image_r10']:.4f}"
                    f"{geo_score_msg}{lg_score_msg} duration={val_duration:.1f}s",
                    flush=True,
                )
                if wandb_logger is not None:
                    wandb_logger.log_metrics(val_metrics, step=global_step)

            epoch_record = {
                "epoch": float(epoch + 1),
                "loss": avg_loss,
                "text_loss": avg_text_loss,
                "geo_loss": avg_geo_loss,
                "lg_loss": avg_lg_loss,
                "steps": float(epoch_steps),
                "duration_sec": epoch_duration,
            }
            epoch_record.update(val_metrics)
            history.append(epoch_record)
            save_training_history(history, history_path)
            print(
                f"[align-marsclip] epoch={epoch + 1}/{args.epochs} complete "
                f"loss={avg_loss:.6f} (text={avg_text_loss:.4f} geo={avg_geo_loss:.4f} "
                f"lg={avg_lg_loss:.4f}) duration={epoch_duration:.1f}s",
                flush=True,
            )
            if wandb_logger is not None:
                wandb_logger.log_metrics(
                    {
                        "epoch/train_loss": avg_loss,
                        "epoch/train_text_loss": avg_text_loss,
                        "epoch/train_geo_loss": avg_geo_loss,
                        "epoch/train_lg_loss": avg_lg_loss,
                        "epoch/train_steps": float(epoch_steps),
                        "epoch/steps_in_epoch": float(steps_per_epoch),
                        "epoch/global_step_end": float(global_step),
                        "epoch/duration_sec": epoch_duration,
                        **val_metrics,
                    },
                    step=global_step,
                )

            _save_marsclip_checkpoint(
                checkpoint_path,
                config=config,
                history=history,
                aligner=aligner,
                text_encoder=text_encoder,
                mae_encoder=mae_encoder,
                args=args,
                image_dim=image_dim,
                text_dim=text_dim,
                geo_dim=int(geo_dim),
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
                    _save_marsclip_checkpoint(
                        best_checkpoint_path,
                        config=config,
                        history=history,
                        aligner=aligner,
                        text_encoder=text_encoder,
                        mae_encoder=mae_encoder,
                        args=args,
                        image_dim=image_dim,
                        text_dim=text_dim,
                        geo_dim=int(geo_dim),
                        wandb_logger=wandb_logger,
                        epoch=epoch + 1,
                        ema=ema,
                    )
                finally:
                    if ema is not None and best_ema_backup is not None:
                        ema.restore(aligner, best_ema_backup)
            if args.checkpoint_every > 0 and (
                ((epoch + 1) % args.checkpoint_every == 0) or ((epoch + 1) == args.epochs)
            ):
                _save_marsclip_checkpoint(
                    checkpoints_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                    config=config,
                    history=history,
                    aligner=aligner,
                    text_encoder=text_encoder,
                    mae_encoder=mae_encoder,
                    args=args,
                    image_dim=image_dim,
                    text_dim=text_dim,
                    geo_dim=int(geo_dim),
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
                    "latest_text_loss": avg_text_loss,
                    "latest_geo_loss": avg_geo_loss,
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
        print(f"[align-marsclip] saved checkpoint: {checkpoint_path}")
        if summary.get("wandb_mode") != "disabled" and summary.get("wandb_run_dir") is not None:
            print(f"[align-marsclip] saved W&B run: {summary['wandb_run_dir']}")

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
                _save_marsclip_checkpoint(
                    checkpoints_dir / "interrupted_checkpoint.pt",
                    config=config,
                    history=history,
                    aligner=aligner,
                    text_encoder=text_encoder,
                    mae_encoder=mae_encoder,
                    args=args,
                    image_dim=image_dim,
                    text_dim=text_dim,
                    geo_dim=int(geo_dim),
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
