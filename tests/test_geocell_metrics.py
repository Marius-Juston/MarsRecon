"""Geocell softness metrics for image→geo retrieval."""

from __future__ import annotations

import torch

from stage_b.align_marsclip import compute_geocell_image_to_geo_metrics


def test_geocell_identity_map_perfect_top1() -> None:
    n = 8
    torch.manual_seed(0)
    img = torch.randn(n, 16)
    geo = torch.randn(n, 16)
    img = img / img.norm(dim=-1, keepdim=True)
    geo = geo / geo.norm(dim=-1, keepdim=True)
    lat_lon = torch.randn(n, 2) * 10.0
    m = compute_geocell_image_to_geo_metrics(
        img, geo, lat_lon, cell_degs=(5.0,), ks=(1,)
    )
    k = "image_to_geo_geocell_5p0deg_top1_any_neighbor"
    assert k in m
    assert m[k] <= 1.0
    assert m[k] >= 0.0


def test_geocell_all_same_cell_equals_one() -> None:
    n = 4
    image_emb = torch.eye(n)
    geo_emb = torch.eye(n)
    lat_lon = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    m = compute_geocell_image_to_geo_metrics(
        image_emb,
        geo_emb,
        lat_lon,
        cell_degs=(360.0,),  # entire planet one cell
        ks=(1,),
    )
    k = "image_to_geo_geocell_360p0deg_top1_any_neighbor"
    assert m[k] == 1.0
