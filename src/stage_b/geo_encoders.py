"""Location encoders for Stage B1a-geo+.

This module vendors and composes a small set of published location encoder
building blocks so we can swap geo encoders by name from the trainer CLI
without pulling in the upstream packages (which carry pretrained-weight
wheels, Lightning deps, etc.).

References (all MIT-licensed):

* Sitzmann et al. "Implicit Neural Representations with Periodic Activation
  Functions" (NeurIPS 2020), arXiv:2006.09661 — defines the SIREN
  architecture used by both SatCLIP and Russwurm et al.
* Tancik et al. "Fourier Features Let Networks Learn High Frequency
  Functions in Low Dimensional Domains" (NeurIPS 2020), arXiv:2006.10739 —
  motivates random Fourier features (RFF) for low-dim coordinate inputs.
* Vivanco Cepeda et al. "GeoCLIP: Clip-Inspired Alignment between
  Locations and Images for Effective Worldwide Geo-localization"
  (NeurIPS 2023), arXiv:2309.16020. Repo:
  https://github.com/VicenteVivan/geo-clip — defines the hierarchical RFF
  capsule architecture replicated here as ``HierarchicalRFFEncoder``.
* Klemmer et al. "SatCLIP: Global, General-Purpose Location Embeddings
  with Satellite Imagery" (AAAI 2025), arXiv:2311.17179. Repo:
  https://github.com/microsoft/satclip — uses ``Siren(SH)`` location
  encoders; this module follows their factoring of
  ``LocationEncoder = posenc -> nn``.
* Rußwurm et al. "Geographic Location Encoding with Spherical Harmonics
  and Sinusoidal Representation Networks" (ICLR 2024), arXiv:2310.06743.
  Repo: https://github.com/MarcCoru/locationencoder — original ``Siren``
  PyTorch implementation, copied here essentially verbatim.

The SatCLIP / Russwurm recommendation for global-scale geographic data is
``Siren(SphericalHarmonics)``. At our Mars-Olympus bbox (~12° × 32°),
spherical-harmonic basis functions are overkill, so the recipe we surface
by default is ``Siren(HierarchicalRFF)`` which matches GeoCLIP and is the
cheapest way to defeat MLP spectral bias on raw lat/lon. ``Siren(SH)``
remains available behind ``--geo-encoder-type sh_siren`` for completeness
and Mars-spherical ablation studies.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


__all__ = [
    "Sine",
    "Siren",
    "SirenNet",
    "GaussianRFF",
    "HierarchicalRFF",
    "SphericalHarmonics",
    "MLPHead",
    "LocationEncoder",
    "build_location_encoder",
    "GEO_ENCODER_CHOICES",
]


# ---------------------------------------------------------------------------
# SIREN (Sitzmann et al. 2020; vendored from MarcCoru/locationencoder)
# ---------------------------------------------------------------------------


class Sine(nn.Module):
    """Sinusoidal activation ``sin(w0 * x)`` used by SIREN."""

    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    """Single SIREN linear layer with the canonical Sitzmann-2020 init."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)
        self.is_first = bool(is_first)
        weight = torch.empty(dim_out, dim_in)
        bias = torch.empty(dim_out) if use_bias else None
        w_std = (1.0 / dim_in) if is_first else (math.sqrt(c / dim_in) / w0)
        nn.init.uniform_(weight, -w_std, w_std)
        if bias is not None:
            nn.init.uniform_(bias, -w_std, w_std)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(torch.nn.functional.linear(x, self.weight, self.bias))


class SirenNet(nn.Module):
    """Stacked SIREN MLP. Mirrors Russwurm et al. 2024 / Sitzmann et al. 2020."""

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        *,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("SirenNet requires num_layers >= 1.")
        layers: list[nn.Module] = []
        prev = int(dim_in)
        for ind in range(int(num_layers)):
            is_first = ind == 0
            layer_w0 = float(w0_initial) if is_first else float(w0)
            layers.append(
                Siren(
                    prev,
                    int(dim_hidden),
                    w0=layer_w0,
                    is_first=is_first,
                    use_bias=use_bias,
                )
            )
            prev = int(dim_hidden)
        layers.append(
            Siren(
                prev,
                int(dim_out),
                w0=float(w0),
                use_bias=use_bias,
                activation=nn.Identity(),
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Random Fourier Features (Tancik et al. 2020; GeoCLIP capsule)
# ---------------------------------------------------------------------------


class GaussianRFF(nn.Module):
    """Random Fourier Features ``[cos(2π Bx), sin(2π Bx)]`` with fixed Gaussian B."""

    def __init__(
        self,
        input_dim: int,
        encoded_size: int,
        sigma: float,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        b = torch.normal(0.0, float(sigma), size=(int(encoded_size), int(input_dim)), generator=gen)
        self.register_buffer("b", b)

    @property
    def out_dim(self) -> int:
        return int(self.b.shape[0]) * 2

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_proj = 2.0 * math.pi * v @ self.b.t()
        return torch.cat([torch.cos(v_proj), torch.sin(v_proj)], dim=-1)


class HierarchicalRFF(nn.Module):
    """Concatenates ``GaussianRFF`` blocks at multiple sigmas (GeoCLIP-style)."""

    def __init__(
        self,
        input_dim: int,
        encoded_size: int,
        sigmas: Sequence[float],
        *,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        if not sigmas:
            raise ValueError("HierarchicalRFF requires at least one sigma value.")
        self.blocks = nn.ModuleList(
            [
                GaussianRFF(
                    input_dim=int(input_dim),
                    encoded_size=int(encoded_size),
                    sigma=float(sigma),
                    seed=None if seed is None else int(seed) + i,
                )
                for i, sigma in enumerate(sigmas)
            ]
        )
        self.sigmas = tuple(float(s) for s in sigmas)

    @property
    def out_dim(self) -> int:
        return sum(int(block.out_dim) for block in self.blocks)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(v) for block in self.blocks], dim=-1)


# ---------------------------------------------------------------------------
# Spherical harmonics (low-degree analytic; SatCLIP / Russwurm style)
# ---------------------------------------------------------------------------


class SphericalHarmonics(nn.Module):
    """Real spherical harmonics up to degree ``L`` evaluated analytically.

    Inputs are ``(lat_deg, lon_deg)``. The block emits ``(L+1)**2`` features.
    Implementation uses the standard real-SH formulae for low ``L``; we cap
    ``L`` at 10 to stay closed-form and match SatCLIP's default. Following
    Russwurm et al. (2024), spherical harmonics are well-behaved on the
    sphere (no polar artefacts) and are the recommended PE for global data.
    """

    def __init__(self, legendre_polys: int = 10) -> None:
        super().__init__()
        L = int(legendre_polys)
        if L < 0 or L > 10:
            raise ValueError("SphericalHarmonics supports legendre_polys in [0, 10].")
        self.L = L

    @property
    def out_dim(self) -> int:
        return (self.L + 1) ** 2

    def forward(self, lat_lon_deg: torch.Tensor) -> torch.Tensor:
        if lat_lon_deg.shape[-1] != 2:
            raise ValueError("SphericalHarmonics expects (lat_deg, lon_deg) inputs.")
        lat_rad = torch.deg2rad(lat_lon_deg[..., 0])
        lon_rad = torch.deg2rad(lat_lon_deg[..., 1])
        theta = math.pi / 2.0 - lat_rad
        phi = lon_rad
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        feats: list[torch.Tensor] = []
        for L in range(self.L + 1):
            for m in range(-L, L + 1):
                feats.append(
                    _real_spherical_harmonic(L, m, cos_t, sin_t, phi)
                )
        return torch.stack(feats, dim=-1)


def _real_spherical_harmonic(
    L: int,
    m: int,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Real spherical harmonic Y_{L,m}(theta, phi) for L in [0, 10].

    Uses condensed closed-form expressions for low ``L``; for higher orders
    the formulae below grow long but remain tractable and stable. Output is
    a tensor matching the broadcast shape of the angular inputs.
    """
    pi = math.pi
    sq = math.sqrt
    if L == 0:
        return torch.full_like(cos_t, 0.5 * sq(1.0 / pi))
    if L == 1:
        if m == -1:
            return sq(3.0 / (4.0 * pi)) * sin_t * torch.sin(phi)
        if m == 0:
            return sq(3.0 / (4.0 * pi)) * cos_t
        if m == 1:
            return sq(3.0 / (4.0 * pi)) * sin_t * torch.cos(phi)
    if L == 2:
        if m == -2:
            return 0.25 * sq(15.0 / pi) * sin_t**2 * torch.sin(2 * phi)
        if m == -1:
            return 0.5 * sq(15.0 / pi) * sin_t * cos_t * torch.sin(phi)
        if m == 0:
            return 0.25 * sq(5.0 / pi) * (3 * cos_t**2 - 1)
        if m == 1:
            return 0.5 * sq(15.0 / pi) * sin_t * cos_t * torch.cos(phi)
        if m == 2:
            return 0.25 * sq(15.0 / pi) * sin_t**2 * torch.cos(2 * phi)
    P = _associated_legendre(L, m, cos_t)
    norm = math.sqrt(
        (2 * L + 1) / (4 * pi) * math.factorial(L - abs(m)) / math.factorial(L + abs(m))
    )
    if m == 0:
        return norm * P
    if m > 0:
        return math.sqrt(2.0) * norm * P * torch.cos(m * phi)
    return math.sqrt(2.0) * norm * P * torch.sin(abs(m) * phi)


def _associated_legendre(L: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """Associated Legendre function P_L^|m|(x) computed by the recurrence."""
    m = abs(int(m))
    if m > L:
        return torch.zeros_like(x)
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))
        fact = 1.0
        for _ in range(m):
            pmm = -pmm * fact * somx2
            fact += 2.0
    if L == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if L == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, L + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


# ---------------------------------------------------------------------------
# Heads (MLP / SIREN) used after the positional encoding
# ---------------------------------------------------------------------------


class MLPHead(nn.Module):
    """ReLU MLP with optional dropout used as the post-PE backbone."""

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        *,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(depth) < 1:
            raise ValueError("MLPHead requires depth >= 1.")
        layers: list[nn.Module] = []
        prev = int(dim_in)
        for _ in range(int(depth) - 1):
            layers.append(nn.Linear(prev, int(dim_hidden)))
            layers.append(nn.ReLU(inplace=True))
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(dim_hidden)
        layers.append(nn.Linear(prev, int(dim_out)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Composite location encoder (PE + head, optional auxiliary feature concat)
# ---------------------------------------------------------------------------


class LocationEncoder(nn.Module):
    """``LocationEncoder(coords) = head(PE(coords) || aux)``.

    ``coords`` must be ``(B, 2)`` lat/lon (degrees). When ``aux_dim > 0`` the
    forward call expects a separate ``aux`` tensor of shape ``(B, aux_dim)``
    that is concatenated with the positional encoding before the head, so we
    can mix lat/lon with viewing/scale features without pushing them through
    the positional encoder.
    """

    def __init__(
        self,
        posenc: nn.Module,
        head: nn.Module,
        *,
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        self.posenc = posenc
        self.head = head
        self.aux_dim = int(aux_dim)

    def forward(self, coords: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        x = self.posenc(coords)
        if self.aux_dim > 0:
            if aux is None:
                raise ValueError(
                    f"LocationEncoder configured with aux_dim={self.aux_dim} but aux is None."
                )
            x = torch.cat([x, aux], dim=-1)
        return self.head(x)


GEO_ENCODER_CHOICES: tuple[str, ...] = (
    "mlp",
    "rff_mlp",
    "rff_siren",
    "sh_siren",
    "sh_linear",
)


def build_location_encoder(
    *,
    encoder_type: str,
    embed_dim: int,
    aux_dim: int = 0,
    hidden_dim: int = 256,
    depth: int = 2,
    dropout: float = 0.0,
    rff_sigmas: Sequence[float] = (1.0, 4.0, 16.0, 64.0),
    rff_encoded_size: int = 128,
    siren_w0: float = 1.0,
    siren_w0_initial: float = 30.0,
    sh_legendre_polys: int = 10,
) -> LocationEncoder:
    """Factory for the ``--geo-encoder-type`` CLI choices.

    ``mlp`` / ``rff_mlp`` / ``rff_siren`` operate on raw ``(lat_deg, lon_deg)``
    after rescaling by the head; the ``sh_*`` variants use spherical harmonics
    on the same ``(lat_deg, lon_deg)`` input. ``aux_dim`` controls whether
    auxiliary scale/viewing features are concatenated post-PE.
    """
    encoder_type = str(encoder_type).lower()
    if encoder_type not in GEO_ENCODER_CHOICES:
        raise ValueError(
            f"Unknown geo encoder type '{encoder_type}'. Choose from {GEO_ENCODER_CHOICES}."
        )
    if encoder_type == "mlp":
        posenc = nn.Identity()
        pe_dim = 2
    elif encoder_type in {"rff_mlp", "rff_siren"}:
        posenc = HierarchicalRFF(
            input_dim=2,
            encoded_size=int(rff_encoded_size),
            sigmas=tuple(float(s) for s in rff_sigmas),
        )
        pe_dim = posenc.out_dim
    elif encoder_type == "sh_siren":
        posenc = SphericalHarmonics(legendre_polys=int(sh_legendre_polys))
        pe_dim = posenc.out_dim
    elif encoder_type == "sh_linear":
        posenc = SphericalHarmonics(legendre_polys=int(sh_legendre_polys))
        pe_dim = posenc.out_dim
    else:  # pragma: no cover - guarded above
        raise AssertionError(encoder_type)

    head_in = pe_dim + int(aux_dim)
    if encoder_type == "sh_linear":
        head: nn.Module = nn.Linear(head_in, int(embed_dim))
    elif encoder_type in {"rff_siren", "sh_siren"}:
        head = SirenNet(
            dim_in=head_in,
            dim_hidden=int(hidden_dim),
            dim_out=int(embed_dim),
            num_layers=int(depth),
            w0=float(siren_w0),
            w0_initial=float(siren_w0_initial),
        )
    else:
        head = MLPHead(
            dim_in=head_in,
            dim_hidden=int(hidden_dim),
            dim_out=int(embed_dim),
            depth=int(depth),
            dropout=float(dropout),
        )
    return LocationEncoder(posenc=posenc, head=head, aux_dim=int(aux_dim))
