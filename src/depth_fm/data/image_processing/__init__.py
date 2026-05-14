"""Image-processing helpers used by the DepthFM HiRISE adapter.

Originally lived in `depth_fm/data/adapter.py` (1908 LOC, mostly these
helpers). Extracted into focused submodules so the adapter file itself stays
small and each algorithm is locatable.

Public API:

* `mask_ops`        — `erode_valid_mask`
* `void_filling`    — `fill_invalid_nearest_neighbor`, `fill_voids_kriging`,
                      `fill_dtm_smart_diffusion`, `fill_voids_gmrf`
* `seam_detection`  — `SeamResult`, `detect_seam_artifact`, `is_tin_artifact`,
                      `compute_piecewise_linearity`, `compute_artifact_multipliers`,
                      `compute_spatial_isolation`
* `sun_vector`      — `estimate_sun_vector_ols`, `estimate_sun_vector_irls`
* `terrain`         — `compute_topographic_residual`
"""

from depth_fm.data.image_processing.mask_ops import erode_valid_mask
from depth_fm.data.image_processing.void_filling import (
    fill_invalid_nearest_neighbor,
    fill_voids_kriging,
    fill_dtm_smart_diffusion,
    fill_voids_gmrf,
)
from depth_fm.data.image_processing.seam_detection import (
    SeamResult,
    detect_seam_artifact,
    is_tin_artifact,
    compute_piecewise_linearity,
    compute_artifact_multipliers,
    compute_spatial_isolation,
)
from depth_fm.data.image_processing.sun_vector import (
    estimate_sun_vector_ols,
    estimate_sun_vector_irls,
)
from depth_fm.data.image_processing.terrain import compute_topographic_residual

__all__ = [
    "erode_valid_mask",
    "fill_invalid_nearest_neighbor",
    "fill_voids_kriging",
    "fill_dtm_smart_diffusion",
    "fill_voids_gmrf",
    "SeamResult",
    "detect_seam_artifact",
    "is_tin_artifact",
    "compute_piecewise_linearity",
    "compute_artifact_multipliers",
    "compute_spatial_isolation",
    "estimate_sun_vector_ols",
    "estimate_sun_vector_irls",
    "compute_topographic_residual",
]
