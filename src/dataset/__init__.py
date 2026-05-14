"""MarsRecon dataset package — HiRISE imagery and DTM datasets.

Public API re-exports the user-facing classes so callers can write:

    from dataset import MarsHiRISE, MarsHiRISEDTM, HiRISEGeoSampler

Internal modules are organized into subpackages:

* `core/`           — dataset classes (base, rdr, dtm)
* `sampling/`       — geo-sampler and patch-packing geometry
* `preprocessing/`  — JP2/IMG → COG conversion
* `stats/`          — dataset statistics (per-channel, histograms)
* `validation/`     — diagnostic visualizations for sampler/coverage
"""

from dataset.core.base import MarsHiRISEBase, ProductMeta, MARS_PROJECTED_CRS
from dataset.core.rdr import MarsHiRISE
from dataset.core.dtm import MarsHiRISEDTM
from dataset.sampling.sampler import HiRISEGeoSampler

__all__ = [
    "MarsHiRISEBase",
    "MarsHiRISE",
    "MarsHiRISEDTM",
    "HiRISEGeoSampler",
    "ProductMeta",
    "MARS_PROJECTED_CRS",
]
