"""I/O helpers shared by visualization and analysis scripts.

Functions:
* `load_geotiff(path)`  — load a (multi-band) GeoTIFF as a NumPy array.
* `load_json(path)`     — read JSON, return dict.
* `load_csv(path)`      — read a CSV with a header, return list[dict].
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_geotiff(path: str | Path) -> np.ndarray:
    """Load a GeoTIFF/TIFF as a NumPy array.

    Tries rasterio first, falls back to tifffile, then PIL.
    Single-band rasters are returned as `(H, W)`; multi-band as `(C, H, W)`.
    """
    try:
        import rasterio
        with rasterio.open(str(path)) as src:
            data = src.read()
            return data[0] if data.shape[0] == 1 else data
    except ImportError:
        pass
    try:
        import tifffile
        return tifffile.imread(str(path))
    except ImportError:
        pass
    from PIL import Image
    return np.array(Image.open(str(path)))


def load_json(path: str | Path) -> Any:
    """Read a JSON file and return the parsed object."""
    with open(path, "r") as f:
        return json.load(f)


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file with a header row, return a list of dicts."""
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))
