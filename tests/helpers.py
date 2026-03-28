"""Shared test helpers (not pytest fixtures) for MarsRecon tests."""

import pathlib
import sys
import types

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def make_mock_dataset(
        geometries: list[Polygon],
        crs: CRS,
        res: tuple[float, float] = (8.44e-6, 8.44e-6),
) -> types.SimpleNamespace:
    """Build a minimal object satisfying the GeoSampler interface.

    Args:
        geometries: Shapely geometries for the index.
        crs: CRS for the GeoDataFrame.
        res: ``(xres, yres)`` resolution in CRS units.

    Returns:
        A namespace with ``.index``, ``.res``, and ``.bounds``.
    """
    n = len(geometries)
    t_starts = pd.to_datetime(["2006-01-01T00:00:00"] * n, utc=True)
    t_stops = pd.to_datetime(["2006-01-02T00:00:00"] * n, utc=True)
    interval_index = pd.IntervalIndex.from_arrays(
        t_starts, t_stops, closed="both", name="datetime"
    )
    gdf = gpd.GeoDataFrame(
        {"obs_id": [f"obs_{i}" for i in range(n)]},
        geometry=geometries,
        crs=crs,
        index=interval_index,
    )

    xmin, ymin, xmax, ymax = gdf.total_bounds
    xres, yres = res
    tmin = t_starts.min()
    tmax = t_stops.max()

    ds = types.SimpleNamespace()
    ds.index = gdf
    ds.res = res
    ds.bounds = (
        slice(xmin, xmax, xres),
        slice(ymin, ymax, yres),
        slice(tmin, tmax, 1),
    )
    return ds
