# Coordinate system

## The Mars CRS hub

All MarsRecon datasets expose a single common CRS:

```
+proj=longlat +a=3396190 +b=3376200
```

This is the **IAU 2000 Mars geographic CRS** — ellipsoidal latitude/longitude on the IAU 2000
reference ellipsoid (semi-major 3 396 190 m, semi-minor 3 376 200 m). Longitudes are normalized
to `[-180°, 180°]` (positive east).

## Per-observation projections

Each individual JP2 or IMG raster uses its **own per-observation Equirectangular projection**
centred on the local target. Reprojection into the common hub is handled automatically inside
`__getitem__` via `rasterio.warp.transform_bounds`, so calling code never needs to deal with
per-observation CRSs.

## Why a geographic hub?

- Strips span large enough swaths that any single projected CRS introduces unacceptable
  distortion.
- A geographic hub lets `HiRISEGeoSampler` reason about strip polygons in a single coordinate
  system across the planet.
- The cost — non-uniform metric pixel sizes near the poles — is acceptable because HiRISE
  coverage at high latitudes is sparse.

## Patch units

When using `HiRISEGeoSampler`, the `size` parameter is in CRS units:

| `units`      | Interpretation                                |
|--------------|-----------------------------------------------|
| `Units.CRS`  | Size in degrees (geographic hub).             |
| `Units.PIX`  | Size in pixels of the source raster.          |

A typical training patch uses `size=0.018` CRS units (~1 km at the equator).
