# MarsHiRISE dataset — technical reference

A systematic documentation of the data structures, coordinate systems, projection equations, file formats, and processing pipeline for the MarsHiRISE TorchGeo dataset built on NASA PDS HiRISE Reduced Data Records.

---

## 1  Coordinate reference system for Mars

### 1.1  Ellipsoid parameters

The Mars reference ellipsoid is defined in the HiRISE RDR Software Interface Specification and encoded in every per-product LBL file (`IMAGE_MAP_PROJECTION` object). The authoritative values are:

| Parameter             | Value       | Source                                   |
|-----------------------|-------------|------------------------------------------|
| Equatorial radius (a) | 3 396 190 m | `HIRISE_RDR_SIS.PDF §3.5.1`; `DSMAP.CAT` |
| Polar radius (b)      | 3 376 200 m | `DSMAP.CAT`                              |

*Reference: [DSMAP.CAT](https://hirise-pds.lpl.arizona.edu/PDS/CATALOG/DSMAP.CAT) — "equatorial radius of 3396.190000 km and polar radius of 3376.200000 kilometers."*

Each per-product LBL file records `A_AXIS_RADIUS = B_AXIS_RADIUS = C_AXIS_RADIUS` as a **single local radius** at the observation's `CENTER_LATITUDE`, computed from the ellipsoid:

```
R = (a × b) / sqrt((b × cos(LatP))² + (a × sin(LatP))²)
```

*Reference: [DSMAP.CAT](https://hirise-pds.lpl.arizona.edu/PDS/CATALOG/DSMAP.CAT) — "The value recorded in the three radii is the local radius at the CENTER_LATITUDE on the Mars ellipsoid."*

For example, [ESP_011261_1960_COLOR.LBL](https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/ORB_011200_011299/ESP_011261_1960/ESP_011261_1960_COLOR.LBL) at `CENTER_LATITUDE = 15.000°` records `A_AXIS_RADIUS = 3394.8398133163 km`.

### 1.2  Geographic CRS definition

The dataset CRS is a Mars IAU 2000 **geographic** CRS (units: decimal degrees), defined in PROJ4 as:

```
+proj=longlat +a=3396190 +b=3376200 +no_defs
```

**Why geographic, not projected?** Each HiRISE JP2 uses its own per-observation Equirectangular projection with a unique `CENTER_LATITUDE` (differing per image). There is no single projected CRS that correctly represents all files. The geographic CRS is the natural common hub: rasterio reads each file's embedded CRS and reprojects into geographic degrees automatically.

The cumulative index bounding-box columns (`MINIMUM_LATITUDE`, `MAXIMUM_LONGITUDE`, `CORNER1_LATITUDE`, etc.) are all in decimal degrees. Assigning degree values to a projected (metre-unit) CRS silently mislabels them, causing the reprojection destination window to land at the wrong surface location.

### 1.3  Native resolution

HiRISE resolution at the equator: `MAP_RESOLUTION = 118 502.26 pix/deg` (≈ 0.25 m/pixel for RED at full resolution). The reciprocal is the dataset's scalar `res`:

```
res = 1 / 118502.26 ≈ 8.44 × 10⁻⁶ °/pixel
```

This value is used throughout the sampler and data loader to convert between degrees and pixels.

*Reference: [ESP_011261_1960_COLOR.LBL](https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/ORB_011200_011299/ESP_011261_1960/ESP_011261_1960_COLOR.LBL) — `MAP_RESOLUTION = 237004.52928064 <PIX/DEG>` (this product is at 0.25 m/pixel, so double the RED-only resolution of 118502).*

---

## 2  PDS cumulative index (RDRCUMINDEX)

### 2.1  What it represents

The cumulative index is a fixed-length ASCII table listing **every RDR product ever released** — one row per JP2 file. It is the authoritative catalogue for discovering HiRISE observations.

*Reference: [RDRCUMINDEX.LBL](https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.LBL) — 54 columns, 198 464 rows (as of the current release).*

Files:
- `RDRCUMINDEX.LBL` — PDS3 label describing the table schema (column names, byte offsets, data types).
- `RDRCUMINDEX.TAB` — the table itself (comma-delimited, fixed-length records of 821 bytes).

### 2.2  Key columns

| Column                                         | Type          | Description                                        | Used for                                                        |
|------------------------------------------------|---------------|----------------------------------------------------|-----------------------------------------------------------------|
| `PRODUCT_ID`                                   | CHARACTER(21) | Unique ID, e.g. `ESP_011261_1960_COLOR`            | Distinguish `_COLOR` vs `_RED` products; extract observation ID |
| `FILE_NAME_SPECIFICATION`                      | CHARACTER(67) | PDS-relative path to the JP2 file                  | Resolve local file paths; construct download URLs               |
| `MINIMUM_LATITUDE`                             | REAL          | Southern edge of projected image (°)               | Spatial filtering (bbox query)                                  |
| `MAXIMUM_LATITUDE`                             | REAL          | Northern edge of projected image (°)               | Spatial filtering                                               |
| `MINIMUM_LONGITUDE`                            | REAL          | Western edge (0°–360° convention)                  | Spatial filtering (after normalisation)                         |
| `MAXIMUM_LONGITUDE`                            | REAL          | Eastern edge (0°–360° convention)                  | Spatial filtering (after normalisation)                         |
| `CORNER1_LATITUDE` through `CORNER4_LONGITUDE` | REAL          | Four corners of the projected image                | **See §2.3**                                                    |
| `START_TIME` / `STOP_TIME`                     | TIME          | UTC acquisition window                             | Temporal index for TorchGeo's `IntervalIndex`                   |
| `MAP_SCALE`                                    | REAL          | Metres per pixel                                   | Resolution metadata                                             |
| `MAP_PROJECTION_TYPE`                          | CHARACTER(19) | `EQUIRECTANGULAR` or `POLAR STEREOGRAPHIC`         | CRS determination                                               |
| `NORTH_AZIMUTH`                                | REAL          | Angle from image right-edge to north (° clockwise) | Image orientation                                               |

*Reference: [RDRCUMINDEX.LBL](https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.LBL) — complete schema definition.*

### 2.3  The CORNER1–4 misconception

The RDRCUMINDEX.LBL defines each `CORNERn_LATITUDE` / `CORNERn_LONGITUDE` as:

> "Latitude/Longitude of corner N of the **projected image**."

The projected image is an Equirectangular grid with `MAP_PROJECTION_ROTATION = 0.0`. Lines run at constant latitude; samples run at constant longitude. The four pixel corners therefore always form an **axis-aligned rectangle** — algebraically identical to the `MINIMUM/MAXIMUM_LATITUDE/LONGITUDE` bounding box. They do **not** encode the rotated strip footprint.

This can be verified by applying the inverse projection equations from [DSMAP.CAT](https://hirise-pds.lpl.arizona.edu/PDS/CATALOG/DSMAP.CAT) (see §3) to the four image corners `(1,1)`, `(1,SAMPLE_LAST)`, `(LINE_LAST,1)`, `(LINE_LAST,SAMPLE_LAST)` — the resulting lat/lon values match `MINIMUM/MAXIMUM` to floating-point precision.

**Consequence for sampling:** Any geometry built from CORNER1–4 is a bounding box, not a strip polygon. The `HiRISEGeoSampler`'s intersection test against it provides zero waste reduction — it accepts every candidate centre inside the bbox, exactly as a naive `RandomGeoSampler` would.

### 2.4  Longitude normalisation

PDS uses the [0°, 360°] east-positive longitude convention. The dataset normalises to [−180°, 180°]:

```
lon_normalised = ((lon_pds + 180) mod 360) − 180
```

**Justification:** TorchGeo and Shapely expect signed longitudes. Without normalisation, an observation at 350°E (= −10°E) would appear on the wrong side of the prime meridian, causing spatial queries to fail and bounding-box overlap tests to produce false negatives.

**Antimeridian guard:** After normalisation, if `lon_min > lon_max`, the observation straddles the antimeridian and is skipped (no HiRISE observations in the Olympus Mons region are affected).

---

## 3  Equirectangular projection equations

Every HiRISE RDR product (latitude range −65° to +65°) uses the Equirectangular projection. The equations relating map coordinates `(x, y)` in metres to geographic coordinates `(Lat, Lon)` are:

```
x = R × (Lon − LonP) × cos(LatP)
y = R × Lat
```

The conversion from pixel coordinates `(Sample, Line)` to `(x, y)`:

```
x = (Sample − S0 − 1) × Scale
y = (1 − L0 − Line) × Scale
```

Therefore the full pixel-to-geographic equations are:

```
Lat = ((1 − L0 − Line) × Scale / R) × (180 / π)
Lon = LonP + ((Sample − S0 − 1) × Scale / (R × cos(LatP))) × (180 / π)
```

| Symbol  | LBL keyword                | Meaning                                                |
|---------|----------------------------|--------------------------------------------------------|
| `LonP`  | `CENTER_LONGITUDE`         | Centre longitude of projection (always 180.0°)         |
| `LatP`  | `CENTER_LATITUDE`          | Centre latitude of projection (varies per observation) |
| `L0`    | `LINE_PROJECTION_OFFSET`   | Line offset of projection origin from pixel (1,1)      |
| `S0`    | `SAMPLE_PROJECTION_OFFSET` | Sample offset of projection origin from pixel (1,1)    |
| `Scale` | `MAP_SCALE`                | Map scale in metres/pixel                              |
| `R`     | `A_AXIS_RADIUS`            | Local ellipsoid radius at `LatP` (km; convert to m)    |

*Reference: [DSMAP.CAT](https://hirise-pds.lpl.arizona.edu/PDS/CATALOG/DSMAP.CAT) — full derivation with worked equations.*

**Why CENTER_LATITUDE varies per observation:** The Equirectangular projection's scale is true only at `LatP`. HiRISE sets `LatP` to a standard latitude band (e.g. 0°, 5°, 10°, 15°, ...) close to the observation to minimise scale distortion. Each product therefore has a slightly different projection, which is why there is no single projected CRS for the whole dataset.

---

## 4  Per-product LBL files

### 4.1  Structure

Each JP2 has a detached PDS3 label (same stem, `.LBL` extension) containing all metadata needed to interpret the image data. The label is a plain-text key=value file.

*Reference: [TRA_000835_1670_COLOR.LBL](https://hirise-pds.lpl.arizona.edu/PDS/RDR/TRA/ORB_000800_000899/TRA_000835_1670/TRA_000835_1670_COLOR.LBL)*

### 4.2  Radiometric calibration constants

The `IMAGE` object in each LBL provides the constants for converting raw DN (digital number) values to physical I/F (radiance factor):

```
I/F = DN × SCALING_FACTOR + OFFSET
```

| LBL keyword                 | Typical value             | Meaning                                           |
|-----------------------------|---------------------------|---------------------------------------------------|
| `SCALING_FACTOR`            | 1.24 × 10⁻⁴ – 2.38 × 10⁻⁴ | Multiplicative calibration factor                 |
| `OFFSET`                    | 0.030 – 0.038             | Additive offset                                   |
| `SAMPLE_BITS`               | 16                        | Bits per pixel (stored as `MSB_UNSIGNED_INTEGER`) |
| `SAMPLE_BIT_MASK`           | `2#0000001111111111#`     | Effective 10-bit data (0–1023)                    |
| `CORE_NULL`                 | 0                         | DN value representing nodata                      |
| `CORE_LOW_REPR_SATURATION`  | 1                         | Lowest valid DN                                   |
| `CORE_HIGH_REPR_SATURATION` | 1023                      | Highest valid DN                                  |

The calibrated I/F is clipped to `[0, 1]`.

**Critical: nodata handling.** `CORE_NULL = 0` means DN = 0 is the designated fill value. After the `reproject()` call (which sets `dst_nodata=0.0`), a nodata mask must be saved **before** applying the calibration arithmetic, then restored **after** — otherwise the offset transforms nodata from 0.0 to ~0.038, contaminating merge logic and percentile stretches.

### 4.3  Band structure

| LBL keyword                | Example value                            | Meaning               |
|----------------------------|------------------------------------------|-----------------------|
| `BANDS`                    | 3 (COLOR) or 1 (RED)                     | Number of image bands |
| `FILTER_NAME`              | `("NEAR-INFRARED", "RED", "BLUE-GREEN")` | Band assignment       |
| `CENTER_FILTER_WAVELENGTH` | `(900, 700, 500)` nm                     | Wavelength centres    |
| `BAND_STORAGE_TYPE`        | `BAND_SEQUENTIAL`                        | BSQ interleaving      |

**Channel selection logic:** When only `"RED"` is requested, the `_RED.JP2` file is preferred (higher fidelity — more TDI lines). When any colour channel (NIR or BG) is requested, the `_COLOR.JP2` is used. The `FILTER_NAME` tuple in the LBL determines the band-index mapping; the fallback `_COLOR_BAND` dictionary (`NIR→1, RED→2, BG→3`) is used when the LBL cannot be parsed.

### 4.4  CCD configuration

The `INSTRUMENT_SETTING_PARAMETERS` group records which CCDs were active and their acquisition settings:

| Keyword        | Meaning                                                          |
|----------------|------------------------------------------------------------------|
| `MRO:CCD_FLAG` | 14-element array (ON/OFF for each CCD)                           |
| `MRO:BINNING`  | Per-CCD pixel binning factor (1, 2, or 4; −9998 = inactive)      |
| `MRO:TDI`      | Time-delay integration stages (32, 64, or 128; −9998 = inactive) |

The `SOURCE_PRODUCT_ID` list names the specific EDR channel products that were mosaicked into the RDR. For example, `ESP_011261_1960_COLOR` uses `BG12_0, BG12_1, RED4_0, RED4_1, IR10_0, IR10_1, BG13_0, BG13_1, RED5_0, RED5_1, IR11_0, IR11_1` — 6 CCD pairs (12 EDR halves). These CCDs are physically staggered on the HiRISE focal plane in both the cross-track and along-track directions, producing an irregular data footprint within the axis-aligned projected image rectangle.

---

## 5  JP2 files — JPEG2000 image products

### 5.1  What they are

HiRISE RDR products are stored as JPEG2000 Part-1 (ISO/IEC 15444-1:2004) compressed images. Each JP2 is a stand-alone image file with embedded coordinate reference system metadata (GeoJP2 or GMLJP2).

*Reference: [AAREADME.TXT](https://hirise-pds.lpl.arizona.edu/PDS/AAREADME.TXT) — "The image is organized according to the JPEG2000 JP2 file format standard."*

### 5.2  Data characteristics

| Property                | COLOR products                             | RED products                                  |
|-------------------------|--------------------------------------------|-----------------------------------------------|
| Bands                   | 3 (NIR, RED, BG)                           | 1 (RED)                                       |
| Bit depth               | 16-bit unsigned (10-bit effective)         | 16-bit unsigned (10-bit effective)            |
| Typical dimensions      | 2 000–18 000 samples × 5 000–110 000 lines | 20 000–70 000 samples × 50 000–200 000+ lines |
| Typical compressed size | 50 MB – 2.5 GB                             | 200 MB – 5 GB                                 |
| Compression ratio       | ~3–6×                                      | ~3–6×                                         |

### 5.3  Random-access performance problem

JPEG2000 uses a wavelet-based codec with large codeblocks. Decompressing a small spatial window requires decoding entire codeblocks that overlap it — for HiRISE's large images, this means decompressing hundreds of megabytes to read a single 593×593 pixel patch. During ML training, where thousands of random patches are loaded per epoch, this makes JP2 files 10–100× slower than tiled formats.

---

## 6  Cloud-Optimised GeoTIFF (COG) conversion

### 6.1  Why COG

A Cloud-Optimised GeoTIFF stores pixel data in internal 512×512 tiles with built-in overviews. Rasterio can decompress only the tiles that overlap a query window, enabling O(patch_area) random-access reads instead of O(image_area) full-codeblock decodes.

### 6.2  Conversion pipeline (`preprocessing.py`)

The conversion proceeds in two passes:

1. **Intermediate GeoTIFF:** Write the JP2 data to a temporary tiled GeoTIFF (no compression — compression is the bottleneck, and this file is immediately discarded). Build overviews at levels `[2, 4, 8, 16]` using average resampling.

2. **Final COG:** Copy the intermediate file to the output path with `copy_src_overviews=True`, applying deflate compression with horizontal-differencing predictor.

### 6.3  COG creation options

```python
_COG_CREATION_OPTIONS = {
    "driver": "GTiff",
    "compress": "deflate",
    "predictor": 2,          # horizontal differencing
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "copy_src_overviews": True,
    "bigtiff": "IF_SAFER",   # 64-bit offsets when >4 GiB
}
```

**Why GeoTIFF and not another format:**
- Native rasterio/GDAL support with zero additional dependencies.
- Internal tiling is the critical feature — it enables windowed reads.
- `IF_SAFER` activates BigTIFF (64-bit offsets) only when the estimated output exceeds the Classic TIFF 4 GiB limit. Some HiRISE COLOR products decompress to >4 GiB; without BigTIFF, writes silently corrupt past the 4 GiB boundary (`TIFFAppendToStrip: Maximum TIFF file size exceeded`).

### 6.4  COG sidecar convention

The COG is written alongside the source JP2 with the same stem and a `.tif` extension. The `_prefer_cog()` method transparently returns the `.tif` path when it exists, falling back to the `.JP2`. This means COG conversion is optional — the dataset works with raw JP2 files, just slower.

### 6.5  Memory management for batch conversion

Each worker fully decompresses one JP2 at a time. `_safe_worker_count()` estimates per-worker memory as `5× on-disk JP2 size` (conservative upper bound for JPEG2000 decompression) and caps the worker count so total usage stays within 75% of available RAM. `max_tasks_per_child=1` recycles worker processes after each conversion, forcing the OS to reclaim heap memory that glibc's `malloc` would otherwise retain.

---

## 7  Spatial index and caching

### 7.1  What the index stores

The spatial index is a `GeoDataFrame` with one row per unique observation, indexed by a `pd.IntervalIndex` of `[START_TIME, STOP_TIME]` intervals. Columns:

| Column       | Type        | Content                                 |
|--------------|-------------|-----------------------------------------|
| `obs_id`     | str         | Observation ID (e.g. `ESP_011261_1960`) |
| `color_path` | str or None | Local path to the `_COLOR.JP2` file     |
| `red_path`   | str or None | Local path to the `_RED.JP2` file       |
| `geometry`   | Polygon     | Strip footprint in geographic CRS       |

### 7.2  Geometry resolution priority

The geometry for each observation is resolved in priority order:

| Case | Source                                                | Shape                    | When used                                 |
|------|-------------------------------------------------------|--------------------------|-------------------------------------------|
| A    | Convex hull of non-zero pixels from JP2/COG           | True strip parallelogram | JP2 file exists and is readable           |
| B    | JP2 bounding box (from `rasterio.transform_bounds`)   | Axis-aligned rectangle   | JP2 exists but footprint extraction fails |
| C    | Cumulative index `MINIMUM/MAXIMUM_LATITUDE/LONGITUDE` | Axis-aligned rectangle   | JP2 not yet downloaded                    |

The footprint extraction (Case A) reads band 1 at the coarsest available overview level (typically 16× reduction), computes the convex hull of non-zero pixel coordinates in pixel space, then reprojects only the hull vertices (typically 4–20 points) to the geographic CRS. This captures the actual data boundary created by CCD stagger without requiring knowledge of the focal plane geometry.

### 7.3  GeoPackage cache

The spatial index is cached as a GeoPackage file (`spatial_cache{suffix}_v3.gpkg`). The suffix encodes any `target`/`bbox` filters. The temporal index is stored as `t_start`/`t_stop` string columns and reconstructed into an `IntervalIndex` on load. The cache version is bumped when the geometry resolution logic changes (v2 used corner polygons; v3 uses pixel-derived footprints).

**Legacy cache detection:** On load, the first 5 geometries are checked against `box(*g.bounds)`. If all match (axis-aligned rectangles), the cache is classified as legacy and rebuilt.

### 7.4  Spatial tolerance

A tolerance of `_SPATIAL_TOL = 1e-5°` (≈ 0.6 m on Mars — well below the 0.25 m pixel size) is applied to the JP2 bounds early-exit check in `_load_from_jp2`. This absorbs floating-point rounding between the index geometry (computed at construction time) and the bounds rasterio recomputes at load time.

---

## 8  Radiometric calibration pipeline

### 8.1  Calibration equation

```
I/F = DN × SCALING_FACTOR + OFFSET
```

Where `I/F` is the radiance factor (reflectance), `DN` is the raw digital number (0–1023 for 10-bit data), and `SCALING_FACTOR` / `OFFSET` are per-product constants from the LBL file.

*Reference: every LBL file's IMAGE object — "The conversion from DN to I/F (intensity/flux) is: I/F = (DN * SCALING_FACTOR) + OFFSET. I/F is defined as the ratio of the observed radiance and the radiance of a 100% lambertian reflector with the sun and camera orthogonal to the observing surface."*

### 8.2  Typical value ranges

For a DN range of 3–1021 (excluding null/saturation) and typical calibration constants:

```
SCALING_FACTOR ≈ 1.24 × 10⁻⁴
OFFSET         ≈ 0.033

I/F_min ≈ 3 × 1.24e-4 + 0.033 ≈ 0.034
I/F_max ≈ 1021 × 1.24e-4 + 0.033 ≈ 0.160
```

Mars surface I/F values are physically low (the surface is dark). A fully calibrated image has pixel values roughly in [0.03, 0.20] — requiring a percentile stretch for display.

### 8.3  Nodata mask requirement

The `reproject()` call sets `dst_nodata=0.0`, so pixels outside the source image's coverage receive value 0.0. The calibration arithmetic then transforms these to `0.0 × SCALING_FACTOR + OFFSET = OFFSET ≈ 0.038`, destroying the zero sentinel.

**Correct sequence:**
```python
nodata_mask = (dest == 0.0)       # save before calibration
dest *= meta.scaling_factor
dest += meta.offset
np.clip(dest, 0.0, 1.0, out=dest)
dest[nodata_mask] = 0.0           # restore nodata
```

Without this, three downstream systems break:

1. **`_merge_tiles`** uses `merged == 0.0` to detect empty pixels. Contaminated nodata (0.038 ≠ 0.0) makes merge think all pixels have data — later tiles can never fill gaps.

2. **`plot()` percentile stretch** uses `band > 0` to select data pixels. Contaminated nodata (0.038 > 0) includes the entire image in the percentile calculation. If 90%+ of pixels are nodata-at-0.038, then `p2 ≈ p98 ≈ 0.038` and no stretch is applied — raw I/F values [0, 0.28] render as near-black.

3. **Any threshold-based data-vs-nodata logic** (overlap computation, coverage statistics) is corrupted.

### 8.4  Default fallback constants

When the LBL file is missing or unparseable, the dataset falls back to representative default values:

```python
_DEFAULT_SCALING_FACTOR = 2.37936949017414e-04
_DEFAULT_OFFSET         = 0.037954361744101
_DEFAULT_SAMPLE_BITS    = 16
_EFFECTIVE_BIT_DEPTH    = 10  # from SAMPLE_BIT_MASK
```

---

## 9  HiRISEGeoSampler — strip-aware sampling

### 9.1  Problem: bounding-box waste

HiRISE strips are long, narrow swaths that run at a slight angle within their axis-aligned projected image. Sampling uniformly from the bounding box wastes patches that fall in the nodata corners. The waste fraction depends on the strip's rotation and aspect ratio; for typical HiRISE geometry it ranges from 20% to 60%.

### 9.2  Solution: polygon-intersection grid

At construction time, the sampler:

1. For each strip polygon in `dataset.index`:
   a. Apply a 5% edge inset (`buffer(-inset)`) to the polygon, absorbing floating-point boundary mismatches between the index geometry and rasterio's recomputed bounds at load time.
   b. Generate a regular grid of candidate centres within the polygon's bounding box, inset by half the patch size so every patch fits within the bbox.
   c. For each candidate, construct the patch rectangle and test `effective.intersects(patch)`.
   d. Keep only centres that pass.

2. Store all valid centres as `(cx, cy, pd.Interval)` tuples.

At each epoch, `__iter__` draws `length` indices uniformly at random from the valid set and yields `(x_slice, y_slice, t_slice)` tuples.

### 9.3  Why `intersects` and not `contains`

Using `contains` would require the entire patch to fall within the strip polygon, discarding a significant fringe of valid data along every diagonal strip edge. `intersects` allows patches that partially overlap the strip, which is the correct trade-off for training — some nodata pixels at patch edges are acceptable, but excluding all edge data is not. The `min_overlap` parameter (when present) provides a tunable threshold.

### 9.4  Why sampling fails — the CORNER polygon problem

When the spatial index uses CORNER1–4 from the cumulative index (which are axis-aligned bbox corners, not strip corners — see §2.3), the polygon is identical to the bounding box. The `intersects` test accepts every candidate, providing zero waste reduction. Patches in the bbox corners contain mostly nodata because the actual data strip doesn't extend there.

**Fix:** The spatial index must use actual strip footprints extracted from the JP2 pixel data (convex hull of non-zero pixels at reduced resolution — see §7.2, Case A).

### 9.5  Nodata sources within valid patches

Even with correct strip footprints, some patches near the strip edge will contain nodata due to:

1. **CCD stagger:** The COLOR product is mosaicked from 6 CCD pairs (IR10/11, RED4/5, BG12/13) that are physically offset on the focal plane. The union has irregular edges — not a clean parallelogram. The convex hull slightly overestimates the data boundary near stagger transitions.

2. **Convex hull approximation:** The true data boundary is concave at CCD stagger transitions; the convex hull envelopes these concavities.

3. **Bilinear resampling bleed:** The `reproject()` call with `Resampling.bilinear` creates intermediate values near nodata boundaries (partial interpolation between zero and real DNs).

---

## 10  Tile loading and reprojection

### 10.1  Load path

```
__getitem__(index)
  → spatial query against index GeoDataFrame
  → _load_tile(color_path, red_path, x, y)
    → _prefer_cog(path)           # use .tif sidecar if available
    → _load_from_jp2(path, band_map, meta, x, y)
      → rasterio.open(path)
      → early-exit overlap check   # _SPATIAL_TOL = 1e-5°
      → reproject(bilinear)        # source CRS → geographic CRS
      → calibrate (I/F)            # with nodata mask
  → _merge_tiles(tiles)            # first-non-zero-wins mosaic
```

### 10.2  Reprojection details

Each JP2/COG has its own Equirectangular CRS (embedded via GeoJP2 metadata). The reprojection is:

```python
reproject(
    source=rasterio.band(src, band_idx),
    destination=dest,               # pre-allocated float32 array
    src_transform=src.transform,
    src_crs=src.crs,                # per-observation Equirectangular
    dst_transform=dst_transform,    # from query bounds + native res
    dst_crs=MARS_GEOGRAPHIC_CRS,    # geographic lon/lat degrees
    resampling=Resampling.bilinear,
    dst_nodata=0.0,
)
```

The destination grid is defined by the query slice `(x, y)` in degrees and the dataset's native resolution. This means every loaded patch is in a consistent geographic grid regardless of which observation it came from.

### 10.3  Early-exit overlap check

Before reprojecting, the loader transforms the JP2's bounds to geographic coordinates and checks for overlap with the query window:

```python
fl, fb, fr, ft = transform_bounds(src_crs, dst_crs, *src.bounds)
fl = ((fl + 180.0) % 360.0) - 180.0  # normalise longitude
fr = ((fr + 180.0) % 360.0) - 180.0
```

The longitude normalisation is critical because HiRISE JP2s use Equirectangular with `CENTER_LONGITUDE = 180.0°`, so `transform_bounds` may return 0°–360° values. Without normalisation, the check produces false negatives (rejects patches that actually overlap).

The check applies `_SPATIAL_TOL = 1e-5°` tolerance to absorb floating-point rounding between the index geometry and the bounds rasterio recomputes at load time.

### 10.4  Tile merging

When multiple observations cover the same query window, `_merge_tiles` mosaics them with a first-non-zero-wins strategy:

```python
empty = merged[:, :h, :w] == 0.0
merged[:, :h, :w][empty] = tile[empty]
```

This depends on nodata pixels being exactly 0.0 — hence the criticality of the nodata mask fix (§8.3).

---

## 11  File layout and path resolution

### 11.1  PDS directory structure

```
<root>/
    RDRCUMINDEX.LBL
    RDRCUMINDEX.TAB
    MROHR_0001/
        DATA/
            PSP/
                ORB_001400_001499/
                    PSP_001430_1780/
                        PSP_001430_1780_COLOR.JP2
                        PSP_001430_1780_COLOR.LBL
                        PSP_001430_1780_RED.JP2
                        PSP_001430_1780_RED.LBL
```

*Reference: [AAREADME.TXT](https://hirise-pds.lpl.arizona.edu/PDS/AAREADME.TXT)*

### 11.2  Local path convention

The dataset flattens the PDS hierarchy into a single `images/` directory under `root`:

```python
def _pds_local_path(self, spec: str) -> pathlib.Path:
    return self.root / 'images' / pathlib.Path(PurePosixPath(spec.strip())).name
```

So `MROHR_0001/DATA/PSP/.../PSP_001430_1780_COLOR.JP2` becomes `<root>/images/PSP_001430_1780_COLOR.JP2`.

### 11.3  COG sidecar paths

COGs sit alongside their source JP2s:
- Source: `<root>/images/ESP_011261_1960_COLOR.JP2`
- COG: `<root>/images/ESP_011261_1960_COLOR.tif`
- LBL: `<root>/images/ESP_011261_1960_COLOR.LBL`

---

## 12  Geographic train/test split

### 12.1  Purpose

Standard random splits risk spatial data leakage — a model may see training craters immediately adjacent to its test craters. `geographic_split()` prevents this by assigning whole geographic blocks to train or test.

### 12.2  Algorithm

1. Project geometries to a planar CRS (`+proj=eqc +a=3396190 +b=3376200`) for centroid computation.
2. Extract the chosen coordinate (longitude or latitude) of each centroid.
3. Divide the coordinate range into `n_blocks = max(5, round(1/test_fraction))` equally-populated blocks using percentile edges.
4. Assign each observation to a block via `np.digitize`.
5. Randomly shuffle blocks and assign the first `round(n_blocks × test_fraction)` blocks to the test set.

This keeps geographically adjacent observations on the same side of the split.

---

## 13  Summary of constants

| Constant                      | Value                | Source                                  |
|-------------------------------|----------------------|-----------------------------------------|
| Mars equatorial radius        | 3 396 190 m          | DSMAP.CAT                               |
| Mars polar radius             | 3 376 200 m          | DSMAP.CAT                               |
| Native resolution (RED)       | 118 502.26 pix/deg   | Per-product LBL                         |
| Effective bit depth           | 10 bits (0–1023)     | `SAMPLE_BIT_MASK = 2#0000001111111111#` |
| COG tile size                 | 512 × 512 pixels     | _COG_CREATION_OPTIONS                   |
| COG overview levels           | [2, 4, 8, 16]        | _OVERVIEW_LEVELS                        |
| Spatial tolerance             | 1 × 10⁻⁵ ° (≈ 0.6 m) | _SPATIAL_TOL                            |
| Sampler edge inset            | 5% of patch size     | _edge_inset in HiRISEGeoSampler         |
| Center longitude (projection) | 180.000°             | All per-product LBLs                    |
| CORE_NULL (nodata DN)         | 0                    | Per-product LBL IMAGE object            |
| Minimum free disk (downloads) | 100 GB               | _MIN_FREE_BYTES                         |
