# MarsHiRISE DTM dataset — technical reference

A systematic documentation of the data structures, coordinate systems, file formats, stereo-pair grouping, ortho
selection, and processing pipeline for the **MarsHiRISEDTM** TorchGeo dataset built on NASA PDS HiRISE Digital Terrain
Models (DTMs) and their accompanying orthoimages.

This document is the DTM counterpart to `MarsHiRISE_Technical_Reference.md` (which covers the RDR JP2 imagery). Where
the RDR pipeline deals with a single radiometric raster per observation, the DTM pipeline deals with **stereo pairs** —
each pair contributing one 32-bit elevation raster (`.IMG`) and up to four orthorectified image rasters (`.JP2`) at
varying resolutions and colour contents. The section numbering below parallels the RDR reference; cross-references point
to the shared base sections when the behaviour is identical.

---

## 1  Coordinate reference system for Mars

### 1.1  Ellipsoid parameters

DTM products use the same Mars IAU 2000 ellipsoid as the RDR products (`a = 3 396 190 m`, `b = 3 376 200 m`), and each
ortho LBL encodes the local radius at the observation's `CENTER_LATITUDE` in
`A_AXIS_RADIUS = B_AXIS_RADIUS = C_AXIS_RADIUS`. For
example, [ESP_011265_1560_RED_A_01_ORTHO.LBL](https://hirise-pds.lpl.arizona.edu/PDS/DTM/ESP/ORB_011200_011299/ESP_011265_1560_ESP_011331_1560/ESP_011265_1560_RED_A_01_ORTHO.LBL)
at `CENTER_LATITUDE = −20.0°` records `A_AXIS_RADIUS = 3393.83 km`.

*Reference: [HiRISE RDR SIS v1.3 §3.5.1](https://www.uahirise.org/pdf/HiRISE_RDR_v13_DTM.pdf) — "The ellipsoid is
defined as, equatorial radius of 3396.190000 km and polar radius of 3376.200000 kilometers. Using the local radius of
the ellipsoid implies that the MAP_SCALE and MAP_RESOLUTION are true at the CENTER_LATITUDE."*

The dataset CRS is the shared Mars geographic CRS:

```
+proj=longlat +a=3396190 +b=3376200 +no_defs
```

See the RDR reference §1.1–1.2 for the rationale.

### 1.2  Native resolution

DTM products are generated at a **post spacing roughly 4× the pixel scale** of the input stereo images — so a HiRISE
observation at 0.25–0.5 m/pixel yields a DTM at 1–2 m post spacing with vertical precision in the tens of centimetres.

The dataset uses 1 m/pixel as its canonical DTM resolution, giving:

```
res_dtm = 1 / 59 251.13 ≈ 1.687 × 10⁻⁵ °/pixel
```

at the equator. This is **roughly 2× coarser** than the RDR dataset's `8.44 × 10⁻⁶ °/pixel`. The scalar `res` passed
into the TorchGeo base class is derived from this value:

```python
super().__init__(..., res=1.0 / 59_251.13)
```

*Reference: [MarsHiRISEDTM.__init__](src/dataset/core/dtm.py) — "DTM typical resolution ≈ 1 m/pix → ~1/(59 251) deg/pix at
equator."*

**Important caveat.** Individual DTMs can actually be produced at any of four canonical post spacings (see §5.3). The
dataset's `res` is a single value used for sampling-grid computations; the per-file `MAP_SCALE` in each ortho LBL is the
authoritative resolution for that file, and the loader honours it when computing pixel windows (§10).

---

## 2  PDS cumulative index (DTMCUMINDEX)

### 2.1  What it represents

The DTM cumulative index is a separate catalogue from RDRCUMINDEX — it lists **DTM products and their orthoimages**
rather than RDR JP2 strips. Each row describes either a DTM file (`.IMG`) or one orthoimage product (`.JP2`); stereo
pairs are *implicit* in the table (one DTM row + 1–4 ortho rows sharing the same `LEFT_OBSERVATION_ID` /
`RIGHT_OBSERVATION_ID`).

*Reference: [DTMCUMINDEX.LBL](https://hirise-pds.lpl.arizona.edu/PDS/INDEX/DTMCUMINDEX.LBL) — 54 columns, 11,647 rows,
610 bytes per record (as of April 2026).*

Files:

- `DTMCUMINDEX.LBL` — PDS3 label describing the table schema.
- `DTMCUMINDEX.TAB` — the table itself (ASCII, fixed-length 610-byte records).

Two sibling tables exist: `DTMINDEX.LBL`/`DTMINDEX.TAB` (same schema, current release only). The dataset uses the
cumulative version because it exposes every ever-released product.

### 2.2  Key columns — how DTMCUMINDEX differs from RDRCUMINDEX

All of the geometric columns of RDRCUMINDEX (§2.2 of the RDR reference) are present here too: `MINIMUM_LATITUDE`,
`CORNER1–4_LATITUDE/LONGITUDE`, `MAP_SCALE`, `MAP_PROJECTION_TYPE`, and so on. Four additional columns are unique to
DTMs and drive the stereo-pair reconstruction logic:

| Column                  | Type          | Description                                                                   | Used for                             |
|-------------------------|---------------|-------------------------------------------------------------------------------|--------------------------------------|
| `LEFT_OBSERVATION_ID`   | CHARACTER(15) | Observation ID of the left stereo image                                       | Pair-key construction                |
| `RIGHT_OBSERVATION_ID`  | CHARACTER(15) | Observation ID of the right stereo image                                      | Pair-key construction                |
| `SOURCE_DTM_PRODUCT_ID` | CHARACTER(33) | For orthos, the DTM used to orthorectify this image; `NA` for DTM rows        | Ortho-to-DTM traceability            |
| `DATA_TYPE`             | CHARACTER(16) | One of `DTM`, `ORTHOIMAGE`, `LEFT ORTHOIMAGE`, `RIGHT ORTHOIMAGE`, `FOM`, ... | Row classification (§2.3)            |
| `RATIONALE_DESC`        | CHARACTER(75) | Scientific rationale (e.g. `"Possible MSL rover landing site Eberswalde..."`) | `target` substring filter, metadata  |
| `PRODUCT_ID`            | CHARACTER(33) | Product identifier (33 bytes, longer than RDR's 21)                           | Ortho parsing regex `_ORTHO_PATTERN` |

*Reference: [DTMCUMINDEX.LBL](https://hirise-pds.lpl.arizona.edu/PDS/INDEX/DTMCUMINDEX.LBL) — column definitions.*

### 2.3  The `DATA_TYPE` column and row classification

DTMCUMINDEX rows are a *union* of different product kinds. The dataset's `_build_spatial_index` classifies each row by
its `DATA_TYPE` value (trimmed, upper-cased) into one of two sets:

```python
DTM_DATA_TYPES   = frozenset({"DTM"})
ORTHO_DATA_TYPES = frozenset({"ORTHOIMAGE"})
```

Rows whose `DATA_TYPE` is neither of these (e.g. `LEFT ORTHOIMAGE`, `FOM MAP`) are dropped. This is deliberate — the
Figure-of-Merit maps (`DTF...JP2`) and the historical `LEFT/RIGHT ORTHOIMAGE` variants are not needed for training and
would confuse the ortho regex parser (§4.3).

### 2.4  Corners, bounding boxes, and antimeridian handling

The `CORNER1–4_LATITUDE/LONGITUDE` and `MINIMUM/MAXIMUM_LATITUDE/LONGITUDE` columns behave identically to the RDR
index — the "corners" are axis-aligned because DTM map projections also have `MAP_PROJECTION_ROTATION = 0.0` (see RDR
reference §2.3).

Longitude normalisation from PDS `[0°, 360°]` to `[−180°, 180°]` and the antimeridian guard are inherited from the base
class (RDR reference §2.4). The DTM-specific ortho-overlap check (§7.4) additionally *splits* bounding boxes at the
antimeridian rather than rejecting them, because DTMs near longitude 180° (e.g. certain landing-site studies) are rare
but real:

```python
if fl_norm > fr_norm:
    ortho_geom = shapely.ops.unary_union([
        box(fl_norm, fb, 180.0, ft),
        box(-180.0, fb, fr_norm, ft),
    ])
```

---

## 3  Equirectangular projection equations

DTM elevations and orthoimages use the same Equirectangular projection as RDR products (see RDR reference §3) —
including the five-degree CENTER_LATITUDE binning and the local-radius convention. The equations:

```
Lat = ((1 − L0 − Line) × Scale / R) × (180 / π)
Lon = LonP + ((Sample − S0 − 1) × Scale / (R × cos(LatP))) × (180 / π)
```

relate pixel coordinates `(Sample, Line)` to geographic `(Lat, Lon)` via the per-product `LINE_PROJECTION_OFFSET`,
`SAMPLE_PROJECTION_OFFSET`, `MAP_SCALE`, and `A_AXIS_RADIUS` values in each LBL. See the `IMAGE_MAP_PROJECTION` object
of any ortho LBL for the stored keywords.

Polar DTMs (|lat| > 65°) use Polar Stereographic;
see [HiRISE RDR SIS v1.3 §3.5.2](https://www.uahirise.org/pdf/HiRISE_RDR_v13_DTM.pdf). The dataset code is agnostic to
the choice because it delegates to rasterio, which reads whichever CRS is embedded in each file's GeoTIFF UUID box.

---

## 4  File-level metadata — DTM vs ortho

DTM products differ fundamentally from RDR products in how their metadata is attached. Understanding this is essential
for the loader.

### 4.1  DTM `.IMG` files — attached PDS3 labels

DTM elevations are stored as **single-file PDS3 products with the label embedded at the start of the binary file** (an "
attached" label in PDS parlance). There is no detached `.LBL` sidecar. The format is:

```
Bytes 0   .. L-1  : ASCII PDS3 label (terminated by `END`)
Bytes L   .. N-1  : IEEE 754 single-precision (32-bit) float raster, MSB order
```

Where `L` is given by the label's `^IMAGE = <byte-offset>` or `RECORD_BYTES × (^IMAGE − 1)` pointer. The raster is
`BAND_SEQUENTIAL` with `BANDS = 1`, `SAMPLE_TYPE = IEEE_REAL`, `SAMPLE_BITS = 32`.

**Consequences for the pipeline:**

1. **No separate `.LBL` to download.** The `_build_download_tasks` method adds a companion `.LBL` only for `.JP2`
   products — for `.IMG` files, everything is in the one file.
2. **Rasterio/GDAL handles the attached label transparently** via the PDS driver; `rasterio.open('DTEEC_*.IMG')` returns
   a properly-CRS'd, properly-scaled float32 raster.
3. **No radiometric calibration.** Unlike the RDR `DN → I/F` conversion, DTM pixels are *already* in physical units (
   metres). The label does still carry `OFFSET` and `SCALING_FACTOR` keywords per the PDS3 schema, but they are
   typically `0.0` and `1.0` — the loader never applies them to elevation data.

### 4.2  Ortho `.JP2` files — detached labels with DTM-specific quirks

The orthoimages attached to each stereo pair are JPEG2000 files with detached `.LBL` sidecars, analogous to RDR
products — but with two critical differences from the RDR ortho convention:

| LBL keyword                 | RDR value             | DTM ortho value      |
|-----------------------------|-----------------------|----------------------|
| `SAMPLE_BITS`               | 16                    | **8**                |
| `SAMPLE_BIT_MASK`           | `2#0000001111111111#` | **`2#11111111#`**    |
| `CORE_HIGH_REPR_SATURATION` | 1023                  | **255**              |
| Effective max DN            | 1023 (10-bit)         | **255 (full 8-bit)** |
| Typical `SCALING_FACTOR`    | ~1.24e-4 – 2.38e-4    | **~8.49e-5**         |
| Typical `OFFSET`            | ~0.033                | **~0.069**           |

*Reference: [ESP_011265_1560_RED_A_01_ORTHO.LBL](https://hirise-pds.lpl.arizona.edu/PDS/DTM/ESP/ORB_011200_011299/ESP_011265_1560_ESP_011331_1560/ESP_011265_1560_RED_A_01_ORTHO.LBL).*

**Why 8-bit?** DTM orthoimages are produced by the orthorectification stage of the SOCET Set pipeline, which outputs
8-bit brightness values rather than the 10-bit instrument DN range preserved in the RDR path. The I/F conversion formula
is unchanged —

```
I/F = DN × SCALING_FACTOR + OFFSET
```

— but the constants are calibrated against the different DN range. **The loader cannot naively reuse RDR calibration
defaults** for DTM orthos; `ProductMeta.from_lbl()` parses the actual values from each ortho's LBL on every load, and
the defaults (`_DEFAULT_SCALING_FACTOR`, `_DEFAULT_OFFSET` in the base class) are only a last-resort fallback.

### 4.3  Ortho `PRODUCT_ID` — the `_ORTHO_PATTERN` regex

Every ortho row's `PRODUCT_ID` follows a strict schema:

```
XSP_xxxxxx_xxxx_CCC_S_NN_ORTHO
```

| Token         | Meaning                                                                                        | Example            |
|---------------|------------------------------------------------------------------------------------------------|--------------------|
| `XSP`         | Mission phase (`PSP`, `ESP`, `TRA`, `AEB`)                                                     | `ESP`              |
| `xxxxxx_xxxx` | Orbit number + latitude/target code                                                            | `011265_1560`      |
| `CCC`         | Colour content: `RED` (1-band) or `IRB` (3-band, NIR/RED/BG)                                   | `RED` or `IRB`     |
| `S`           | Grid spacing letter (see §5.3 table)                                                           | `A`, `B`, `C`, `D` |
| `NN`          | Sequence number for multiple orthos from the same observation rectified against different DTMs | `01`, `02`, ...    |
| `ORTHO`       | Literal suffix                                                                                 | `ORTHO`            |

Full regex used in the dataset:

```python
_ORTHO_PATTERN = re.compile(
    r"(\w+_\d+_\d+)_(RED|IRB)_([A-Z])_(\d+)_ORTHO\s*$"
)
```

The four captured groups are stored in `_ortho_obs_id`, `_ortho_color`, `_ortho_scale`, and an unused sequence number.
An ortho row whose `PRODUCT_ID` fails to match (e.g. a FOM map `DTF...` or a legacy `_RED_ORTHO` without a scale letter)
is silently dropped — the resulting `_ortho_color` is `None` and `_assign_ortho_path` returns early.

*Reference: [UA HiRISE DTM About page](https://www.uahirise.org/dtm/about.php) — full product-ID conventions.*

---

## 5  DTM product naming and stereo-pair keys

### 5.1  DTM `PRODUCT_ID` schema

DTM elevation products follow a different convention from ortho products:

```
aabcd_xxxxxx_xxxx_yyyyyy_yyyy_Vnn
```

| Token         | Meaning                                                                              | Example       |
|---------------|--------------------------------------------------------------------------------------|---------------|
| `aa`          | Always `DT` (DTM)                                                                    | `DT`          |
| `b`           | Data type: `E`=areoid elevation, `R`=planetary radii, `F`=FOM                        | `E`           |
| `c`           | Projection: `E`=Equirectangular, `P`=Polar Stereographic                             | `E`           |
| `d`           | Grid spacing letter (see §5.3)                                                       | `C` (= 1.0 m) |
| `xxxxxx_xxxx` | Orbit + latitude code of the **left** source image                                   | `011265_1560` |
| `yyyyyy_yyyy` | Orbit + latitude code of the **right** source image                                  | `011331_1560` |
| `V`           | Producing institution: `U`=USGS, `A`=UA, `C`=Caltech, `P`=PSI, `O`=Open Uni, `L`=UCL | `U`           |
| `nn`          | Two-digit version number                                                             | `01`          |

Example: `DTEEC_011265_1560_011331_1560_U01` = DT, **E**levation, **E**quirectangular projection, **C** (1 m) post
spacing, left `ESP_011265_1560`, right `ESP_011331_1560`, produced by **U**SGS, version `01`.

*Reference: [HiRISE DTM About](https://www.uahirise.org/dtm/about.php) — naming conventions.*

### 5.2  The "pair key"

The dataset builds a canonical stereo-pair key by concatenating the two observation IDs from the DTM row:

```python
pair_key = LEFT_OBSERVATION_ID + "__" + RIGHT_OBSERVATION_ID
```

All ortho rows that reference the same `left_id` or `right_id` are then associated with this pair. Every row in the
final `GeoDataFrame` is uniquely identified by `pair_key` — exactly one DTM plus 0–4 orthos per row.

**Why both IDs?** A single observation can participate in multiple stereo pairs (e.g. three observations of the same
crater produce three pairs). The obs-ID → pair-key map is therefore many-to-many:

```python
obs_to_pairs: dict[str, list[str]]
# e.g. "ESP_011265_1560" → [
#   "ESP_011265_1560__ESP_011331_1560",
#   "ESP_011265_1560__ESP_014298_1560",
# ]
```

When assigning orthos to pair records, each ortho row is "exploded" into one copy per containing pair so the same ortho
file can legitimately be attached to multiple pair records.

### 5.3  Grid-spacing letters

The scale letter (`A`–`D`) in both DTM and ortho product IDs encodes the pixel/post spacing. Most DTMs are produced at
`C` (1 m) post spacing from `A` (0.25 m) input stereo, which is the "4× post spacing" rule of thumb from the HiRISE
team.

| Letter | Metres/pixel | Typical use                                      |
|--------|--------------|--------------------------------------------------|
| `A`    | 0.25         | Full-resolution orthos (matches RDR 0.25 m)      |
| `B`    | 0.50         | Half-binned orthos                               |
| `C`    | 1.00         | DTM-resolution orthos and most DTMs              |
| `D`    | 2.00         | Coarse DTMs from high-altitude / binned-2 stereo |

*Reference: [HiRISE DTM About](https://www.uahirise.org/dtm/about.php) — grid-spacing definitions.*

A single stereo pair may have orthos at multiple scales (e.g. both `RED_A` and `RED_C`). The dataset's `ortho_scale`
argument lets the user request a specific letter; with `ortho_scale=None` the loader picks the finest available scale
per pair (see §6.2).

---

## 6  Stereo-pair spatial index

### 6.1  Schema

The per-pair GeoDataFrame has one row per stereo pair, with these columns in addition to the `geometry` and
`IntervalIndex` temporal key inherited from the base class:

| Column                            | Type  | Description                                    |
|-----------------------------------|-------|------------------------------------------------|
| `pair_key`                        | str   | `<left_obs>__<right_obs>`                      |
| `left_obs_id`, `right_obs_id`     | str   | The two observation IDs of the stereo pair     |
| `dtm_path`                        | str   | Local path to the `DTEEC_*.IMG` file           |
| `dtm_product_id`                  | str   | Full DTM product ID                            |
| `data_elevation_type`             | str   | Elevation (`DTM`) vs radii vs FOM              |
| `map_scale`                       | float | DTM pixel scale in metres (from `MAP_SCALE`)   |
| `rationale_desc`                  | str   | Scientific rationale (used by `target` filter) |
| `left_red_path`, `right_red_path` | str   | RED ortho paths (left and right observations)  |
| `left_irb_path`, `right_irb_path` | str   | IRB ortho paths (left and right observations)  |
| `*_scale`                         | str   | Chosen scale letter per ortho slot             |

Only the `dtm_path`, plus whichever ortho paths were assigned, participate in tile loading. The other columns exist for
metadata inspection and filtering.

### 6.2  Ortho path assignment — `_assign_ortho_path`

For each stereo pair, up to four ortho slots are filled (`{left,right} × {red,irb}`). The assignment logic handles three
wrinkles:

1. **Left vs right disambiguation.** The `DATA_TYPE` may read `LEFT ORTHOIMAGE` / `RIGHT ORTHOIMAGE` explicitly, but
   older products just say `ORTHOIMAGE`. In that case, the ortho's `_ortho_obs_id` is compared against the pair's
   `left_obs_id` / `right_obs_id` to decide which slot to fill.

2. **Scale preference.** If the user passed `ortho_scale="A"` (for example), rows with `_ortho_scale == "A"` win
   outright. Otherwise the logic picks the lexicographically smallest scale letter — `A < B < C < D` — which happens to
   match "finest wins":

```python
if current_path is None or (current_scale and scale < current_scale):
    rec[col_key] = orow["_local_path"]
    rec[scale_key] = scale
```

3. **Colour gating.** If `include_ortho=False`, no orthos are assigned. If `ortho_type=["RED"]`, IRB slots are still
   populated internally (for possible future use) but only RED slots are tile-loaded in `__getitem__`.

### 6.3  Stereo-pair completeness filter — *why pairs get dropped*

Before adding a pair to the index, the loader enforces **both-sides-complete** for every requested ortho type:

```python
for otype in self.ortho_types:
    color_key = otype.lower()
    if rec.get(f"left_{color_key}_path") is None or \
       rec.get(f"right_{color_key}_path") is None:
        missing_required_ortho = True
        break
if missing_required_ortho:
    continue  # skip this stereo pair entirely
```

This is deliberately strict: a stereo pair with only a left RED ortho but no right RED ortho is dropped because
downstream training code (e.g. `DepthFMHiRISEAdapter` which randomly picks left or right per sample) depends on both
sides being loadable.

Consequences:

- Asking for `ortho_type=["RED", "IRB"]` yields a **smaller** index than `ortho_type=["RED"]` because pairs missing IRB
  orthos are now excluded.
- The index size depends on what's on disk — a partially-downloaded mirror may produce a much smaller index than a
  complete one. Log messages during indexing note the drop count.

---

## 7  Footprint extraction and validation

### 7.1  Why only the DTM is used for the footprint

In the RDR dataset, each observation's footprint is computed from the convex hull of non-zero pixels in the JP2 (RDR
reference §7.2, Case A). For DTM stereo pairs, the same pixel-hull approach applies — but **only to the DTM `.IMG` file
**, never to the orthos.

Reason: orthoimages in the HiRISE DTM archive contain substantial **zero-padded border regions** from the
orthorectification process. These padded zeros are indistinguishable (pixel-value-wise) from nodata in the convex-hull
computation, which would inflate the ortho's inferred footprint far beyond its true data coverage. In practice the
padding is rectangular and larger than the true strip, so the resulting "ghost geometry" covers parts of Mars the ortho
has no information about. Using the DTM — which is float32 with a proper nodata sentinel — avoids this.

Code site:

```python
# Only use the DTM to compute the footprint. Orthoimages contain
# unreliable padding that generates "ghost" geometries.
if dtm_p is not None and pathlib.Path(dtm_p).exists():
    paths = [dtm_p]
```

### 7.2  DTM nodata predicate

The RDR footprint extractor treats pixel > 0 as valid. For float32 DTMs, this is wrong — valid elevations can be zero or
negative (Mars elevations are quoted relative to the areoid). The DTM dataset passes a custom predicate:

```python
def _dtm_valid(data: np.ndarray) -> np.ndarray:
    return np.isfinite(data) & (data > -1e30)
```

The `> -1e30` guard catches the IEEE `FLT_MIN` sentinel `−3.4028226550889045 × 10³⁸` used by SOCET Set to mark nodata.
The `np.isfinite` guard additionally catches `NaN`/`±inf` that rasterio may introduce after reprojection.

### 7.3  Ortho–DTM misalignment filter (≥ 75 % overlap)

After the DTM-based geometry is computed, each assigned ortho is cross-checked against it:

```python
overlap_ratio = self._get_ortho_overlap(geom, ortho_path)
if overlap_ratio < 0.75:
    logger.error("Misalignment detected! ... Dropping pair from index.")
    ortho_validation = True
    break
```

`_get_ortho_overlap` transforms the ortho's rasterio bounds into the geographic CRS, normalises longitudes, builds a
Shapely `box`, and computes `intersection.area / dtm_geom.area`.

**Why 75 %?** The HiRISE orthorectification process aligns each ortho to *its specific* DTM. When a stereo pair has
multiple DTM versions (e.g. `V01`, `V02`), an ortho in the archive might be aligned to a version not present in the
local mirror, producing a 10–50 % overlap with the DTM the dataset has. A 100 % threshold is too strict (numerical
rounding alone can cost a few percent on edge-only overlaps); < 50 % is clearly wrong; 75 % catches obvious mismatches
while tolerating legitimate boundary fuzz.

Observationally, misaligned pairs are rare (< 1 % of pairs in the current index) but cause catastrophic training
failures when present — the ortho appears to overlap the elevation query window but returns unrelated pixels, creating
silent label noise.

### 7.4  Ortho LBL sanity check

A second validation rejects orthos whose LBL parses to a degenerate calibration:

```python
if abs(meta.offset) < _EPS and abs(meta.scaling_factor - 1) < _EPS:
    logger.error("Incorrect scaling_factor and offset! ... Dropping pair from index.")
```

That is — if an ortho's label reports `OFFSET = 0.0` and `SCALING_FACTOR = 1.0`, the calibration would be a no-op (
`I/F = DN`), producing pixel values in `[0, 255]` instead of physically meaningful reflectance in `[0, 1]`. A small
number of ortho LBLs in the archive exhibit this degeneracy, likely an artefact of a reprocessing run where the
calibration step was skipped. The dataset drops the entire stereo pair rather than silently feeding uncalibrated orthos
into training.

### 7.5  Geometry resolution priority

Summarised from `_build_spatial_index`, the per-pair geometry resolves in this order:

| Case | Source                                         | Shape                    | When used                            |
|------|------------------------------------------------|--------------------------|--------------------------------------|
| A    | Convex hull of valid DTM pixels (from `.IMG`)  | True strip parallelogram | DTM file exists and is readable      |
| B    | DTM bounding box (`rasterio.transform_bounds`) | Axis-aligned rectangle   | DTM exists but hull extraction fails |
| C    | CORNER1–4 from the cumulative index            | Axis-aligned rectangle   | DTM not yet downloaded               |

Unlike the RDR pipeline, **orthoimage footprints are never used** as a fallback (see §7.1).

---

## 8  DTM elevation semantics

### 8.1  Physical units

Each DTM pixel is a **scalar elevation in metres**. The reference surface depends on the DTM type letter (second
character of the `PRODUCT_ID`):

| Type letter | `data_elevation_type` | Reference surface                                                     |
|-------------|-----------------------|-----------------------------------------------------------------------|
| `E`         | areoid elevation      | The Mars areoid (geoid-equivalent; MOLA-derived)                      |
| `R`         | planetary radii       | Distance from the centre of Mars (add/subtract 3 389.5 km to convert) |
| `F`         | figure of merit       | *Not a DTM* — stereo correlation quality map                          |

The cumulative index's `DATA_TYPE = "DTM"` covers `E`-type products; `R`-type and `F`-type products have different
`DATA_TYPE` strings and are filtered out in §2.3.

**Conversion between areoid elevation and planetary radius:** `radius_km = 3 389.5 + elevation_m / 1 000` (approximate;
the precise areoid–ellipsoid offset varies by location). The dataset works natively in areoid elevation and never
performs this conversion.

### 8.2  Nodata encoding — `FLT_MIN`, not zero

Unlike RDR DN rasters (where `CORE_NULL = 0`), DTM rasters use the IEEE 754 single-precision minimum as their nodata
sentinel:

```python
_DTM_NODATA: float = -3.4028226550889045e+38  # ≈ −FLT_MAX
```

The load path converts this to `NaN` immediately after reading:

```python
data = src.read(1, window=window, ...)
data = data.astype(np.float32)
valid = (data > -1e30) & np.isfinite(data)
data[~valid] = np.nan
```

**Consequences throughout the pipeline:**

1. **Merging** (§10.3) uses `torch.isnan(merged)` instead of `merged == 0.0` to find empty pixels.
2. **Plotting** uses `np.isfinite(elev_np)` (in the `plot` and `plot3d` methods) to mask the colour map; the percentile
   stretch runs only over finite pixels.
3. **Normalisation** (`normalize_elevation=True`) must preserve the `NaN` mask through the z-scoring:
   ```python
   nodata_mask = torch.isnan(elev)
   elev = (elev - self._elev_mean) / self._elev_std
   elev[nodata_mask] = float("nan")
   ```
4. **Downstream adapters** (e.g. `DepthFMHiRISEAdapter`) must be `NaN`-aware. A naive `torch.mean(elevation)` over a
   patch containing any nodata returns `NaN`, silently poisoning the rest of the batch.

### 8.3  Typical elevation ranges

Mars surface elevations span roughly `−8 500 m` (Hellas Basin floor) to `+21 230 m` (Olympus Mons summit), a 30-km total
range. Per-DTM ranges are tighter — a typical 10 × 10 km DTM covers 100–500 m of relief. The dataset's hardcoded
fallback quantiles (from `src/depth_fm/data/adapter.py`) for a 52k-patch Olympus-region sample are:

```
_DEFAULT_ELEV_P02 = −4 396.57 m
_DEFAULT_ELEV_P98 = 20 757.66 m
```

These are useful as absolute bounds for normalisation; for patch-relative normalisation the 98th-percentile of *centred*
elevation distribution (≈ 45.9 m) is more useful and also hard-coded there.

### 8.4  Vertical precision and accuracy

From the HiRISE DTM literature ([Kirk et al. 2008](https://doi.org/10.1029/2007JE003000)):

- **Precision** (pixel-to-pixel noise): ~10–30 cm for DTMs from bin-1 (0.25 m/pixel) stereo pairs.
- **Accuracy** (absolute): controlled by the MOLA tie used to seed the bundle adjustment; typically ~1 m vertically
  and ~20 m horizontally.

These numbers matter because they set a floor on what a depth-estimation model can possibly learn. Attempting to train
sub-decimetre prediction is pointless — the labels themselves carry ~30 cm noise.

---

## 9  Known DTM artefacts

DTMs are not clean elevation rasters. The HiRISE DTM About page lists several artefact classes that appear in a subset
of products. The dataset does not attempt to auto-filter them — detection requires shaded-relief inspection, which is
outside the per-pair load budget — but users training on patches should be aware and should apply downstream filters if
their task is sensitive to them.

### 9.1  "Boxes" — SOCET Set processing artefacts

Square regions roughly 0.5–1 m offset from the surrounding terrain. These are artefacts of the automated stereo
correlation in SOCET Set (BAE Systems) and are nearly impossible to edit out. They are most visible in shaded-relief
maps of smooth terrain; in patches dominated by real topography (dunes, crater walls) they are within the noise.

*Impact on training:* introduces ~1 m step-function labels into otherwise smooth regions. For absolute-elevation
regression this is a real label noise floor; for slope- or curvature-based tasks the step edges appear as spurious
high-gradient features.

### 9.2  CCD seams

A HiRISE RED image is mosaicked from 10 CCDs (RED0–RED9). The DTM inherits any residual mis-registration at CCD seams as
**sub-metre vertical jumps running along lines at regular cross-track intervals**. The spacing is roughly the CCD
width (~2 048 pixels × `MAP_SCALE` metres/pixel); for a 0.25 m/pixel DTM that's a seam every ~500 m.

*Impact on training:* appears as a repeating anisotropic texture; can be partially absorbed by random patch rotation
augmentation.

### 9.3  Jitter

High-frequency spacecraft jitter (above the frequencies captured by reconstructed pointing kernels) modulates the
pointing vector during the along-track scan. In the DTM it appears as **cross-track elevation ripples** at along-track
frequencies of ~1–10 Hz (metres-to-tens-of-metres wavelength on the surface).

*Impact on training:* periodic high-pass noise superimposed on the true topography. For absolute elevation tasks the
amplitude (~10–50 cm) is small; for slope tasks it can dominate at short wavelengths.

### 9.4  Long-baseline tilts and undulations

Slow drifts of the stereo model, typically arising from imperfect MOLA control or systematic camera-distortion
residuals, produce **large-wavelength (~kilometre) tilts across the DTM**. Relative elevations are preserved within a
tile; absolute elevations drift by up to several metres across the full 5–10 km DTM extent.

*Impact on training:* crucial if training for *absolute* elevation at pair boundaries; benign if training for *relative*
topography within a patch smaller than ~1 km. The `DepthFMHiRISEAdapter`'s `dtm_normalization="relative"` mode
explicitly centres each patch to sidestep this.

### 9.5  Data gaps (usually edited)

The standard HiROC workflow edits out obvious correlation failures (shadows, saturated regions, insufficient stereo
overlap) by interpolating from neighbouring valid posts. The interpolated regions are marked in the companion *
*Figure-of-Merit** map (`DTF...JP2`) but *not* flagged in the DTM itself. Users who need to exclude interpolated regions
must fetch and co-load the FOM map — the current dataset does not expose this.

*Impact on training:* interpolated regions carry zero information about real topography but contribute like real labels.
For smooth interpolation over small gaps (≤ 10 posts) the effect is minor; for large shadowed regions it can create
entire patches of synthetic terrain.

### 9.6  Figure-of-Merit (FOM) map categories

For reference, the FOM map colour legend encodes SOCET Set correlation quality:

| FOM value(s)       | Category                       |
|--------------------|--------------------------------|
| 1                  | No data, outside boundary      |
| 2                  | Shadow                         |
| 3, 5–20, 28, 31–39 | Suspicious / did not correlate |
| 4, 30              | Interpolated/extrapolated      |
| 21                 | Saturated in source            |
| 22–27, 29          | Manually edited                |
| 40–59              | Low end of good correlation    |
| 60–99              | Good correlation               |

*Reference: sample README
from [the PDS Extras directory](https://www.uahirise.org/PDS/EXTRAS/DTM/ESP/ORB_026400_026499/ESP_026404_2565_ESP_026457_2565/README_DTEPC_026404_2565_026457_2565_A01.TXT).*

---

## 10  Tile loading — native-pixel window vs reprojection

### 10.1  Why DTM loading diverges from the RDR loader

The RDR loader reprojects every JP2 tile into the dataset's geographic CRS using
`rasterio.warp.reproject(..., Resampling.bilinear)`. For DTMs, this is the wrong default for two reasons:

1. **Bilinear interpolation across NaN is undefined.** `reproject` treats nodata as a participating value unless given
   an explicit `src_nodata`, and mixing a valid −500 m elevation with a nearby NaN produces a contaminated result that
   looks "close to −500 m" but isn't. Using nearest-neighbour resampling avoids this but produces staircase-visible
   terrain.

2. **Elevation values have physical meaning** — a reprojected elevation interpolates surface height, which is only
   approximately correct if the reprojection is area-preserving, which Equirectangular → Equirectangular is only exactly
   at the source CENTER_LATITUDE.

The DTM loader therefore **reads native-pixel windows without reprojection** for both DTMs and orthos:

```python
native_bounds = rasterio.warp.transform_bounds(
    self.mars_crs, src.crs, x.start, y.start, x.stop, y.stop
)
window = rasterio.windows.from_bounds(*native_bounds, transform=src.transform)
data = src.read(1, window=window, out_shape=(h, w), boundless=True, fill_value=_DTM_NODATA)
```

Steps:

1. Transform the geographic query bounds into the file's native (per-observation Equirectangular) CRS.
2. Compute the native-pixel window with `rasterio.windows.from_bounds`.
3. Round to integer pixel dimensions, fail-fast if the window is empty.
4. Read the window directly, using `fill_value=_DTM_NODATA` for `boundless=True` reads outside the file bounds.

This trades strict geographic alignment between sibling observations for correctness of elevation values. Two
overlapping DTMs queried in the same window may return slightly offset native grids — but neither has had its elevations
interpolated across nodata boundaries.

### 10.2  Load path

```
__getitem__(slice)
  → spatial query against self.index (shapely intersects on pair geometry)
  → for each candidate pair:
      → _load_dtm_tile(dtm_path, x, y)        # 32-bit float, NaN-masked
      → for side in (left, right):
          for otype in ortho_types:
              → _load_ortho_tile(path, otype, x, y)  # calibrated I/F, [0, 1]
  → _merge_elevation_tiles(dtm_tiles)         # first-valid-wins with NaN
  → merge_tiles(ortho_tiles) for each slot    # first-non-zero-wins
  → (optional) z-score normalisation of elevation
  → return sample dict
```

### 10.3  Tile merging

**Elevation tiles** merge with NaN semantics — the first DTM to cover each pixel wins; subsequent tiles fill remaining
NaN holes:

```python
merged = torch.full((1, max_h, max_w), float("nan"))
for tile in tiles:
    empty = torch.isnan(merged[:, :h, :w])
    valid = ~torch.isnan(tile)
    merged[:, :h, :w][empty & valid] = tile[empty & valid]
```

**Ortho tiles** use the shared zero-based merge logic from the base class (first-non-zero-wins), because calibrated
orthos have `0.0` as their nodata sentinel (after applying the nodata mask post-calibration, per the base class's
`_load_from_jp2` fix).

### 10.4  Out-of-bounds behaviour

When a query slice partially overhangs the DTM/ortho bounds, `boundless=True, fill_value=_DTM_NODATA` returns the
requested window at the correct shape, padded with the nodata sentinel outside the file. This preserves the expected
`(1, H, W)` patch shape regardless of how close to the strip edge the sampler landed — a property the `HiRISEGeoSampler`
and downstream stacking code depend on.

---

## 11  File layout and path resolution

### 11.1  PDS directory structure

DTMs live in a dedicated top-level `DTM/` tree, separate from the `RDR/` JP2s:

```
<root>/
    DTMCUMINDEX.LBL
    DTMCUMINDEX.TAB
    DTM/
        ESP/
            ORB_011200_011299/
                ESP_011265_1560_ESP_011331_1560/
                    DTEEC_011265_1560_011331_1560_U01.IMG
                    ESP_011265_1560_RED_A_01_ORTHO.JP2
                    ESP_011265_1560_RED_A_01_ORTHO.LBL
                    ESP_011265_1560_RED_C_01_ORTHO.JP2
                    ESP_011265_1560_RED_C_01_ORTHO.LBL
                    ESP_011331_1560_RED_A_01_ORTHO.JP2
                    ESP_011331_1560_RED_A_01_ORTHO.LBL
                    ESP_011331_1560_RED_C_01_ORTHO.JP2
                    ESP_011331_1560_RED_C_01_ORTHO.LBL
        PSP/
            ...
```

Each pair directory contains **one DTM** plus **up to four orthos** (RED at two scales × 2 observations), optionally
with four more orthos if IRB colour was produced.

*Reference: [hirise-pds.lpl.arizona.edu/PDS/DTM/](https://hirise-pds.lpl.arizona.edu/PDS/DTM/).*

### 11.2  Local path flattening

As with the RDR dataset, the DTM dataset flattens the PDS hierarchy into a single `images/` directory:

```python
def _pds_local_path(self, spec: str) -> pathlib.Path:
    return self.root / 'images' / pathlib.Path(PurePosixPath(spec.strip())).name
```

So `DTM/ESP/ORB_011200_011299/.../DTEEC_011265_1560_011331_1560_U01.IMG` becomes
`<root>/images/DTEEC_011265_1560_011331_1560_U01.IMG`.

### 11.3  Download companions

For JP2 ortho files, the downloader fetches both `.JP2` and `.LBL`. For `.IMG` files, only the one file is fetched — the
label is attached. The `_build_download_tasks` method:

```python
if spec.upper().endswith(".JP2"):
    lbl_spec = spec[:-4] + ".LBL"
    lbl_local = self._pds_local_path(lbl_spec)
    if not lbl_local.exists():
        tasks.append((f"{self.url}/{lbl_spec}", lbl_local))
```

### 11.4  COG sidecars

As with RDR, each `.JP2` / `.IMG` can optionally have a `.tif` Cloud-Optimised GeoTIFF sidecar for fast random access.
`prefer_cog()` returns the `.tif` if it exists, falling back to the original. COG conversion of DTM `.IMG` files
requires `GDAL_PAM_ENABLED` and BigTIFF output (most DTMs are under 4 GB, but some landing-site DTMs exceed it).

---

## 12  Data-volume considerations

The full DTM cumulative index contains ~11,647 products (as of April 2026), of which roughly 3,000 are DTMs and the
remainder are orthoimages. A back-of-the-envelope volume estimate:

| Product type      | Typical size  | Count  | Subtotal     |
|-------------------|---------------|--------|--------------|
| DTM `.IMG`        | 300–700 MB    | ~3,000 | ~1.5 TB      |
| Ortho `_A_`       | 500 MB – 2 GB | ~3,500 | ~4.4 TB      |
| Ortho `_C_`       | 30–80 MB      | ~3,500 | ~0.2 TB      |
| Ortho `_B_`/`_D_` | variable      | small  | ~0.5 TB      |
| **Total**         |               |        | **~6–10 TB** |

An unfiltered `download=True` run over the full index is therefore prohibitive on most workstations. The dataset's
`main()` entrypoint warns on this explicitly:

```python
if args.target is None and bbox_tuple is None:
    logger.warning(
        "No --target or --bbox filter specified. The full DTM index "
        "contains ~11,600 products totalling >10 TB. Using a default "
        "bbox for Eberswalde Crater as a demo."
    )
```

Recommended usage patterns:

- **Regional studies**: pass `bbox=(lon_min, lat_min, lon_max, lat_max)` to restrict to a geographic window. A single
  landing-site study typically needs < 10 GB.
- **Thematic studies**: pass `target="Eberswalde"` (or any substring) to filter on `RATIONALE_DESC`, `TARGET_NAME`, or
  any character column of the index. Jezero, Gale, and other named sites are well-covered.
- **Full-index research**: mirror the archive separately (`wget -m`, `rclone`, etc.) and pass the mirror root — the
  dataset will index from the existing files without re-downloading.

---

## 13  Sample dict — what `__getitem__` returns

Each call to `dataset[slice]` returns a dict with the following keys:

| Key         | Shape / Type          | Semantics                                                              |
|-------------|-----------------------|------------------------------------------------------------------------|
| `elevation` | `(1, H, W)` float32   | Metres relative to the areoid. Nodata = `NaN`.                         |
| `left_red`  | `(1, H, W)` float32   | Left observation's calibrated RED ortho I/F in `[0, 1]`. Nodata = 0.0. |
| `right_red` | `(1, H, W)` float32   | Right observation's RED ortho, same conventions.                       |
| `left_irb`  | `(3, H, W)` float32   | Left IRB ortho: `(NIR, RED, BG)` in I/F `[0, 1]`.                      |
| `right_irb` | `(3, H, W)` float32   | Right IRB ortho.                                                       |
| `bounds`    | tensor `(4,)` float64 | Query bounds `(xmin, ymin, xmax, ymax)` in geographic degrees.         |
| `crs`       | str (WKT)             | Mars geographic CRS.                                                   |
| `meta`      | list[dict] (optional) | Per-overlapping-pair metadata if `return_meta=True` (§13.1).           |

Ortho keys are conditional: `left_red` / `right_red` appear only when `"RED" ∈ ortho_types`, and similarly for IRB.
Setting `include_ortho=False` suppresses all ortho keys and reduces load time substantially.

The sample **never** contains `None` values — missing ortho coverage (e.g. one side failed to load) is handled by
omission at the dict level. Downstream code should use `sample.get("left_red")` rather than assume the key exists.

### 13.1  Metadata return (`return_meta=True`)

When `return_meta=True`, each sample carries a `meta` list with one entry per overlapping stereo pair:

```python
{
    "dtm_product_id": "DTEEC_011265_1560_011331_1560_U01",
    "left_obs_id": "ESP_011265_1560",
    "right_obs_id": "ESP_011331_1560",
    "left_red_meta": {
        "incidence_angle": 48.3,
        "solar_azimuth": 105.2,
        "scaling_factor": 8.488e-05,
        "offset": 0.0685,
    },
    "right_red_meta": { ... },
}
```

The incidence angle and solar azimuth are essential for downstream **sun-vector estimation** in the
`DepthFMHiRISEAdapterCached` pipeline, which uses them as a prior for the OLS sun-direction fit. Without them, the
sun-vector estimate is unconstrained and frequently wrong in low-texture patches.

---

## 14  Elevation normalisation

The dataset supports optional per-patch z-score normalisation of the elevation raster via `normalize_elevation=True`,
backed by a JSON stats file:

```json
{
  "mean": -1234.56,
  "std": 876.54
}
```

When enabled:

```python
nodata_mask = torch.isnan(elev)
elev = (elev - self._elev_mean) / self._elev_std
elev[nodata_mask] = float("nan")   # preserve NaN through z-score
sample["elevation"] = elev
```

The stats file is computed offline from a representative sample of patches (the reference stats at
`dataset_stats/dtm/dataset_stats.json` use 52k patches from the Olympus Mons region). Using stats from a different
geographic region produces biased normalisation — e.g. normalising an Eberswalde patch with Olympus stats centres it at
−5 km instead of at 0.

**Why patch-level z-score instead of per-pair?** Z-score with **global** stats preserves inter-patch elevation
differences, which matters for learning spatial trends. Per-patch centring (which the `DepthFMHiRISEAdapter`'s
`relative` mode does instead) destroys this but makes individual patches scale-invariant. The choice depends on the
downstream task.

---

## 15  Geographic train/test split

Inherits from the base class (`geographic_split()`). For DTMs specifically, block sizes may need to be larger because
DTM density is much lower than RDR density — a 5% latitude block at low latitudes may contain only a handful of DTMs,
producing noisy test-set statistics. Empirically, `n_blocks = max(10, round(1/test_fraction))` produces more stable
splits for DTMs than the default `max(5, ...)`, but the base class's default is retained for consistency.

---

## 16  Summary of DTM-specific constants

| Constant                         | Value                                       | Source                                         |
|----------------------------------|---------------------------------------------|------------------------------------------------|
| DTM native resolution            | ~1 m/pixel (letter `C`)                     | Most DTM products                              |
| DTM resolution in degrees        | 1 / 59 251.13 ≈ 1.687 × 10⁻⁵ °/pixel        | Equator, 1 m/pixel                             |
| DTM sample type                  | IEEE_REAL, 32-bit, MSB                      | Attached PDS3 label                            |
| DTM nodata sentinel              | −3.4028226550889045 × 10³⁸                  | `_DTM_NODATA` (= `−FLT_MAX`)                   |
| DTM nodata in-memory             | `NaN` (float32)                             | Converted on read                              |
| DTM nodata validity threshold    | `> −1 × 10³⁰`                               | `_dtm_valid` predicate                         |
| Ortho effective bit depth        | 8-bit (0–255)                               | `SAMPLE_BIT_MASK = 2#11111111#`                |
| Ortho typical SCALING_FACTOR     | ~8.49 × 10⁻⁵                                | Per-product LBL                                |
| Ortho typical OFFSET             | ~0.069                                      | Per-product LBL                                |
| Ortho scale letters              | A=0.25 m, B=0.5 m, C=1 m, D=2 m             | HiRISE DTM About page                          |
| Ortho-to-DTM misalignment reject | < 75 % overlap                              | `_get_ortho_overlap` threshold                 |
| Ortho LBL sanity reject          | `                                           | offset                                         | < 1e-6` AND `|scaling − 1| < 1e-6` | `_EPS` check          |
| Stereo-pair completeness         | Both sides required for each requested type | `_build_spatial_index` filter                  |
| Cumulative index rows (Apr 2026) | 11,647                                      | `DTMCUMINDEX.LBL / FILE_RECORDS`               |
| Cumulative index row bytes       | 610                                         | `DTMCUMINDEX.LBL / RECORD_BYTES`               |
| Full DTM index size              | ~6–10 TB                                    | Estimated from `FILE_NAME_SPECIFICATION` sizes |
| Producing institution codes      | U, A, C, P, O, L                            | USGS, UA, Caltech, PSI, Open Uni, UCL          |
| Cache version                    | `dtm_v1`                                    | `MarsHiRISEDTM._cache_version()`               |

---

## 17  References

- **HiRISE RDR SIS v1.3** (Dec 2011), Section 5 "Addendum to HiRISE RDR – DTM
  Products": <https://www.uahirise.org/pdf/HiRISE_RDR_v13_DTM.pdf>
- **HiRISE DTM About**: <https://www.uahirise.org/dtm/about.php>
- **DTM cumulative index**: <https://hirise-pds.lpl.arizona.edu/PDS/INDEX/DTMCUMINDEX.LBL>
- **PDS DTM archive root**: <https://hirise-pds.lpl.arizona.edu/PDS/DTM/>
- **Kirk et al. (2008)**, "Ultrahigh resolution topographic mapping of Mars with MRO HiRISE stereo images", JGR-Planets,
  doi:10.1029/2007JE003000 — DTM precision/accuracy characterisation.
- **Mattson et al. (2022)**, "Revealing Active Mars with HiRISE Digital Terrain Models", Remote Sensing 14(10):2403 —
  end-to-end DTM production workflow and quality assessment.
- Sample pair directory used throughout this
  document: <https://hirise-pds.lpl.arizona.edu/PDS/DTM/ESP/ORB_011200_011299/ESP_011265_1560_ESP_011331_1560/>
