"""
hirise_rationale_analysis.py
============================

Analyse and visualise the RATIONALE_DESC field of the HiRISE PDS index files.

Works with either:
  - The official PDS fixed-width text file:
      https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.TAB  (~155 MB)
      https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRINDEX.TAB     (~140 KB, latest volume)
  - A pre-extracted CSV with at least: rationale_desc, latitude, longitude

USAGE
-----
    python hirise_rationale_analysis.py path/to/RDRCUMINDEX.TAB --outdir figures/
    python hirise_rationale_analysis.py path/to/sample.csv --outdir figures/

The module is also import-able for use in notebooks.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud
import logging
import seaborn as sns


logger = logging.getLogger(__name__)

# ---------- visual identity ---------------------------------------------------

MARS_COLORS = ["#1a0a05", "#3d1810", "#7a2d18", "#c0421f", "#e87a3a", "#f3b86b"]
MARS_CMAP = LinearSegmentedColormap.from_list("mars", MARS_COLORS, N=256)

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "#222",
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.titlepad":    14,
    "xtick.color":      "#333",
    "ytick.color":      "#333",
    "font.family":      "DejaVu Sans",
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})

# ---------- I/O ---------------------------------------------------------------

# Per RDRCUMINDEX.LBL: fixed-width fields. Byte offsets are 1-indexed.
# We expose the columns we actually use.
TAB_FIELDS = [
    # (name,             start_byte_1indexed, length_bytes, dtype)
    ("volume_id",        2,   10, "str"),
    ("file_name",        15,  67, "str"),
    ("observation_id",   100, 15, "str"),
    ("product_id",       118, 21, "str"),
    ("target_name",      148, 32, "str"),
    ("orbit_number",     182, 6,  "int"),
    ("mission_phase",    190, 30, "str"),
    ("rationale_desc",   223, 75, "str"),
    ("start_time",       347, 24, "str"),
    ("incidence_angle",  461, 7,  "float"),
    ("min_latitude",     609, 10, "float"),
    ("max_latitude",     620, 10, "float"),
    ("min_longitude",    631, 10, "float"),
    ("max_longitude",    642, 10, "float"),
]


def read_tab(path: str | Path) -> pd.DataFrame:
    """Parse a HiRISE PDS RDR(CUM)INDEX.TAB file.

    The TAB file uses comma-separated quoted fields padded to a fixed total
    record width (821 bytes per record incl. CRLF). We can parse it either
    by fixed-width slicing or by csv.reader. We use csv.reader because the
    fields are reliably comma-separated with quoted strings.
    """
    import csv
    rows = []
    needed_idx = {
        "observation_id": 4,
        "target_name":    7,
        "orbit_number":   8,
        "mission_phase":  9,
        "rationale_desc": 10,
        "start_time":     13,
        "incidence_angle": 19,
        "min_latitude":   35,
        "max_latitude":   36,
        "min_longitude":  37,
        "max_longitude":  38,
    }
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                fields = next(csv.reader([line], skipinitialspace=True))
            except Exception:
                continue
            if len(fields) < 39:
                continue
            try:
                rec = {
                    "observation_id":  fields[needed_idx["observation_id"]].strip(),
                    "target_name":     fields[needed_idx["target_name"]].strip(),
                    "orbit_number":    int(fields[needed_idx["orbit_number"]]),
                    "mission_phase":   fields[needed_idx["mission_phase"]].strip(),
                    "rationale_desc":  fields[needed_idx["rationale_desc"]].strip(),
                    "start_time":      fields[needed_idx["start_time"]].strip(),
                    "incidence_angle": float(fields[needed_idx["incidence_angle"]]),
                    "min_latitude":    float(fields[needed_idx["min_latitude"]]),
                    "max_latitude":    float(fields[needed_idx["max_latitude"]]),
                    "min_longitude":   float(fields[needed_idx["min_longitude"]]),
                    "max_longitude":   float(fields[needed_idx["max_longitude"]]),
                }
            except (ValueError, IndexError):
                continue
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["latitude"]  = (df["min_latitude"]  + df["max_latitude"])  / 2.0
    df["longitude"] = (df["min_longitude"] + df["max_longitude"]) / 2.0
    # Each observation has up to two products (RED + COLOR). Dedupe.
    df = df.drop_duplicates("observation_id", keep="first").reset_index(drop=True)
    df["year"] = df["start_time"].str[:4].astype("Int64", errors="ignore")
    return df


def read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "rationale_desc" not in df.columns:
        raise ValueError("CSV must contain a 'rationale_desc' column")
    return df


def load(path: str | Path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".tab"):
        df = read_tab(path)
    else:
        df = read_csv(path)
    if df.empty:
        raise SystemExit(f"no rows parsed from {path}")
    print(f"loaded {len(df):,} unique observations from {path}")
    return df


# ---------- text processing ---------------------------------------------------
 
# Words that aren't science-meaningful in this context. We strip these so the
# word cloud reflects what scientists actually study, not connective tissue.
STOPWORDS = {
    "and", "or", "in", "the", "of", "a", "an", "on", "at", "to", "with",
    "for", "from", "by", "near", "into", "around", "between", "as", "is",
    "are", "be", "this", "that", "these", "those", "its", "it", "no",
    # generic-ish HiRISE phrasing
    "site", "area", "region", "location", "feature", "features", "terrain",
    "possible", "candidate", "small", "large", "long",
    "monitoring", "monitor", "survey", "survey", "sample",
}
 
# Mars geographic feature types — we'll count these specifically.
GEO_FEATURES = [
    "crater", "chasma", "chasmata", "vallis", "valles", "planitia",
    "planum", "terra", "fossa", "fossae", "tholus", "mons", "montes",
    "patera", "labyrinthus", "sulci", "rupes", "dorsa", "cavi", "scopulus",
    "colles", "mensa", "mensae", "fluctus",
]
 
# Science themes — keyword sets tuned to the actual vocabulary HiRISE
# scientists use in RATIONALE_DESC, not the textbook geology vocabulary.
# Order matters only for the "primary theme" picker on overlapping matches.
THEMES = {
    "Impact craters":             [
        "crater", "craters", "impact", "ejecta", "rayed",
        "pedestal crater", "secondary crater", "central peak",
        "central uplift", "fresh crater", "fresh impact",
        "candidate impact", "candidate recent impact", "newly formed",
        "rayed crater", "crater rim", "crater wall", "crater floor",
    ],
    "Gullies & slopes":           [
        "gully", "gullies", "slope", "slopes", "scarp", "scarps",
        "RSL", "recurring", "lineae", "slope streak", "slope monitoring",
        "slope feature", "pole-facing", "pole facing",
    ],
    "Mass wasting & landslides":  [
        "landslide", "rockfall", "debris flow", "mass wasting", "slump",
        "avalanche", "rock avalanche",
    ],
    "Polar / ice / frost":        [
        "polar", "ice", "frost", "defrosting", "araneiform", "araneiforms",
        "spider", "geyser", "seasonal", "cryptic", "residual cap",
        "polar layered", "PLD", "polar deposit", "polar dune",
        "polar gypsum", "polar erg", "icy",
    ],
    "Glacial / periglacial":      [
        "periglacial", "polygon", "polygons", "patterned ground",
        "lobate", "lobate debris", "debris apron", "concentric crater fill",
        "concentric", "viscous flow", "glacier", "glacial", "ice-rich",
        "ice-cemented", "thermokarst",
    ],
    "Aeolian (wind)":             [
        "dune", "dunes", "ripple", "ripples", "yardang", "TAR",
        "transverse aeolian", "dust devil", "devil track", "barchan",
        "erg", "aeolian", "wind", "active dune", "dust",
    ],
    "Fluvial / channels":         [
        "channel", "channels", "valley", "valleys", "vallis", "valles",
        "fluvial", "alluvial", "delta", "inverted channel", "outflow",
        "tributary", "fan delta", "alluvial fan", "drainage", "sinuous ridge",
    ],
    "Layers & stratigraphy":      [
        "layered", "layer", "layers", "bedrock", "stratigraphy",
        "outcrop", "exposure", "exposures", "deposits", "deposit",
        "stratified", "layered deposits", "layered deposit",
    ],
    "Mineralogy / hydrated":      [
        "clay", "clays", "sulfate", "sulfates", "olivine", "olivine-rich",
        "iron-rich", "mafic", "phyllosilicate", "phyllosilicates",
        "phyllosilicate-rich", "carbonate", "carbonates", "hydrated",
        "polyhydrated", "monohydrated", "gypsum", "hematite", "smectite",
        "nontronite", "jarosite", "pyroxene", "altered", "alteration",
        "spectral signature", "iron",
    ],
    "Light/dark-toned":           [
        "light-toned", "light toned", "dark-toned", "dark toned",
        "toned material", "toned bedrock", "toned deposit", "toned outcrop",
        "albedo", "bright deposit", "bright material", "dark streak",
        "slope streak",
    ],
    "Volcanic":                   [
        "lava", "lava flow", "volcanic", "vent", "tholus", "patera",
        "caldera", "flow front", "fissure vent", "pyroclastic",
    ],
    "Tectonic / structural":      [
        "fault", "faulted", "graben", "grabens", "fracture", "fractured",
        "wrinkle ridge", "wrinkle", "horst", "tectonic", "thrust",
    ],
    "Landing sites & rovers":     [
        "landing site", "landing", "rover", "InSight", "Curiosity",
        "Perseverance", "MSL", "Phoenix", "ExoMars", "Spirit",
        "Opportunity", "Pathfinder", "Viking", "future landing",
        "candidate landing", "Beagle",
    ],
    "Calibration / engineering":  [
        "calibration", "ADC", "ADC settings", "settings test",
        "test observation", "MOC image", "validation", "engineering",
        "stray light", "geometry", "test image", "ride-along",
    ],
}


def save_fig(fig: plt.Figure, path: Path, formats: tuple[str, ...] = (".png", ".pdf", ".svg"), **kwargs):
    for f in formats:
        new_path = path.with_suffix(f)
        fig.savefig(new_path, **kwargs)
        logger.info(f"Saved {new_path}")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    # Keep hyphenated terms as single tokens (e.g. "iron-rich", "sulfate-rich")
    tokens = re.findall(r"[a-z][a-z\-]+[a-z]|[a-z]", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def all_tokens(df: pd.DataFrame) -> list[str]:
    out = []
    for r in df["rationale_desc"].astype(str):
        out.extend(tokenize(r))
    return out


def all_bigrams(df: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    for r in df["rationale_desc"].astype(str):
        toks = tokenize(r)
        out.extend(zip(toks, toks[1:]))
    return out


def theme_assignments(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each rationale with one or more themes."""
    rows = []
    for txt in df["rationale_desc"].astype(str):
        low = txt.lower()
        hits = []
        for theme, kws in THEMES.items():
            for kw in kws:
                if re.search(rf"\b{re.escape(kw.lower())}\b", low):
                    hits.append(theme)
                    break
        rows.append(hits if hits else ["Other"])
    return pd.DataFrame({"themes": rows})


# ---------- figures -----------------------------------------------------------

def fig_wordcloud(df: pd.DataFrame, out: Path) -> None:
    text = " ".join(df["rationale_desc"].astype(str))
    text = re.sub(r"[^A-Za-z\- ]", " ", text)
    wc = WordCloud(
        width=2400, height=1200,
        background_color="white",
        colormap=MARS_CMAP,
        prefer_horizontal=0.92,
        max_words=300,
        relative_scaling=0.45,
        min_font_size=8,
        collocations=True,
        stopwords=STOPWORDS,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"What HiRISE looks at on Mars\n"
        f"word cloud over {len(df):,} observation rationales",
        fontsize=15, pad=16,
    )
    save_fig(fig, out / "01_wordcloud.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out/'01_wordcloud.png'}")


def fig_wordcloud_mars_disk(df: pd.DataFrame, out: Path) -> None:
    """Word cloud constrained to a Mars-disk silhouette."""
    text = " ".join(df["rationale_desc"].astype(str))
    text = re.sub(r"[^A-Za-z\- ]", " ", text)
    # Build a circular mask the size of a Mars disk.
    H = W = 1400
    yy, xx = np.ogrid[:H, :W]
    cx, cy, r = W // 2, H // 2, 600
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 > r ** 2).astype(np.uint8) * 255
    wc = WordCloud(
        width=W, height=H,
        background_color="white",
        colormap=MARS_CMAP,
        prefer_horizontal=0.85,
        max_words=400,
        mask=mask,
        contour_color="#7a2d18", contour_width=3,
        relative_scaling=0.5,
        stopwords=STOPWORDS,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(8.4, 8.4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("HiRISE targeting rationale — Mars-disk word cloud",
                 fontsize=14, pad=10)
    save_fig(fig, out / "02_wordcloud_mars_disk.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out/'02_wordcloud_mars_disk.png'}")


def fig_top_words_and_phrases(df: pd.DataFrame, out: Path) -> None:
    tokens = all_tokens(df)
    bigrams = all_bigrams(df)
    top_unigrams = Counter(tokens).most_common(20)
    top_bigrams_raw = Counter(bigrams).most_common(40)
    # Filter trivial bigrams (where one half is purely a stopword-y word).
    top_bigrams = [(f"{a} {b}", n) for (a, b), n in top_bigrams_raw][:20]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, data, title in zip(
        axes,
        [top_unigrams, top_bigrams],
        ["Top single words", "Top two-word phrases"],
    ):
        labels, counts = zip(*data)
        y = np.arange(len(labels))
        colors = MARS_CMAP(np.linspace(0.25, 0.85, len(labels)))
        ax.barh(y, counts, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("mentions")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, (lbl, c) in enumerate(zip(labels, counts)):
            ax.text(c + max(counts) * 0.01, i, str(c),
                    va="center", fontsize=9, color="#222")
    fig.suptitle(f"Most-mentioned terms in HiRISE rationales (n={len(df):,})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, out / "03_top_words_phrases.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out/'03_top_words_phrases.png'}")


def fig_geographic_features(df: pd.DataFrame, out: Path) -> None:
    """How often each Mars feature type (Crater, Chasma, Vallis…) is named."""
    counts = Counter()
    pat = {f: re.compile(rf"\b{f}\b", re.I) for f in GEO_FEATURES}
    for txt in df["rationale_desc"].astype(str):
        for f, p in pat.items():
            if p.search(txt):
                counts[f.title()] += 1
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        return
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = MARS_CMAP(np.linspace(0.25, 0.85, len(labels)))
    bars = ax.barh(labels, vals, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("rationales mentioning this feature type")
    ax.set_title("Mars geographic feature types named in HiRISE rationales")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                str(v), va="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out / "04_geo_features.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out/'04_geo_features.png'}")


def fig_themes_bar(df: pd.DataFrame, out: Path) -> None:
    th = theme_assignments(df)
    flat = [t for ts in th["themes"] for t in ts]
    counts = Counter(flat)
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = MARS_CMAP(np.linspace(0.25, 0.9, len(labels)))
    bars = ax.barh(labels, vals, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("observations matching this theme")
    ax.set_title("Science themes inferred from HiRISE targeting rationales")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                str(v), va="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out / "05_science_themes.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out/'05_science_themes.png'}")


def fig_map_themed(df: pd.DataFrame, out: Path) -> None:
    """Mars-map of observations colored by primary science theme."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return
        
    # Use the centralized high-contrast palette
    themes, color_for, counts, primary = _theme_palette(df)
    
    df = df.copy()
    df["primary_theme"] = primary

    # Normalise longitude to -180..180 for plotting
    lon = df["longitude"].copy()
    lon = ((lon + 180) % 360) - 180
    lat = df["latitude"]

    # Auto-tune scatter style for the dataset size
    n = len(df)
    if   n < 500:    style = dict(s=40,  alpha=0.95, ew=0.6,  raster=False)
    elif n < 5000:   style = dict(s=12,  alpha=0.85, ew=0.0,  raster=False)
    elif n < 30000:  style = dict(s=4,   alpha=0.65, ew=0.0,  raster=False)
    else:            style = dict(s=1.5, alpha=0.52, ew=0.0,  raster=False)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.add_patch(plt.Rectangle((-180, -90), 360, 180,
                               facecolor="white", edgecolor="none", zorder=0))
    for x in range(-180, 181, 30):
        ax.axvline(x, color="#e0e0e0", lw=0.6, zorder=1)
    for y in range(-90, 91, 30):
        ax.axhline(y, color="#e0e0e0", lw=0.6, zorder=1)
    ax.axhline(0, color="#7a2d18", lw=0.8, zorder=2, alpha=0.6)

    # Plot using the ordered themes to keep the legend cleanly sorted
    for theme in reversed(themes):
        c = color_for[theme]
        m = df["primary_theme"] == theme
        ax.scatter(lon[m], lat[m],
                   s=style["s"], c=[c], alpha=style["alpha"],
                   edgecolor=("white" if style["ew"] > 0 else "none"),
                   linewidth=style["ew"],
                   # Ensure legend label order stays correct later
                   label=f"{theme} ({m.sum():,})", 
                   # Increment zorder so smaller categories stay on top
                   zorder=3 + (1 - m.sum()/len(df)), 
                   rasterized=style["raster"])
                   
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("east longitude (°)")
    ax.set_ylabel("planetocentric latitude (°)")
    ax.set_title("HiRISE observations on Mars, coloured by primary science theme")
    leg = ax.legend(loc="lower left", fontsize=8, framealpha=0.95,
                    ncol=2, bbox_to_anchor=(1.005, 0), borderaxespad=0)
    # Force legend dots to be readable even when the scatter is tiny.
    for h in leg.legend_handles:
        try:    h.set_sizes([36])
        except Exception:  pass
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_fig(fig, out / "06_themed_map.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out/'06_themed_map.png'}")

def _theme_palette(df: pd.DataFrame):
    """Return ordered theme list and a stable theme→color mapping."""
    th = theme_assignments(df)
    primary = [ts[0] for ts in th["themes"]]
    counts = pd.Series(primary).value_counts()
    themes = counts.index.tolist()
    
    n_themes = len(themes)
    
    # Strategy: Vibrant & Pastel
    # Prevents alpha-blending confusion by using distinct, low-saturation 
    # hues for the long tail, rather than grays.
    
    # 1. Grab up to 5 bold, highly saturated colors for the leaders
    # 'Set1' gives strong, distinct primary/secondary colors
    num_highlight = min(5, n_themes)
    top_colors = sns.color_palette("Set1", num_highlight)
    
    # 2. Grab pale, pastel hues for the remaining categories
    # 'Set3' provides 12 distinct but desaturated/pastel colors
    num_muted = max(0, n_themes - 5)
    if num_muted > 0:
        bottom_colors = sns.color_palette("Set3", num_muted)
    else:
        bottom_colors = []
        
    # 3. Combine them into the custom palette
    custom_palette = list(top_colors) + list(bottom_colors)
    
    return themes, dict(zip(themes, custom_palette)), counts, primary

def fig_map_themed_smallmultiples(df: pd.DataFrame, out: Path) -> None:
    """One Mars hex-density map per theme (panels share the same map background).

    This is the readable answer to the overplotting that hits the single-
    panel scatter at 100K+ observations.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return
    df = df.copy()
    themes, color_for, counts, primary = _theme_palette(df)
    df["primary_theme"] = primary

    lon = ((df["longitude"] + 180) % 360) - 180
    lat = df["latitude"]

    # Resolution scales with dataset size — finer hexes when there's more data.
    # if   len(df) < 1000:    grid = (40, 22)
    # elif len(df) < 20000:   grid = (60, 30)
    # else:                   grid = (90, 45)
    grid = (45, 22)

    n = len(themes)
    ncols = 5 if n >= 5 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 2.0 * nrows + 0.6),
                             gridspec_kw={"hspace": 0.32, "wspace": 0.10})
    axes = np.atleast_1d(axes).flatten()

    for i, theme in enumerate(themes):
        ax = axes[i]
        m = (df["primary_theme"] == theme).values
        c = color_for[theme]

        # 2. Raise the colormap floor: start at a light gray instead of pure white
        cmap = LinearSegmentedColormap.from_list(
                    f"th_{i}", ["#e5e5e5", c, "#111111"], N=256)

        ax.add_patch(plt.Rectangle((-180, -90), 360, 180,
                                   facecolor="white",
                                   edgecolor="none", zorder=0))
        if m.sum() > 0:
            ax.hexbin(lon[m].values, lat[m].values,
                      gridsize=grid,
                      extent=(-180, 180, -90, 90),
                      cmap=cmap, mincnt=1,
                      bins='log', # 3. THE MAGIC BULLET: Logarithmic color scaling
                      linewidths=0, edgecolors="none",
                      zorder=2)
        ax.axhline(0, color="#7a2d18", lw=0.4, alpha=0.45, zorder=3)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_title(f"{theme} — {int(m.sum()):,}",
                     fontsize=10, color="#222", weight="bold", pad=4)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=7, length=2)
        if i % ncols != 0:           ax.set_yticklabels([])
        if i // ncols != nrows - 1:  ax.set_xticklabels([])
    for j in range(n, nrows * ncols):
        axes[j].set_visible(False)

    fig.suptitle("Where on Mars is each science theme studied? — hex-density per theme",
                 fontsize=13, weight="bold", y=1.0)
    save_fig(fig, out / "06b_themed_map_smallmultiples.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out/'06b_themed_map_smallmultiples.png'}")


def fig_map_themed_dominant(df: pd.DataFrame, out: Path) -> None:
    """Single map: each hex coloured by the LOCALLY DOMINANT theme.

    Opacity scales with log(observations) so empty desert stays light, and
    high-density regions read crisply. This keeps the cognitive simplicity
    of one map while still surfacing per-region differences.
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return
    df = df.copy()
    themes, color_for, counts, primary = _theme_palette(df)
    df["primary_theme"] = primary

    lon = (((df["longitude"].values + 180) % 360) - 180)
    lat = df["latitude"].values

    # 2-degree bins everywhere — fine enough to show structure, coarse
    # enough that each cell gets a meaningful sample at full-corpus scale.
    nx, ny = 180, 90
    lon_edges = np.linspace(-180, 180, nx + 1)
    lat_edges = np.linspace( -90,  90, ny + 1)

    # Build a per-theme 2D histogram on the common grid.
    H = np.zeros((len(themes), ny, nx), dtype=np.int32)
    for i, t in enumerate(themes):
        m = df["primary_theme"] == t
        h, _, _ = np.histogram2d(lat[m.values], lon[m.values],
                                 bins=[lat_edges, lon_edges])
        H[i] = h.astype(np.int32)
    total = H.sum(axis=0)
    dominant = np.where(total > 0, H.argmax(axis=0), -1)

    # Compose the RGBA image.
    img = np.zeros((ny, nx, 4), dtype=float)
    for i, t in enumerate(themes):
        c = color_for[t]
        sel = dominant == i
        img[sel, 0] = c[0]; img[sel, 1] = c[1]; img[sel, 2] = c[2]
    log_total = np.log1p(total)
    norm = log_total / log_total.max() if log_total.max() > 0 else log_total
    # Slightly higher base opacity (0.40) to pop against white
    img[..., 3] = np.where(dominant >= 0, 0.40 + 0.60 * norm, 0.0)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    
    # --- REPLACE THE BACKGROUND ---
    ax.add_patch(plt.Rectangle((-180, -90), 360, 180,
                               facecolor="white",
                               edgecolor="none", zorder=0))
    ax.imshow(img,
              extent=(-180, 180, -90, 90), origin="lower",
              interpolation="nearest", aspect="auto", zorder=2)
    ax.axhline(0, color="#7a2d18", lw=0.6, alpha=0.5, zorder=3)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("east longitude (°)")
    ax.set_ylabel("planetocentric latitude (°)")
    ax.set_title("Dominant science theme per region of Mars\n"
                 "(hue = theme that wins the hex; opacity = observation density)",
                 fontsize=13)
    ax.set_aspect("equal", adjustable="box")

    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=color_for[t],
                             edgecolor="white", linewidth=0.5)
               for t in themes]
    labels = [f"{t}  ({counts[t]:,})" for t in themes]
    ax.legend(handles, labels, loc="lower left", fontsize=8.5,
              ncol=2, bbox_to_anchor=(1.005, 0),
              frameon=True, framealpha=0.95)
    fig.tight_layout()
    save_fig(fig, out / "06c_themed_map_dominant.png", bbox_inches="tight", )
    plt.close(fig)
    print(f"  -> {out/'06c_themed_map_dominant.png'}")


def fig_treemap(df: pd.DataFrame, out: Path) -> None:
    """Squarified treemap of theme proportions."""
    try:
        import squarify
    except ImportError:
        os.system(f"{sys.executable} -m pip install squarify --break-system-packages --quiet")
        import squarify

    th = theme_assignments(df)
    flat = [t for ts in th["themes"] for t in ts]
    counts = Counter(flat)
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels, vals = zip(*items)
    colors = MARS_CMAP(np.linspace(0.25, 0.9, len(labels)))

    fig, ax = plt.subplots(figsize=(12, 7))
    norm = squarify.normalize_sizes(vals, 100, 100)
    rects = squarify.squarify(norm, 0, 0, 100, 100)
    for r, lbl, v, c in zip(rects, labels, vals, colors):
        ax.add_patch(plt.Rectangle((r["x"], r["y"]), r["dx"], r["dy"],
                                   facecolor=c, edgecolor="white", linewidth=2.5))
        if r["dx"] * r["dy"] > 60:
            fs = max(8, min(22, int((r["dx"] * r["dy"]) ** 0.5 / 4)))
            ax.text(r["x"] + r["dx"] / 2, r["y"] + r["dy"] / 2,
                    f"{lbl}\n{v}",
                    ha="center", va="center", fontsize=fs,
                    color="white", weight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Treemap of HiRISE science themes — area = number of observations",
                 fontsize=13)
    fig.tight_layout()
    save_fig(fig, out / "07_themes_treemap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out/'07_themes_treemap.png'}")


def fig_themes_by_latitude_band(df: pd.DataFrame, out: Path) -> None:
    """Stacked-bar showing how themes shift between equator, mid-lat, and polar."""
    if "latitude" not in df.columns:
        return
    bands = pd.cut(
        df["latitude"],
        bins=[-90, -65, -30, 30, 65, 90],
        labels=["S polar", "S mid-lat", "Equatorial", "N mid-lat", "N polar"],
        ordered=True
    )
    th = theme_assignments(df)
    rows = []
    for band, themes in zip(bands, th["themes"]):
        for t in themes:
            rows.append((band, t))
    crosstab = (pd.DataFrame(rows, columns=["band", "theme"])
                  .pivot_table(index="band", columns="theme",
                               aggfunc=len, fill_value=0,
                               observed=False))
    # Order columns by overall popularity for a clean look
    order = crosstab.sum().sort_values(ascending=False).index
    crosstab = crosstab[order]

    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Grab the centralized mapping
    themes, color_for, counts, primary = _theme_palette(df)
    
    bottom = np.zeros(len(crosstab))
    for col in crosstab.columns:
        vals = crosstab[col].values
        # Look up the globally consistent color
        c = color_for.get(col, "#888888") 
        ax.bar(crosstab.index.astype(str), vals, bottom=bottom,
               label=col, color=c, edgecolor="white", linewidth=0.5)
        bottom += vals
        
    ax.set_ylabel("observations")
    ax.set_title("Where on Mars does each theme cluster? — themes per latitude band")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out / "08_themes_by_latitude.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out/'08_themes_by_latitude.png'}")


# ---------- main --------------------------------------------------------------

def make_all(df: pd.DataFrame, outdir: str | Path) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {out.resolve()}")
    fig_wordcloud(df, out)
    fig_wordcloud_mars_disk(df, out)
    fig_top_words_and_phrases(df, out)
    fig_geographic_features(df, out)
    fig_themes_bar(df, out)
    fig_map_themed(df, out)
    fig_map_themed_smallmultiples(df, out)
    fig_map_themed_dominant(df, out)
    fig_treemap(df, out)
    fig_themes_by_latitude_band(df, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="/scratch/mars_hirise/RDRCUMINDEX.TAB", help="path to .TAB or .csv")
    p.add_argument("--outdir", default="outputs/figures/rational",
                   help="where to write the .png figures (default: figures/)")
    args = p.parse_args()
    df = load(args.input)
    make_all(df, args.outdir)


if __name__ == "__main__":
    main()