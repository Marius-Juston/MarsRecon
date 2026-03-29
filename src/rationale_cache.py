"""Offline rationale-expansion cache utilities for MarsCLIP."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable

import pandas as pd

DEFAULT_PROMPT_TEMPLATE = """Expand this HiRISE observation rationale into a cautious Mars geological description.

Rationale: {rationale}

Include:
1. likely surface morphology
2. plausible geological context
3. possible scientific significance

Do not invent specific mineral detections or facts not supported by the rationale.
Separate direct information from reasonable inference.
Explicitly mention uncertainty when needed.
"""

_CACHE_COLUMNS = [
    "obs_id",
    "rationale_raw",
    "rationale_expanded",
    "expansion_model",
    "prompt_version",
    "prompt_template",
    "expansion_status",
    "expansion_error",
    "expansion_timestamp",
]


def _clean_text(value: str | None) -> str | None:
    """Normalize whitespace in cache text fields."""
    if value is None:
        return None
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def render_expansion_prompt(
    rationale: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    """Render the LLM prompt for one observation rationale."""
    return prompt_template.format(rationale=_clean_text(rationale) or "")


def _empty_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=_CACHE_COLUMNS)


def load_rationale_cache(
    cache: pd.DataFrame | pathlib.Path | str | None,
) -> pd.DataFrame:
    """Load a rationale-expansion cache from a DataFrame or on-disk file."""
    if cache is None:
        return _empty_cache()

    if isinstance(cache, pd.DataFrame):
        df = cache.copy()
    else:
        path = pathlib.Path(cache)
        if not path.exists():
            return _empty_cache()
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, parse_dates=["expansion_timestamp"])

    for col in _CACHE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[_CACHE_COLUMNS].copy()
    if "expansion_timestamp" in df.columns:
        df["expansion_timestamp"] = pd.to_datetime(
            df["expansion_timestamp"], utc=True, errors="coerce"
        )
    return df


def save_rationale_cache(
    cache_df: pd.DataFrame,
    out_path: pathlib.Path | str,
) -> None:
    """Persist a rationale-expansion cache to parquet or CSV."""
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        cache_df.to_parquet(path, index=False)
    else:
        cache_df.to_csv(path, index=False)


def build_rationale_expansion_cache(
    manifest: pd.DataFrame,
    expander: Callable[[str], str],
    *,
    expansion_model: str,
    cache: pd.DataFrame | pathlib.Path | str | None = None,
    out_path: pathlib.Path | str | None = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    prompt_version: str = "v1",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build or extend an offline rationale-expansion cache."""
    existing = load_rationale_cache(cache)
    rows = manifest[["obs_id", "rationale_desc"]].drop_duplicates("obs_id").copy()
    rows["rationale_desc"] = rows["rationale_desc"].map(_clean_text)

    cache_key = (
        existing["obs_id"].astype(str)
        + "||"
        + existing["rationale_raw"].fillna("").astype(str)
        + "||"
        + existing["expansion_model"].fillna("").astype(str)
        + "||"
        + existing["prompt_version"].fillna("").astype(str)
    )
    existing_lookup = {key: row for key, (_, row) in zip(cache_key, existing.iterrows())}

    built_rows: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        obs_id = str(row["obs_id"])
        rationale_raw = _clean_text(row["rationale_desc"]) or ""
        key = f"{obs_id}||{rationale_raw}||{expansion_model}||{prompt_version}"

        existing_row = existing_lookup.get(key)
        if (
            not overwrite
            and existing_row is not None
            and existing_row["expansion_status"] == "ok"
            and pd.notna(existing_row["rationale_expanded"])
        ):
            built_rows.append(existing_row.to_dict())
            continue

        prompt = render_expansion_prompt(
            rationale=rationale_raw,
            prompt_template=prompt_template,
        )
        timestamp = pd.Timestamp.now("UTC")
        try:
            expanded = _clean_text(expander(prompt))
            built_rows.append(
                {
                    "obs_id": obs_id,
                    "rationale_raw": rationale_raw,
                    "rationale_expanded": expanded,
                    "expansion_model": expansion_model,
                    "prompt_version": prompt_version,
                    "prompt_template": prompt_template,
                    "expansion_status": "ok",
                    "expansion_error": None,
                    "expansion_timestamp": timestamp,
                }
            )
        except Exception as exc:  # noqa: BLE001
            built_rows.append(
                {
                    "obs_id": obs_id,
                    "rationale_raw": rationale_raw,
                    "rationale_expanded": None,
                    "expansion_model": expansion_model,
                    "prompt_version": prompt_version,
                    "prompt_template": prompt_template,
                    "expansion_status": "error",
                    "expansion_error": str(exc),
                    "expansion_timestamp": timestamp,
                }
            )

    cache_df = pd.DataFrame.from_records(built_rows, columns=_CACHE_COLUMNS)
    if out_path is not None:
        save_rationale_cache(cache_df, out_path)
    return cache_df


def merge_rationale_cache(
    manifest: pd.DataFrame,
    cache: pd.DataFrame | pathlib.Path | str,
) -> pd.DataFrame:
    """Merge the latest successful rationale expansion onto a manifest."""
    cache_df = load_rationale_cache(cache)
    if cache_df.empty:
        merged = manifest.copy()
        merged["rationale_expanded"] = None
        merged["has_rationale_expanded"] = False
        return merged

    ok_rows = cache_df[cache_df["expansion_status"] == "ok"].copy()
    ok_rows = ok_rows.sort_values("expansion_timestamp")
    ok_rows = ok_rows.drop_duplicates("obs_id", keep="last")

    keep_cols = [
        "obs_id",
        "rationale_expanded",
        "expansion_model",
        "prompt_version",
        "expansion_timestamp",
    ]
    merged = manifest.merge(ok_rows[keep_cols], on="obs_id", how="left")
    merged["rationale_expanded"] = merged["rationale_expanded"].astype(object)
    merged["rationale_expanded"] = merged["rationale_expanded"].where(
        pd.notna(merged["rationale_expanded"]), None
    )
    merged["has_rationale_expanded"] = merged["rationale_expanded"].notna()
    return merged
