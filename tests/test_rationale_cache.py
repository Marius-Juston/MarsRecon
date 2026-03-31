"""Tests for offline rationale-expansion cache utilities."""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rationale_cache import (
    DEFAULT_PROMPT_TEMPLATE,
    _clean_text,
    build_rationale_expansion_cache,
    load_rationale_cache,
    merge_rationale_cache,
    render_expansion_prompt,
    save_rationale_cache,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"obs_id": "OBS_A", "rationale_desc": "Olympus Mons lava channels"},
            {"obs_id": "OBS_B", "rationale_desc": "Karzok Crater on Olympus Mons"},
        ]
    )


def test_render_prompt_includes_rationale_text():
    prompt = render_expansion_prompt("Olympus Mons lava channels")
    assert "Olympus Mons lava channels" in prompt
    assert "cautious Mars geological description" in prompt


def test_build_cache_creates_expanded_rows():
    seen_prompts: list[str] = []

    def expander(prompt: str) -> str:
        seen_prompts.append(prompt)
        return f"Expanded: {prompt.splitlines()[2]}"

    cache_df = build_rationale_expansion_cache(
        _manifest(),
        expander,
        expansion_model="mock-llm",
    )

    assert len(cache_df) == 2
    assert cache_df["expansion_status"].tolist() == ["ok", "ok"]
    assert cache_df["expansion_model"].tolist() == ["mock-llm", "mock-llm"]
    assert len(seen_prompts) == 2
    assert cache_df.loc[0, "rationale_expanded"].startswith("Expanded:")


def test_build_cache_reuses_existing_success_rows():
    calls = 0

    def expander(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "new expansion"

    existing = pd.DataFrame(
        [
            {
                "obs_id": "OBS_A",
                "rationale_raw": "Olympus Mons lava channels",
                "rationale_expanded": "cached expansion",
                "expansion_model": "mock-llm",
                "prompt_version": "v1",
                "prompt_template": DEFAULT_PROMPT_TEMPLATE,
                "expansion_status": "ok",
                "expansion_error": None,
                "expansion_timestamp": pd.Timestamp("2026-03-29T00:00:00Z"),
            }
        ]
    )

    cache_df = build_rationale_expansion_cache(
        _manifest(),
        expander,
        expansion_model="mock-llm",
        cache=existing,
    )

    assert calls == 1
    obs_a = cache_df.loc[cache_df["obs_id"] == "OBS_A"].iloc[0]
    obs_b = cache_df.loc[cache_df["obs_id"] == "OBS_B"].iloc[0]
    assert obs_a["rationale_expanded"] == "cached expansion"
    assert obs_b["rationale_expanded"] == "new expansion"


def test_build_cache_records_errors():
    def expander(prompt: str) -> str:
        raise RuntimeError("boom")

    cache_df = build_rationale_expansion_cache(
        _manifest().iloc[[0]],
        expander,
        expansion_model="mock-llm",
    )

    assert cache_df.loc[0, "expansion_status"] == "error"
    assert cache_df.loc[0, "rationale_expanded"] is None
    assert "boom" in cache_df.loc[0, "expansion_error"]


def test_cache_can_roundtrip_to_parquet(tmp_path):
    out_path = tmp_path / "rationale_cache.parquet"

    cache_df = build_rationale_expansion_cache(
        _manifest().iloc[[0]],
        lambda prompt: "expanded text",
        expansion_model="mock-llm",
        out_path=out_path,
    )
    loaded = load_rationale_cache(out_path)

    assert out_path.exists()
    assert loaded.loc[0, "rationale_expanded"] == "expanded text"
    assert pd.Timestamp(loaded.loc[0, "expansion_timestamp"]).tz is not None
    assert list(loaded.columns) == list(cache_df.columns)


def test_merge_rationale_cache_adds_expanded_text():
    manifest = pd.DataFrame(
        [
            {"obs_id": "OBS_A", "rationale_desc": "raw a"},
            {"obs_id": "OBS_B", "rationale_desc": "raw b"},
        ]
    )
    cache = pd.DataFrame(
        [
            {
                "obs_id": "OBS_A",
                "rationale_raw": "raw a",
                "rationale_expanded": "expanded a",
                "expansion_model": "mock-llm",
                "prompt_version": "v1",
                "prompt_template": DEFAULT_PROMPT_TEMPLATE,
                "expansion_status": "ok",
                "expansion_error": None,
                "expansion_timestamp": pd.Timestamp("2026-03-29T00:00:00Z"),
            },
            {
                "obs_id": "OBS_B",
                "rationale_raw": "raw b",
                "rationale_expanded": None,
                "expansion_model": "mock-llm",
                "prompt_version": "v1",
                "prompt_template": DEFAULT_PROMPT_TEMPLATE,
                "expansion_status": "error",
                "expansion_error": "boom",
                "expansion_timestamp": pd.Timestamp("2026-03-29T00:00:00Z"),
            },
        ]
    )

    merged = merge_rationale_cache(manifest, cache)

    obs_a = merged.loc[merged["obs_id"] == "OBS_A"].iloc[0]
    obs_b = merged.loc[merged["obs_id"] == "OBS_B"].iloc[0]
    assert obs_a["rationale_expanded"] == "expanded a"
    assert bool(obs_a["has_rationale_expanded"])
    assert obs_b["rationale_expanded"] is None
    assert not bool(obs_b["has_rationale_expanded"])


def test_clean_text_returns_none_for_none_input():
    assert _clean_text(None) is None


def test_load_rationale_cache_returns_empty_for_nonexistent_path(tmp_path):
    result = load_rationale_cache(tmp_path / "missing.parquet")
    assert result.empty


def test_load_rationale_cache_reads_parquet_file(tmp_path):
    df = pd.DataFrame([{
        "obs_id": "OBS_A",
        "expansion_status": "ok",
        "rationale_expanded": "Some expanded text.",
        "expansion_model": "test-model",
        "expansion_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
        "prompt_version": "v1",
    }])
    path = tmp_path / "cache.parquet"
    df.to_parquet(path, index=False)
    loaded = load_rationale_cache(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["obs_id"] == "OBS_A"


def test_load_rationale_cache_fills_missing_columns(tmp_path):
    df = pd.DataFrame([{
        "obs_id": "OBS_B",
        "expansion_status": "ok",
        "expansion_timestamp": "2026-01-01T00:00:00+00:00",
    }])
    path = tmp_path / "partial.csv"
    df.to_csv(path, index=False)
    loaded = load_rationale_cache(path)
    assert "rationale_expanded" in loaded.columns
    assert "expansion_model" in loaded.columns


def test_save_rationale_cache_writes_parquet(tmp_path):
    cache = pd.DataFrame([{
        "obs_id": "OBS_A",
        "expansion_status": "ok",
        "rationale_expanded": "text",
        "expansion_model": "m",
        "expansion_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
        "prompt_version": "v1",
    }])
    out_path = tmp_path / "cache.parquet"
    save_rationale_cache(cache, out_path)
    assert out_path.exists()
    loaded = pd.read_parquet(out_path)
    assert loaded.iloc[0]["obs_id"] == "OBS_A"


def test_save_rationale_cache_writes_csv(tmp_path):
    """Line 98: save_rationale_cache uses CSV for non-parquet extensions."""
    cache = pd.DataFrame([{
        "obs_id": "OBS_A",
        "expansion_status": "ok",
        "rationale_expanded": "text",
        "expansion_model": "m",
        "expansion_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
        "prompt_version": "v1",
    }])
    out_path = tmp_path / "cache.csv"
    save_rationale_cache(cache, out_path)
    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert loaded.iloc[0]["obs_id"] == "OBS_A"


def test_merge_rationale_cache_with_empty_cache_adds_none_columns():
    manifest = pd.DataFrame([{"obs_id": "OBS_A", "rationale_desc": "text"}])
    result = merge_rationale_cache(manifest, None)
    assert "rationale_expanded" in result.columns
    assert "has_rationale_expanded" in result.columns
    assert result.iloc[0]["rationale_expanded"] is None
    assert not bool(result.iloc[0]["has_rationale_expanded"])
