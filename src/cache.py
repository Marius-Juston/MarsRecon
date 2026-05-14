"""Canonical cache-key computation and path conventions for MarsRecon.

All hashing logic lives here. Other modules import from this module to
ensure identical cache locations are computed regardless of which pipeline
component constructs the path.

Cache directory layout relative to <data_root>:
  .cache/
    litdata/<hash>/              # LitData streaming chunks (train/ val/ test/)
    litdata_raw/<hash>/          # build_litdata_raw output (train/ val/ test/)
    manifests/<hash>.parquet     # Adapter manifest parquet files
    sampler_splits/<hash>.json   # Train/val/test split assignments
    spatial/                     # GeoPackage spatial indexes
      spatial_cache[_<suffix>]_<version>.gpkg
  spatial_cache_*.gpkg           # Legacy root-level location (not written by new code)

Usage:
    from cache import (
        compute_hash, litdata_cache_key, litdata_cache_root,
        manifest_cache_dir, sampler_split_cache_dir, spatial_cache_dir,
        write_manifest, validate_cache, run_id_hash,
    )
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HASH_LENGTH = 16  # hex chars (64-bit entropy) — matches all existing caches

# ---------------------------------------------------------------------------
# Core hash primitive
# ---------------------------------------------------------------------------

def compute_hash(data: dict[str, Any], length: int = HASH_LENGTH) -> str:
    """SHA-256 of a JSON-serialised dict with sorted keys.

    Stable across Python restarts and dict insertion orders. This is the
    single hashing primitive for the entire project — never call hashlib
    directly in other modules.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    """SHA-256 checksum of a file, read in streaming 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cache key functions
# ---------------------------------------------------------------------------

def litdata_cache_key(config) -> str:
    """Deterministic hash over all fields that affect LitData chunk content.

    Always includes clip — this fixes a bug in the old litdata_datamodule.py
    where clip was omitted, causing it to look for a different directory than
    build_litdata.py produced.
    """
    from omegaconf import OmegaConf
    key_parts: dict[str, Any] = {
        "hirise": OmegaConf.to_container(config.data.hirise, resolve=True),
        "sampler": OmegaConf.to_container(config.data.sampler, resolve=True),
        "resolution": config.data.get("resolution", 512),
        "dtm_normalization": config.data.get("dtm_normalization", "relative"),
        "clip": config.data.get("clip", False),
    }
    return compute_hash(key_parts)


def litdata_cache_root(config) -> Path:
    """Canonical root directory for the LitData streaming cache."""
    return Path(config.data.hirise.root) / ".cache" / "litdata" / litdata_cache_key(config)


def litdata_split_dir(config, split: str) -> Path:
    return litdata_cache_root(config) / split


def litdata_tmp_dir(config, split: str) -> Path:
    """Temporary extraction directory for LitData build (auto-deleted on success)."""
    return litdata_cache_root(config).parent / f"_tmp_{litdata_cache_key(config)}_{split}"


def manifest_cache_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / ".cache" / "manifests"


def sampler_split_cache_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / ".cache" / "sampler_splits"


def spatial_cache_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / ".cache" / "spatial"


def run_id_hash(config, length: int = 8) -> str:
    """Short identifier for training-run logs — NOT a cache key.

    Uses 8 hex chars (32-bit) to avoid collisions across a handful of
    concurrent runs without the verbosity of a full 16-char hash.
    """
    from omegaconf import OmegaConf
    raw = json.dumps(
        OmegaConf.to_container(config, resolve=True),
        sort_keys=True, default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


# ---------------------------------------------------------------------------
# Reproducibility manifest
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _key_package_versions() -> dict[str, str]:
    packages = ["torch", "lightning", "litdata", "numpy", "rasterio", "omegaconf"]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def _select_checksum_files(cache_path: Path) -> list[Path]:
    """Pick a deterministic, representative subset of files to checksum.

    For a directory: checksums the _SUCCESS marker + up to 8 evenly-spaced
    chunk files so the selection is stable (same files chosen for the same
    cache regardless of when validation runs).

    For a single file: checksums just that file.
    """
    if cache_path.is_file():
        return [cache_path]

    targets: list[Path] = []
    success = cache_path / "_SUCCESS"
    if success.exists():
        targets.append(success)

    chunks = sorted(f for f in cache_path.rglob("*") if f.is_file() and f.name != "_SUCCESS")
    if chunks:
        step = max(1, len(chunks) // 8)
        targets.extend(chunks[i] for i in range(0, len(chunks), step))

    return targets[:9]  # cap at 9 (1 _SUCCESS + 8 chunks)


@dataclass
class CacheManifest:
    """Reproducibility metadata written alongside every cache entry."""

    cache_hash: str
    config_snapshot: dict
    created_at: str
    git_commit: str
    python_version: str
    platform: str
    key_packages: dict[str, str]
    file_checksums: dict[str, str] = field(default_factory=dict)


def write_manifest(
        cache_path: Path,
        cache_hash: str,
        config_snapshot: dict,
        checksum_files: list[Path] | None = None,
) -> Path:
    """Write a _MANIFEST.json sidecar alongside cache_path.

    For a directory: writes <cache_path>/_MANIFEST.json
    For a file:      writes <cache_path.parent>/<stem>_MANIFEST.json

    Returns the path of the written manifest.
    """
    if checksum_files is None:
        checksum_files = _select_checksum_files(cache_path)

    checksums: dict[str, str] = {}
    for f in checksum_files:
        try:
            rel = str(f.relative_to(cache_path.parent if cache_path.is_file() else cache_path))
        except ValueError:
            rel = f.name
        try:
            checksums[rel] = file_sha256(f)
        except OSError as e:
            logger.warning("Could not checksum %s: %s", f, e)

    manifest = CacheManifest(
        cache_hash=cache_hash,
        config_snapshot=config_snapshot,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_commit=_git_commit(),
        python_version=sys.version,
        platform=platform.platform(),
        key_packages=_key_package_versions(),
        file_checksums=checksums,
    )

    if cache_path.is_dir():
        manifest_path = cache_path / "_MANIFEST.json"
    else:
        manifest_path = cache_path.parent / f"{cache_path.stem}_MANIFEST.json"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    logger.info("Wrote cache manifest to %s", manifest_path)
    return manifest_path


def _manifest_path(cache_path: Path) -> Path:
    if cache_path.is_dir():
        return cache_path / "_MANIFEST.json"
    return cache_path.parent / f"{cache_path.stem}_MANIFEST.json"


def validate_cache(
        cache_path: Path,
        config_snapshot: dict | None = None,
        verify_checksums: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a cache entry against its manifest.

    Args:
        cache_path:       Directory or file that was cached.
        config_snapshot:  If provided, verify the manifest's config matches.
        verify_checksums: If True, recompute SHA-256 for recorded files
                          (slow for large caches — off by default).

    Returns:
        (is_valid, issues) — issues is an empty list on success.
    """
    issues: list[str] = []
    mp = _manifest_path(cache_path)

    if not mp.exists():
        issues.append(f"Manifest not found: {mp}")
        return False, issues

    try:
        data = json.loads(mp.read_text())
    except json.JSONDecodeError as e:
        issues.append(f"Manifest JSON invalid: {e}")
        return False, issues

    stored_hash = data.get("cache_hash", "")
    if config_snapshot is not None:
        expected_hash = compute_hash(config_snapshot)
        if stored_hash != expected_hash:
            issues.append(
                f"Hash mismatch: manifest has {stored_hash!r}, "
                f"current config produces {expected_hash!r}"
            )
    elif not stored_hash:
        issues.append("Manifest missing cache_hash field")

    if verify_checksums:
        base = cache_path if cache_path.is_dir() else cache_path.parent
        for rel, expected in data.get("file_checksums", {}).items():
            fpath = base / rel
            if not fpath.exists():
                issues.append(f"Checksummed file missing: {fpath}")
                continue
            actual = file_sha256(fpath)
            if actual != expected:
                issues.append(
                    f"Checksum mismatch for {rel}: "
                    f"expected {expected[:16]}…, got {actual[:16]}…"
                )

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_validate(args: argparse.Namespace) -> int:
    cache_path = Path(args.path)
    if not cache_path.exists():
        print(f"ERROR: path does not exist: {cache_path}", file=sys.stderr)
        return 1

    ok, issues = validate_cache(
        cache_path,
        verify_checksums=args.checksums,
    )
    if ok:
        print(f"OK  {cache_path}")
        mp = _manifest_path(cache_path)
        data = json.loads(mp.read_text())
        print(f"    created : {data.get('created_at', '?')}")
        print(f"    git     : {data.get('git_commit', '?')[:12]}")
        print(f"    hash    : {data.get('cache_hash', '?')}")
        return 0
    else:
        print(f"INVALID  {cache_path}")
        for issue in issues:
            print(f"  • {issue}")
        return 1


def main() -> None:
    """CLI entry point: python -m cache validate <path> [--checksums]"""
    parser = argparse.ArgumentParser(
        prog="python -m cache",
        description="MarsRecon cache utility",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    val_parser = sub.add_parser("validate", help="Validate a cache entry against its manifest")
    val_parser.add_argument("path", help="Cache directory or file to validate")
    val_parser.add_argument(
        "--checksums", action="store_true",
        help="Recompute SHA-256 for recorded files (slow for large caches)",
    )

    args = parser.parse_args()
    sys.exit(_cli_validate(args))


if __name__ == "__main__":
    main()
