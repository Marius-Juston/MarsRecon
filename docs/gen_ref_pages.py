"""Auto-generate API reference pages and a SUMMARY.md nav for the `src/` tree.

Run automatically by the mkdocs-gen-files plugin during `mkdocs build`.
For each Python module under `src/`, emit a one-line stub that mkdocstrings
expands into a full API page via griffe's static-analysis backend (no runtime
import of optional heavy deps required).
"""
from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

SRC_ROOT = Path("src").resolve()
REF_ROOT = Path("reference")

SKIP_PARTS = {"__pycache__"}
SKIP_PREFIXES = (
    "depth_fm/models/unet",  # vendored CompVis LDM — frozen upstream code
)
# Explicit module skips. These mirror the [tool.coverage.run].omit list in
# pyproject.toml — heavy entry-point scripts whose docstrings reference
# missing imports or external CLI usage and that griffe can fail to collect
# in some environments.
SKIP_MODULES = {
    "dataset.validation.sampling_diagnostics",
    "dataset.stats.compute_stats",
    "dataset.stats.compute_stats_litdata",
    "clip.build_marsclip_split_manifest",
}

nav = mkdocs_gen_files.Nav()

for path in sorted(SRC_ROOT.rglob("*.py")):
    rel = path.relative_to(SRC_ROOT)
    parts = rel.with_suffix("").parts

    if any(p in SKIP_PARTS for p in parts):
        continue

    rel_posix = rel.with_suffix("").as_posix()
    if any(rel_posix.startswith(prefix) for prefix in SKIP_PREFIXES):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue
        doc_path = Path(*parts, "index.md")
    elif parts[-1] == "__main__":
        continue
    else:
        doc_path = Path(*parts).with_suffix(".md")

    module_path = ".".join(parts)
    if module_path in SKIP_MODULES:
        continue
    full_doc_path = REF_ROOT / doc_path

    nav[tuple(parts)] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{module_path}`\n\n")
        fd.write(f"::: {module_path}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("..") / "src" / rel)

with mkdocs_gen_files.open(REF_ROOT / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
