"""Shared helpers for files in `scripts/`.

These utilities exist so that visualization, analysis, and figure-generation
scripts don't each re-implement matplotlib boilerplate, `save_fig`, or JSON/CSV
loading. New scripts should prefer importing from here; existing scripts that
inline these helpers can be migrated opportunistically.
"""
