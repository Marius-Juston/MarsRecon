---
name: project-test-infra
description: MarsRecon test-environment traps - ROS/lark pollution, xdist coverage, manifest glob
metadata:
  type: project
---

Non-obvious test-infra facts for MarsRecon (not in CLAUDE.md):

- **Run pytest as** `PYTHONPATH= .venv/bin/python -m pytest <args> -p no:cacheprovider`.
  A system ROS install injects a setuptools entrypoint; without the empty
  `PYTHONPATH=` collection dies with `ModuleNotFoundError: No module named 'lark'`.
  `uv run pytest` does NOT work here.
  **Why:** machine-level ROS contamination.
  **How to apply:** always use this exact invocation; coverage with
  `--cov=src/dataset -n auto`.

- **Single-file coverage reports "No data to report".** pytest-cov + the
  src-root layout only produces a report when running a directory
  (`tests/dataset_pkg/`), not one file/class. To check progress, run the whole
  package dir; per-test debugging must rely on pass/fail, not term-missing.

- **xdist suppresses `--cov-report=term-missing` detail** unless the run
  completes; grep the TOTAL/per-file lines from the tail.

- **Manifest glob trap:** `cache.write_manifest` writes
  `split_<hash>_MANIFEST.json` next to `split_<hash>.json`. Test code that does
  `tmp_path.rglob("split_*.json")` + `next(...)` is order-dependent and flaky
  under xdist (sometimes grabs the manifest, KeyError 'assignments'). Always
  filter out `*_MANIFEST.json`. Fixed in test_sampler_cache.py.

- Sampler `HiRISEGeoSampler.__init__` defaults `center_mode="optimal"` (the
  classmethod/`__init__` default), even though the field annotation elsewhere
  shows `"simple"`. Tests must pass `center_mode="simple"` explicitly to
  exercise `_build_valid_centers_simple`.

- Geometry brentq/buffer(0) edge branches are only reachable by monkeypatching
  `dataset.sampling.geometry.brentq` / `_get_optimized_angles`; realistic
  polygons don't trigger them.

See [[user-role]].
