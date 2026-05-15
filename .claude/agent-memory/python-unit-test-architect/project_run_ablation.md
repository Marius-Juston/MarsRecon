---
name: project-run-ablation
description: run_ablation.py test notes - script-load pattern, real BadZipFile bug, watchdog dropna gotcha
metadata:
  type: project
---

Testing `scripts/training/run_ablation.py` (not importable; load via importlib
from file path under name "run_ablation_under_test"). Suite at
`tests/test_run_ablation.py` (63 pass, 1 xfail). Run with the standard
`PYTHONPATH= .venv/bin/python -m pytest -p no:cacheprovider` invocation.

**Real bug found (source intentionally NOT changed):**
`_load_per_patch` catches `(OSError, ValueError, EOFError)` but a truncated
npz — the documented "job killed mid-write" scenario — raises
`zipfile.BadZipFile` (also `pickle.UnpicklingError` on some truncations),
neither of which subclasses the caught types. So one corrupt rank file aborts
the entire aggregation. Locked down with a `strict=True` xfail
(`test_truncated_npz_is_a_real_bug`). Fix would be to add `zipfile.BadZipFile`
+ `pickle.UnpicklingError` (or just `Exception`) to the except tuple.
**Why:** [[user-role]] wants real bugs reported, not silently patched.

**Watchdog gotcha:** `watchdog_check` calls `df[col].dropna()` BEFORE the
finite check, so a trailing literal NaN is silently removed and never trips
the guard. Only `+/-Inf` (survives dropna) actually triggers
"val/rmse_mean is NaN/Inf". Tests must use Inf, not NaN.

Watchdog stall thresholds: grace=1800s hardcoded, default
val_stall_timeout=7200s -> no-val-row kill at >9000s elapsed since
started_at; post-first-row stall kill at >timeout since last_val_seen_at.
Monkeypatch `ra.time.monotonic` to drive these deterministically.

`RunningJob` dataclass has NO `csv_path` field (removed) — current fields:
job, proc, lane, run_dir, log_file, last_val_seen_at, last_val_row_count,
started_at. `_NVSMI_WARNED` / `_SHUTDOWN` are module globals; reset in an
autouse fixture for order-independence under -n auto.

pytest-randomly is NOT installed in this venv (can't enforce random order).

See [[project-test-infra]], [[user-role]].
