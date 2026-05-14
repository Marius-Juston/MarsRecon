# Testing

## Running the suite

```bash
uv run pytest tests/ -m "not integration" -n auto                   # unit only
uv run pytest tests/ --cov=src --cov-report=term-missing -n auto    # with coverage
```

`tests/conftest.py` adds `src/` to `sys.path` so tests don't need `PYTHONPATH=src`.

## Markers

| Marker        | Meaning                                                                   |
|---------------|---------------------------------------------------------------------------|
| `integration` | Requires real HiRISE data at `/scratch/mars_hirise`. Excluded by default. |

```bash
uv run pytest tests/ -m integration                # run only integration tests
```

## Coverage config

`pyproject.toml` excludes a few heavy entry-point modules from coverage to keep numbers
meaningful:

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "src/dataset/validation/sampling_diagnostics.py",
    "src/build_marsclip_split_manifest.py",
    "src/dataset/stats/compute_stats.py",
    "src/dataset/stats/compute_stats_litdata.py",
]
```

## Known broken tests (post-refactor)

See [Refactor notes](../technical-notes/refactor-notes.md) for the exact list of tests whose
string-based `mock.patch(...)` paths still reference pre-refactor module names.
**Functionality is unchanged — only the test patches need updating.**
