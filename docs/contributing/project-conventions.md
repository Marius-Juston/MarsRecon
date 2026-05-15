# Project conventions

## Package layout rules

- `src/` is the import root. Tests handle this in `tests/conftest.py`; scripts need
  `PYTHONPATH=src` set explicitly.
- The three top-level packages are `dataset/`, `depth_fm/`, and `clip/`. Cross-package
  imports flow `clip → dataset` and `depth_fm → dataset`, never the reverse.
- Entry-point scripts live under `scripts/` and are **not part of the importable library**.
  Library code lives in `src/`.

## Where to put new code

Use the routing table from the [overview](../architecture/overview.md). A few of the most
common entries:

| Task                            | File                                        |
|---------------------------------|---------------------------------------------|
| Add a loss                      | `src/depth_fm/objectives/losses.py`         |
| Add a metric                    | `src/depth_fm/objectives/metrics.py`        |
| Change training loop            | `src/depth_fm/training/lightning_module.py` |
| Change sampling / split logic   | `src/dataset/sampling/sampler.py`           |
| Change PDS download / footprint | `src/dataset/core/base.py`                  |
| New publication figure          | `scripts/visualization/`                    |

## Style

- Public APIs use **Google-style docstrings** — `mkdocstrings` is configured to render them.
- Type hints on all public function signatures.
- Prefer dataclasses over kwargs-on-kwargs.
- No emojis in code or comments unless explicitly requested.

## Commits & PRs

- Branch from `master`. Open PRs into `master`.
- CI runs the docs build on every PR that touches `src/`, `docs/`, `properdocs.yml`, or
  `scripts/architecture/`. A failing build blocks merge.
- Tests run under `uv run pytest tests/ -m "not integration" -n auto`.
