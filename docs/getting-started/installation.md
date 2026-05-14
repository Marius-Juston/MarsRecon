# Installation

## Prerequisites

- **Python 3.12** (enforced by `pyproject.toml`).
- **[uv](https://docs.astral.sh/uv/)** as the package manager.
- **GDAL ≥ 3.6** available on the system (needed by `rasterio`). On Ubuntu:
  `sudo apt-get install gdal-bin libgdal-dev`.
- **Graphviz** (optional) for rendering pipeline diagrams: `sudo apt-get install graphviz`.

## Install

```bash
git clone https://github.com/Marius-Juston/MarsRecon.git
cd MarsRecon

# Runtime only
uv sync

# With test dependencies
uv sync --extra dev

# Full training stack (Lightning, rasterio, transformers, ...)
uv sync --extra depthfm

# Documentation site
uv sync --extra docs
```

## Optional dependency groups

| Group        | What it adds                                                                  |
|--------------|-------------------------------------------------------------------------------|
| `dev`        | `pytest`, `pytest-cov`, `pytest-mock`, `pytest-asyncio`, `pytest-xdist`       |
| `experiment` | `wandb`                                                                       |
| `depthfm`    | Lightning, LitData, rasterio, transformers, diffusers, xgboost, scikit-image… |
| `viz`        | `graphviz` (Python bindings), `cairosvg`                                      |
| `docs`       | `properdocs`, `mkdocs-material`, `mkdocstrings`, mermaid / lightbox plugins   |

You can combine groups: `uv sync --extra depthfm --extra dev --extra docs`.

## Verify the install

```bash
uv run pytest tests/ -m "not integration" -n auto
```

For the documentation site (built with [ProperDocs](https://properdocs.org/), a drop-in
continuation of MkDocs 1.x):

```bash
uv run properdocs serve
# open http://127.0.0.1:8000
```
