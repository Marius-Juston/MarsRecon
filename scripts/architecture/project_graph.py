"""Generate full-project import graphs as SVGs for the documentation site.

Renders one SVG per top-level package into ``docs/diagrams/`` using `pydeps`_.
The graphs include cross-package edges (e.g. ``depth_fm → dataset``) so the
collection acts as a "full project graph" — taken together they cover every
internal import in the codebase.

Outputs:

* ``project_graph_dataset.svg``  — every module under ``dataset``.
* ``project_graph_depth_fm.svg`` — every module under ``depth_fm`` plus
                                   inbound/outbound edges to ``dataset``.
* ``project_graph_clip.svg``     — every module under ``clip`` plus edges
                                   to ``dataset``.

pydeps parses imports statically (AST-based) so this script does **not**
require the heavy training stack to be installed — only ``pydeps`` and
Graphviz's ``dot`` need to be on PATH.

Run directly::

    uv run --extra docs python scripts/architecture/project_graph.py

.. _pydeps: https://pydeps.readthedocs.io/
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
OUT_DIR = REPO_ROOT / "docs" / "diagrams"

# Shared pydeps flags:
#   --noshow            don't open a browser
#   --no-config         ignore any user ~/.pydeps file
#   --cluster           visually group modules by package
#   --max-bacon 16      "show everything within 16 hops" — effectively unlimited
#                        for our codebase; ``0`` would mean "root only".
#   --rankdir LR        left-to-right layout (more readable for tall graphs)
#   --reverse           edges point from importee to importer (visually nicer)
COMMON_OPTS = [
    "--noshow",
    "--no-config",
    "--cluster",
    "--max-bacon",
    "16",
    "--rankdir",
    "LR",
]


def _check_prerequisites() -> None:
    if shutil.which("dot") is None:
        sys.exit(
            "error: 'dot' (Graphviz) is not on PATH. "
            "Install with `sudo apt-get install graphviz` (Linux) or "
            "`brew install graphviz` (macOS)."
        )


def _render(target: Path, out_name: str, only: list[str], extra: list[str] | None = None) -> bool:
    """Run pydeps for `target`. Return True on success.

    Args:
        target: Directory of the package to analyse.
        out_name: Output filename under OUT_DIR.
        only: Restrict the graph to edges within these top-level packages
            (passed as repeated ``--only`` flags).
        extra: Additional pydeps flags.
    """
    out_path = OUT_DIR / out_name
    only_flags: list[str] = []
    for pkg in only:
        only_flags.extend(["--only", pkg])
    cmd = [
        sys.executable,
        "-m",
        "pydeps",
        str(target),
        "-o",
        str(out_path),
        "-T",
        "svg",
        *COMMON_OPTS,
        *only_flags,
        *(extra or []),
    ]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(
            f"warning: pydeps failed for {target} (exit {result.returncode}).",
            file=sys.stderr,
        )
        return False
    if not out_path.exists() or out_path.stat().st_size < 2048:
        print(
            f"warning: pydeps produced a suspiciously small SVG at {out_path} "
            f"({out_path.stat().st_size if out_path.exists() else 0} bytes).",
            file=sys.stderr,
        )
        return False
    print(f"  wrote {out_path.relative_to(REPO_ROOT)} "
          f"({out_path.stat().st_size // 1024} KB)")
    return True


def main() -> int:
    _check_prerequisites()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    successes = 0
    successes += _render(
        SRC_ROOT / "dataset",
        "project_graph_dataset.svg",
        only=["dataset"],
    )
    successes += _render(
        SRC_ROOT / "depth_fm",
        "project_graph_depth_fm.svg",
        only=["depth_fm", "dataset"],
        extra=["--exclude", "depth_fm.models.unet.*"],
    )
    successes += _render(
        SRC_ROOT / "clip",
        "project_graph_clip.svg",
        only=["clip", "dataset"],
    )
    print(f"\nGenerated {successes}/3 project graphs in {OUT_DIR}")
    return 0 if successes == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
