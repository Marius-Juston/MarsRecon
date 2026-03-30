"""Small terminal monitor for queued MarsCLIP training runs."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Any


def load_queue_status(queue_dir: pathlib.Path | str) -> dict[str, Any]:
    """Load a queue status JSON payload."""
    queue_path = pathlib.Path(queue_dir)
    status_path = queue_path / "queue_status.json"
    return json.loads(status_path.read_text())


def extract_out_dir(argv: list[str]) -> pathlib.Path | None:
    """Extract a trainer output directory from a queued argv list."""
    for idx, token in enumerate(argv):
        if token == "--out-dir" and idx + 1 < len(argv):
            return pathlib.Path(argv[idx + 1])
    return None


def _is_pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _latest_checkpoint_step(out_dir: pathlib.Path) -> int | None:
    checkpoints = sorted(out_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None
    latest = checkpoints[-1].stem.split("_")[-1]
    try:
        return int(latest)
    except ValueError:
        return None


def _load_progress(out_dir: pathlib.Path) -> dict[str, Any] | None:
    progress_path = out_dir / "progress.json"
    if not progress_path.exists():
        return None
    return json.loads(progress_path.read_text())


def _tail_last_line(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return None


def build_job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """Enrich one queued job record with filesystem/runtime state."""
    out_dir = extract_out_dir(list(job.get("argv", [])))
    log_path = pathlib.Path(job["log_path"])
    progress = _load_progress(out_dir) if out_dir is not None and out_dir.exists() else None
    return {
        "name": str(job["name"]),
        "status": str(job["status"]),
        "pid": job.get("pid"),
        "pid_alive": _is_pid_alive(job.get("pid")),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "heartbeat_at": job.get("heartbeat_at"),
        "exit_code": job.get("exit_code"),
        "out_dir": str(out_dir) if out_dir is not None else None,
        "progress": progress,
        "latest_checkpoint_step": _latest_checkpoint_step(out_dir) if out_dir is not None and out_dir.exists() else None,
        "has_summary": bool(out_dir is not None and (out_dir / "summary.json").exists()),
        "has_preview_start": bool(out_dir is not None and (out_dir / "preview_step_000000.png").exists()),
        "last_log_line": _tail_last_line(log_path),
    }


def render_queue_snapshot(status: dict[str, Any]) -> str:
    """Render a human-readable terminal snapshot for the queue."""
    lines = [
        f"Queue: {status['queue_name']}",
        f"Status: {status['status']}",
        f"Queue Updated: {status.get('updated_at')}",
        "",
    ]
    for job in status.get("jobs", []):
        snapshot = build_job_snapshot(job)
        lines.append(f"[{snapshot['status']}] {snapshot['name']}")
        lines.append(
            f"  pid={snapshot['pid']} alive={snapshot['pid_alive']} exit={snapshot['exit_code']} heartbeat={snapshot['heartbeat_at']}"
        )
        lines.append(f"  started={snapshot['started_at']} finished={snapshot['finished_at']}")
        lines.append(f"  out_dir={snapshot['out_dir']}")
        progress = snapshot["progress"]
        if progress is not None:
            current_step = progress.get("current_step")
            target_step = progress.get("target_step")
            latest_loss = progress.get("latest_loss")
            best_loss = progress.get("best_loss")
            latest_lr = progress.get("latest_lr")
            lines.append(
                "  progress="
                f"{progress.get('phase')} step={current_step}/{target_step} "
                f"loss={latest_loss} best={best_loss} lr={latest_lr}"
            )
        else:
            lines.append(
                f"  artifacts=summary:{snapshot['has_summary']} preview0:{snapshot['has_preview_start']} checkpoint_step:{snapshot['latest_checkpoint_step']}"
            )
        if snapshot["last_log_line"] is not None:
            lines.append(f"  last_log={snapshot['last_log_line']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def monitor_queue(
    queue_dir: pathlib.Path | str,
    *,
    refresh_seconds: float = 5.0,
    once: bool = False,
) -> None:
    """Print a periodically refreshed queue snapshot."""
    queue_path = pathlib.Path(queue_dir)
    while True:
        status = load_queue_status(queue_path)
        rendered = render_queue_snapshot(status)
        if once:
            print(rendered, end="")
            return
        print("\033[2J\033[H", end="")
        print(rendered, end="", flush=True)
        time.sleep(refresh_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor queued MarsCLIP jobs in the terminal.")
    parser.add_argument("--queue-dir", type=pathlib.Path, required=True)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument("--once", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    monitor_queue(args.queue_dir, refresh_seconds=args.refresh_seconds, once=args.once)


if __name__ == "__main__":
    main()
