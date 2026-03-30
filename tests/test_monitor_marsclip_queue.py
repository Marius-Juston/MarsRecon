"""Tests for the terminal MarsCLIP queue monitor."""

from __future__ import annotations

import json
import pathlib
import sys

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from monitor_marsclip_queue import build_job_snapshot, extract_out_dir, render_queue_snapshot


def test_extract_out_dir_finds_training_output_dir():
    argv = [
        ".venv/bin/python",
        "src/train_marsclip_mae.py",
        "--out-dir",
        "/tmp/marsclip_artifacts/run_a",
        "--batch-size",
        "128",
    ]

    out_dir = extract_out_dir(argv)

    assert out_dir == pathlib.Path("/tmp/marsclip_artifacts/run_a")


def test_build_job_snapshot_reads_progress_and_summary(tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True)
    (out_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "phase": "training",
                "current_step": 12,
                "target_step": 100,
                "latest_loss": 0.42,
                "best_loss": 0.4,
                "latest_lr": 1e-4,
            }
        )
    )
    (out_dir / "checkpoint_step_000100.pt").write_text("ckpt")
    log_path = tmp_path / "queue.log"
    log_path.write_text("line 1\nlast line\n")
    job = {
        "name": "demo",
        "status": "running",
        "pid": None,
        "started_at": "2026-03-30 10:00:00",
        "finished_at": None,
        "heartbeat_at": "2026-03-30 10:05:00",
        "exit_code": None,
        "log_path": str(log_path),
        "argv": [".venv/bin/python", "src/train_marsclip_mae.py", "--out-dir", str(out_dir)],
    }

    snapshot = build_job_snapshot(job)

    assert snapshot["out_dir"] == str(out_dir)
    assert snapshot["progress"]["current_step"] == 12
    assert snapshot["latest_checkpoint_step"] == 100
    assert snapshot["last_log_line"] == "last line"


def test_render_queue_snapshot_includes_progress_fields(tmp_path):
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True)
    (out_dir / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "phase": "training",
                "current_step": 25,
                "target_step": 200,
                "latest_loss": 0.123,
                "best_loss": 0.111,
                "latest_lr": 5e-5,
            }
        )
    )
    log_path = tmp_path / "job.log"
    log_path.write_text("hello\nworld\n")
    status = {
        "queue_name": "demo_queue",
        "status": "running",
        "updated_at": "2026-03-30 10:06:00",
        "jobs": [
            {
                "name": "demo_job",
                "status": "running",
                "pid": None,
                "started_at": "2026-03-30 10:00:00",
                "finished_at": None,
                "heartbeat_at": "2026-03-30 10:05:00",
                "exit_code": None,
                "log_path": str(log_path),
                "argv": [".venv/bin/python", "src/train_marsclip_mae.py", "--out-dir", str(out_dir)],
            }
        ],
    }

    rendered = render_queue_snapshot(status)

    assert "Queue: demo_queue" in rendered
    assert "[running] demo_job" in rendered
    assert "step=25/200" in rendered
    assert "loss=0.123" in rendered
