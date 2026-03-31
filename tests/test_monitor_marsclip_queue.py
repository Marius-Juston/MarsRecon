"""Tests for the terminal MarsCLIP queue monitor."""

from __future__ import annotations

import json
import os
import pathlib
import sys
from unittest.mock import patch

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from monitor_marsclip_queue import (
    _is_pid_alive,
    _latest_checkpoint_step,
    _load_progress,
    _tail_last_line,
    build_job_snapshot,
    extract_out_dir,
    load_queue_status,
    monitor_queue,
    render_queue_snapshot,
)


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


def test_extract_out_dir_returns_none_when_flag_not_found():
    assert extract_out_dir(["python", "train.py", "--batch-size", "32"]) is None


def test_is_pid_alive_returns_false_for_invalid_pid():
    assert _is_pid_alive(99999999) is False


def test_latest_checkpoint_step_returns_none_for_invalid_stem(tmp_path):
    (tmp_path / "checkpoint_step_abc.pt").write_text("bad")
    assert _latest_checkpoint_step(tmp_path) is None


def test_load_progress_returns_none_when_missing(tmp_path):
    assert _load_progress(tmp_path) is None


def test_tail_last_line_returns_none_for_missing_file(tmp_path):
    assert _tail_last_line(tmp_path / "missing.log") is None


def test_tail_last_line_returns_none_for_all_blank_lines(tmp_path):
    log = tmp_path / "blank.log"
    log.write_text("\n   \n\n")
    assert _tail_last_line(log) is None


def test_load_queue_status_reads_existing_status(tmp_path):
    import json
    status = {"queue_name": "q", "status": "running", "jobs": []}
    (tmp_path / "queue_status.json").write_text(json.dumps(status))
    loaded = load_queue_status(tmp_path)
    assert loaded["queue_name"] == "q"


def test_render_queue_snapshot_shows_artifacts_when_no_progress(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("")
    status = {
        "queue_name": "q",
        "status": "completed",
        "updated_at": "2026-01-01",
        "jobs": [
            {
                "name": "job_done",
                "status": "completed",
                "pid": None,
                "started_at": None,
                "finished_at": None,
                "heartbeat_at": None,
                "exit_code": 0,
                "log_path": str(log_path),
                "argv": ["python", "train.py"],
            }
        ],
    }
    rendered = render_queue_snapshot(status)
    assert "artifacts=" in rendered
    assert "[completed] job_done" in rendered


def test_is_pid_alive_returns_true_for_current_process():
    """Line 35: _is_pid_alive returns True when PID is alive."""
    assert _is_pid_alive(os.getpid()) is True


def test_tail_last_line_returns_none_on_read_error(tmp_path):
    """Lines 61-62: _tail_last_line returns None on OSError during read."""
    log = tmp_path / "readable.log"
    log.write_text("some content")
    with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
        result = _tail_last_line(log)
    assert result is None


def test_monitor_queue_once_prints_snapshot(tmp_path, capsys):
    """Lines 138-144: monitor_queue with once=True prints snapshot and returns."""
    status = {
        "queue_name": "test_q",
        "status": "completed",
        "updated_at": "2026-01-01",
        "jobs": [],
    }
    (tmp_path / "queue_status.json").write_text(json.dumps(status))
    monitor_queue(tmp_path, once=True)
    out = capsys.readouterr().out
    assert "Queue: test_q" in out
