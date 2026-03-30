"""Tests for the MarsCLIP queue runner."""

from __future__ import annotations

import json
import pathlib
import sys

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from run_marsclip_queue import initialize_queue_status, load_queue_spec, run_queue


def test_load_queue_spec_requires_jobs(tmp_path):
    spec_path = tmp_path / "queue.json"
    spec_path.write_text(json.dumps({"name": "empty", "jobs": []}))

    try:
        load_queue_spec(spec_path)
    except ValueError as exc:
        assert "non-empty 'jobs' list" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty job list")


def test_initialize_queue_status_creates_queued_jobs(tmp_path):
    spec = {
        "name": "unit-queue",
        "jobs": [
            {"name": "job_a", "argv": ["python", "--version"]},
            {"name": "job_b", "argv": ["python", "--version"]},
        ],
    }

    status = initialize_queue_status(spec, queue_dir=tmp_path / "queue")

    assert status["queue_name"] == "unit-queue"
    assert status["status"] == "queued"
    assert [job["status"] for job in status["jobs"]] == ["queued", "queued"]
    assert status["jobs"][0]["log_path"].endswith("00_job_a.log")


def test_run_queue_records_successful_jobs(tmp_path):
    marker_a = tmp_path / "job_a.txt"
    marker_b = tmp_path / "job_b.txt"
    spec = {
        "name": "success-queue",
        "jobs": [
            {
                "name": "job_a",
                "argv": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path(r'" + str(marker_a) + "').write_text('a'); print('job_a_ok')",
                ],
            },
            {
                "name": "job_b",
                "argv": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path(r'" + str(marker_b) + "').write_text('b'); print('job_b_ok')",
                ],
            },
        ],
    }

    status = run_queue(spec, queue_dir=tmp_path / "queue", stream_child_output=False)

    assert status["status"] == "completed"
    assert [job["status"] for job in status["jobs"]] == ["completed", "completed"]
    assert marker_a.read_text() == "a"
    assert marker_b.read_text() == "b"
    queue_status = json.loads((tmp_path / "queue" / "queue_status.json").read_text())
    assert queue_status["status"] == "completed"
    assert queue_status["jobs"][0]["heartbeat_at"] is not None
    assert queue_status["jobs"][0]["pid"] is None
    first_log = pathlib.Path(queue_status["jobs"][0]["log_path"])
    assert "job_a_ok" in first_log.read_text()


def test_run_queue_fail_fast_skips_remaining_jobs(tmp_path):
    marker = tmp_path / "should_not_exist.txt"
    spec = {
        "name": "fail-queue",
        "fail_fast": True,
        "jobs": [
            {
                "name": "job_fail",
                "argv": [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
            },
            {
                "name": "job_skip",
                "argv": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path(r'" + str(marker) + "').write_text('x')",
                ],
            },
        ],
    }

    status = run_queue(spec, queue_dir=tmp_path / "queue", stream_child_output=False)

    assert status["status"] == "failed"
    assert status["jobs"][0]["status"] == "failed"
    assert status["jobs"][0]["exit_code"] == 3
    assert status["jobs"][1]["status"] == "skipped"
    assert not marker.exists()
