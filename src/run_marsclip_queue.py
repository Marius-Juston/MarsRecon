"""Run queued MarsCLIP jobs sequentially with terminal and JSON status logging."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


def _timestamp() -> str:
    """Return a compact local timestamp for queue status records."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_queue_spec(path: pathlib.Path | str) -> dict[str, Any]:
    """Load and lightly validate a queue spec JSON file."""
    spec_path = pathlib.Path(path)
    spec = json.loads(spec_path.read_text())
    jobs = spec.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Queue spec must include a non-empty 'jobs' list.")
    for idx, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"Job {idx} must be a JSON object.")
        argv = job.get("argv")
        if not isinstance(argv, list) or not argv:
            raise ValueError(f"Job {idx} must include a non-empty 'argv' list.")
        if not all(isinstance(part, str) for part in argv):
            raise ValueError(f"Job {idx} argv entries must all be strings.")
    return spec


def write_queue_status(status: dict[str, Any], path: pathlib.Path | str) -> pathlib.Path:
    """Persist queue status to JSON."""
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    status = dict(status)
    status["updated_at"] = _timestamp()
    out_path.write_text(json.dumps(status, indent=2))
    return out_path


def initialize_queue_status(spec: dict[str, Any], *, queue_dir: pathlib.Path | str) -> dict[str, Any]:
    """Create the initial queue status payload."""
    queue_path = pathlib.Path(queue_dir)
    jobs = []
    for idx, job in enumerate(spec["jobs"]):
        name = str(job.get("name") or f"job_{idx:02d}")
        jobs.append(
            {
                "index": idx,
                "name": name,
                "status": "queued",
                "cwd": str(job.get("cwd")) if job.get("cwd") is not None else None,
                "argv": list(job["argv"]),
                "log_path": str(queue_path / "logs" / f"{idx:02d}_{name}.log"),
                "started_at": None,
                "finished_at": None,
                "duration_seconds": None,
                "exit_code": None,
                "pid": None,
                "heartbeat_at": None,
                "last_output_at": None,
                "last_output_line": None,
            }
        )
    return {
        "queue_name": str(spec.get("name") or "marsclip_queue"),
        "queue_dir": str(queue_path),
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "fail_fast": bool(spec.get("fail_fast", True)),
        "status": "queued",
        "jobs": jobs,
    }


def _format_command(argv: list[str]) -> str:
    return " ".join(argv)


def _emit_terminal(message: str) -> None:
    print(message, flush=True)


def run_queue(
    spec: dict[str, Any],
    *,
    queue_dir: pathlib.Path | str,
    stream_child_output: bool = False,
) -> dict[str, Any]:
    """Run queued jobs sequentially, updating JSON status and log files."""
    queue_path = pathlib.Path(queue_dir)
    logs_dir = queue_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = queue_path / "queue_status.json"

    status = initialize_queue_status(spec, queue_dir=queue_path)
    write_queue_status(status, status_path)

    fail_fast = bool(spec.get("fail_fast", True))
    status["status"] = "running"
    write_queue_status(status, status_path)

    for job_status, job_spec in zip(status["jobs"], spec["jobs"], strict=True):
        name = str(job_status["name"])
        log_path = pathlib.Path(job_status["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)

        launch_time = time.time()
        job_status["status"] = "running"
        job_status["started_at"] = _timestamp()
        write_queue_status(status, status_path)

        cwd = pathlib.Path(job_spec["cwd"]) if job_spec.get("cwd") is not None else None
        env = os.environ.copy()
        for key, value in dict(job_spec.get("env", {})).items():
            env[str(key)] = str(value)

        _emit_terminal(f"[queue] starting {name}")
        _emit_terminal(f"[queue] command: {_format_command(job_status['argv'])}")
        _emit_terminal(f"[queue] log: {log_path}")

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[queue] started_at={job_status['started_at']}\n")
            handle.write(f"[queue] command={_format_command(job_status['argv'])}\n")
            handle.flush()
            if stream_child_output:
                process = subprocess.Popen(
                    job_status["argv"],
                    cwd=str(cwd) if cwd is not None else None,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                job_status["pid"] = int(process.pid)
                job_status["heartbeat_at"] = _timestamp()
                write_queue_status(status, status_path)

                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    job_status["heartbeat_at"] = _timestamp()
                    job_status["last_output_at"] = job_status["heartbeat_at"]
                    job_status["last_output_line"] = line.rstrip()[:500]
                    write_queue_status(status, status_path)
                    _emit_terminal(f"[{name}] {line.rstrip()}")
                exit_code = int(process.wait())
            else:
                process = subprocess.Popen(
                    job_status["argv"],
                    cwd=str(cwd) if cwd is not None else None,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    close_fds=True,
                )
                job_status["pid"] = int(process.pid)
                job_status["heartbeat_at"] = _timestamp()
                write_queue_status(status, status_path)

                while True:
                    exit_code = process.poll()
                    if exit_code is not None:
                        exit_code = int(exit_code)
                        break
                    job_status["heartbeat_at"] = _timestamp()
                    write_queue_status(status, status_path)
                    time.sleep(1.0)

            job_status["exit_code"] = exit_code
            job_status["finished_at"] = _timestamp()
            job_status["duration_seconds"] = round(time.time() - launch_time, 3)
            job_status["status"] = "completed" if exit_code == 0 else "failed"
            job_status["pid"] = None
            job_status["heartbeat_at"] = job_status["finished_at"]
            handle.write(f"[queue] exit_code={exit_code}\n")
            handle.flush()
            write_queue_status(status, status_path)

        if exit_code == 0:
            _emit_terminal(f"[queue] finished {name} exit=0")
            continue

        _emit_terminal(f"[queue] failed {name} exit={exit_code}")
        if fail_fast:
            for remaining in status["jobs"][job_status["index"] + 1 :]:
                if remaining["status"] == "queued":
                    remaining["status"] = "skipped"
            status["status"] = "failed"
            write_queue_status(status, status_path)
            return status

    status["status"] = "completed"
    write_queue_status(status, status_path)
    _emit_terminal("[queue] all jobs completed")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequential queue of MarsCLIP jobs with status logging.")
    parser.add_argument("--spec", type=pathlib.Path, required=True, help="Path to a queue spec JSON file.")
    parser.add_argument("--queue-dir", type=pathlib.Path, required=True, help="Directory for queue logs and status.")
    parser.add_argument(
        "--stream-child-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stream child process stdout/stderr to the terminal in addition to per-job log files.",
    )
    args = parser.parse_args()

    spec = load_queue_spec(args.spec)
    run_queue(spec, queue_dir=args.queue_dir, stream_child_output=args.stream_child_output)


if __name__ == "__main__":
    main()
