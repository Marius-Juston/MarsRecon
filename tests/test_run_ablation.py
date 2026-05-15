"""Unit tests for scripts/training/run_ablation.py — the loss-ablation orchestrator.

`run_ablation.py` is a *script*, not part of the importable library, so it is
loaded by file path via importlib (see the `ra` module-scoped fixture).

Everything here is hermetic: no GPUs, no network, no real training launches,
no real nvidia-smi. `subprocess.Popen`/`subprocess.run` and `time.monotonic`/
`time.sleep` are monkeypatched wherever they would otherwise touch the system.

Regression coverage for the three bug classes fixed this session:
  1. `_find_metrics_csv` deep-glob (false-killed healthy runs).
  2. Absolute `LAUNCH_SCRIPT` / `_abs_config` path robustness.
  3. GPU-free barrier / nvidia-smi helpers degrade gracefully when unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import pathlib
import signal
import sys
import types
import unittest.mock as mock

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "training" / "run_ablation.py"


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ra() -> types.ModuleType:
    """Load run_ablation.py by file path under a stable module name."""
    if str(_REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("run_ablation_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _reset_module_globals(ra: types.ModuleType):
    """Reset module-global state so tests are order-independent under -n auto."""
    ra._NVSMI_WARNED["done"] = False
    ra._SHUTDOWN["requested"] = False
    yield
    ra._NVSMI_WARNED["done"] = False
    ra._SHUTDOWN["requested"] = False


@pytest.fixture
def _restore_logging():
    """Snapshot/restore root logging handlers so setup_logging tests don't leak.

    Critical under -n auto / repeated runs: RotatingFileHandler instances would
    otherwise stack across tests and across workers.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for h in list(root.handlers):
        if h not in saved_handlers:
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    for h in saved_handlers:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(**over) -> dict:
    spec = {
        "base_config": "configs/train_hirise.yaml",
        "output_root": "outputs/ablation_test",
        "project_name": "mars-ablation",
        "lanes": ["0,1", "2,3"],
        "seeds": [42, 43],
        "variants": [
            {"id": "L0", "label": "baseline", "overrides": ["loss.a=1"]},
            {"id": "L1", "overrides": ["loss.b=0"]},
        ],
        "common_overrides": ["trainer.max_steps=10"],
    }
    spec.update(over)
    return spec


def _make_job(ra, job_id="L0_s42", variant_id="L0", label="baseline",
              seed=42, overrides=None):
    return ra.Job(job_id=job_id, variant_id=variant_id,
                  variant_label=label, seed=seed,
                  overrides=overrides or [])


def _running_job(ra, job, run_dir, *, last_seen=0.0, row_count=0, started=0.0):
    """Build a RunningJob with the *current* dataclass fields (no csv_path)."""
    return ra.RunningJob(
        job=job,
        proc=mock.MagicMock(),
        lane="0,1",
        run_dir=pathlib.Path(run_dir),
        log_file=mock.MagicMock(),
        last_val_seen_at=last_seen,
        last_val_row_count=row_count,
        started_at=started,
    )


def _write_metrics_csv(path: pathlib.Path, rmse=None, photo=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {}
    if rmse is not None:
        cols["val/rmse_mean"] = rmse
    if photo is not None:
        cols["val/photo_consistency_mean"] = photo
    import pandas as pd
    pd.DataFrame(cols).to_csv(path, index=False)


def _completed(stdout="", returncode=0):
    return types.SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


# ---------------------------------------------------------------------------
# Spec + job model
# ---------------------------------------------------------------------------


class TestBuildJobs:
    def test_cartesian_product_of_variants_and_seeds(self, ra):
        jobs = ra.build_jobs(_make_spec())
        assert len(jobs) == 4  # 2 variants x 2 seeds
        assert {j.job_id for j in jobs} == {"L0_s42", "L0_s43", "L1_s42", "L1_s43"}

    def test_job_id_format_is_id_underscore_s_seed(self, ra):
        jobs = ra.build_jobs(_make_spec())
        j = next(j for j in jobs if j.variant_id == "L1" and j.seed == 43)
        assert j.job_id == "L1_s43"

    def test_common_overrides_appended_after_variant_overrides(self, ra):
        jobs = ra.build_jobs(_make_spec())
        j = next(j for j in jobs if j.variant_id == "L1")
        assert j.overrides == ["loss.b=0", "trainer.max_steps=10"]

    def test_label_falls_back_to_id_when_missing(self, ra):
        # L1 has no "label" key in _make_spec.
        jobs = ra.build_jobs(_make_spec())
        j = next(j for j in jobs if j.variant_id == "L1")
        assert j.variant_label == "L1"

    def test_seed_coerced_to_int(self, ra):
        jobs = ra.build_jobs(_make_spec(seeds=["7"]))
        assert all(isinstance(j.seed, int) for j in jobs)
        assert jobs[0].seed == 7


class TestLoadSpec:
    def test_yaml_round_trip(self, ra, tmp_path):
        import yaml

        spec = _make_spec()
        p = tmp_path / "spec.yaml"
        p.write_text(yaml.safe_dump(spec))
        assert ra.load_spec(p) == spec


# ---------------------------------------------------------------------------
# Path robustness
# ---------------------------------------------------------------------------


class TestPathRobustness:
    def test_launch_script_is_absolute_and_exists(self, ra):
        assert ra.LAUNCH_SCRIPT.is_absolute()
        assert ra.LAUNCH_SCRIPT == _REPO_ROOT / "scripts" / "training" / "launch_train.sh"
        assert ra.LAUNCH_SCRIPT.exists()

    def test_abs_config_resolves_relative_under_repo_root(self, ra):
        out = ra._abs_config({"base_config": "configs/foo.yaml"})
        assert out == str(_REPO_ROOT / "configs" / "foo.yaml")
        assert pathlib.Path(out).is_absolute()

    def test_abs_config_passes_absolute_unchanged(self, ra, tmp_path):
        abs_in = str(tmp_path / "x.yaml")
        assert ra._abs_config({"base_config": abs_in}) == abs_in


class TestLaunchTrainArgv:
    def test_launch_train_builds_absolute_argv_and_lane_env(self, ra, tmp_path,
                                                             monkeypatch):
        spec = _make_spec(output_root=str(tmp_path / "out"))
        job = _make_job(ra, overrides=["loss.a=1"])
        store = ra.StateStore(tmp_path / "state.json")

        captured = {}

        class FakePopen:
            def __init__(self, cmd, **kw):
                captured["cmd"] = cmd
                captured["kw"] = kw
                self.pid = 4321

        monkeypatch.setattr(ra.subprocess, "Popen", FakePopen)
        # Avoid an unseeded token so the assertion below is deterministic.
        monkeypatch.setattr(ra.secrets, "token_hex", lambda n: "deadbeef")
        monkeypatch.setattr(ra.time, "monotonic", lambda: 100.0)

        rj = ra.launch_train(job, "2,3", spec, store)

        cmd = captured["cmd"]
        assert cmd[0] == "bash"
        assert cmd[1] == str(ra.LAUNCH_SCRIPT)
        cfg = cmd[cmd.index("--config") + 1]
        assert pathlib.Path(cfg).is_absolute()
        assert cfg == str(_REPO_ROOT / "configs" / "train_hirise.yaml")
        assert captured["kw"]["cwd"] == str(ra.REPO_ROOT)
        assert captured["kw"]["env"]["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert captured["kw"]["env"]["WANDB_RUN_ID"] == "deadbeef"
        assert rj.proc.pid == 4321
        rj.log_file.close()

    def test_run_test_only_uses_launch_script_and_absolute_config(self, ra,
                                                                   tmp_path,
                                                                   monkeypatch):
        spec = _make_spec(output_root=str(tmp_path / "out"))
        job = _make_job(ra)
        (tmp_path / "out" / job.job_id).mkdir(parents=True)
        ckpt = tmp_path / "best.ckpt"
        ckpt.write_text("x")

        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["kw"] = kw
            return _completed(returncode=0)

        monkeypatch.setattr(ra.subprocess, "run", fake_run)
        rc = ra.run_test_only(job, "0,1", spec, ckpt, "rmse_ckpt")

        assert rc == 0
        cmd = captured["cmd"]
        assert cmd[0] == "bash" and cmd[1] == str(ra.LAUNCH_SCRIPT)
        assert pathlib.Path(cmd[cmd.index("--config") + 1]).is_absolute()
        assert captured["kw"]["cwd"] == str(ra.REPO_ROOT)
        assert captured["kw"]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert captured["kw"]["env"]["WANDB_MODE"] == "disabled"


# ---------------------------------------------------------------------------
# _find_metrics_csv + watchdog
# ---------------------------------------------------------------------------


class TestFindMetricsCsv:
    def test_returns_deeply_nested_path(self, ra, tmp_path):
        nested = (tmp_path / "depthfm" / "abc123" / "run_0"
                  / "csv" / "version_0" / "metrics.csv")
        nested.parent.mkdir(parents=True)
        nested.write_text("a\n1\n")
        assert ra._find_metrics_csv(tmp_path) == nested

    def test_returns_none_when_absent(self, ra, tmp_path):
        assert ra._find_metrics_csv(tmp_path) is None

    def test_picks_a_hit_when_multiple_present(self, ra, tmp_path):
        for h in ("h1", "h2"):
            p = tmp_path / "depthfm" / h / "run_0" / "csv" / "version_0" / "metrics.csv"
            p.parent.mkdir(parents=True)
            p.write_text("a\n1\n")
        hit = ra._find_metrics_csv(tmp_path)
        assert hit is not None and hit.name == "metrics.csv"


class TestWatchdogCheck:
    def _nested_csv(self, run_dir: pathlib.Path) -> pathlib.Path:
        return run_dir / "depthfm" / "h" / "run_0" / "csv" / "version_0" / "metrics.csv"

    def test_nan_in_rmse_returns_nan_reason(self, ra, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        # NOTE: source calls .dropna() first, so a trailing literal NaN would
        # be silently removed. A diverged run writes +Inf; that survives
        # dropna() and is the value the non-finite guard actually catches.
        _write_metrics_csv(self._nested_csv(run_dir),
                           rmse=[0.1, float("inf")])
        monkeypatch.setattr(ra.time, "monotonic", lambda: 10.0)
        rj = _running_job(ra, _make_job(ra), run_dir, started=0.0, last_seen=0.0)

        reason = ra.watchdog_check(rj, {})
        assert reason == "val/rmse_mean is NaN/Inf"

    def test_fresh_no_rows_under_grace_returns_none(self, ra, tmp_path,
                                                    monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Started 100s ago; grace+timeout is far larger -> not yet stalled.
        monkeypatch.setattr(ra.time, "monotonic", lambda: 100.0)
        rj = _running_job(ra, _make_job(ra), run_dir, started=0.0, last_seen=0.0)

        assert ra.watchdog_check(rj, {}) is None

    def test_no_val_row_after_grace_plus_timeout_returns_reason(self, ra,
                                                                tmp_path,
                                                                monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # grace=1800, timeout default 7200 -> threshold 9000s. Elapsed 9001s.
        monkeypatch.setattr(ra.time, "monotonic", lambda: 9001.0)
        rj = _running_job(ra, _make_job(ra), run_dir, started=0.0,
                          last_seen=0.0, row_count=0)

        reason = ra.watchdog_check(rj, {})
        assert reason is not None and reason.startswith("no val/* row produced")

    def test_advancing_rows_updates_count_and_seen_at(self, ra, tmp_path,
                                                      monkeypatch):
        run_dir = tmp_path / "run"
        _write_metrics_csv(self._nested_csv(run_dir), rmse=[0.5, 0.4, 0.3])
        monkeypatch.setattr(ra.time, "monotonic", lambda: 555.0)
        rj = _running_job(ra, _make_job(ra), run_dir, started=0.0,
                          last_seen=0.0, row_count=0)

        reason = ra.watchdog_check(rj, {})
        assert reason is None
        assert rj.last_val_row_count == 3
        assert rj.last_val_seen_at == 555.0

    def test_seen_rows_then_stall_beyond_timeout_returns_reason(self, ra,
                                                                tmp_path,
                                                                monkeypatch):
        run_dir = tmp_path / "run"
        _write_metrics_csv(self._nested_csv(run_dir), rmse=[0.5])
        monkeypatch.setattr(ra.time, "monotonic", lambda: 8000.0)
        # Already saw 1 row long ago; elapsed since last_seen = 8000 > 7200.
        rj = _running_job(ra, _make_job(ra), run_dir, started=0.0,
                          last_seen=0.0, row_count=1)

        reason = ra.watchdog_check(rj, {})
        assert reason is not None and reason.startswith("val/* stalled")


# ---------------------------------------------------------------------------
# GPU-free barrier / nvidia-smi helpers
# ---------------------------------------------------------------------------


class TestLaneGpuIds:
    @pytest.mark.parametrize("lane,expected", [
        ("0,1", ["0", "1"]),
        ("0, 1 ,2", ["0", "1", "2"]),
        ("", []),
        ("0,,1", ["0", "1"]),
        ("3", ["3"]),
    ])
    def test_parsing(self, ra, lane, expected):
        assert ra._lane_gpu_ids(lane) == expected


class TestNvidiaSmiHelpers:
    def test_gpu_used_mib_parses_good_output(self, ra, monkeypatch):
        monkeypatch.setattr(ra.subprocess, "run",
                            lambda *a, **k: _completed(stdout="123\n4096\n"))
        assert ra._gpu_used_mib(["0", "1"]) == {"0": 123, "1": 4096}

    def test_gpu_used_mib_returncode_nonzero_returns_empty(self, ra,
                                                           monkeypatch):
        monkeypatch.setattr(ra.subprocess, "run",
                            lambda *a, **k: _completed(returncode=9))
        assert ra._gpu_used_mib(["0"]) == {}

    def test_gpu_used_mib_filenotfound_returns_empty_and_warns_once(
            self, ra, monkeypatch, caplog):
        def boom(*a, **k):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(ra.subprocess, "run", boom)
        with caplog.at_level(logging.WARNING, logger="ablation"):
            assert ra._gpu_used_mib(["0"]) == {}
            assert ra._gpu_used_mib(["0"]) == {}  # second call: no 2nd warning
        warnings = [r for r in caplog.records
                    if "nvidia-smi unavailable" in r.message]
        assert len(warnings) == 1

    def test_gpu_compute_pids_parses_pids(self, ra, monkeypatch):
        monkeypatch.setattr(ra.subprocess, "run",
                            lambda *a, **k: _completed(stdout="111\n222\n"))
        assert ra._gpu_compute_pids(["0"]) == {111, 222}

    def test_gpu_compute_pids_failure_returns_empty_set(self, ra, monkeypatch):
        monkeypatch.setattr(ra.subprocess, "run",
                            lambda *a, **k: _completed(returncode=1))
        assert ra._gpu_compute_pids(["0"]) == set()

    def test_nvidia_smi_returns_none_on_empty_gpu_ids(self, ra):
        assert ra._nvidia_smi("--query-gpu=memory.used", []) is None

    def test_reap_lane_gpu_pids_swallows_kill_errors(self, ra, monkeypatch):
        monkeypatch.setattr(ra.subprocess, "run",
                            lambda *a, **k: _completed(stdout="999\n888\n"))
        killed = []

        def fake_kill(pid, sig):
            killed.append((pid, sig))
            if pid == 888:
                raise ProcessLookupError
            raise PermissionError  # 999

        monkeypatch.setattr(ra.os, "kill", fake_kill)
        # Must not raise despite both pids erroring.
        ra._reap_lane_gpu_pids("0,1")
        assert {p for p, _ in killed} == {999, 888}
        assert all(s == signal.SIGKILL for _, s in killed)


class TestWaitForLaneFree:
    def test_noop_true_when_nvidia_smi_unavailable(self, ra, monkeypatch):
        monkeypatch.setattr(ra.subprocess, "run",
                            lambda *a, **k: _completed(returncode=127))
        # Should return immediately without touching time.sleep.
        monkeypatch.setattr(ra.time, "sleep",
                            lambda *_: pytest.fail("slept on no-op path"))
        assert ra.wait_for_lane_free("0,1", 2048, timeout=1.0) is True

    def test_returns_true_fast_when_below_threshold(self, ra, monkeypatch):
        monkeypatch.setattr(ra.subprocess, "run",
                            lambda *a, **k: _completed(stdout="10\n20\n"))
        monkeypatch.setattr(ra.time, "monotonic", lambda: 0.0)
        assert ra.wait_for_lane_free("0,1", 2048, timeout=1.0) is True

    def test_busy_until_timeout_reaps_and_returns_false(self, ra, monkeypatch):
        # Memory stays high forever -> never below threshold.
        monkeypatch.setattr(ra.subprocess, "run",
                            lambda *a, **k: _completed(stdout="9000\n9000\n"))
        clock = {"t": 0.0}

        def fake_monotonic():
            return clock["t"]

        def fake_sleep(s):
            clock["t"] += 100.0  # jump well past the timeout

        monkeypatch.setattr(ra.time, "monotonic", fake_monotonic)
        monkeypatch.setattr(ra.time, "sleep", fake_sleep)
        reaped = []
        monkeypatch.setattr(ra, "_reap_lane_gpu_pids",
                            lambda lane: reaped.append(lane))

        result = ra.wait_for_lane_free("0,1", 2048, timeout=5.0, poll=1.0)
        assert result is False
        assert reaped == ["0,1"]


# ---------------------------------------------------------------------------
# StateStore
# ---------------------------------------------------------------------------


class TestStateStore:
    def test_create_and_persist_round_trip(self, ra, tmp_path):
        path = tmp_path / "state.json"
        s1 = ra.StateStore(path)
        s1.set("L0_s42", status=ra.STATUS_RUNNING, attempts=2)

        s2 = ra.StateStore(path)  # fresh load from disk
        st = s2.get("L0_s42")
        assert st.status == ra.STATUS_RUNNING
        assert st.attempts == 2

    def test_load_ignores_unknown_keys(self, ra, tmp_path):
        path = tmp_path / "state.json"
        path.write_text(json.dumps({
            "L0_s42": {"status": ra.STATUS_DONE, "totally_new_field": 99},
        }))
        st = ra.StateStore(path).get("L0_s42")
        assert st.status == ra.STATUS_DONE
        assert not hasattr(st, "totally_new_field")

    def test_jobstate_defaults(self, ra):
        st = ra.JobState()
        assert st.status == ra.STATUS_QUEUED
        assert st.attempts == 0
        assert st.resumes == 0
        assert st.run_dir is None
        assert st.wandb_run_id is None

    def test_get_creates_default_entry(self, ra, tmp_path):
        s = ra.StateStore(tmp_path / "state.json")
        st = s.get("never_seen")
        assert st.status == ra.STATUS_QUEUED


# ---------------------------------------------------------------------------
# Control file
# ---------------------------------------------------------------------------


class TestControlFile:
    def test_read_missing_returns_empty(self, ra, tmp_path):
        assert ra.read_control(tmp_path) == {"disabled": [], "drain": []}

    def test_read_corrupt_json_returns_empty_and_warns(self, ra, tmp_path,
                                                       caplog):
        p = ra._control_path(tmp_path)
        p.parent.mkdir(parents=True)
        p.write_text("{ not json")
        with caplog.at_level(logging.WARNING, logger="ablation"):
            assert ra.read_control(tmp_path) == {"disabled": [], "drain": []}
        assert any("unreadable" in r.message for r in caplog.records)

    def test_write_then_read_round_trip(self, ra, tmp_path):
        ra.write_control(tmp_path, ["0,1"], ["2,3"])
        assert ra.read_control(tmp_path) == {"disabled": ["0,1"], "drain": ["2,3"]}

    def test_add_disabled_discards_from_drain(self, ra, tmp_path):
        ra.write_control(tmp_path, [], ["2,3"])
        out = ra.mutate_control(tmp_path, add_disabled="2,3")
        assert out == {"disabled": ["2,3"], "drain": []}

    def test_add_drain_discards_from_disabled(self, ra, tmp_path):
        ra.write_control(tmp_path, ["2,3"], [])
        out = ra.mutate_control(tmp_path, add_drain="2,3")
        assert out == {"disabled": [], "drain": ["2,3"]}

    def test_enable_lane_clears_both(self, ra, tmp_path):
        ra.write_control(tmp_path, ["2,3"], ["2,3"])
        out = ra.mutate_control(tmp_path, remove_disabled="2,3",
                                remove_drain="2,3")
        assert out == {"disabled": [], "drain": []}


# ---------------------------------------------------------------------------
# OrchestratorLock
# ---------------------------------------------------------------------------


class TestOrchestratorLock:
    def test_second_acquire_raises_systemexit_with_holder_pid(self, ra,
                                                              tmp_path):
        lock_path = tmp_path / "orchestrator.lock"
        with ra.OrchestratorLock(lock_path):
            assert lock_path.read_text().strip() == str(ra.os.getpid())
            with pytest.raises(SystemExit, match=r"already holds"):
                with ra.OrchestratorLock(lock_path):
                    pass

    def test_releases_on_exit_so_reacquire_succeeds(self, ra, tmp_path):
        lock_path = tmp_path / "orchestrator.lock"
        with ra.OrchestratorLock(lock_path):
            pass
        # Must be re-acquirable now.
        with ra.OrchestratorLock(lock_path):
            pass


# ---------------------------------------------------------------------------
# find_best_checkpoint
# ---------------------------------------------------------------------------


class TestFindBestCheckpoint:
    def _mk(self, run_dir, name):
        p = run_dir / "depthfm" / "h" / "run_0" / "checkpoints" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        return p

    def test_rmse_picks_min_trailing_value(self, ra, tmp_path):
        self._mk(tmp_path, "depthfm-best-rmse-100-0.0900.ckpt")
        best = self._mk(tmp_path, "depthfm-best-rmse-200-0.0100.ckpt")
        assert ra.find_best_checkpoint(tmp_path, "rmse") == best

    def test_photo_picks_max_trailing_value(self, ra, tmp_path):
        self._mk(tmp_path, "depthfm-best-photo-100-0.10.ckpt")
        best = self._mk(tmp_path, "depthfm-best-photo-200-0.90.ckpt")
        assert ra.find_best_checkpoint(tmp_path, "photo") == best

    def test_unparseable_filename_falls_back_to_sentinel(self, ra, tmp_path):
        # An unparseable name maps to +inf for rmse, so a parseable lower one
        # is chosen as the minimum.
        good = self._mk(tmp_path, "depthfm-best-rmse-1-0.0500.ckpt")
        self._mk(tmp_path, "depthfm-best-rmse-junk.ckpt")
        assert ra.find_best_checkpoint(tmp_path, "rmse") == good

    def test_none_when_no_candidates(self, ra, tmp_path):
        assert ra.find_best_checkpoint(tmp_path, "rmse") is None


# ---------------------------------------------------------------------------
# _load_per_patch
# ---------------------------------------------------------------------------


class TestLoadPerPatch:
    def _npz(self, run_dir, tag, rank, tile_ids, rmse):
        p = (run_dir / "depthfm" / "h" / "run_0"
             / f"test_per_patch_{tag}_rank{rank}.npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, tile_id=np.array(tile_ids), rmse=np.array(rmse,
                                                              dtype=float))
        return p

    @staticmethod
    def _corrupt_npz(path: pathlib.Path) -> None:
        """A valid zip whose .npy member is corrupt -> np.load raises ValueError.

        ValueError is one of the exception types _load_per_patch catches. See
        test_truncated_npz_is_a_real_bug below for the *uncaught* case.
        """
        import zipfile

        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("rmse.npy", b"\x93NUMPY\x01\x00corrupt-header")

    def test_concat_and_tile_id_dedupe_across_ranks(self, ra, tmp_path):
        self._npz(tmp_path, "rmse_ckpt", 0, ["a", "b"], [0.1, 0.2])
        self._npz(tmp_path, "rmse_ckpt", 1, ["b", "c"], [0.2, 0.3])
        df = ra._load_per_patch(tmp_path, "rmse_ckpt")
        assert df is not None
        assert sorted(df["tile_id"].tolist()) == ["a", "b", "c"]

    def test_none_when_no_files(self, ra, tmp_path):
        assert ra._load_per_patch(tmp_path, "rmse_ckpt") is None

    def test_corrupt_npz_skipped_not_fatal(self, ra, tmp_path, caplog):
        self._npz(tmp_path, "rmse_ckpt", 0, ["a"], [0.1])
        self._corrupt_npz(tmp_path / "depthfm" / "h" / "run_0"
                          / "test_per_patch_rmse_ckpt_rank1.npz")
        with caplog.at_level(logging.ERROR, logger="ablation"):
            df = ra._load_per_patch(tmp_path, "rmse_ckpt")
        assert df is not None and df["tile_id"].tolist() == ["a"]
        assert any("Skipping unreadable" in r.message for r in caplog.records)

    def test_all_bad_returns_none_and_warns(self, ra, tmp_path, caplog):
        self._corrupt_npz(tmp_path / "depthfm" / "h" / "run_0"
                          / "test_per_patch_rmse_ckpt_rank0.npz")
        with caplog.at_level(logging.WARNING, logger="ablation"):
            assert ra._load_per_patch(tmp_path, "rmse_ckpt") is None
        assert any("No readable per-patch files" in r.message
                   for r in caplog.records)

    def test_truncated_npz_skipped_not_fatal(self, ra, tmp_path, caplog):
        """A truncated npz (the documented 'job killed mid-write' case)
        raises zipfile.BadZipFile; _load_per_patch must skip it, not abort
        the whole aggregation. Regression for the broadened except clause."""
        self._npz(tmp_path, "rmse_ckpt", 0, ["a"], [0.1])
        good = (tmp_path / "depthfm" / "h" / "run_0"
                / "test_per_patch_rmse_ckpt_rank0.npz")
        raw = good.read_bytes()
        truncated = (tmp_path / "depthfm" / "h" / "run_0"
                     / "test_per_patch_rmse_ckpt_rank1.npz")
        truncated.write_bytes(raw[: len(raw) // 2])
        with caplog.at_level(logging.ERROR, logger="ablation"):
            df = ra._load_per_patch(tmp_path, "rmse_ckpt")
        assert df is not None and df["tile_id"].tolist() == ["a"]
        assert any("Skipping unreadable" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_creates_log_and_single_debug_file_handler(self, ra, tmp_path,
                                                        _restore_logging):
        from logging.handlers import RotatingFileHandler

        log_path = ra.setup_logging(tmp_path)
        assert log_path == tmp_path / "orchestrator.log"
        assert log_path.exists()
        fh = [h for h in logging.getLogger().handlers
              if isinstance(h, RotatingFileHandler)]
        assert len(fh) == 1
        assert fh[0].level == logging.DEBUG

    def test_idempotent_does_not_stack_file_handlers(self, ra, tmp_path,
                                                     _restore_logging):
        from logging.handlers import RotatingFileHandler

        ra.setup_logging(tmp_path)
        ra.setup_logging(tmp_path)
        ra.setup_logging(tmp_path)
        fh = [h for h in logging.getLogger().handlers
              if isinstance(h, RotatingFileHandler)]
        assert len(fh) == 1

    def test_console_handler_stays_info(self, ra, tmp_path, _restore_logging):
        from logging.handlers import RotatingFileHandler

        ra.setup_logging(tmp_path)
        consoles = [h for h in logging.getLogger().handlers
                    if isinstance(h, logging.StreamHandler)
                    and not isinstance(h, RotatingFileHandler)]
        assert consoles  # basicConfig installed one
        assert all(h.level == logging.INFO for h in consoles)


# ---------------------------------------------------------------------------
# aggregate (happy path) + degenerate-stats degradation
# ---------------------------------------------------------------------------


class TestAggregate:
    def _npz(self, run_dir, job_id, tag, n=12):
        p = (run_dir / job_id / "depthfm" / "h" / "run_0"
             / f"test_per_patch_{tag}_rank0.npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash(job_id)) % (2**32))
        np.savez(
            p,
            tile_id=np.array([f"t{i}" for i in range(n)]),
            rmse=rng.normal(0.1, 0.01, n),
            abs_rel=rng.normal(0.05, 0.005, n),
            si_log=rng.normal(0.2, 0.01, n),
            delta_1=rng.normal(0.9, 0.01, n),
            normal_angular_error=rng.normal(10.0, 0.5, n),
            photo_consistency=rng.normal(0.7, 0.02, n),
            ms_ssim_topo=rng.normal(0.8, 0.02, n),
            dbf_score=rng.normal(0.5, 0.02, n),
        )

    @pytest.mark.slow
    def test_happy_path_writes_expected_csvs(self, ra, tmp_path):
        spec = _make_spec(output_root=str(tmp_path))
        jobs = ra.build_jobs(spec)
        store = ra.StateStore(tmp_path / "state.json")
        for j in jobs:
            for tag in ("rmse_ckpt", "photo_ckpt"):
                self._npz(tmp_path, j.job_id, tag, n=12)

        ra.aggregate(spec, store, jobs)

        rt = tmp_path / "results_table.csv"
        assert rt.exists()
        import pandas as pd
        cols = set(pd.read_csv(rt).columns)
        assert {"variant", "metric", "mean", "n_patches"} <= cols
        assert (tmp_path / "results_rmse_track.csv").exists()
        assert (tmp_path / "results_photo_track.csv").exists()
        stats = pd.read_csv(tmp_path / "results_stats_rmse.csv")
        assert {"variant", "metric", "t_pvalue",
                "holm_pvalue"} <= set(stats.columns)

    def test_empty_results_writes_table_then_skips_stats(self, ra, tmp_path):
        # No per-patch npz anywhere -> long_df empty, early return.
        spec = _make_spec(output_root=str(tmp_path))
        jobs = ra.build_jobs(spec)
        store = ra.StateStore(tmp_path / "state.json")
        ra.aggregate(spec, store, jobs)
        rt = tmp_path / "results_table.csv"
        assert rt.exists()
        assert not (tmp_path / "results_stats_rmse.csv").exists()

    def test_degenerate_stats_degrade_to_nan_without_raising(self, ra,
                                                             tmp_path):
        """1 baseline + 1 variant, 2 seeds, identical constant metrics.

        Wilcoxon raises on all-zero differences; the `except Exception`
        guards must degrade to NaN instead of propagating.
        """
        spec = _make_spec(
            output_root=str(tmp_path),
            seeds=[1, 2],
            variants=[
                {"id": "L0", "label": "baseline", "overrides": []},
                {"id": "L1", "overrides": []},
            ],
        )
        jobs = ra.build_jobs(spec)
        store = ra.StateStore(tmp_path / "state.json")
        for j in jobs:
            for tag in ("rmse_ckpt", "photo_ckpt"):
                p = (tmp_path / j.job_id / "depthfm" / "h" / "run_0"
                     / f"test_per_patch_{tag}_rank0.npz")
                p.parent.mkdir(parents=True, exist_ok=True)
                # Constant -> zero paired differences -> wilcoxon ValueError.
                np.savez(
                    p,
                    tile_id=np.array([f"t{i}" for i in range(12)]),
                    rmse=np.full(12, 0.1),
                    photo_consistency=np.full(12, 0.7),
                )

        # Must not raise.
        ra.aggregate(spec, store, jobs)
        assert (tmp_path / "results_table.csv").exists()


# ---------------------------------------------------------------------------
# launch_train.sh wrapper (no training executed)
# ---------------------------------------------------------------------------


class TestLaunchScript:
    SCRIPT = _REPO_ROOT / "scripts" / "training" / "launch_train.sh"

    def test_exists_and_executable(self):
        assert self.SCRIPT.exists()
        import os as _os

        assert _os.access(self.SCRIPT, _os.X_OK)

    def test_bash_syntax_check_passes(self):
        import subprocess

        r = subprocess.run(
            ["bash", "-n", str(self.SCRIPT)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr

    def test_honors_preset_cuda_visible_devices_guard(self):
        text = self.SCRIPT.read_text()
        assert ': "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"' in text
        assert "export CUDA_VISIBLE_DEVICES" in text
