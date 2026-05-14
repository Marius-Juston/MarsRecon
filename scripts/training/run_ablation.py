#!/usr/bin/env python
"""Mars DepthFM — Loss-Component Ablation Orchestrator.

Owns the queue for the loss drop-one ablation campaign. Reads a declarative
spec (configs/ablation/loss_dropone.yaml), schedules jobs across multiple
GPU lanes, watches for crashes, and runs a post-hoc aggregator that produces
paper-ready statistical summaries (paired Wilcoxon + Holm-Bonferroni +
hierarchical bootstrap CIs) for both headline metrics (RMSE, photo).

Usage:
    # End-to-end (preflight + queue + aggregate)
    uv run python scripts/run_ablation.py --spec configs/ablation/loss_dropone.yaml

    # Plan only — no launches
    uv run python scripts/run_ablation.py --spec ... --dry-run

    # Run only one specific job (smoke test)
    uv run python scripts/run_ablation.py --spec ... --only L0_s42

    # Re-run the aggregator on existing results
    uv run python scripts/run_ablation.py --spec ... --aggregate-only

State lives in <output_root>/state.json (fsync'd on every transition).
Restart the script after a crash; it picks up where it left off.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ablation")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Status enum (strings to keep state.json human-readable)
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CRASHED = "crashed"
TERMINAL = {STATUS_DONE, STATUS_FAILED}

# Metrics tracked in the per-patch npz (must match lightning_module dump).
METRICS_RMSE_TRACK = ["rmse", "abs_rel", "si_log", "delta_1", "normal_angular_error"]
METRICS_PHOTO_TRACK = ["photo_consistency", "ms_ssim_topo", "dbf_score"]
ALL_TRACKED_METRICS = sorted(set(METRICS_RMSE_TRACK + METRICS_PHOTO_TRACK))


# --------------------------------------------------------------------------
# Spec + job model
# --------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str  # e.g. "L1_s42"
    variant_id: str  # e.g. "L1"
    variant_label: str
    seed: int
    overrides: list[str]  # OmegaConf dotlist (variant + common)


@dataclass
class JobState:
    status: str = STATUS_QUEUED
    attempts: int = 0
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    run_dir: Optional[str] = None
    lane: Optional[str] = None
    pid: Optional[int] = None
    exit_reason: Optional[str] = None
    final_rmse: Optional[float] = None
    final_photo: Optional[float] = None
    best_rmse_ckpt: Optional[str] = None
    best_photo_ckpt: Optional[str] = None
    test_done_rmse_ckpt: bool = False
    test_done_photo_ckpt: bool = False


def load_spec(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_jobs(spec: dict) -> list[Job]:
    jobs: list[Job] = []
    common = list(spec.get("common_overrides", []))
    for v in spec["variants"]:
        for seed in spec["seeds"]:
            jobs.append(Job(
                job_id=f"{v['id']}_s{seed}",
                variant_id=v["id"],
                variant_label=v.get("label", v["id"]),
                seed=int(seed),
                overrides=list(v.get("overrides", [])) + common,
            ))
    return jobs


# --------------------------------------------------------------------------
# State persistence
# --------------------------------------------------------------------------

class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._states: dict[str, JobState] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text())
        self._states = {jid: JobState(**v) for jid, v in raw.items()}
        logger.info("Loaded state for %d jobs from %s", len(self._states), self.path)

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({jid: asdict(s) for jid, s in self._states.items()}, indent=2))
        os.replace(tmp, self.path)
        # fsync the directory to make the rename durable
        try:
            fd = os.open(self.path.parent, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass

    def get(self, job_id: str) -> JobState:
        return self._states.setdefault(job_id, JobState())

    def set(self, job_id: str, **kwargs) -> None:
        st = self.get(job_id)
        for k, v in kwargs.items():
            setattr(st, k, v)
        self._flush()

    def snapshot(self) -> dict[str, JobState]:
        return dict(self._states)


# --------------------------------------------------------------------------
# Cache preflight (LitData _SUCCESS markers)
# --------------------------------------------------------------------------

def cache_preflight(base_config_path: Path, force_rebuild: bool = False) -> None:
    """Ensure LitData chunks exist before any training launch.

    Calls into src/cache.py::litdata_cache_root to compute the canonical
    cache dir for this config, then checks _SUCCESS markers for train/val/test.
    If anything is missing (or --rebuild-cache), invokes build_litdata.py.
    """
    # Defer imports so this script can run --aggregate-only without the
    # full depthfm extra.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from omegaconf import OmegaConf
    from cache import litdata_cache_root

    config = OmegaConf.load(base_config_path)
    cache_root = litdata_cache_root(config)
    missing = []
    for split in ("train", "val", "test"):
        if not (cache_root / split / "_SUCCESS").exists():
            missing.append(split)

    if not missing and not force_rebuild:
        logger.info("LitData cache OK at %s (all splits have _SUCCESS).", cache_root)
        return

    if force_rebuild:
        logger.warning("--rebuild-cache: removing existing _SUCCESS markers under %s", cache_root)
        for split in ("train", "val", "test"):
            marker = cache_root / split / "_SUCCESS"
            if marker.exists():
                marker.unlink()

    logger.warning(
        "LitData cache incomplete (missing splits: %s). Rebuilding via build_litdata.py …",
        missing or "[force]",
    )
    cmd = [
        "uv", "run", "python", "-m", "depth_fm.build_litdata",
        "--config", str(base_config_path),
        "--workers", "96",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode
    if rc != 0:
        raise SystemExit(f"build_litdata.py exited with code {rc}; aborting")
    # Re-check after rebuild
    for split in ("train", "val", "test"):
        if not (cache_root / split / "_SUCCESS").exists():
            raise SystemExit(f"Cache rebuild reported success but {split}/_SUCCESS is still missing")
    logger.info("LitData cache rebuilt OK at %s.", cache_root)


# --------------------------------------------------------------------------
# Job launching + watchdog
# --------------------------------------------------------------------------

@dataclass
class RunningJob:
    job: Job
    proc: subprocess.Popen
    lane: str
    run_dir: Path
    csv_path: Path
    log_file: object  # open file handle
    last_val_seen_at: float  # monotonic time of last new val/* row
    last_val_row_count: int = 0
    started_at: float = 0.0


def launch_train(job: Job, lane: str, spec: dict) -> RunningJob:
    base_config = spec["base_config"]
    output_root = REPO_ROOT / spec["output_root"]
    run_dir = output_root / job.job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    overrides = list(job.overrides) + [
        f"training.run_name=ablation_{job.job_id}",
        f"training.output_dir={run_dir}",
        f"training.project_name={spec['project_name']}",
    ]

    cmd = [
        "bash", "scripts/launch_train.sh",
        "--config", base_config,
        "--n_runs", "1",
        "--seed", str(job.seed),
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = lane
    env["WANDB_RUN_GROUP"] = spec.get("group", spec.get("study_name", "ablation"))

    log_path = run_dir / "job.log"
    log_file = open(log_path, "ab")
    log_file.write(f"\n=== launched at {time.strftime('%Y-%m-%dT%H:%M:%S')} lane={lane} ===\n".encode())
    log_file.write(f"cmd: {' '.join(cmd)}\n".encode())
    log_file.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group → reliable group kill
    )
    (run_dir / "pid").write_text(str(proc.pid))

    csv_path = run_dir / "csv" / "version_0" / "metrics.csv"
    return RunningJob(
        job=job, proc=proc, lane=lane, run_dir=run_dir,
        csv_path=csv_path, log_file=log_file,
        last_val_seen_at=time.monotonic(), last_val_row_count=0,
        started_at=time.monotonic(),
    )


def kill_running(rj: RunningJob, reason: str) -> None:
    logger.warning("Killing job %s: %s", rj.job.job_id, reason)
    try:
        os.killpg(rj.proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        rj.proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(rj.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def watchdog_check(rj: RunningJob, watchdog_cfg: dict) -> Optional[str]:
    """Return a non-None failure reason string if the job should be killed.

    Crash detection only — never performance pruning.
    """
    # Read CSV if available
    csv_path = rj.csv_path
    val_rmse_seen, val_photo_seen = [], []
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "val/rmse_mean" in df.columns:
                v = df["val/rmse_mean"].dropna().to_numpy()
                val_rmse_seen = v.tolist()
                if v.size and not np.isfinite(v[-1]):
                    return "val/rmse_mean is NaN/Inf"
            if "val/photo_consistency_mean" in df.columns:
                v = df["val/photo_consistency_mean"].dropna().to_numpy()
                val_photo_seen = v.tolist()
                if v.size and not np.isfinite(v[-1]):
                    return "val/photo_consistency_mean is NaN/Inf"
            new_count = len(val_rmse_seen)
            if new_count > rj.last_val_row_count:
                rj.last_val_row_count = new_count
                rj.last_val_seen_at = time.monotonic()
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass  # CSV being written; try again next tick

    # Stall detection (only after we've seen at least one val row, or after a grace period)
    grace = 1800.0  # 30 min grace for first val tick
    timeout = float(watchdog_cfg.get("val_stall_timeout_seconds", 7200))
    elapsed = time.monotonic() - rj.last_val_seen_at
    if rj.last_val_row_count == 0:
        if (time.monotonic() - rj.started_at) > grace + timeout:
            return f"no val/* row produced after {grace + timeout:.0f}s"
    else:
        if elapsed > timeout:
            return f"val/* stalled for {elapsed:.0f}s (>{timeout:.0f}s)"

    return None


def find_best_checkpoint(run_dir: Path, kind: str) -> Optional[Path]:
    """Locate the best-{rmse,photo} checkpoint under run_dir.

    train_lightning.py adds an extra <model_type>/<config_hash>/run_0 nesting,
    so we glob to find it.
    """
    assert kind in ("rmse", "photo")
    pattern = f"depthfm-best-{kind}-*.ckpt"
    candidates = list(run_dir.rglob(f"checkpoints/{pattern}"))
    if not candidates:
        return None

    # Filename embeds the metric value; pick lowest RMSE or highest photo.
    def metric_from(p: Path) -> float:
        # e.g. depthfm-best-rmse-1234-0.0456.ckpt
        try:
            return float(p.stem.rsplit("-", 1)[-1])
        except ValueError:
            return float("inf") if kind == "rmse" else float("-inf")

    if kind == "rmse":
        return min(candidates, key=metric_from)
    else:
        return max(candidates, key=metric_from)


def run_test_only(job: Job, lane: str, spec: dict, ckpt: Path, tag: str) -> int:
    """Synchronously run the test-only entry point on `ckpt` with the given tag."""
    base_config = spec["base_config"]
    run_dir = REPO_ROOT / spec["output_root"] / job.job_id
    overrides = list(job.overrides) + [
        f"training.run_name=ablation_{job.job_id}",
        f"training.output_dir={run_dir}",
        f"training.project_name={spec['project_name']}",
        # We want the same num_gpus as during training for sampler partition consistency.
    ]
    # Direct invocation of the python module via torchrun so DDP test loop is sharded.
    cmd = [
        "bash", "scripts/launch_train.sh",
        "--config", base_config,
        "--seed", str(job.seed),
        *overrides,
        "--test_only",
        "--ckpt", str(ckpt),
        "--test_tag", tag,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = lane

    log_path = run_dir / f"test_{tag}.log"
    with open(log_path, "ab") as log_file:
        log_file.write(f"\n=== test-only {tag} at {time.strftime('%Y-%m-%dT%H:%M:%S')} lane={lane} ===\n".encode())
        log_file.write(f"cmd: {' '.join(cmd)}\n".encode())
        log_file.flush()
        rc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
        ).returncode
    return rc


# --------------------------------------------------------------------------
# Main scheduler loop
# --------------------------------------------------------------------------

def schedule(spec: dict, store: StateStore, jobs: list[Job], only: Optional[set[str]]) -> None:
    lanes: list[str] = list(spec["lanes"])
    watchdog_cfg = spec.get("watchdog", {})
    max_attempts = int(watchdog_cfg.get("max_attempts", 3))

    # Build the work queue from state (skip done; re-queue failed/crashed under retry budget).
    pending: list[Job] = []
    for j in jobs:
        if only is not None and j.job_id not in only:
            continue
        st = store.get(j.job_id)
        if st.status == STATUS_DONE:
            logger.info("Skipping %s (already done)", j.job_id)
            continue
        if st.status == STATUS_FAILED and st.attempts >= max_attempts:
            logger.warning("Skipping %s (permanently failed after %d attempts)", j.job_id, st.attempts)
            continue
        if st.status == STATUS_RUNNING:
            logger.warning("State says %s was running — will relaunch (PID may be dead).", j.job_id)
            store.set(j.job_id, status=STATUS_QUEUED)
        pending.append(j)

    logger.info("Scheduling %d jobs across %d lanes", len(pending), len(lanes))

    running: dict[str, RunningJob] = {}  # lane -> RunningJob
    queue_idx = 0
    poll_interval = 60.0

    def free_lane() -> Optional[str]:
        for lane in lanes:
            if lane not in running:
                return lane
        return None

    while queue_idx < len(pending) or running:
        # Launch into free lanes
        while queue_idx < len(pending):
            lane = free_lane()
            if lane is None:
                break
            j = pending[queue_idx]
            queue_idx += 1
            st = store.get(j.job_id)
            store.set(
                j.job_id,
                status=STATUS_RUNNING,
                attempts=(st.attempts or 0) + 1,
                started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                ended_at=None,
                lane=lane,
                exit_reason=None,
            )
            rj = launch_train(j, lane, spec)
            running[lane] = rj
            store.set(j.job_id, pid=rj.proc.pid, run_dir=str(rj.run_dir))
            logger.info("Launched %s on lane=%s pid=%d", j.job_id, lane, rj.proc.pid)

        # Poll running jobs
        time.sleep(poll_interval)
        for lane in list(running.keys()):
            rj = running[lane]
            rc = rj.proc.poll()
            if rc is None:
                reason = watchdog_check(rj, watchdog_cfg)
                if reason is not None:
                    kill_running(rj, reason)
                    rj.log_file.close()
                    store.set(rj.job.job_id, status=STATUS_CRASHED, exit_reason=reason,
                              ended_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    del running[lane]
                    # Retry?
                    st = store.get(rj.job.job_id)
                    if st.attempts < max_attempts:
                        logger.warning("Re-queuing %s (attempt %d/%d)", rj.job.job_id, st.attempts, max_attempts)
                        pending.append(rj.job)
                        store.set(rj.job.job_id, status=STATUS_QUEUED)
                    else:
                        store.set(rj.job.job_id, status=STATUS_FAILED)
                continue

            # Process exited on its own
            rj.log_file.close()
            if rc != 0:
                logger.warning("%s exited with rc=%d", rj.job.job_id, rc)
                st = store.get(rj.job.job_id)
                if st.attempts < max_attempts:
                    store.set(rj.job.job_id, status=STATUS_CRASHED,
                              exit_reason=f"exit code {rc}",
                              ended_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    logger.warning("Re-queuing %s (attempt %d/%d)", rj.job.job_id, st.attempts, max_attempts)
                    pending.append(rj.job)
                    store.set(rj.job.job_id, status=STATUS_QUEUED)
                else:
                    store.set(rj.job.job_id, status=STATUS_FAILED,
                              exit_reason=f"exit code {rc}",
                              ended_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                del running[lane]
                continue

            # Success path → locate checkpoints, run test-only twice on the SAME lane (now free briefly)
            store.set(rj.job.job_id, ended_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            best_rmse = find_best_checkpoint(rj.run_dir, "rmse")
            best_photo = find_best_checkpoint(rj.run_dir, "photo")
            store.set(rj.job.job_id,
                      best_rmse_ckpt=str(best_rmse) if best_rmse else None,
                      best_photo_ckpt=str(best_photo) if best_photo else None)

            if best_rmse is None and best_photo is None:
                logger.error("%s finished but no checkpoints found under %s", rj.job.job_id, rj.run_dir)
                store.set(rj.job.job_id, status=STATUS_FAILED, exit_reason="no checkpoints")
                del running[lane]
                continue

            # Test on best-rmse
            if best_rmse is not None:
                logger.info("%s: testing best-rmse ckpt %s", rj.job.job_id, best_rmse.name)
                rc1 = run_test_only(rj.job, lane, spec, best_rmse, "rmse_ckpt")
                store.set(rj.job.job_id, test_done_rmse_ckpt=(rc1 == 0))
                if rc1 != 0:
                    logger.error("%s: test-only on best-rmse rc=%d", rj.job.job_id, rc1)
            if best_photo is not None and best_photo != best_rmse:
                logger.info("%s: testing best-photo ckpt %s", rj.job.job_id, best_photo.name)
                rc2 = run_test_only(rj.job, lane, spec, best_photo, "photo_ckpt")
                store.set(rj.job.job_id, test_done_photo_ckpt=(rc2 == 0))
                if rc2 != 0:
                    logger.error("%s: test-only on best-photo rc=%d", rj.job.job_id, rc2)
            else:
                # If best_rmse == best_photo, copy artifacts under the photo_ckpt tag.
                store.set(rj.job.job_id, test_done_photo_ckpt=True)

            store.set(rj.job.job_id, status=STATUS_DONE)
            del running[lane]
            logger.info("Job %s DONE", rj.job.job_id)


# --------------------------------------------------------------------------
# Aggregator
# --------------------------------------------------------------------------

def _load_per_patch(run_dir: Path, tag: str) -> Optional[pd.DataFrame]:
    """Concatenate all per-rank npz files for a given tag into one DataFrame."""
    files = sorted(run_dir.rglob(f"test_per_patch_{tag}_rank*.npz"))
    if not files:
        return None
    frames = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True)
    # Deduplicate on tile_id across ranks (DDP may double-count last batch with samplers)
    if "tile_id" in df.columns:
        df = df.drop_duplicates(subset=["tile_id"]).reset_index(drop=True)
    return df


def aggregate(spec: dict, store: StateStore, jobs: list[Job]) -> None:
    from scipy import stats as sps

    output_root = REPO_ROOT / spec["output_root"]
    by_variant: dict[str, list[Job]] = {}
    for j in jobs:
        by_variant.setdefault(j.variant_id, []).append(j)

    # Build long-form table
    rows = []
    per_patch_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for j in jobs:
        st = store.get(j.job_id)
        run_dir = output_root / j.job_id
        for tag in ("rmse_ckpt", "photo_ckpt"):
            df = _load_per_patch(run_dir, tag)
            if df is None:
                continue
            per_patch_cache[(j.job_id, tag)] = df
            for m in ALL_TRACKED_METRICS:
                if m in df.columns:
                    rows.append({
                        "variant": j.variant_id,
                        "label": j.variant_label,
                        "seed": j.seed,
                        "ckpt": tag,
                        "metric": m,
                        "mean": float(df[m].mean()),
                        "median": float(df[m].median()),
                        "n_patches": int(len(df)),
                    })
    long_df = pd.DataFrame(rows)
    long_df.to_csv(output_root / "results_table.csv", index=False)
    logger.info("Wrote %s", output_root / "results_table.csv")

    if long_df.empty:
        logger.warning("No per-patch results found — skipping summary/stats")
        return

    baseline_id = spec["variants"][0]["id"]  # convention: first variant is baseline

    # Per-track summaries with paired stats vs baseline (matched on seed + tile_id)
    def write_track(track_name: str, metrics: list[str], ckpt_tag: str) -> None:
        summary_rows = []
        stats_rows = []
        boot_rows = []

        # Baseline mean ± std across seeds for each metric
        b_means_by_metric = {m: [] for m in metrics}
        for j in by_variant.get(baseline_id, []):
            df = per_patch_cache.get((j.job_id, ckpt_tag))
            if df is None:
                continue
            for m in metrics:
                if m in df.columns:
                    b_means_by_metric[m].append(float(df[m].mean()))

        for vid, variant_jobs in by_variant.items():
            for m in metrics:
                # Variant means across seeds
                v_means = []
                paired_deltas_run = []  # per-seed (variant_mean - baseline_mean_same_seed)
                paired_deltas_patch_pvals = []
                for vj in variant_jobs:
                    vdf = per_patch_cache.get((vj.job_id, ckpt_tag))
                    if vdf is None or m not in vdf.columns:
                        continue
                    v_means.append(float(vdf[m].mean()))
                    # Find baseline job with same seed
                    bj = next((b for b in by_variant.get(baseline_id, []) if b.seed == vj.seed), None)
                    if bj is None:
                        continue
                    bdf = per_patch_cache.get((bj.job_id, ckpt_tag))
                    if bdf is None or m not in bdf.columns:
                        continue
                    paired_deltas_run.append(float(vdf[m].mean()) - float(bdf[m].mean()))
                    # Patch-level paired test on shared tile_ids
                    try:
                        merged = vdf[["tile_id", m]].merge(
                            bdf[["tile_id", m]], on="tile_id", suffixes=("_v", "_b"),
                        )
                        if len(merged) >= 10:
                            stat, p = sps.wilcoxon(merged[f"{m}_v"], merged[f"{m}_b"])
                            paired_deltas_patch_pvals.append(float(p))
                    except Exception:
                        pass

                if not v_means:
                    continue

                row = {
                    "variant": vid,
                    "metric": m,
                    "ckpt": ckpt_tag,
                    "variant_mean": float(np.mean(v_means)),
                    "variant_std": float(np.std(v_means, ddof=0 if len(v_means) < 2 else 1)),
                    "baseline_mean": float(np.mean(b_means_by_metric[m])) if b_means_by_metric[m] else None,
                    "baseline_std": float(
                        np.std(b_means_by_metric[m], ddof=0 if len(b_means_by_metric[m]) < 2 else 1)) if
                    b_means_by_metric[m] else None,
                    "delta_mean": float(np.mean(paired_deltas_run)) if paired_deltas_run else None,
                    "n_seeds": len(v_means),
                }
                summary_rows.append(row)

                # Run-level paired t-test on the per-seed deltas (n=3 typically)
                if vid != baseline_id and len(paired_deltas_run) >= 2:
                    try:
                        t_stat, t_p = sps.ttest_1samp(paired_deltas_run, popmean=0.0)
                    except Exception:
                        t_stat, t_p = float("nan"), float("nan")
                    # Combine patch-level Wilcoxon p-values via Fisher's method
                    if paired_deltas_patch_pvals:
                        pp = np.clip(paired_deltas_patch_pvals, 1e-300, 1.0)
                        chi2 = -2.0 * np.sum(np.log(pp))
                        fisher_p = float(sps.chi2.sf(chi2, df=2 * len(pp)))
                    else:
                        fisher_p = float("nan")
                    stats_rows.append({
                        "variant": vid, "metric": m, "ckpt": ckpt_tag,
                        "t_stat": float(t_stat), "t_pvalue": float(t_p),
                        "wilcoxon_patch_fisher_pvalue": fisher_p,
                        "n_seed_pairs": len(paired_deltas_run),
                    })

                # Hierarchical bootstrap CI (resample seeds, then patches within seed)
                if v_means:
                    boot = []
                    rng = np.random.default_rng(0)
                    for _ in range(2000):
                        # resample seeds with replacement
                        sampled_seeds = rng.choice(len(variant_jobs), size=len(variant_jobs), replace=True)
                        per_seed_means = []
                        for s_idx in sampled_seeds:
                            vj = variant_jobs[s_idx]
                            vdf = per_patch_cache.get((vj.job_id, ckpt_tag))
                            if vdf is None or m not in vdf.columns:
                                continue
                            patches = vdf[m].to_numpy()
                            if patches.size == 0:
                                continue
                            sampled_patches = rng.choice(patches, size=patches.size, replace=True)
                            per_seed_means.append(float(np.mean(sampled_patches)))
                        if per_seed_means:
                            boot.append(float(np.mean(per_seed_means)))
                    if boot:
                        lo, hi = np.percentile(boot, [2.5, 97.5])
                        boot_rows.append({
                            "variant": vid, "metric": m, "ckpt": ckpt_tag,
                            "ci95_lo": float(lo), "ci95_hi": float(hi),
                            "n_boot": len(boot),
                        })

        # Holm-Bonferroni correction across variants (per metric) within this track
        if stats_rows:
            sdf = pd.DataFrame(stats_rows)
            adj_pvals = []
            for m in metrics:
                sub = sdf[sdf["metric"] == m].copy()
                if sub.empty:
                    continue
                # Use Fisher's combined p-value when available, else t-test p
                src = sub["wilcoxon_patch_fisher_pvalue"].fillna(sub["t_pvalue"]).to_numpy()
                order = np.argsort(src)
                k = len(src)
                adj = np.empty(k)
                running_max = 0.0
                for i, idx in enumerate(order):
                    val = (k - i) * src[idx]
                    running_max = max(running_max, val)
                    adj[idx] = min(running_max, 1.0)
                sub["holm_pvalue"] = adj
                adj_pvals.append(sub)
            if adj_pvals:
                sdf_out = pd.concat(adj_pvals, ignore_index=True)
                sdf_out.to_csv(output_root / f"results_stats_{track_name}.csv", index=False)
                logger.info("Wrote %s", output_root / f"results_stats_{track_name}.csv")

        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(output_root / f"results_{track_name}_track.csv", index=False)
            logger.info("Wrote %s", output_root / f"results_{track_name}_track.csv")
        if boot_rows:
            pd.DataFrame(boot_rows).to_csv(output_root / f"results_bootstrap_{track_name}.csv", index=False)
            logger.info("Wrote %s", output_root / f"results_bootstrap_{track_name}.csv")

    write_track("rmse", METRICS_RMSE_TRACK, "rmse_ckpt")
    write_track("photo", METRICS_PHOTO_TRACK, "photo_ckpt")

    # Co-optimization table: both metric families evaluated from the best-rmse checkpoint.
    coopt_rows = []
    for j in jobs:
        df = per_patch_cache.get((j.job_id, "rmse_ckpt"))
        if df is None:
            continue
        row = {"variant": j.variant_id, "seed": j.seed, "ckpt": "rmse_ckpt"}
        for m in ("rmse", "photo_consistency"):
            if m in df.columns:
                row[m] = float(df[m].mean())
        coopt_rows.append(row)
    if coopt_rows:
        pd.DataFrame(coopt_rows).to_csv(output_root / "results_cooptimization.csv", index=False)
        logger.info("Wrote %s", output_root / "results_cooptimization.csv")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Mars DepthFM loss-component ablation orchestrator")
    p.add_argument("--spec", required=True, type=Path, help="Path to ablation spec YAML")
    p.add_argument("--dry-run", action="store_true", help="Print planned queue, exit")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated job_ids to run (e.g. 'L0_s42,L1_s42')")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force LitData cache rebuild before scheduling")
    p.add_argument("--skip-cache-check", action="store_true",
                   help="Don't preflight the cache (assume it's ready)")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip scheduling; just regenerate the summary CSVs")
    args = p.parse_args()

    spec = load_spec(args.spec)
    jobs = build_jobs(spec)
    output_root = REPO_ROOT / spec["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    store = StateStore(output_root / "state.json")

    if args.dry_run:
        print(f"Spec: {args.spec}")
        print(f"Output root: {output_root}")
        print(f"Lanes: {spec['lanes']}")
        print(f"Total jobs: {len(jobs)}")
        print(f"Estimated wall-clock at 24h/run, {len(spec['lanes'])} lanes: "
              f"{len(jobs) * 24 / max(1, len(spec['lanes'])):.1f}h "
              f"(~{len(jobs) * 24 / max(1, len(spec['lanes'])) / 24:.1f} days)")
        print("\nPlanned jobs:")
        for j in jobs:
            st = store.get(j.job_id)
            print(f"  {j.job_id:14}  seed={j.seed}  status={st.status:8}  overrides={j.overrides}")
        return 0

    only_set: Optional[set[str]] = None
    if args.only:
        only_set = {s.strip() for s in args.only.split(",") if s.strip()}

    if not args.aggregate_only:
        if not args.skip_cache_check:
            cache_preflight(Path(spec["base_config"]), force_rebuild=args.rebuild_cache)
        try:
            schedule(spec, store, jobs, only=only_set)
        except KeyboardInterrupt:
            logger.warning("Interrupted; state is persisted. Restart to resume.")
            return 130

    try:
        aggregate(spec, store, jobs)
    except Exception:
        logger.exception("Aggregator failed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
