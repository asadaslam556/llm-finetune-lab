"""Runner behavior: the loop, the lock, the failure path."""

import shutil
import time

import pytest

import finetune_lab.pipeline.runner as runner
from finetune_lab.core.errors import PipelineBusy
from finetune_lab.pipeline.runner import _latest_run_dir, get_store, start_run
from finetune_lab.pipeline.status import STAGE_ORDER


def wait_done(timeout=30.0):
    store = get_store()
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = store.read()
        # the runner writes final status a hair before releasing its lock,
        # so wait for both, otherwise back-to-back starts flake with Busy
        if status["state"] in ("done", "failed") and not runner.is_running():
            return status
        time.sleep(0.05)
    raise TimeoutError(f"pipeline still {status['state']} after {timeout}s")


class TestRunner:
    def test_full_dry_run_completes_all_seven(self, env):
        info = start_run(dry_run=True)
        assert info["stages"] == STAGE_ORDER
        status = wait_done()
        assert status["state"] == "done"
        assert all(s["state"] == "done" for s in status["stages"].values())
        assert status["run_id"] == info["run_id"]
        assert all(s["seconds"] is not None for s in status["stages"].values())

    def test_subset_runs_only_requested_stages(self, env):
        start_run(dry_run=True, stages=["ingest", "prepare"])
        status = wait_done()
        assert status["stages"]["prepare"]["state"] == "done"
        assert status["stages"]["finetune"]["state"] == "skipped"

    def test_failure_halts_and_reports(self, env, monkeypatch):
        def boom(ctx):
            raise RuntimeError("disk full, allegedly")

        monkeypatch.setitem(runner.STAGE_FUNCS, "prepare", boom)
        start_run(dry_run=True)
        status = wait_done()
        assert status["state"] == "failed"
        assert "disk full" in status["error"]
        assert status["stages"]["prepare"]["state"] == "failed"
        # nothing after the failure should have started
        assert status["stages"]["pull_base"]["state"] == "pending"

    def test_second_start_while_running_raises_busy(self, env, monkeypatch):
        def slow(ctx):
            time.sleep(1.0)
            return "slept", {}

        monkeypatch.setitem(runner.STAGE_FUNCS, "ingest", slow)
        start_run(dry_run=True, stages=["ingest"])
        with pytest.raises(PipelineBusy):
            start_run(dry_run=True)
        wait_done()  # let it finish so the lock is free for the next test

    def test_run_ids_are_unique(self, env):
        ids = set()
        for _ in range(3):
            ids.add(start_run(dry_run=True, stages=["ingest"])["run_id"])
            wait_done()
        assert len(ids) == 3

    def test_invalid_stage_names_rejected_before_locking(self, env):
        with pytest.raises(ValueError, match="Unknown stage"):
            start_run(dry_run=True, stages=["compile_kernel"])
        # and the lock must not be stuck held after the rejection
        start_run(dry_run=True, stages=["ingest"])
        assert wait_done()["state"] == "done"


class TestSubsetRuns:
    """Re-running part of the pipeline has to land in the previous run's
    directory. A fresh empty directory guarantees a FileNotFoundError three
    stages later, which is a rotten way to find out."""

    def test_subset_without_a_previous_run_is_rejected(self, env):
        with pytest.raises(ValueError, match="is not one yet"):
            start_run(dry_run=True, stages=["evaluate"])

    def test_subset_reuses_the_previous_run_directory(self, env):
        first = start_run(dry_run=True)
        wait_done()
        again = start_run(dry_run=True, stages=["evaluate"])
        assert again["run_dir"] == first["run_dir"]
        assert wait_done()["state"] == "done"

    def test_missing_inputs_are_named_up_front(self, env):
        start_run(dry_run=True, stages=["ingest"])
        wait_done()
        # prepare ran? no. so finetune has neither train.jsonl nor profile.json
        with pytest.raises(ValueError, match="finetune needs"):
            start_run(dry_run=True, stages=["finetune"])

    def test_real_run_demands_an_adapter_before_deploying(self, env):
        start_run(dry_run=True)
        wait_done()
        # a dry run leaves a simulated adapter dir, so remove it and ask for real
        shutil.rmtree(_latest_run_dir() / "adapter")
        with pytest.raises(ValueError, match="export_deploy needs adapter"):
            start_run(dry_run=False, stages=["export_deploy"])


class TestRunnerRobustness:
    def test_a_crash_in_run_setup_does_not_wedge_the_lock(self, env, monkeypatch):
        """Regression: the lock used to be acquired in the caller and released
        in the worker's finally. Anything that blew up before that finally was
        reached left the app returning 409 forever."""
        import finetune_lab.pipeline.runner as r

        def boom(*a, **k):
            raise PermissionError("status.json is locked")

        monkeypatch.setattr(r, "get_store", boom)
        start_run(dry_run=True, stages=["ingest"])
        for _ in range(100):
            if not r.is_running():
                break
            time.sleep(0.05)
        monkeypatch.undo()
        assert r.is_running() is False, "lock leaked after a setup crash"
        start_run(dry_run=True, stages=["ingest"])  # must not raise PipelineBusy
        assert wait_done()["state"] == "done"

    def test_a_thread_that_wont_start_releases_the_flag(self, env, monkeypatch):
        import finetune_lab.pipeline.runner as r

        class DeadThread:
            def __init__(self, *a, **k):
                pass

            def start(self):
                raise RuntimeError("can't start new thread")

        monkeypatch.setattr(r.threading, "Thread", DeadThread)
        with pytest.raises(RuntimeError):
            start_run(dry_run=True, stages=["ingest"])
        monkeypatch.undo()
        assert r.is_running() is False

    def test_status_records_the_run_directory(self, env):
        info = start_run(dry_run=True)
        status = wait_done()
        assert status["run_dir"] == info["run_dir"]
