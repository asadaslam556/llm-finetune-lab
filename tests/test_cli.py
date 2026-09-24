"""The finetune-lab command line, driven through main() the way a shell would."""

import json

import pytest

import finetune_lab.cli as cli
import finetune_lab.pipeline.runner as runner
from finetune_lab.pipeline.status import STAGE_LABELS


def test_dry_run_prints_every_stage_and_exits_zero(env, capsys):
    assert cli.main(["run"]) == 0
    out = capsys.readouterr().out
    assert "(dry)" in out
    for label in STAGE_LABELS.values():
        assert label in out
    assert "Run finished." in out


def test_second_run_reports_its_own_results_not_the_previous_ones(env, capsys, monkeypatch):
    """Regression: the CLI polled status.json before the new run had written
    to it, printed the previous run's stages, and then skipped the real ones."""
    assert cli.main(["run"]) == 0
    capsys.readouterr()

    def fails(ctx):
        raise RuntimeError("second run broke on purpose")

    monkeypatch.setitem(runner.STAGE_FUNCS, "ingest", fails)
    assert cli.main(["run"]) == 1
    captured = capsys.readouterr()
    assert "[FAIL] Ingest data" in captured.out
    assert "second run broke on purpose" in captured.err
    assert "[ok  ]" not in captured.out


def test_subset_run(env, capsys):
    assert cli.main(["run", "--stages", "ingest", "prepare"]) == 0
    out = capsys.readouterr().out
    assert "Prepare & format" in out
    assert "Profile & plan" not in out


def test_unknown_stage_is_rejected_by_argparse(env):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--stages", "compile_kernel"])
    assert e.value.code == 2


def test_start_failure_is_one_line_and_exit_two(env, capsys):
    # A subset with no earlier run to continue from.
    assert cli.main(["run", "--stages", "evaluate"]) == 2
    assert "Could not start" in capsys.readouterr().err


def test_plan_prints_json(env, capsys):
    assert cli.main(["plan"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert body["plan"]["strategy"] in ("qlora", "lora")
    assert body["plan"]["reason"]
    assert body["vram_estimate"]["params_b"] == 0.5


def test_a_command_is_required(env):
    with pytest.raises(SystemExit):
        cli.main([])
