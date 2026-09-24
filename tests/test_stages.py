"""The seven stages, exercised in dry mode against the real seed dataset."""

import json

import pytest

from finetune_lab.core.errors import StageError
from finetune_lab.pipeline.context import StageContext
from finetune_lab.pipeline.stages import (
    s1_ingest,
    s2_prepare,
    s3_pull_base,
    s4_profile,
    s5_finetune,
    s6_evaluate,
    s7_export_deploy,
)
from finetune_lab.pipeline.stages.s6_evaluate import keyword_hit, overlap_f1


@pytest.fixture
def ctx(settings, run_dir):
    return StageContext(settings=settings, run_dir=run_dir, dry_run=True)


def run_through(ctx, *stages):
    out = None
    for stage in stages:
        out = stage.run(ctx)
    return out


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestIngest:
    def test_drops_malformed_and_incomplete_rows(self, ctx):
        _, m = s1_ingest.run(ctx)
        # seed data ships 140 lines: 137 valid, 1 non-JSON, 2 missing fields
        assert m["rows_seen"] == 140
        assert m["rows_kept"] == 137
        assert m["rows_dropped"] == 3

    def test_writes_clean_jsonl(self, ctx):
        s1_ingest.run(ctx)
        rows = ctx.read_jsonl("ingested.jsonl")
        assert all(r["question"].strip() and r["answer"].strip() for r in rows)

    def test_empty_raw_dir_fails_loudly(self, ctx):
        for f in (ctx.settings.data_dir / "raw").glob("*.jsonl"):
            f.unlink()
        with pytest.raises(StageError, match=r"No \.jsonl files"):
            s1_ingest.run(ctx)


class TestPrepare:
    def test_dedupes_and_drops_short_answers(self, ctx):
        _, m = run_through(ctx, s1_ingest, s2_prepare)
        assert m["duplicates_dropped"] == 8  # baked into the seed generator
        assert m["too_short_dropped"] == 5
        assert m["train_rows"] + m["val_rows"] == 137 - 8 - 5
        assert m["val_rows"] >= 10  # enough for the eval numbers to mean something

    def test_chat_format_shape(self, ctx):
        run_through(ctx, s1_ingest, s2_prepare)
        row = ctx.read_jsonl("train.jsonl")[0]
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert "Nimbus" in row["messages"][0]["content"]

    def test_split_is_deterministic_given_seed(self, ctx, settings, tmp_path):
        run_through(ctx, s1_ingest, s2_prepare)
        first = ctx.path("val.jsonl").read_text(encoding="utf-8")
        ctx2 = StageContext(settings=settings, run_dir=tmp_path / "again", dry_run=True)
        ctx2.run_dir.mkdir()
        run_through(ctx2, s1_ingest, s2_prepare)
        assert ctx2.path("val.jsonl").read_text(encoding="utf-8") == first


class TestPullBaseDry:
    def test_reports_without_network(self, ctx):
        msg, m = s3_pull_base.run(ctx)
        assert m["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert m["already_cached"] is False
        assert "[dry]" in msg


class TestProfile:
    def test_writes_plan_with_required_keys(self, ctx):
        run_through(ctx, s1_ingest, s2_prepare)
        _, metrics = s4_profile.run(ctx)
        report = read_json(ctx.path("profile.json"))
        for key in ("device", "plan", "per_device_batch", "grad_accum", "estimated_steps"):
            assert key in report
        assert report["plan"]["strategy"] in ("qlora", "lora")
        assert metrics["strategy"] == report["plan"]["strategy"]

    def test_plan_always_carries_a_reason(self, ctx):
        """A downgrade to LoRA must never be silent. The reason lands on the rail."""
        run_through(ctx, s1_ingest, s2_prepare)
        _, metrics = s4_profile.run(ctx)
        assert metrics["reason"]
        if metrics["downgraded"]:
            assert metrics["strategy"] == "lora"

    def test_no_torch_means_cpu_and_a_lora_downgrade(self, ctx, monkeypatch):
        # torch may well be installed here, but the probe must survive without
        # it, and a box with no torch can obviously not do 4-bit.
        import builtins

        real_import = builtins.__import__

        def no_torch(name, *a, **k):
            if name in ("torch", "bitsandbytes"):
                raise ImportError("nope")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_torch)
        run_through(ctx, s1_ingest, s2_prepare)
        _, metrics = s4_profile.run(ctx)
        report = read_json(ctx.path("profile.json"))
        assert metrics["device"] == "cpu"
        assert metrics["strategy"] == "lora"
        assert metrics["downgraded"] is True
        assert report["hardware"]["torch"] is None


class TestFinetuneDry:
    def test_simulated_run_writes_adapter_info(self, ctx):
        run_through(ctx, s1_ingest, s2_prepare, s4_profile)
        _, m = s5_finetune.run(ctx)
        assert m["mode"] == "simulated"
        info = read_json(ctx.path("adapter") / "ADAPTER_INFO.json")
        assert info["mode"] == "simulated"
        assert info["config"]["lora"]["r"] == ctx.settings.lora_r
        assert len(info["loss_curve"]) == info["steps"]

    def test_adapter_info_records_the_strategy(self, ctx):
        """The saved config has to be enough to reproduce the run, which means
        recording whether the base was quantized and how."""
        run_through(ctx, s1_ingest, s2_prepare, s4_profile)
        s5_finetune.run(ctx)
        cfg = read_json(ctx.path("adapter") / "ADAPTER_INFO.json")["config"]
        assert cfg["strategy"] in ("qlora", "lora")
        if cfg["strategy"] == "qlora":
            assert cfg["quantization"]["bits"] == 4
            assert cfg["quantization"]["quant_type"] == "nf4"
        else:
            assert cfg["quantization"] is None

    def test_targets_the_mlp_projections_too(self, ctx):
        """The QLoRA paper's practical finding: covering every linear layer
        matters more than rank. Dropping the MLP targets loses most of it."""
        run_through(ctx, s1_ingest, s2_prepare, s4_profile)
        s5_finetune.run(ctx)
        targets = read_json(ctx.path("adapter") / "ADAPTER_INFO.json")["config"]["lora"][
            "target_modules"
        ]
        assert {"q_proj", "k_proj", "v_proj", "o_proj"} <= set(targets)
        assert {"gate_proj", "up_proj", "down_proj"} <= set(targets)

    def test_loss_curve_trends_down(self, ctx):
        run_through(ctx, s1_ingest, s2_prepare, s4_profile)
        s5_finetune.run(ctx)
        curve = read_json(ctx.path("adapter") / "ADAPTER_INFO.json")["loss_curve"]
        assert curve[0] > curve[-1]  # it is fake, but it should at least look sane

    def test_missing_profile_names_the_stage_to_run(self, ctx):
        run_through(ctx, s1_ingest, s2_prepare)
        with pytest.raises(StageError, match="profile"):
            s5_finetune.run(ctx)


class TestEvaluate:
    def test_metric_functions(self):
        assert overlap_f1("pin the note to top", "pin the note to top") == 1.0
        assert overlap_f1("completely unrelated words", "pin the note") == 0.0
        assert keyword_hit("Long-press and choose Pin to top", "Pin to top of the list") > 0

    def test_dry_eval_writes_labeled_report(self, ctx):
        run_through(ctx, s1_ingest, s2_prepare, s4_profile, s5_finetune)
        _, m = s6_evaluate.run(ctx)
        report = read_json(ctx.path("eval_report.json"))
        assert "simulated" in report["mode"]  # a fake score must say it is fake
        assert 0 <= report["overlap_f1"] <= 1
        assert len(report["examples"]) == min(5, report["n_eval"])
        assert m["n_eval"] == report["n_eval"]


class TestExportDeployDry:
    def test_writes_modelfile_and_plan(self, ctx):
        _, m = s7_export_deploy.run(ctx)
        modelfile = ctx.path("Modelfile").read_text(encoding="utf-8")
        assert modelfile.startswith("FROM ")
        assert "Nimbus" in modelfile
        plan = ctx.path("DEPLOY_PLAN.md").read_text(encoding="utf-8")
        assert "ollama create nimbus-support" in plan
        assert m["deploy_name"] == "nimbus-support"

    def test_plan_warns_against_merging_into_quantized_weights(self, ctx):
        """The single easiest QLoRA mistake. The plan has to say so, because
        the plan is what people copy-paste when the automated path fails."""
        s7_export_deploy.run(ctx)
        plan = ctx.path("DEPLOY_PLAN.md").read_text(encoding="utf-8")
        assert "NOT loaded in 4-bit" in plan

    def test_dry_mode_does_not_write_deploy_marker(self, ctx, settings):
        s7_export_deploy.run(ctx)
        assert not (settings.artifacts_dir / "deployed.txt").exists()
