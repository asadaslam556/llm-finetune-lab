"""Real-mode logic that can be checked without torch, a GPU, or the network.

The heavy calls themselves only run on a GPU box, but the decisions around
them (what to train on, which error line to show, where llama.cpp lives,
what advice a failed download gets) are plain Python and belong in CI.
"""

import subprocess
import sys

import pytest

from finetune_lab.core.errors import StageError
from finetune_lab.pipeline.context import StageContext
from finetune_lab.pipeline.stages import s3_pull_base, s5_finetune, s7_export_deploy
from finetune_lab.pipeline.stages.s5_finetune import IGNORE_INDEX, build_example


class CharTokenizer:
    """One token per character, with a chat template shaped like Qwen's."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</s>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


CONVO = [
    {"role": "system", "content": "Be helpful."},
    {"role": "user", "content": "Pin?"},
    {"role": "assistant", "content": "Long-press it."},
]


class TestLossMasking:
    def test_only_the_reply_is_scored(self):
        ex = build_example(CharTokenizer(), CONVO, max_len=1000)
        scored = "".join(chr(t) for t in ex["labels"] if t != IGNORE_INDEX)
        assert scored == "Long-press it.</s>"

    def test_labels_line_up_with_inputs(self):
        ex = build_example(CharTokenizer(), CONVO, max_len=1000)
        assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
        for tok, label in zip(ex["input_ids"], ex["labels"], strict=True):
            assert label in (IGNORE_INDEX, tok)

    def test_truncation_keeps_lengths_consistent(self):
        ex = build_example(CharTokenizer(), CONVO, max_len=40)
        assert len(ex["input_ids"]) == len(ex["labels"]) == 40

    def test_prompt_longer_than_max_len_leaves_nothing_to_learn(self):
        """The training stage filters these out; a batch of them is a NaN loss."""
        ex = build_example(CharTokenizer(), CONVO, max_len=5)
        assert all(label == IGNORE_INDEX for label in ex["labels"])


@pytest.fixture
def real_ctx(settings, run_dir):
    return StageContext(settings=settings, run_dir=run_dir, dry_run=False)


class TestFinetuneReal:
    def test_missing_training_extras_is_a_readable_error(self, real_ctx, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_torch(name, *a, **k):
            if name.split(".")[0] in ("torch", "datasets", "transformers"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_torch)
        plan = s5_finetune.QuantPlan(
            strategy="lora",
            requested="lora",
            bits=None,
            quant_type=None,
            double_quant=False,
            compute_dtype="float32",
            gradient_checkpointing=False,
            optimizer="adamw_torch",
            reason="test",
        )
        with pytest.raises(StageError, match=r"pip install -e '\.\[train\]'"):
            s5_finetune._run_real(real_ctx, plan)


class TestPullBaseHints:
    @pytest.mark.parametrize(
        ("error", "has_token", "fragment"),
        [
            ("401 Client Error", False, "no token set"),
            ("GatedRepoError: 403", True, "accept them"),
            ("RepositoryNotFoundError", False, "does not resolve"),
            ("ConnectTimeout", False, "network problem"),
            ("something unheard of", False, ""),
        ],
    )
    def test_advice_matches_the_failure(self, error, has_token, fragment):
        hint = s3_pull_base._download_hint(error, "org/model", has_token)
        assert fragment in hint if fragment else hint == ""

    def test_real_pull_without_hub_library_says_what_to_install(self, real_ctx, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(StageError, match="huggingface_hub is not installed"):
            s3_pull_base.run(real_ctx)


class TestExportHelpers:
    def test_best_error_line_skips_traceback_noise(self):
        proc = subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr='Traceback\n  File "x.py", line 3\n    ^^^^\nValueError: bad tensor shape\n',
        )
        assert s7_export_deploy._best_error_line(proc) == "ValueError: bad tensor shape"

    def test_best_error_line_with_no_output(self):
        proc = subprocess.CompletedProcess([], 1, stdout="", stderr="")
        assert s7_export_deploy._best_error_line(proc) == "no output"

    def test_run_tool_reports_the_reason_not_just_the_exit_code(self):
        cmd = [sys.executable, "-c", "import sys; sys.exit('RuntimeError: converter says no')"]
        with pytest.raises(StageError, match="converter says no"):
            s7_export_deploy._run_tool(cmd, "GGUF conversion")

    def test_run_tool_missing_binary(self):
        with pytest.raises(StageError, match="Is it installed"):
            s7_export_deploy._run_tool(["definitely-not-a-real-binary-xyz"], "thing")

    def test_run_tool_returns_stdout(self):
        assert (
            s7_export_deploy._run_tool([sys.executable, "-c", "print('hi')"], "x").strip() == "hi"
        )

    def test_configured_convert_script_wins(self, real_ctx, tmp_path, monkeypatch):
        script = tmp_path / "convert_hf_to_gguf.py"
        script.write_text("", encoding="utf-8")
        monkeypatch.setattr(real_ctx.settings, "gguf_convert_script", str(script))
        assert s7_export_deploy._find_convert_script(real_ctx) == script

    def test_configured_but_missing_script_is_none_not_a_guess(
        self, real_ctx, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(real_ctx.settings, "gguf_convert_script", str(tmp_path / "nope.py"))
        assert s7_export_deploy._find_convert_script(real_ctx) is None

    def test_real_export_without_adapter_names_the_stage(self, real_ctx, monkeypatch):
        import types

        # peft and transformers only need to be importable for this path.
        monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))
        monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
        with pytest.raises(StageError, match="Run the fine-tune stage first"):
            s7_export_deploy.run(real_ctx)
