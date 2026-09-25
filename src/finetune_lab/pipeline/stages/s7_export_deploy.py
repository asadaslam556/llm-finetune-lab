"""Stage 7, export and deploy.

The finish line: adapter -> merged model -> GGUF -> `ollama create` -> a
chattable local model named after settings.deploy_name.

Dry mode writes the exact Modelfile plus DEPLOY_PLAN.md with the literal
commands real mode runs, so even a dry run leaves you a copy-pasteable path
to a deployed model.

The QLoRA detail that trips people up: you cannot merge an adapter into a
4-bit base. Merging means adding the adapter delta into the base weights, and
4-bit weights have nowhere near the precision to absorb it. Dequantizing
first gives you a model whose weights are 4-bit values in a 16-bit container,
which is worse than either option.

So the merge here deliberately reloads the base at full precision, adapter on
top, merge, then quantize once at the end with llama.cpp. Train in 4-bit,
merge in 16-bit, ship in GGUF. That ordering is the whole trick and it is why
this stage does not read the quantization plan at all.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from ...core.errors import StageError
from ..context import StageContext
from ..modeling import base_source, dtype_kwarg

MODELFILE_TEMPLATE = """FROM {gguf}
PARAMETER temperature 0.3
SYSTEM \"\"\"You are the support assistant for Nimbus, a note-taking app. Answer clearly and concisely using only Nimbus features.\"\"\"
"""

CONVERT_TIMEOUT_S = 3600


def _find_convert_script(ctx: StageContext) -> Path | None:
    """Locate llama.cpp's convert_hf_to_gguf.py.

    shutil.which alone is a bad bet: the file usually is not marked executable
    on Linux and macOS, and on Windows it is only found if .PY happens to be
    in PATHEXT. So check the explicit setting first, then PATH, then the
    handful of places people actually clone llama.cpp into.
    """
    configured = ctx.settings.gguf_convert_script.strip()
    if configured:
        p = Path(configured).expanduser()
        return p if p.is_file() else None

    on_path = shutil.which("convert_hf_to_gguf.py")
    if on_path:
        return Path(on_path)

    home = Path.home()
    for candidate in (
        home / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.cwd() / "llama.cpp" / "convert_hf_to_gguf.py",
        ctx.settings.artifacts_dir.parent / "llama.cpp" / "convert_hf_to_gguf.py",
    ):
        if candidate.is_file():
            return candidate
    return None


def _best_error_line(proc: subprocess.CompletedProcess) -> str:
    """Pick the one line worth showing out of a subprocess's output.

    Blindly taking the last few lines drags in traceback carets and file
    paths, which is how you end up with a rail that reads "^^^^^^^^ | File
    base.py line 1840". Python puts the actual reason on the last non-empty
    line, so prefer a line that names an error and fall back to that.
    """
    lines = [ln.strip() for ln in (proc.stderr or proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return "no output"
    for ln in reversed(lines):
        if ("Error" in ln or "error:" in ln) and not ln.startswith(("File ", "^")):
            return ln[:300]
    return lines[-1][:300]


def _run_tool(cmd: list[str], what: str) -> str:
    """Run an external command and, if it fails, put the actual reason in the
    error instead of "returned non-zero exit status 1". The tail of stderr is
    almost always the useful part, and it is what ends up on the rail."""
    try:
        # Explicit UTF-8: on Windows the default is the ANSI code page, and
        # ollama's progress spinner bytes crash the output reader with a
        # UnicodeDecodeError, which would also swallow any real error text.
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CONVERT_TIMEOUT_S,
        )
    except FileNotFoundError as e:
        raise StageError(f"{what}: could not run '{cmd[0]}'. Is it installed and on PATH?") from e
    except subprocess.TimeoutExpired as e:
        raise StageError(f"{what}: gave up after an hour.") from e
    if proc.returncode != 0:
        raise StageError(f"{what} failed (exit {proc.returncode}): {_best_error_line(proc)}")
    return proc.stdout


def _write_plan(ctx: StageContext, modelfile: str) -> None:
    s = ctx.settings
    plan = f"""# Deploy plan (written by the export stage)

Real mode runs these, in order:

1. Reload the base model at full precision and merge the adapter into it
   (done in-process with peft). The base is NOT loaded in 4-bit here, even
   though training used 4-bit. Merging into quantized weights loses the
   adapter.
2. Convert the merged model to GGUF. Set LFL_GGUF_CONVERT_SCRIPT to skip the
   search:
   {sys.executable} <llama.cpp>/convert_hf_to_gguf.py {ctx.path("merged")} --outfile {ctx.path("model.gguf")} --outtype {s.gguf_outtype}
3. Register it with Ollama:
   {s.ollama_bin} create {s.deploy_name} -f {ctx.path("Modelfile")}
4. Chat with it:
   ollama run {s.deploy_name}

The Modelfile below is already written next to this plan. You can hand-edit
it and re-run step 3 to change the system prompt without retraining anything.

```
{modelfile}```
"""
    ctx.path("DEPLOY_PLAN.md").write_text(plan, encoding="utf-8")


def _merge(ctx: StageContext, adapter_dir: Path, merged_dir: Path) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = base_source(ctx.settings)
    try:
        # dtype="auto" keeps the checkpoint's own precision, normally bf16 or
        # fp16. No quantization_config on purpose, see the module docstring.
        model = AutoModelForCausalLM.from_pretrained(src, **dtype_kwarg("auto"))
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.merge_and_unload().save_pretrained(merged_dir)
        AutoTokenizer.from_pretrained(src).save_pretrained(merged_dir)
    except Exception as e:  # peft and transformers both throw freely here
        raise StageError(
            f"Could not merge the adapter into the base model: {type(e).__name__}: {e}"
        ) from e


def run(ctx: StageContext) -> tuple[str, dict]:
    s = ctx.settings
    modelfile = MODELFILE_TEMPLATE.format(gguf=ctx.path("model.gguf"))
    ctx.path("Modelfile").write_text(modelfile, encoding="utf-8")
    _write_plan(ctx, modelfile)

    if ctx.dry_run:
        return (
            f"[dry] Wrote Modelfile and deploy plan for '{s.deploy_name}'. See DEPLOY_PLAN.md.",
            {"deploy_name": s.deploy_name, "modelfile": str(ctx.path("Modelfile"))},
        )

    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise StageError("Training extras missing. Run: pip install -e '.[train]'") from e

    adapter_dir = ctx.path("adapter")
    if not adapter_dir.is_dir():
        raise StageError(f"No adapter at {adapter_dir}. Run the fine-tune stage first.")

    merged_dir = ctx.path("merged")
    _merge(ctx, adapter_dir, merged_dir)

    convert = _find_convert_script(ctx)
    if convert is None:
        raise StageError(
            "Cannot find llama.cpp's convert_hf_to_gguf.py. Clone llama.cpp and point "
            "LFL_GGUF_CONVERT_SCRIPT at the file, or run the commands in DEPLOY_PLAN.md by "
            f"hand. The merged model is already saved at {merged_dir}."
        )
    # sys.executable, not "python": inside a venv, or on any box where only
    # python3 exists, "python" is either missing or the wrong interpreter.
    _run_tool(
        [
            sys.executable,
            str(convert),
            str(merged_dir),
            "--outfile",
            str(ctx.path("model.gguf")),
            "--outtype",
            s.gguf_outtype,
        ],
        "GGUF conversion",
    )
    if not ctx.path("model.gguf").exists():
        raise StageError(f"The converter finished but produced no {ctx.path('model.gguf')}.")

    if shutil.which(s.ollama_bin) is None:
        raise StageError(
            f"Ollama CLI not found (looked for '{s.ollama_bin}'). Install it, then run: "
            f"{s.ollama_bin} create {s.deploy_name} -f {ctx.path('Modelfile')}"
        )
    _run_tool(
        [s.ollama_bin, "create", s.deploy_name, "-f", str(ctx.path("Modelfile"))],
        f"ollama create {s.deploy_name}",
    )

    # This marker flips the Ollama provider's default model to the fine-tune.
    (s.artifacts_dir / "deployed.txt").write_text(s.deploy_name, encoding="utf-8")
    return (
        f"Deployed '{s.deploy_name}' to Ollama. Try it in the chat panel.",
        {"deploy_name": s.deploy_name, "gguf": str(ctx.path("model.gguf"))},
    )
