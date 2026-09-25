"""Shared model-loading helpers.

The training, evaluation and export stages all need to answer the same two
questions: where do the base weights live, and what keyword does this version
of transformers want for dtype. Doing it in one place keeps the three stages
from drifting apart.

Everything here imports torch and transformers lazily. The API server runs
without the training extras and should stay that way.
"""

from __future__ import annotations

from ..core.config import Settings


def base_source(settings: Settings) -> str:
    """Local snapshot if the pull stage has run, otherwise the hub id.

    Falling back to the hub id means a run started at the training stage still
    works; transformers will just download on demand.
    """
    local = settings.local_base_dir()
    return str(local) if local.exists() else settings.base_model_hf


def wants_dtype_keyword(version: str) -> bool:
    """transformers 4.56 renamed `torch_dtype` to `dtype` and warns on the old
    spelling from then on."""
    major, minor = (int(p) for p in version.split(".")[:2])
    return (major, minor) >= (4, 56)


def dtype_kwarg(dtype) -> dict:
    """The dtype keyword the installed transformers actually wants."""
    import transformers

    key = "dtype" if wants_dtype_keyword(transformers.__version__) else "torch_dtype"
    return {key: dtype}


def load_tokenizer(settings: Settings):
    """Tokenizer with a pad token guaranteed.

    Most causal LMs ship without one, and the collator needs it. Reusing EOS
    is the conventional fix and costs nothing at this scale.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_source(settings))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
