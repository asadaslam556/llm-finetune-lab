"""App configuration.

Every tunable lives here and can be overridden with an LFL_* env var or a
.env file. Defaults are picked so a dry run works on any laptop with zero
setup. Only a real training run needs a GPU and the training extras.

Provider credentials are a special case: they also read the vendor's own
standard variable names (ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN and
friends). If you already have those exported for the official SDKs, this
project picks them up without you configuring anything twice.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from dotenv import dotenv_values
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# config.py -> core -> finetune_lab -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

Strategy = Literal["qlora", "lora"]


def _env(*names: str) -> str:
    """First non-empty value among these names, from the shell first and then
    the .env files. pydantic-settings only reads LFL_* names out of .env, so
    without the second half ANTHROPIC_API_KEY in a .env was silently ignored.
    """
    files = Settings.model_config.get("env_file") or ()
    if isinstance(files, (str, Path)):
        files = (files,)
    sources = [os.environ, *(dotenv_values(f) for f in files if Path(f).is_file())]
    for src in sources:
        for n in names:
            v = (src.get(n) or "").strip()
            if v:
                return v
    return ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LFL_",
        # Look in the working directory and the repo root, so it works whether
        # you launch uvicorn from the top level or from somewhere inside.
        env_file=(".env", str(REPO_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------- server
    # NoDecode stops pydantic-settings from JSON-parsing this before our
    # validator sees it, which is what lets the comma form work at all.
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    # Host headers the API answers to. Blocks DNS rebinding: a web page that
    # re-points its own domain at 127.0.0.1 still sends its own Host header,
    # so it cannot drive this API (or spend your provider credit) from a tab.
    allowed_hosts: Annotated[list[str], NoDecode] = ["localhost", "127.0.0.1"]
    request_timeout_s: float = 60.0
    max_output_tokens: int = 1024

    # ------------------------------------------------------------- providers
    # ollama | anthropic | openai | deepseek | custom | mock
    default_provider: str = "ollama"

    ollama_host: str = "http://localhost:11434"
    ollama_bin: str = "ollama"

    # Anthropic. Leave the base URL alone for api.anthropic.com, or point it
    # at a gateway (corporate proxy, LiteLLM, Bedrock shim) and everything
    # else stays the same.
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    anthropic_model: str = ""

    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = ""

    deepseek_api_key: str = ""
    deepseek_base_url: str = ""
    deepseek_model: str = ""

    # Escape hatch for any other OpenAI-compatible endpoint: Together, Groq,
    # OpenRouter, a local vLLM, whatever. Set three vars, get a provider.
    custom_api_key: str = ""
    custom_base_url: str = ""
    custom_model: str = ""
    custom_label: str = "custom OpenAI-compatible endpoint"

    # Only needed for gated or private HF repos. Public models download
    # without it. Falls back to HF_TOKEN, which `hf auth login` already writes.
    hf_token: str = ""

    # ---------------------------------------------------------------- models
    # The base model we fine-tune. 0.5B keeps a full run on a normal laptop.
    # QLoRA earns its keep at 7B and up, see docs/qlora.md for the numbers.
    base_model_hf: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Tag Ollama already knows for the same base, for A/B chat before deploy.
    ollama_base_tag: str = "qwen2.5:0.5b-instruct"
    # What the fine-tune gets called inside Ollama after the last stage.
    deploy_name: str = "nimbus-support"

    # ----------------------------------------------------------------- paths
    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"

    # ---------------------------------------------------------- quantization
    # "qlora" loads the base in 4-bit and trains adapters on top. "lora" keeps
    # the base in bf16/fp32. QLoRA is the default; it falls back to LoRA with
    # a logged reason when the machine cannot do 4-bit (no CUDA, no
    # bitsandbytes). See pipeline/quantization.py for that decision.
    finetune_strategy: Strategy = "qlora"
    quant_bits: Literal[4, 8] = 4
    quant_type: Literal["nf4", "fp4"] = "nf4"
    # Quantizes the quantization constants too. Roughly 0.4 bits per parameter
    # back for no measurable quality cost, so it is on by default.
    double_quant: bool = True
    # auto picks bfloat16 on Ampere and newer, float16 below that.
    compute_dtype: Literal["auto", "bfloat16", "float16"] = "auto"
    # Trades compute for memory by recomputing activations in the backward
    # pass. Worth it on anything that is not tiny.
    gradient_checkpointing: bool = True

    # ------------------------------------------------------------ LoRA knobs
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    # The QLoRA paper's main practical finding: covering every linear layer
    # matters more than a big rank. The MLP projections are in here for that
    # reason, and dropping them is how you quietly lose most of the benefit.
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # --------------------------------------------------------------- trainer
    learning_rate: float = 2e-4
    num_epochs: int = 1
    batch_size: int = 4
    max_seq_len: int = 512
    seed: int = 42
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    # Paged optimizers spill optimizer state to host RAM instead of dying on a
    # VRAM spike. Needs bitsandbytes, so the trainer downgrades to plain
    # adamw_torch when it is not installed.
    optimizer: str = "paged_adamw_8bit"

    # ------------------------------------------------------------- data prep
    val_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    min_answer_chars: int = 20  # tickets with one-word answers teach nothing

    # ---------------------------------------------------- export (last stage)
    gguf_convert_script: str = ""
    gguf_outtype: str = "q8_0"

    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def _split_origins(cls, v):
        # Accept a comma-separated string as well as a JSON list, because
        # everyone types the comma version into a .env first.
        if isinstance(v, str) and not v.strip().startswith("["):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    # -------------------------------------------------------------- resolvers
    # These exist so the vendor's own env var names keep working. Our config
    # stays namespaced, but nobody has to set the same secret twice.

    def resolved_hf_token(self) -> str | None:
        return self.hf_token or _env("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") or None

    def anthropic(self) -> tuple[str, str, str]:
        """Returns (api_key, base_url, model)."""
        base = self.anthropic_base_url or _env("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        return (
            self.anthropic_api_key or _env("ANTHROPIC_API_KEY"),
            base.rstrip("/"),
            self.anthropic_model or _env("ANTHROPIC_MODEL") or "claude-sonnet-5",
        )

    def openai(self) -> tuple[str, str, str]:
        base = self.openai_base_url or _env("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        return (
            self.openai_api_key or _env("OPENAI_API_KEY"),
            base.rstrip("/"),
            self.openai_model or _env("OPENAI_MODEL") or "gpt-4o-mini",
        )

    def deepseek(self) -> tuple[str, str, str]:
        base = self.deepseek_base_url or _env("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        return (
            self.deepseek_api_key or _env("DEEPSEEK_API_KEY"),
            base.rstrip("/"),
            self.deepseek_model or _env("DEEPSEEK_MODEL") or "deepseek-chat",
        )

    def custom(self) -> tuple[str, str, str]:
        return (
            self.custom_api_key or _env("CUSTOM_API_KEY"),
            (self.custom_base_url or _env("CUSTOM_BASE_URL")).rstrip("/"),
            self.custom_model or _env("CUSTOM_MODEL"),
        )

    def local_base_dir(self) -> Path:
        """Where the pull stage drops the base model snapshot."""
        return self.artifacts_dir / "models" / self.base_model_hf.replace("/", "__")

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
