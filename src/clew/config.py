"""Environment-driven configuration.

All settings load from environment variables (prefix ``CLEW_``) or a local
``.env`` file. No provider, model, or credential is hardcoded anywhere else in
the codebase — everything funnels through this module and the ``llm`` provider
abstraction.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLEW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database (the only source of truth) ---
    database_url: str = Field(
        default="postgresql+psycopg://clew:clew@localhost:5432/clew",
        description="Write-capable DSN used by extraction pipelines.",
    )
    database_url_ro: str | None = Field(
        default=None,
        description="Optional read-only DSN for the reasoning/API layer.",
    )

    # --- LLM / embeddings provider abstraction (Vercel AI Gateway) ---
    llm_base_url: str = "https://ai-gateway.vercel.sh/v1"
    llm_api_key: str | None = None
    extraction_model: str = "openai/gpt-5"
    reasoning_model: str = "openai/gpt-5"
    embedding_model: str = "local:Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024

    # Vector backend: "pgvector" (default; lives in Postgres) or "qdrant"
    # (embedded, on-disk via qdrant-client — no server, works without Docker).
    vector_backend: str = "pgvector"
    qdrant_path: str = "data/qdrant"

    # ER backend: "default" (CIK + Jaro-Winkler union-find) or "splink".
    # Kept as "default" because it beats Splink on the gold at current data volume
    # (see `clew eval er-compare`); flip to "splink" once EM training wins.
    er_backend: str = "default"

    # --- SEC EDGAR ---
    sec_user_agent: str = "clew-research research@example.com"

    # --- Paths ---
    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def has_llm(self) -> bool:
        """True when a gateway API key is configured for live extraction."""
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
