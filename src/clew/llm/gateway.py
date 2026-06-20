"""Concrete provider implementations behind the :mod:`clew.llm` protocols.

* :class:`GatewayLLMClient` — OpenAI-compatible client pointed at the Vercel AI
  Gateway. Model ids are provider-prefixed (``openai/gpt-5``, ``google/gemini-2.5-pro``)
  and chosen via config, so the provider is never hardcoded.
* :class:`LocalEmbeddingClient` — Qwen3-Embedding via sentence-transformers (CPU).
* :class:`GatewayEmbeddingClient` — embeddings through the gateway.

Use the factory helpers :func:`get_llm`, :func:`get_extractor`, :func:`get_embedder`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel

from clew.config import Settings, get_settings
from clew.llm.base import EmbeddingClient, LLMClient

T = TypeVar("T", bound=BaseModel)


class LLMNotConfiguredError(RuntimeError):
    """Raised when an LLM call is attempted without a gateway API key."""


class GatewayLLMClient(LLMClient):
    def __init__(self, model: str, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.settings.llm_api_key:
            raise LLMNotConfiguredError(
                "CLEW_LLM_API_KEY is not set; configure the Vercel AI Gateway key "
                "to enable LLM-based extraction/reasoning."
            )
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
        )
        return self._client

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def extract(self, system: str, user: str, schema: type[T], *, temperature: float = 0.0) -> T:
        """Structured extraction via JSON-schema response format."""
        client = self._ensure_client()
        schema_json = schema.model_json_schema()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": _strict_schema(schema_json),
                    "strict": True,
                },
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return schema.model_validate_json(content)


def _strict_schema(schema: dict) -> dict:
    """Make a Pydantic JSON schema acceptable for OpenAI strict mode.

    Strict mode requires ``additionalProperties: false`` on every object and all
    properties listed in ``required``.
    """

    def walk(node: dict) -> dict:
        if not isinstance(node, dict):
            return node
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for key in ("properties", "$defs", "definitions"):
            if key in node and isinstance(node[key], dict):
                for k, v in node[key].items():
                    node[key][k] = walk(v)
        for key in ("items", "anyOf", "allOf", "oneOf"):
            if key in node:
                if isinstance(node[key], list):
                    node[key] = [walk(v) for v in node[key]]
                else:
                    node[key] = walk(node[key])
        return node

    return walk(json.loads(json.dumps(schema)))


class LocalEmbeddingClient(EmbeddingClient):
    """Qwen3-Embedding (or any sentence-transformers model) on CPU."""

    def __init__(self, model_name: str, dim: int) -> None:
        self.model = model_name
        self.dim = dim
        self._st = None

    def _ensure_model(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.model)
        return self._st

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class GatewayEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str, dim: int, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model
        self.dim = dim
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.llm_base_url, api_key=self.settings.llm_api_key
            )
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()
        resp = client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


# --- factories -------------------------------------------------------------

@lru_cache
def get_llm() -> LLMClient:
    s = get_settings()
    return GatewayLLMClient(s.reasoning_model, s)


@lru_cache
def get_extractor() -> LLMClient:
    s = get_settings()
    return GatewayLLMClient(s.extraction_model, s)


@lru_cache
def get_embedder() -> EmbeddingClient:
    s = get_settings()
    spec = s.embedding_model
    if spec.startswith("local:"):
        return LocalEmbeddingClient(spec.split(":", 1)[1], s.embedding_dim)
    return GatewayEmbeddingClient(spec, s.embedding_dim, s)
