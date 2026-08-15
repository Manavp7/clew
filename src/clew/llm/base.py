"""Provider-agnostic LLM + embedding interfaces.

No concrete provider is referenced outside :mod:`clew.llm`. The rest of the
codebase depends only on these protocols, so providers/models can be swapped via
configuration (Vercel AI Gateway today, anything OpenAI-compatible tomorrow)
without refactoring.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """Text + structured generation. Implementations must be read-only w.r.t. the ledger."""

    model: str

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        """Return a plain-text completion."""
        ...

    def extract(
        self, system: str, user: str, schema: type[T], *, temperature: float = 0.0
    ) -> T:
        """Return a structured object validated against ``schema``."""
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...
