"""Source connector interface.

A connector knows how to fetch raw documents from one source and yield them in a
normalized shape. Persisting to the ledger (dedupe, raw storage) is handled once,
in :func:`clew.ingest.service.ingest_documents`, so every connector behaves the
same.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class RawDocument:
    external_id: str  # source-native id (e.g. EDGAR accession number)
    doc_type: str  # e.g. "SC 13D"
    url: str | None
    retrieved_at: datetime
    text: str  # normalized text; downstream char offsets index into this
    meta: dict = field(default_factory=dict)

    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class Connector(ABC):
    source_name: str

    @abstractmethod
    def fetch(self, limit: int) -> Iterator[RawDocument]:
        """Yield up to ``limit`` raw documents from the source."""
        raise NotImplementedError
