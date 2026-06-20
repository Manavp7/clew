"""SEC EDGAR connector for Schedule 13D / 13G beneficial-ownership filings.

For each filing we capture:

* the normalized cover-page + body text (``Filing.text()``) — downstream offsets
  index into this exact string;
* structured header metadata (filer + subject company with **CIK anchors**),
  which seeds high-precision entity resolution.

The CIK anchors are the backbone of ER: a filer/issuer with a known CIK resolves
deterministically rather than fuzzily.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

from clew.config import get_settings
from clew.ingest.base import Connector, RawDocument

# EDGAR form codes for the wedge.
FORM_MAP = {
    "13D": "SC 13D",
    "13G": "SC 13G",
    "SC 13D": "SC 13D",
    "SC 13G": "SC 13G",
}


def _company_dict(info) -> dict | None:
    if info is None:
        return None
    return {
        "name": getattr(info, "name", None),
        "cik": (str(getattr(info, "cik", "")).lstrip("0") or None),
        "irs_number": getattr(info, "irs_number", None),
        "sic": getattr(info, "sic", None),
        "state_of_incorporation": getattr(info, "state_of_incorporation", None),
    }


class EdgarConnector(Connector):
    source_name = "SEC EDGAR"

    def __init__(self, user_agent: str | None = None) -> None:
        self.user_agent = user_agent or get_settings().sec_user_agent

    def fetch(
        self,
        limit: int,
        *,
        form: str = "13D",
        year: int | None = None,
        quarter: int | None = None,
    ) -> Iterator[RawDocument]:
        from edgar import get_filings, set_identity

        set_identity(self.user_agent)
        form_code = FORM_MAP.get(form, form)

        kwargs: dict = {"form": form_code}
        if year is not None:
            kwargs["year"] = year
        if quarter is not None:
            kwargs["quarter"] = quarter
        filings = get_filings(**kwargs)

        emitted = 0
        for filing in filings:
            if emitted >= limit:
                break
            try:
                doc = self._to_raw(filing, form_code)
            except Exception as exc:  # noqa: BLE001 - skip unparseable filings, keep going
                print(f"  ! skipped {getattr(filing, 'accession_no', '?')}: {exc}")
                continue
            if doc is None:
                continue
            emitted += 1
            yield doc

    def _to_raw(self, filing, form_code: str) -> RawDocument | None:
        text = filing.text()
        if not text or len(text.strip()) < 50:
            return None

        filers: list[dict] = []
        subjects: list[dict] = []
        try:
            header = filing.header
            filers = [c for f in header.filers if (c := _company_dict(f.company_information))]
            subjects = [
                c
                for s in getattr(header, "subject_companies", [])
                if (c := _company_dict(s.company_information))
            ]
        except Exception:  # noqa: BLE001 - header parsing is best-effort
            pass

        filing_date = getattr(filing, "filing_date", None)
        meta = {
            "filers": filers,
            "subject_companies": subjects,
            "filing_date": str(filing_date) if filing_date else None,
            "company": getattr(filing, "company", None),
            "cik": getattr(filing, "cik", None),
        }
        return RawDocument(
            external_id=filing.accession_no,
            doc_type=form_code,
            url=getattr(filing, "filing_url", None),
            retrieved_at=datetime.now(UTC),
            text=text,
            meta=meta,
        )
