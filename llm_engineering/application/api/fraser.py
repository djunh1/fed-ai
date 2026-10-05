"""FRASER (St. Louis Fed) REST client.

FRASER is the secondary source. fed.gov holds the complete modern corpus;
FRASER's value is `location.textUrl`, which gives pre-extracted plain text on
1,032 of its 1,041 FOMC records -- no PDF parsing, no OCR. For the 1936-1967
minutes, where fed.gov serves scans of typewriter pages, that is the
difference between clean prose and whatever pypdf can recover.

Only `/api/*` is metered, at 30 requests/minute. The file hosts that serve
pdfUrl and textUrl are unauthenticated and unthrottled, which is why fetching
documents does not go through this client at all.
"""

import datetime
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import httpx
from loguru import logger

from llm_engineering.application.api.models import FedSourceDocument
from llm_engineering.domain.types import FedInfoType, SourceSystem
from llm_engineering.infrastructure.http import RateLimiter, get
from llm_engineering.settings import settings

#: The FOMC meeting series. 1,041 child records, 1933-07-20 to 2026-09-16.
FOMC_TITLE_ID = 677

#: The four MODS blocks parse_record() reads. Requesting all nine (the
#: default) returns several times the payload for data we discard.
RECORD_FIELDS = ("titleInfo", "originInfo", "location", "physicalDescription")

#: Filename fragment to document type, checked in order, most specific
#: first. Derived from all 28 filename shapes across the 1,041 records.
#:
#: Agendas (437 of them) and staff materials map to UNKNOWN so callers drop
#: them, matching how fed.gov treats the same documents. They are scaffolding
#: around the deliberation rather than a record of it.
_PDF_KINDS: tuple[tuple[str, FedInfoType], ...] = (
    ("agenda", FedInfoType.UNKNOWN),
    ("material", FedInfoType.UNKNOWN),
    ("forecast-errors", FedInfoType.UNKNOWN),
    ("taskforce", FedInfoType.UNKNOWN),
    # "memo" alone would swallow staff memos, which are not Memoranda of
    # Discussion. Only the fomcmod filename is the real series.
    ("fomcmod", FedInfoType.MEMORANDA_OF_DISCUSSION),
    ("moa", FedInfoType.MINUTES_OF_ACTIONS),  # fomcmoa and MoACC
    ("confcall", FedInfoType.TRANSCRIPT),
    ("transcript", FedInfoType.TRANSCRIPT),
    ("beige", FedInfoType.BEIGE_BOOK),
    ("bluebook", FedInfoType.BRIEFING_BOOK),
    ("greenbook", FedInfoType.BRIEFING_BOOK),
    ("gbspecial", FedInfoType.BRIEFING_BOOK),
    ("tealbook", FedInfoType.TEAL_BOOK),
    ("statement", FedInfoType.STATEMENT),
    ("longerrungoals", FedInfoType.STATEMENT),
    ("monetary", FedInfoType.STATEMENT),
    # Minutes variants last: "min" is a substring of several above, and all
    # of these map to MINUTES anyway, so their relative order cannot hurt.
    ("histmin", FedInfoType.MINUTES),
    ("minutes", FedInfoType.MINUTES),
    ("min", FedInfoType.MINUTES),
)


class FraserClient:
    """Authenticated, rate-limited access to the FRASER API.

    One limiter lives on the instance and is shared by every call it makes,
    so the 30/min budget is tracked per client rather than per request. Build
    one client and reuse it; building a fresh one per call hands each a full
    bucket and quietly defeats the throttle.
    """

    BASE = "https://fraser.stlouisfed.org/api"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: Path | None = None,
        rate_per_minute: float = 28.0,
    ) -> None:
        self._api_key = api_key or settings.FED_API_KEY
        if not self._api_key:
            raise RuntimeError(
                "FRASER requires an API key. Set FED_API_KEY in .env "
                "(free key from https://fraser.stlouisfed.org/developers)."
            )

        self._limiter = RateLimiter(rate_per_minute=rate_per_minute)
        self._cache_dir = cache_dir

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key}

    def _get(self, path: str, *, force: bool = False, **params) -> dict:
        """Every API call goes through here: key, throttle, envelope.

        Returns the raw response envelope --
        {format, page, limit, fields, total, records, start} -- because
        callers need `total` to paginate, not just `records`.

        `fields` is bang-delimited in this API, not comma-delimited, so a
        list is joined with "!" on the way out. Passing a comma-joined string
        silently returns every field instead of erroring.

        `force` is keyword-only and separated from **params so it cannot be
        mistaken for a query parameter: it governs the local cache, and
        sending it to FRASER would be meaningless.
        """
        url = f"{self.BASE}/{path.lstrip('/')}"
        query = {
            key: "!".join(value) if isinstance(value, (list, tuple)) else value
            for key, value in params.items()
            if value is not None
        }

        try:
            raw = get(
                url,
                params=query or None,
                headers=self.headers,
                cache_dir=self._cache_dir,
                force=force,
                limiter=self._limiter,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                raise RuntimeError(
                    f"FRASER rejected the API key (401) for {url}. "
                    "Check FED_API_KEY in .env is current."
                ) from exc
            raise

        return json.loads(raw)

    def iter_title_items(
        self,
        title_id: int = FOMC_TITLE_ID,
        *,
        limit: int = 100,
        fields: Sequence[str] | None = RECORD_FIELDS,
        force: bool = False,
    ) -> Iterator[dict]:
        """Stream every child record of a FRASER title, oldest first.

        Title 677 is the FOMC meeting series: 1,041 records, one per meeting,
        1933-07-20 to 2026-09-16. At limit=100 that is 11 calls, well inside
        the 30/min budget, so a full pass costs seconds rather than minutes.

        Pagination is driven by the envelope's `total` rather than by
        "stop when a page comes back short". A final page that happens to be
        exactly `limit` long is indistinguishable from a full one, so the
        short-page heuristic silently truncates the series.

        `force=True` bypasses the HTTP cache. Without it an incremental run
        replays the pages it saw last time and can never observe a new
        meeting -- the same trap fetch_press_index avoids on the fed.gov side.
        """
        page = 1
        seen = 0

        while True:
            envelope = self._get(
                f"title/{title_id}/items",
                force=force,
                page=page,
                limit=limit,
                fields=fields,
            )
            records = envelope.get("records") or []
            if not records:
                break

            yield from records

            seen += len(records)
            total = envelope.get("total")
            if total is None or seen >= total:
                break

            page += 1


def _first(value: object) -> str | None:
    """Unwrap FRASER's inconsistent list-or-scalar values.

    Within one record: `location` values are all lists, `originInfo.sortDate`
    is a bare string, and `titleInfo` is a list of dicts whose values are
    strings. Three conventions, so every read goes through here.
    """
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value) if value else None


def _page_count(extent: str | None) -> int | None:
    """Pages from `physicalDescription.extent`, but only when it says pages.

    912 of 1,041 records read "N files" -- the number of documents in that
    meeting's folder, not a page count. Reading the integer without checking
    the unit would put "10" in page_count for a meeting with ten documents.
    """
    if not extent:
        return None
    match = re.match(r"(\d+)\s+pages?\b", extent.strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _source_id(landing_url: str | None) -> str | None:
    """FRASER's item id, which trails the landing URL as `-22634`.

    `location.apiUrl` would be the obvious source, but the API emits it
    malformed -- `/api//22634`, missing the `/item` segment -- so the id is
    recovered from the one URL that is always well formed.
    """
    if not landing_url:
        return None
    match = re.search(r"-(\d+)/?$", landing_url)
    return match.group(1) if match else None


def classify_pdf_url(pdf_url: str) -> FedInfoType:
    """Map a FRASER document filename onto a document type.

    Checked in order, most specific first. Derived from all 28 filename
    shapes across the 1,041 records, not from documentation.
    """
    name = pdf_url.rsplit("/", 1)[-1].lower()
    for fragment, info_type in _PDF_KINDS:
        if fragment in name:
            return info_type
    return FedInfoType.UNKNOWN


def parse_record(record: dict) -> FedSourceDocument | None:
    """One FRASER API record to a FedSourceDocument.

    Pure -- no I/O, so it unit-tests against a saved sample.

    Returns None when the record carries no `pdfUrl`: 9 of the 1,041 have
    none, meaning there is no document to fetch. A landing page is not a
    document, so emitting one would put an unfetchable URL in the manifest.
    """
    location = record.get("location") or {}
    pdf_url = _first(location.get("pdfUrl"))
    if not pdf_url:
        return None

    sort_date = _first((record.get("originInfo") or {}).get("sortDate"))
    if not sort_date:
        return None
    try:
        meeting_date = datetime.date.fromisoformat(sort_date)
    except ValueError:
        logger.warning(f"Unparseable sortDate {sort_date!r} for {pdf_url}")
        return None

    titles = record.get("titleInfo") or [{}]
    title = (titles[0].get("title") or "").strip()

    return FedSourceDocument(
        source=SourceSystem.FRASER,
        info_type=classify_pdf_url(pdf_url),
        meeting_date=meeting_date,
        title=title or f"FOMC {meeting_date}",
        url=pdf_url,
        text_url=_first(location.get("textUrl")),
        source_id=_source_id(_first(location.get("url"))),
        media_type="application/pdf",
        page_count=_page_count((record.get("physicalDescription") or {}).get("extent")),
    )


def fetch_text(text_url: str, *, cache_dir: Path | None = None) -> str:
    """Fetch FRASER's pre-extracted plain text for one document.

    Deliberately a module function rather than a FraserClient method, and it
    does NOT go through `_get`. The file hosts (/files/text/, /docs/) are
    unauthenticated and unthrottled -- only `/api/*` counts against the
    30/min budget. Routing these through the client would send a pointless
    API key and spend quota the server is not metering.

    This is the whole reason FRASER is in the pipeline: every one of its
    1,032 documents has a text_url, so no PDF parsing is needed for any of
    them, including the 1930s typewriter scans.
    """
    raw = get(text_url, cache_dir=cache_dir)

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # latin-1 maps every byte, so it cannot fail. Worse than correct
        # decoding, but better than losing the document -- and loud, so a
        # file needing different handling is visible rather than silent.
        logger.warning(f"{text_url} is not valid UTF-8; falling back to latin-1")
        return raw.decode("latin-1")
