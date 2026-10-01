"""federalreserve.gov index enumeration, URL classification and index parsing.

The Fed publishes one index page per year at a predictable URL, so the corpus
is enumerable rather than crawlable: no link-following, no rendered-page
scraping. Everything here is pure -- the caller supplies the bytes.

Rule patterns were derived from all 86 cached index pages (60 distinct
date-bearing URL shapes), not from documentation.
"""

import datetime
import html
import io
import json
from pathlib import Path
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import httpx
from loguru import logger
import pypdf

from llm_engineering.application.api.models import FedSourceDocument
from llm_engineering.domain.types import FedInfoType, SourceSystem
from llm_engineering.infrastructure.http import get

_CHROME_TAGS = ["script", "style", "nav", "header", "footer", "form", "noscript"]
_BLANK_RUN = re.compile(r"\n{3,}")

BASE = "https://www.federalreserve.gov"

FIRST_HISTORICAL_YEAR = 1936
LAST_HISTORICAL_YEAR = 2020

CALENDARS_URL = f"{BASE}/monetarypolicy/fomccalendars.htm"

# Trailing "(10.12 MB PDF)" / "611 KB PDF" noise in link text.
_SIZE_SUFFIX = re.compile(r"\s*\(?\s*[\d.]+\s*[KMG]B\s+PDF\s*\)?\s*$", re.IGNORECASE)

# A run of 8 digits can parse as a perfectly valid date and still be nonsense:
# FOMChminec119401101.pdf is a part number followed by 19401101, and reading
# the first 8 gives the year 1194. Anything outside these bounds means the
# pattern latched onto the wrong digits.
EARLIEST_PLAUSIBLE_YEAR = 1913  # the Federal Reserve Act
LATEST_PLAUSIBLE_YEAR = datetime.date.today().year + 3  # calendars list future meetings

PRESS_INDEX_URL = f"{BASE}/json/ne-press.json"

# The two title forms the Fed has used for an FOMC statement release.
_STATEMENT_TITLES = frozenset({"fomc statement", "federal reserve issues fomc statement"})

# Both shapes seen in the feed's `d` field.
_PRESS_DATE_FORMATS = ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y")

def _parse_date(text: str, date_format: str) -> datetime.date | None:
    """Parse a captured date, rejecting anything outside the plausible range."""
    try:
        parsed = datetime.datetime.strptime(text, date_format).date()
    except ValueError:
        return None
    if not EARLIEST_PLAUSIBLE_YEAR <= parsed.year <= LATEST_PLAUSIBLE_YEAR:
        return None
    return parsed

def _press_date(text: str) -> datetime.date | None:
    for date_format in _PRESS_DATE_FORMATS:
        try:
            return datetime.datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    return None


@dataclass(frozen=True, slots=True)
class UrlRule:
    """
    `pattern` must capture the meeting date in group 1. That single constraint
    is what makes site chrome unmatchable: nav links carry no date, so they
    cannot match any rule. The false positives a substring table produces are
    impossible by construction rather than filtered out afterwards.
    """

    pattern: re.Pattern[str]
    info_type: FedInfoType
    date_format: str = "%Y%m%d"
    since: int | None = None  # earliest plausible year; validation, not filtering
    note: str = ""


@dataclass(frozen=True, slots=True)
class Classified:
    """A URL recognised as a document, with the date its pattern captured."""

    info_type: FedInfoType
    meeting_date: datetime.date
    url: str
    rule: UrlRule | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class ExtractedText:
    """Plain text plus enough detail to spot a silent extraction failure."""

    text: str
    page_chars: tuple[int, ...] = ()    # per PDF page; empty for HTML
    failed_pages: tuple[int, ...] = ()  # 0-based indices pypdf could not read

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

_RULES: tuple[UrlRule, ...] = (
    # --- Minutes, across the four URL schemes the Fed has used -------------
    UrlRule(re.compile(r"/fomcminutes(\d{8})"), FedInfoType.MINUTES, since=2007),
    UrlRule(re.compile(r"/fomc/minutes/\d{4}/(\d{8})min\.htm"), FedInfoType.MINUTES, since=1993),
    UrlRule(re.compile(r"/fomc/minutes/(\d{8})\.htm"), FedInfoType.MINUTES, since=1996),
    UrlRule(re.compile(r"/fomchistmin(\d{8})"), FedInfoType.MINUTES, note="pre-1993 scans"),
    # hminec filenames may carry a part number before the date
    # (FOMChminec119401101.pdf = part 1, 1940-11-01), so anchor on .pdf and
    # let \d? absorb it -- otherwise the first 8 digits read as the year 1194.
    UrlRule(
        re.compile(r"/fomchminec\d?(\d{8})\.pdf"),
        FedInfoType.MINUTES,
        note="executive committee",
    ),
    # --- Statements --------------------------------------------------------
    # The /fomc/ prefix is load-bearing: default.htm is the Fed's index
    # filename sitewide, so an unanchored rule swallows ~20 nav links a page.
    UrlRule(re.compile(r"/fomc/(\d{8})default\.htm"), FedInfoType.STATEMENT, since=1994),
    UrlRule(re.compile(r"/pressreleases/monetary(\d{8})[a-z]"), FedInfoType.STATEMENT, since=2006),
    UrlRule(
        re.compile(r"/press/monetary/(\d{8})[a-z]\.htm"),
        FedInfoType.STATEMENT,
        since=2006,
        note="pre-2011 press path",
    ),
    # --- Transcripts -------------------------------------------------------
    UrlRule(re.compile(r"/fomc(\d{8})meeting\.pdf"), FedInfoType.TRANSCRIPT, since=1976),
    UrlRule(
        re.compile(r"/fomc(\d{8})confcall\.pdf"),
        FedInfoType.TRANSCRIPT,
        note="conference calls",
    ),
    # --- Deliberation records ----------------------------------------------
    # mod = Memoranda of Discussion, moa = Minutes of Actions. Different
    # documents; an earlier version of these rules conflated them.
    UrlRule(re.compile(r"/fomcmod(\d{8})"), FedInfoType.MEMORANDA_OF_DISCUSSION),
    UrlRule(re.compile(r"/fomcmoa(\d{8})"), FedInfoType.MINUTES_OF_ACTIONS),
    UrlRule(re.compile(r"/fomcropa(\d{8})"), FedInfoType.RECORD_OF_POLICY_ACTIONS),
    # --- Briefing books ----------------------------------------------------
    # Tealbook A and B replaced Greenbook + Bluebook in 2010.
    UrlRule(re.compile(r"/fomc(\d{8})tealbook[ab]"), FedInfoType.TEAL_BOOK, since=2010),
    UrlRule(
        re.compile(r"/fomc(\d{8})(?:greenbook|bluebook|gbpt|gbsup)"),
        FedInfoType.BRIEFING_BOOK,
    ),
    # --- Beige Books (called Redbook before 1983) --------------------------
    UrlRule(re.compile(r"/fomc(\d{8})(?:beige|redbook)"), FedInfoType.BEIGE_BOOK),
    # 1996-2010 Beige Books live in a dated directory; both the HTML landing
    # page and fullreport{date}.pdf inside it match this one prefix.
    UrlRule(re.compile(r"/fomc/beigebook/\d{4}/(\d{8})/"), FedInfoType.BEIGE_BOOK),
    UrlRule(re.compile(r"/beigebook_?(\d{6})"), FedInfoType.BEIGE_BOOK, date_format="%Y%m"),
    # --- Press conferences --------------------------------------------------
    UrlRule(re.compile(r"/fomcpresconf(\d{8})"), FedInfoType.PRESS_CONFERENCE, since=2011),
)


def iter_index_urls() -> Iterator[str]:
    """Yield every FOMC index page URL, oldest first."""
    for year in range(FIRST_HISTORICAL_YEAR, LAST_HISTORICAL_YEAR + 1):
        yield f"{BASE}/monetarypolicy/fomchistorical{year}.htm"
    yield CALENDARS_URL


def classify_url(url: str) -> Classified | None:
    """Identify a document URL and extract its meeting date.

      * Classified with info_type -- a recognised document.
      * Classified with UNKNOWN          -- carries a date but matched no rule.
                                            A gap in _RULES; this is how the
                                            missing Tealbooks surfaced.
      * None                             -- no date at all. Site chrome.

    """
    # Match against a lowercased copy: the Fed's casing is inconsistent across
    # eras (/fomc/MINUTES/, FOMC19950201meeting.pdf). Every rule pattern is
    # therefore written in lowercase.
    lowered = url.lower()

    for rule in _RULES:
        match = rule.pattern.search(lowered)
        if match is None:
            continue
        meeting_date = _parse_date(match.group(1), rule.date_format)
        if meeting_date is None:
            continue
        return Classified(rule.info_type, meeting_date, url, rule)

    fallback = re.search(r"(\d{8})", lowered)
    if fallback is None:
        return None
    meeting_date = _parse_date(fallback.group(1), "%Y%m%d")
    if meeting_date is None:
        return None
    return Classified(FedInfoType.UNKNOWN, meeting_date, url)


def _clean_title(text: str, classified: Classified) -> str:
    """Link text minus the file-size parenthetical, or a synthesised title.

    Some anchors are nothing but the size ("1.82 MB PDF"), which is not a
    title. Falling back to type + date keeps every document identifiable.
    """
    title = _SIZE_SUFFIX.sub("", text).strip()
    if title:
        return title
    return f"{classified.info_type.value.replace('_', ' ').title()} {classified.meeting_date}"


def _media_type(url: str) -> str:
    return "application/pdf" if url.lower().endswith(".pdf") else "text/html"


def parse_index(html: bytes) -> list[FedSourceDocument]:
    """Extract every recognised document from one index page.

    No `year` parameter: the meeting date comes from the URL, so the page a
    document was found on tells us nothing the URL does not. Pure -- hand it
    bytes from the cache and it never touches the network.
    """
    soup = BeautifulSoup(html, "html.parser")
    docs: dict[str, FedSourceDocument] = {}

    for anchor in soup.find_all("a", href=True):
        url = urljoin(BASE, anchor["href"])
        if url in docs:
            continue

        classified = classify_url(url)
        if classified is None or classified.info_type is FedInfoType.UNKNOWN:
            continue

        docs[url] = FedSourceDocument(
            source=SourceSystem.FED_GOV,
            info_type=classified.info_type,
            meeting_date=classified.meeting_date,
            title=_clean_title(anchor.get_text(strip=True), classified),
            url=url,
            media_type=_media_type(url),
        )

    return list(docs.values())

def iter_documents(
        index_urls: Iterable[str] | None = None,
        *,
        cache_dir: Path | None = None,
        force: bool = False
) -> Iterator[FedSourceDocument]:

    seen: set[str] = set()

    for index_url in iter_index_urls() if index_urls is None else index_urls:
        try:
            html = get(index_url, cache_dir=cache_dir, force=force)
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            logger.warning(f"Skipping index page {index_url}: {exc!r}")
            continue

        for doc in parse_index(html):
            if doc.url in seen:
                continue
            seen.add(doc.url)
            yield doc

    
def fetch_press_index(
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> list[FedSourceDocument]:
    """
    Fed statements 2006 to present
    """

    raw = get(PRESS_INDEX_URL, cache_dir=cache_dir, force=force)
    records = json.loads(raw.decode("utf-8-sig"))

    docs: dict[str, FedSourceDocument] = {}

    for record in records:
        if record.get("pt") != "Monetary Policy":
            continue 

        title = html.unescape(record.get("t", "")).strip()

        if title.lower().rstrip(".") not in _STATEMENT_TITLES:
            continue

        url = urljoin(BASE, record["l"])
        if url in docs:
            continue

        classified = classify_url(url)
        meeting_date = (
            classified.meeting_date if classified else _press_date(record.get("d", ""))
        )
        if meeting_date is None:
            logger.warning(f"No usable date for press record: {record!r}")
            continue

        docs[url] = FedSourceDocument(
            source=SourceSystem.FED_GOV,
            info_type=FedInfoType.STATEMENT,
            meeting_date=meeting_date,
            title=title,
            url=url,
            media_type=_media_type(url),
        )

    return list(docs.values())



def _extract_html(content: bytes) -> ExtractedText:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(_CHROME_TAGS):
        tag.decompose()

    # The Fed wraps body copy in #article; fall back for older layouts.
    node = soup.find(id="article") or soup.body or soup
    text = _BLANK_RUN.sub("\n\n", node.get_text("\n", strip=True))
    return ExtractedText(text)

def _extract_pdf(content: bytes) -> ExtractedText:
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages: list[str] = []
    failed: list[int] = []

    for number, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # pypdf raises many types on damaged streams
            logger.warning(f"Page {number} failed to extract: {exc!r}")
            pages.append("")
            failed.append(number)

    return ExtractedText(
        text="\n\n".join(p for p in pages if p),
        page_chars=tuple(len(p) for p in pages),
        failed_pages=tuple(failed),
    )

def extract_text(content: bytes, media_type: str) -> ExtractedText:
    """Document bytes to plain text. Dispatches on the document's media_type."""
    if media_type == "application/pdf":
        return _extract_pdf(content)
    return _extract_html(content)
