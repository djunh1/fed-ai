"""Unit tests for llm_engineering.application.api.fraser.FraserClient.

No network and no API key. `http.get` is replaced with a stub that serves
canned envelopes and records every call, so pagination, the bang-delimited
fields quirk and the 401 translation are all asserted exactly.

The live counterpart is temp/debug_fraser.py, which hits the real API.

Run: uv run pytest tests/unit/fraser_test.py -v
"""

import datetime
import json

import httpx
import pytest

from llm_engineering.application.api import fraser
from llm_engineering.application.api.fraser import (
    FOMC_TITLE_ID,
    RECORD_FIELDS,
    FraserClient,
    classify_pdf_url,
    fetch_text,
    parse_record,
)
from llm_engineering.domain.types import FedInfoType, SourceSystem
from llm_engineering.infrastructure.http import RateLimiter

KEY = "test-key-0123456789"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _record(n: int) -> dict:
    """Shaped like the real thing, which is not uniform.

    titleInfo is a LIST of dicts holding strings; originInfo is a DICT
    mixing strings and lists; every location value is a list. Flattening
    this in a fixture would hide the shape step 12 has to absorb.
    """
    return {
        "titleInfo": [{"title": f"Meeting, record {n}", "subTitle": ""}],
        "originInfo": {"sortDate": f"19{33 + n % 60}-07-20", "dateIssued": ["July 20"]},
        "location": {
            "url": [f"https://fraser.stlouisfed.org/title/677/meeting-{n}"],
            "pdfUrl": [f"https://fraser.stlouisfed.org/docs/{n}.pdf"],
            "textUrl": [f"https://fraser.stlouisfed.org/files/text/{n}.txt"],
        },
        "physicalDescription": {"extent": "10 pages"},
    }


def _envelope(page: int, limit: int, total: int) -> dict:
    start = (page - 1) * limit
    return {
        "format": "json",
        "page": page,
        "limit": limit,
        "fields": ["*"],
        "total": total,
        "start": start,
        "records": [_record(n) for n in range(start, min(start + limit, total))],
    }


class FakeGet:
    """Stands in for infrastructure.http.get. Serves pages, records calls."""

    def __init__(self, total: int = 1041) -> None:
        self.total = total
        self.calls: list[dict] = []

    def __call__(self, url: str, **kwargs) -> bytes:
        self.calls.append({"url": url, **kwargs})
        params = kwargs.get("params") or {}
        page = int(params.get("page", 1))
        limit = int(params.get("limit", 100))
        return json.dumps(_envelope(page, limit, self.total)).encode()

    @property
    def count(self) -> int:
        return len(self.calls)


@pytest.fixture
def fake_get(monkeypatch: pytest.MonkeyPatch):
    def install(total: int = 1041) -> FakeGet:
        fake = FakeGet(total)
        monkeypatch.setattr(fraser, "get", fake)
        return fake

    return install


@pytest.fixture
def client() -> FraserClient:
    return FraserClient(api_key=KEY)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_explicit_key_wins_over_settings(client: FraserClient) -> None:
    """The parameter exists so the class is testable without monkeypatching
    a module-level settings object."""
    assert client.headers == {"X-API-Key": KEY}


def test_falls_back_to_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fraser.settings, "FED_API_KEY", "from-env")

    assert FraserClient().headers["X-API-Key"] == "from-env"


def test_missing_key_fails_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config mistake should surface during setup, not as a 401 thirty
    lines into a paginated call."""
    monkeypatch.setattr(fraser.settings, "FED_API_KEY", None)

    with pytest.raises(RuntimeError, match="FED_API_KEY"):
        FraserClient()


def test_empty_key_falls_through_to_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """`api_key or settings...` means "" is treated as "not supplied", not as
    an empty key. Surprising enough to pin."""
    monkeypatch.setattr(fraser.settings, "FED_API_KEY", "from-env")

    assert FraserClient(api_key="").headers["X-API-Key"] == "from-env"


def test_each_client_owns_its_limiter() -> None:
    """One bucket per client. A fresh client per call hands each a full
    bucket and quietly defeats the 30/min budget."""
    first, second = FraserClient(api_key=KEY), FraserClient(api_key=KEY)

    assert isinstance(first._limiter, RateLimiter)
    assert first._limiter is not second._limiter


def test_rate_defaults_below_the_published_limit(client: FraserClient) -> None:
    """FRASER allows 30/min. Running at exactly 30 gets counted as 31 when
    the two clocks disagree."""
    assert client._limiter._rate * 60 < 30


# --------------------------------------------------------------------------
# _get
# --------------------------------------------------------------------------


def test_get_returns_the_whole_envelope(client: FraserClient, fake_get) -> None:
    """Not just records: `total` is what drives pagination."""
    fake_get()

    envelope = client._get("title/677/items", limit=3)

    assert envelope["total"] == 1041
    assert {"page", "limit", "records", "start"} <= set(envelope)


def test_get_sends_the_api_key(client: FraserClient, fake_get) -> None:
    fake = fake_get()

    client._get("title/677/items")

    assert fake.calls[0]["headers"] == {"X-API-Key": KEY}


def test_get_passes_the_limiter_to_http(client: FraserClient, fake_get) -> None:
    """The limiter must reach http.get, which acquires after the cache read
    and inside the retry loop. Acquiring here instead would spend a token on
    every cache hit."""
    fake = fake_get()

    client._get("title/677/items")

    assert fake.calls[0]["limiter"] is client._limiter


def test_get_builds_the_url_from_base(client: FraserClient, fake_get) -> None:
    fake = fake_get()

    client._get("title/677/items")

    assert fake.calls[0]["url"] == f"{FraserClient.BASE}/title/677/items"


def test_get_tolerates_a_leading_slash(client: FraserClient, fake_get) -> None:
    """Otherwise the URL doubles its slash and FRASER 404s."""
    fake = fake_get()

    client._get("/title/677/items")

    assert fake.calls[0]["url"] == f"{FraserClient.BASE}/title/677/items"


def test_fields_are_bang_delimited(client: FraserClient, fake_get) -> None:
    """The quirk worth absorbing centrally: FRASER wants a!b!c. Commas do not
    error -- they are ignored and you silently receive every field."""
    fake = fake_get()

    client._get("title/677/items", fields=["titleInfo", "location"])

    assert fake.calls[0]["params"]["fields"] == "titleInfo!location"


def test_scalar_params_pass_through_unchanged(client: FraserClient, fake_get) -> None:
    fake = fake_get()

    client._get("title/677/items", page=2, limit=50)

    assert fake.calls[0]["params"]["page"] == 2
    assert fake.calls[0]["params"]["limit"] == 50


def test_none_params_are_dropped(client: FraserClient, fake_get) -> None:
    """A literal "None" in the query string is worse than omitting the key."""
    fake = fake_get()

    client._get("title/677/items", limit=10, fields=None)

    assert "fields" not in fake.calls[0]["params"]


def test_401_becomes_a_readable_error(client: FraserClient, monkeypatch) -> None:
    """httpx says "Client error '401 Unauthorized'", naming neither the key
    nor the variable that holds it."""

    def unauthorized(url: str, **kwargs):
        raise httpx.HTTPStatusError(
            "401",
            request=httpx.Request("GET", url),
            response=httpx.Response(401, request=httpx.Request("GET", url)),
        )

    monkeypatch.setattr(fraser, "get", unauthorized)

    with pytest.raises(RuntimeError, match="FED_API_KEY"):
        client._get("title/677/items")


def test_other_status_errors_are_not_translated(client: FraserClient, monkeypatch) -> None:
    """A 404 on a bad title id should still look like a 404."""

    def not_found(url: str, **kwargs):
        raise httpx.HTTPStatusError(
            "404",
            request=httpx.Request("GET", url),
            response=httpx.Response(404, request=httpx.Request("GET", url)),
        )

    monkeypatch.setattr(fraser, "get", not_found)

    with pytest.raises(httpx.HTTPStatusError):
        client._get("title/99999/items")


# --------------------------------------------------------------------------
# iter_title_items
# --------------------------------------------------------------------------


def test_yields_every_record(client: FraserClient, fake_get) -> None:
    fake_get(total=1041)

    assert len(list(client.iter_title_items())) == 1041


def test_pagination_uses_the_expected_number_of_calls(client: FraserClient, fake_get) -> None:
    """1,041 records at limit=100 is 11 pages."""
    fake = fake_get(total=1041)

    list(client.iter_title_items())

    assert fake.count == 11
    assert [call["params"]["page"] for call in fake.calls] == list(range(1, 12))


def test_a_full_final_page_does_not_truncate(client: FraserClient, fake_get) -> None:
    """The bug the `total` check exists to prevent.

    With exactly 200 records at limit=100, page 2 comes back FULL. A "stop
    when a page is short" heuristic cannot tell that from a mid-series page,
    so it must fetch a third, empty page to discover it is done.

    Driving off `total` gets the same 200 records in two calls -- it knows
    the series is complete the moment seen reaches total.
    """
    fake = fake_get(total=200)

    assert len(list(client.iter_title_items(limit=100))) == 200
    assert fake.count == 2, "total should end it without a wasted probe page"


def test_stops_on_an_empty_page(client: FraserClient, fake_get) -> None:
    """Belt and braces: if `total` ever lies, an empty page still ends it
    rather than paging forever."""
    fake_get(total=0)

    assert list(client.iter_title_items()) == []


def test_it_streams(client: FraserClient, fake_get) -> None:
    """A generator: the first record costs one page, not eleven."""
    fake = fake_get(total=1041)

    next(client.iter_title_items())

    assert fake.count == 1


def test_requests_only_the_fields_we_parse(client: FraserClient, fake_get) -> None:
    """All nine blocks is several times the payload for data step 12
    discards."""
    fake = fake_get(total=10)

    list(client.iter_title_items())

    assert fake.calls[0]["params"]["fields"] == "!".join(RECORD_FIELDS)


def test_fields_can_be_widened(client: FraserClient, fake_get) -> None:
    fake = fake_get(total=10)

    list(client.iter_title_items(fields=None))

    assert "fields" not in fake.calls[0]["params"]


def test_defaults_to_the_fomc_title(client: FraserClient, fake_get) -> None:
    fake = fake_get(total=10)

    list(client.iter_title_items())

    assert f"title/{FOMC_TITLE_ID}/items" in fake.calls[0]["url"]
    assert FOMC_TITLE_ID == 677


def test_limit_is_honoured(client: FraserClient, fake_get) -> None:
    fake = fake_get(total=250)

    records = list(client.iter_title_items(limit=50))

    assert len(records) == 250
    assert fake.calls[0]["params"]["limit"] == 50
    assert fake.count == 5


def test_records_keep_their_nested_shape(client: FraserClient, fake_get) -> None:
    """Step 12 parses these. titleInfo is a list of dicts, originInfo a dict,
    location values all lists -- three different shapes in one record."""
    fake_get(total=1)

    (record,) = list(client.iter_title_items())

    assert isinstance(record["titleInfo"], list)
    assert isinstance(record["titleInfo"][0]["title"], str)
    assert isinstance(record["originInfo"]["sortDate"], str)
    assert isinstance(record["location"]["textUrl"], list)


# --------------------------------------------------------------------------
# classify_pdf_url
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # Minutes, across the six filename conventions in the series.
        ("rg82_fomcminutes_19330720.pdf", FedInfoType.MINUTES),
        ("fomchistmin19560110.pdf", FedInfoType.MINUTES),
        ("19360318Minutesv.pdf", FedInfoType.MINUTES),
        ("19360319MinutesECv.pdf", FedInfoType.MINUTES),
        ("19550926MinutesCCv.pdf", FedInfoType.MINUTES),
        ("19940418min.pdf", FedInfoType.MINUTES),
        # Everything else.
        ("fomcmod19680314.pdf", FedInfoType.MEMORANDA_OF_DISCUSSION),
        ("fomcmoa19761108.pdf", FedInfoType.MINUTES_OF_ACTIONS),
        ("19780310MoACC.pdf", FedInfoType.MINUTES_OF_ACTIONS),
        ("FOMC19770527confcall.pdf", FedInfoType.TRANSCRIPT),
        ("900917ConfCall.pdf", FedInfoType.TRANSCRIPT),
        ("Beigebook_20170118.pdf", FedInfoType.BEIGE_BOOK),
        ("20080116beige.pdf", FedInfoType.BEIGE_BOOK),
        ("FOMC19721017bluebook19721013.pdf", FedInfoType.BRIEFING_BOOK),
        ("fomc19791006gbspecial19791012.pdf", FedInfoType.BRIEFING_BOOK),
        ("19981015statement.pdf", FedInfoType.STATEMENT),
        ("monetary20191011a1.pdf", FedInfoType.STATEMENT),
        ("FOMC_LongerRunGoals_20250822.pdf", FedInfoType.STATEMENT),
        # Deliberately excluded, matching fed.gov's treatment.
        ("FOMC19640107Agenda.pdf", FedInfoType.UNKNOWN),
        ("FOMC19791005material.pdf", FedInfoType.UNKNOWN),
        ("fomc20140409-historical-forecast-errors.pdf", FedInfoType.UNKNOWN),
        ("marconf.pdf", FedInfoType.UNKNOWN),
    ],
)
def test_classify_pdf_url(filename: str, expected: FedInfoType) -> None:
    assert classify_pdf_url(f"https://fraser.stlouisfed.org/docs/{filename}") == expected


def test_staff_memos_are_not_memoranda_of_discussion() -> None:
    """The ordering trap. Matching on "memo" would sweep up staff memos,
    which are not the Memoranda of Discussion series -- only fomcmod is."""
    memo = "https://fraser.stlouisfed.org/docs/FOMC19900309memo01_taskforce_foreigncurrency.pdf"
    real = "https://fraser.stlouisfed.org/docs/fomcmod19680314.pdf"

    assert classify_pdf_url(memo) is not FedInfoType.MEMORANDA_OF_DISCUSSION
    assert classify_pdf_url(real) is FedInfoType.MEMORANDA_OF_DISCUSSION


def test_classification_ignores_the_directory() -> None:
    """Only the filename is classified. "statement" appearing in a path
    segment must not reclassify a minutes PDF."""
    url = "https://fraser.stlouisfed.org/docs/statement/historical/fomchistmin19560110.pdf"

    assert classify_pdf_url(url) is FedInfoType.MINUTES


# --------------------------------------------------------------------------
# parse_record
# --------------------------------------------------------------------------


def _fraser_record(**overrides) -> dict:
    record = {
        "titleInfo": [{"title": "Meeting, July 20, 1933", "subTitle": ""}],
        "originInfo": {"sortDate": "1933-07-20", "dateIssued": ["July 20, 1933"]},
        "location": {
            "url": ["https://fraser.stlouisfed.org/title/677/meeting-july-20-1933-22634"],
            "pdfUrl": ["https://fraser.stlouisfed.org/docs/rg82_fomcminutes_19330720.pdf"],
            "textUrl": ["https://fraser.stlouisfed.org/files/text/rg82_fomcminutes_19330720.txt"],
        },
        "physicalDescription": {"extent": "10 pages"},
    }
    record.update(overrides)
    return record


def test_parse_record_maps_every_field() -> None:
    doc = parse_record(_fraser_record())

    assert doc is not None
    assert doc.source is SourceSystem.FRASER
    assert doc.info_type is FedInfoType.MINUTES
    assert doc.meeting_date == datetime.date(1933, 7, 20)
    assert doc.title == "Meeting, July 20, 1933"
    assert doc.url.endswith("rg82_fomcminutes_19330720.pdf")
    assert doc.text_url.endswith("rg82_fomcminutes_19330720.txt")
    assert doc.source_id == "22634"
    assert doc.media_type == "application/pdf"
    assert doc.page_count == 10


def test_url_is_the_pdf_not_the_landing_page() -> None:
    """location.url is a human-facing page. The manifest needs something
    fetchable, and merge() dedupes against fed.gov document URLs."""
    doc = parse_record(_fraser_record())

    assert doc is not None
    assert "/docs/" in doc.url
    assert "/title/" not in doc.url


def test_records_without_a_pdf_are_dropped() -> None:
    """Nine of the 1,041 have no pdfUrl. A landing page is not a document,
    so emitting one would put an unfetchable URL in the manifest."""
    record = _fraser_record()
    record["location"] = {"url": record["location"]["url"]}

    assert parse_record(record) is None


def test_missing_sort_date_is_dropped() -> None:
    """meeting_date is not optional on FedSourceDocument, and a document
    that cannot be dated cannot be merged against fed.gov."""
    assert parse_record(_fraser_record(originInfo={})) is None


def test_unparseable_sort_date_is_dropped() -> None:
    assert parse_record(_fraser_record(originInfo={"sortDate": "July 1933"})) is None


def test_page_count_only_from_a_pages_extent() -> None:
    """912 of 1,041 extents read "N files" -- documents in the meeting's
    folder, not pages. Reading the integer without checking the unit puts
    a file count in page_count."""
    assert parse_record(_fraser_record()).page_count == 10
    assert parse_record(_fraser_record(physicalDescription={"extent": "10 files"})).page_count is None
    assert parse_record(_fraser_record(physicalDescription={})).page_count is None


@pytest.mark.parametrize(
    ("extent", "expected"),
    [("1 page", 1), ("10 pages", 10), ("250 Pages", 250), ("10 files", None), ("", None)],
)
def test_page_count_variants(extent: str, expected: int | None) -> None:
    assert parse_record(_fraser_record(physicalDescription={"extent": extent})).page_count == expected


def test_source_id_comes_from_the_landing_url() -> None:
    """location.apiUrl would be the obvious source, but FRASER emits it
    malformed -- "/api//22634", missing the /item segment."""
    assert parse_record(_fraser_record()).source_id == "22634"


def test_missing_source_id_is_tolerated() -> None:
    record = _fraser_record()
    record["location"]["url"] = ["https://fraser.stlouisfed.org/title/677/no-id-here"]

    assert parse_record(record).source_id is None


def test_text_url_is_optional() -> None:
    """9 records lack one; those are the PDF-only fallbacks for step 13."""
    record = _fraser_record()
    del record["location"]["textUrl"]

    doc = parse_record(record)

    assert doc is not None
    assert doc.text_url is None


def test_empty_title_is_synthesised() -> None:
    """A document with no usable title is still identifiable by type+date."""
    doc = parse_record(_fraser_record(titleInfo=[{"title": ""}]))

    assert doc.title == "FOMC 1933-07-20"


def test_parse_record_is_pure() -> None:
    """Same input, same output, and the input is not mutated."""
    record = _fraser_record()
    before = json.dumps(record, sort_keys=True)

    first, second = parse_record(record), parse_record(record)

    assert first == second
    assert json.dumps(record, sort_keys=True) == before


def test_handles_scalar_where_a_list_is_expected() -> None:
    """FRASER mixes lists and bare strings across blocks. _first absorbs
    both so one inconsistent record does not crash a 1,041-record pass."""
    record = _fraser_record()
    record["location"]["pdfUrl"] = "https://fraser.stlouisfed.org/docs/19981015statement.pdf"

    doc = parse_record(record)

    assert doc is not None
    assert doc.info_type is FedInfoType.STATEMENT


# --------------------------------------------------------------------------
# fetch_text
# --------------------------------------------------------------------------

TEXT_URL = "https://fraser.stlouisfed.org/files/text/rg82_fomcminutes_19330720.txt"


class FakeFetch:
    """Returns fixed bytes and records how it was called."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def __call__(self, url: str, **kwargs) -> bytes:
        self.calls.append({"url": url, **kwargs})
        return self.payload


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch):
    def install(payload: bytes) -> FakeFetch:
        fake = FakeFetch(payload)
        monkeypatch.setattr(fraser, "get", fake)
        return fake

    return install


def test_fetch_text_decodes_utf8(fake_fetch) -> None:
    fake_fetch("A meeting of the Federal Open Market Committee".encode())

    assert fetch_text(TEXT_URL).startswith("A meeting of the")


def test_fetch_text_handles_non_ascii(fake_fetch) -> None:
    """Beige Books contain curly quotes and em dashes."""
    fake_fetch("growth was “modest” — per district".encode())

    assert "“modest”" in fetch_text(TEXT_URL)


def test_fetch_text_falls_back_to_latin1(fake_fetch) -> None:
    """latin-1 maps every byte so it cannot fail. Worse than correct
    decoding, better than losing the document."""
    fake_fetch(b"caf\xe9 \xff\xfe not utf-8")

    result = fetch_text(TEXT_URL)

    assert "caf" in result
    assert isinstance(result, str)


def test_fetch_text_sends_no_api_key(fake_fetch) -> None:
    """The file hosts are unauthenticated. Sending a key is pointless and
    leaks it to a host that never asked for it."""
    fake = fake_fetch(b"text")

    fetch_text(TEXT_URL)

    assert fake.calls[0].get("headers") is None


def test_fetch_text_spends_no_rate_limit(fake_fetch) -> None:
    """Only /api/* is metered at 30/min. Throttling file fetches would make
    a 581-document harvest take 20 minutes for no reason."""
    fake = fake_fetch(b"text")

    fetch_text(TEXT_URL)

    assert fake.calls[0].get("limiter") is None


def test_fetch_text_passes_the_cache_dir(fake_fetch, tmp_path) -> None:
    fake = fake_fetch(b"text")

    fetch_text(TEXT_URL, cache_dir=tmp_path)

    assert fake.calls[0]["cache_dir"] == tmp_path


def test_fetch_text_does_not_strip_anything(fake_fetch) -> None:
    """FRASER's text is pre-extracted, not pre-cleaned. Archival headers and
    OCR damage are a downstream cleaning concern -- silently trimming here
    would make the raw corpus unreproducible."""
    raw = "Reproduced from the Unclassified Holdings of the National Archives\n\nA meeting..."
    fake_fetch(raw.encode())

    assert fetch_text(TEXT_URL) == raw


def test_fetch_text_preserves_an_empty_document(fake_fetch) -> None:
    """An empty string is a real signal -- a scan with no text layer. It must
    not become None or raise, or the caller cannot tell it apart."""
    fake_fetch(b"")

    assert fetch_text(TEXT_URL) == ""
