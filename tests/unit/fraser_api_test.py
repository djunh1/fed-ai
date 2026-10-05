"""Tests for FraserApi, the dispatcher adapter over fraser.py.

The adapter holds no parsing logic -- that lives in fraser.py and is covered
by fraser_test.py. What is tested here is the seam: what it filters out, what
it passes through, and the two places it deliberately differs from FedGovApi.
"""

import datetime

import pytest

from llm_engineering.application.api import fedgov
from llm_engineering.application.api.base import BaseApi
from llm_engineering.application.api.dispatcher import ApiDispatcher
from llm_engineering.application.api.fedgov_api import FedGovApi
from llm_engineering.application.api.fraser_api import FraserApi
from llm_engineering.domain.documents import FedDocument
from llm_engineering.domain.types import FedInfoType, SourceSystem

PDF_URL = "https://fraser.stlouisfed.org/docs/historical/FOMC/meetingdocuments/rg82_fomcminutes_19330720.pdf"
TEXT_URL = "https://fraser.stlouisfed.org/files/text/historical/FOMC/meetingdocuments/rg82_fomcminutes_19330720.txt"


def record(pdf_name: str, sort_date: str = "1933-07-20", **overrides) -> dict:
    """A minimal FRASER API record, shaped like the real envelope's entries."""
    base = {
        "titleInfo": [{"title": "Meeting, July 20, 1933"}],
        "originInfo": {"sortDate": sort_date},
        "location": {
            "pdfUrl": [f"https://fraser.stlouisfed.org/docs/{pdf_name}"],
            "textUrl": [f"https://fraser.stlouisfed.org/files/text/{pdf_name}"],
            "url": ["https://fraser.stlouisfed.org/title/677/item/22634"],
        },
        "physicalDescription": {"extent": "12 pages"},
    }
    base.update(overrides)
    return base


class FakeClient:
    """Replaces FraserClient so the adapter can be tested with no key."""

    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self.force_seen: list[bool] = []

    def iter_title_items(self, *, force: bool = False, **_):
        self.force_seen.append(force)
        yield from self.records


# --- what the adapter promises ------------------------------------------


def test_is_a_registered_source_type():
    assert issubclass(FraserApi, BaseApi)
    assert FraserApi.model is FedDocument


def test_no_api_key_needed_to_construct():
    """Only /api/* is authenticated. Registering FRASER purely to route URLs
    in the text step must not demand a credential it will never send."""
    assert FraserApi() is not None


# --- iter_documents: the filtering seam ----------------------------------


def test_yields_parsed_documents():
    api = FraserApi(client=FakeClient([record("rg82_fomcminutes_19330720.pdf")]))

    documents = list(api.iter_documents())

    assert len(documents) == 1
    assert documents[0].source is SourceSystem.FRASER
    assert documents[0].info_type is FedInfoType.MINUTES
    assert documents[0].meeting_date == datetime.date(1933, 7, 20)
    assert documents[0].text_url is not None


def test_drops_records_with_no_pdf():
    """9 of the 1,041 records carry no pdfUrl. A landing page is not a
    document, so emitting one would put an unfetchable URL in the manifest."""
    api = FraserApi(client=FakeClient([record("x.pdf", location={"url": ["https://x/-1"]})]))

    assert list(api.iter_documents()) == []


@pytest.mark.parametrize("name", ["19950201agenda.pdf", "fomc_material_1995.pdf"])
def test_drops_unknown_info_types(name):
    """437 agendas plus staff materials classify as UNKNOWN. fed.gov drops
    the same documents in parse_index, so both sources agree on what counts."""
    api = FraserApi(client=FakeClient([record(name)]))

    assert list(api.iter_documents()) == []


def test_keeps_recognised_alongside_dropped():
    api = FraserApi(
        client=FakeClient(
            [
                record("19950201agenda.pdf"),
                record("rg82_fomcminutes_19330720.pdf"),
                record("19950201transcript.pdf"),
            ]
        )
    )

    types = [doc.info_type for doc in api.iter_documents()]

    assert types == [FedInfoType.MINUTES, FedInfoType.TRANSCRIPT]


def test_force_reaches_the_client():
    """If the adapter swallowed force, a cached page would replay forever and
    an incremental run could never discover a new meeting."""
    client = FakeClient([record("rg82_fomcminutes_19330720.pdf")])
    api = FraserApi(client=client)

    list(api.iter_documents(force=False))
    list(api.iter_documents(force=True))

    assert client.force_seen == [False, True]


def test_iter_documents_is_lazy():
    """A generator, not a list: the manifest step streams 1,041 records."""
    api = FraserApi(client=FakeClient([record("rg82_fomcminutes_19330720.pdf")]))

    documents = api.iter_documents()

    assert next(documents).info_type is FedInfoType.MINUTES
    with pytest.raises(StopIteration):
        next(documents)


# --- fetch: no key, no limiter -------------------------------------------


def test_fetch_sends_no_api_key(monkeypatch):
    """The file hosts are unauthenticated and unmetered. Sending a key would
    leak it to a host that never asked, and spend quota FRASER is not
    charging for."""
    seen = {}

    def fake_get(url, **kwargs):
        seen.update(kwargs, url=url)
        return b"text"

    monkeypatch.setattr("llm_engineering.application.api.fraser_api.get", fake_get)

    FraserApi().fetch(TEXT_URL)

    assert seen["url"] == TEXT_URL
    assert seen.get("headers") is None
    assert seen.get("limiter") is None


def test_fetch_passes_cache_dir(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        "llm_engineering.application.api.fraser_api.get",
        lambda url, **kwargs: seen.update(kwargs) or b"",
    )

    FraserApi().fetch(TEXT_URL, cache_dir=tmp_path)

    assert seen["cache_dir"] == tmp_path


# --- extract_text: the branch fed.gov does not need ----------------------


def test_plain_text_passes_through_unparsed():
    """fedgov.extract_text falls back to HTML, which would run BeautifulSoup
    over prose and silently eat anything shaped like a tag."""
    prose = b"the ratio <adjusted> upward"

    result = FraserApi().extract_text(prose, "text/plain")

    assert result.text == "the ratio <adjusted> upward"
    assert "<adjusted>" not in fedgov.extract_text(prose, "text/html").text


def test_plain_text_honours_charset_suffix():
    result = FraserApi().extract_text(b"minutes", "text/plain; charset=utf-8")

    assert result.text == "minutes"


def test_plain_text_falls_back_to_latin1():
    """latin-1 maps every byte, so a wrongly-encoded file is degraded rather
    than lost."""
    result = FraserApi().extract_text("Système".encode("latin-1"), "text/plain")

    assert result.text == "Syst\xe8me"


def test_plain_text_reports_empty():
    assert FraserApi().extract_text(b"   \n ", "text/plain").is_empty


def test_plain_text_has_no_page_chars():
    """page_chars is per-PDF-page; there are no pages in a text file."""
    result = FraserApi().extract_text(b"minutes", "text/plain")

    assert result.page_chars == ()
    assert result.failed_pages == ()


def test_non_text_delegates_to_fedgov(monkeypatch):
    """PDF parsing is not source-specific, so it is not duplicated here."""
    monkeypatch.setattr(
        "llm_engineering.application.api.fedgov.extract_text",
        lambda content, media_type: fedgov.ExtractedText(text=f"{media_type}:{len(content)}"),
    )

    assert FraserApi().extract_text(b"%PDF-1.4", "application/pdf").text == "application/pdf:8"


# --- dispatcher registration ---------------------------------------------


def test_register_fraser_routes_every_fraser_url():
    """The landing pages, the PDFs and the extracted text share one host, so
    one registration covers all three. The text step fetches
    `text_url or url`, so routing only the /docs/ path would break it."""
    dispatcher = ApiDispatcher.build().register_fraser()

    for url in (PDF_URL, TEXT_URL, "https://fraser.stlouisfed.org/title/677"):
        assert isinstance(dispatcher.get_api(url), FraserApi)


def test_registers_alongside_fedgov():
    dispatcher = ApiDispatcher.build().register_fedgov().register_fraser()

    assert len(dispatcher) == 2
    assert isinstance(dispatcher.get_api(PDF_URL), FraserApi)
    assert isinstance(
        dispatcher.get_api("https://www.federalreserve.gov/monetarypolicy/fomcminutes20150128.htm"),
        FedGovApi,
    )


def test_register_fraser_is_fluent():
    """Chaining is the handbook's registration style; a method returning None
    breaks `.register_fedgov().register_fraser()`."""
    dispatcher = ApiDispatcher.build()

    assert dispatcher.register_fraser() is dispatcher


def test_unknown_host_still_returns_none():
    dispatcher = ApiDispatcher.build().register_fraser()

    assert dispatcher.get_api("https://example.org/nothing.pdf") is None
