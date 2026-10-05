"""FRASER as a registered source.

The mirror of `fedgov_api.FedGovApi`: a thin adapter holding no logic of its
own, so the 67 tests in `fraser_test.py` keep covering the real work.

One asymmetry with fed.gov is worth stating. fed.gov documents are fetched
from the URL in the manifest and parsed; FRASER documents carry a `text_url`
alongside the PDF, pointing at text the St. Louis Fed already extracted. The
text step prefers that URL, which is why `fetch` here is handed a `.txt` far
more often than a `.pdf` -- and why the 1930s typewriter scans never reach
pypdf.
"""

from collections.abc import Iterator
from pathlib import Path

from llm_engineering.application.api import fedgov, fraser
from llm_engineering.application.api.base import BaseApi
from llm_engineering.application.api.models import FedSourceDocument
from llm_engineering.domain.documents import FedDocument
from llm_engineering.domain.types import FedInfoType
from llm_engineering.infrastructure.http import get


class FraserApi(BaseApi):
    model = FedDocument

    #: Used by ApiDispatcher.register_fraser() for host-based lookup. The
    #: landing pages, the PDFs and the extracted text all live on this one
    #: host, so a single registration routes every FRASER URL.
    domain = "https://fraser.stlouisfed.org"

    def __init__(self, client: fraser.FraserClient | None = None) -> None:
        """The client is built lazily, on the first iter_documents() call.

        Constructing one eagerly would raise when FED_API_KEY is unset, and
        that is wrong for the half of this adapter that needs no key: only
        `/api/*` is authenticated, so `fetch` works without one. Registering
        FRASER purely to route URLs in the text step must not demand a
        credential it will never send.

        Passing `client` is for tests, which inject a fake rather than reach
        the network.
        """
        self._client = client

    def iter_documents(
        self, *, cache_dir: Path | None = None, force: bool = False
    ) -> Iterator[FedSourceDocument]:
        """Every FOMC record in title 677 that is an actual document.

        1,041 records in, ~581 out. Two filters account for the difference:
        `parse_record` drops the 9 with no pdfUrl, and UNKNOWN is dropped
        here -- 437 agendas plus staff materials, which are scaffolding
        around the deliberation rather than a record of it.

        That second filter matches what fed.gov already does in
        `parse_index`, so both sources agree on what counts as a document.
        """
        client = self._client or fraser.FraserClient(cache_dir=cache_dir)

        for record in client.iter_title_items(force=force):
            document = fraser.parse_record(record)
            if document is None:
                continue
            if document.info_type is FedInfoType.UNKNOWN:
                continue
            yield document

    def fetch(self, url: str, *, cache_dir: Path | None = None) -> bytes:
        """Raw bytes for one FRASER URL -- usually a text_url, sometimes a PDF.

        Deliberately `http.get` rather than the client: the file hosts are
        unauthenticated and unmetered, so routing this through FraserClient
        would send a pointless API key and spend quota against the 30/min
        budget that FRASER is not charging for.
        """
        return get(url, cache_dir=cache_dir)

    def extract_text(self, content: bytes, media_type: str) -> fedgov.ExtractedText:
        """Bytes to text, with a pass-through branch FRASER needs and fed.gov does not.

        `fedgov.extract_text` dispatches PDF-or-HTML, with HTML as the
        fallback. Handing it plain text would run BeautifulSoup over prose
        and quietly eat anything shaped like a tag, so text/plain is decoded
        here instead. Everything else delegates, since PDF parsing is not
        source-specific.
        """
        if media_type.startswith("text/plain"):
            return fedgov.ExtractedText(text=_decode(content))
        return fedgov.extract_text(content, media_type)


def _decode(content: bytes) -> str:
    """UTF-8, falling back to latin-1, matching `fraser.fetch_text`.

    latin-1 maps every byte, so it cannot raise -- a worse decoding than the
    right codec, but better than losing the document.
    """
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")
