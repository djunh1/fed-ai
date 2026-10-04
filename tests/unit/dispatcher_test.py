"""Unit tests for ApiDispatcher and the FedGovApi adapter.

No network. These cover host normalisation, which is the whole substance of
the dispatcher and the one place a silent miss is possible: an unroutable URL
means a document that never gets fetched, with no error.

Run: uv run pytest tests/unit/dispatcher_test.py -v
"""

import pytest

from llm_engineering.application.api.base import BaseApi
from llm_engineering.application.api.dispatcher import ApiDispatcher, _host
from llm_engineering.application.api.fedgov_api import FedGovApi
from llm_engineering.domain.documents import FedDocument


@pytest.fixture
def dispatcher() -> ApiDispatcher:
    return ApiDispatcher.build().register_fedgov()


# --------------------------------------------------------------------------
# host normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://www.federalreserve.gov", "federalreserve.gov"),
        ("https://federalreserve.gov", "federalreserve.gov"),
        ("http://WWW.FederalReserve.GOV/a.htm", "federalreserve.gov"),
        ("https://www.federalreserve.gov/monetarypolicy/x.pdf", "federalreserve.gov"),
        ("https://fraser.stlouisfed.org/api/title/677", "fraser.stlouisfed.org"),
        ("federalreserve.gov", "federalreserve.gov"),
        ("https://www.federalreserve.gov:443/a.htm", "federalreserve.gov"),
    ],
)
def test_host_normalisation(value: str, expected: str) -> None:
    """register() is called with bare domains and get_api() with full document
    URLs, so both forms must reduce to the same key."""
    assert _host(value) == expected


def test_www_is_stripped_from_the_netloc_not_the_url() -> None:
    """The regression: anchoring the www. strip to ^https?:// means it never
    fires, because urlparse has already removed the scheme. Registration then
    keys on 'www.federalreserve.gov' and a bare-domain URL misses."""
    assert _host("https://www.federalreserve.gov") == _host("https://federalreserve.gov")


# --------------------------------------------------------------------------
# registration and lookup
# --------------------------------------------------------------------------


def test_build_starts_empty() -> None:
    assert len(ApiDispatcher.build()) == 0


def test_register_fedgov_registers_one_source(dispatcher: ApiDispatcher) -> None:
    assert len(dispatcher) == 1
    assert isinstance(dispatcher.sources()[0], FedGovApi)


def test_registration_is_chainable() -> None:
    """The handbook's fluent style: build().register_a().register_b()."""
    result = ApiDispatcher.build().register_fedgov()

    assert isinstance(result, ApiDispatcher)
    assert isinstance(result.register_fedgov(), ApiDispatcher)


def test_registering_twice_does_not_duplicate(dispatcher: ApiDispatcher) -> None:
    dispatcher.register_fedgov()

    assert len(dispatcher) == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20150128.htm",
        "https://federalreserve.gov/fomc/MINUTES/1995/19950201min.htm",
        "https://www.federalreserve.gov/json/ne-press.json",
    ],
)
def test_fedgov_urls_route_to_fedgov(dispatcher: ApiDispatcher, url: str) -> None:
    assert isinstance(dispatcher.get_api(url), FedGovApi)


@pytest.mark.parametrize(
    "url",
    [
        "https://fraser.stlouisfed.org/docs/1.pdf",
        "https://example.org/a.htm",
        "not a url",
        "",
    ],
)
def test_unknown_hosts_return_none(dispatcher: ApiDispatcher, url: str) -> None:
    """None rather than raising: a manifest from an older run can hold hosts
    this dispatcher no longer registers, and one unroutable URL must not end
    a multi-hour harvest."""
    assert dispatcher.get_api(url) is None


def test_sources_returns_every_adapter(dispatcher: ApiDispatcher) -> None:
    assert len(dispatcher.sources()) == len(dispatcher)


# --------------------------------------------------------------------------
# the adapter contract
# --------------------------------------------------------------------------


def test_fedgov_api_is_a_base_api() -> None:
    assert issubclass(FedGovApi, BaseApi)


def test_fedgov_api_declares_its_model() -> None:
    assert FedGovApi.model is FedDocument


def test_base_api_cannot_be_instantiated() -> None:
    """The abstract methods are the point: a registered adapter has to
    actually promise iter_documents and fetch."""
    with pytest.raises(TypeError):
        BaseApi()  # type: ignore[abstract]


def test_incomplete_adapter_is_rejected() -> None:
    class Partial(BaseApi):
        def fetch(self, url: str, *, cache_dir=None) -> bytes:
            return b""

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_fedgov_api_domain_matches_the_module(dispatcher: ApiDispatcher) -> None:
    """The adapter's registration key comes from fedgov.BASE, so the two can
    never drift apart."""
    from llm_engineering.application.api import fedgov

    assert FedGovApi.domain == fedgov.BASE
    assert dispatcher.get_api(fedgov.BASE) is not None
