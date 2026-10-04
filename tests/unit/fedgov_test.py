"""Unit tests for llm_engineering.application.api.fedgov.

Everything under test is pure -- no network, no disk -- so these run instantly
and the assertions are exact.

URL samples are real federalreserve.gov paths, taken from the eras the rules
cover. Casing is preserved verbatim, including the shouted
`19960131DEFAULT.htm`, because that inconsistency is what classify_url's
lowercasing exists to absorb.

Run: uv run pytest tests/unit/fedgov_test.py -v
"""

import datetime
import re
from collections.abc import Iterator

import pytest

from llm_engineering.application.api.fedgov import (
    _RULES,
    BASE,
    CALENDARS_URL,
    FIRST_HISTORICAL_YEAR,
    LAST_HISTORICAL_YEAR,
    Classified,
    UrlRule,
    classify_url,
    iter_index_urls,
    parse_index,
)
from llm_engineering.domain.types import FedInfoType, SourceSystem

EXPECTED_COUNT = (LAST_HISTORICAL_YEAR - FIRST_HISTORICAL_YEAR + 1) + 1


def _date(text: str) -> datetime.date:
    return datetime.datetime.strptime(text, "%Y-%m-%d").date()


# --------------------------------------------------------------------------
# iter_index_urls
# --------------------------------------------------------------------------


def test_is_a_generator_not_a_list() -> None:
    """Laziness is the contract: iter_documents() chains onto this, and an
    eager stage anywhere forces every stage above it to complete."""
    urls = iter_index_urls()

    assert isinstance(urls, Iterator)
    assert not isinstance(urls, list)


def test_yields_one_url_per_year_plus_the_calendars_page() -> None:
    """85, not 86, is the signature of a forgotten `+ 1` on the range."""
    assert len(list(iter_index_urls())) == EXPECTED_COUNT


def test_covers_the_full_year_range_inclusive() -> None:
    joined = "\n".join(iter_index_urls())

    for year in (FIRST_HISTORICAL_YEAR, 1970, LAST_HISTORICAL_YEAR):
        assert f"fomchistorical{year}.htm" in joined, f"{year} missing"


def test_excludes_years_outside_the_range() -> None:
    """2021+ lives on the calendars page; fomchistorical2021.htm is a 404."""
    joined = "\n".join(iter_index_urls())

    assert f"fomchistorical{FIRST_HISTORICAL_YEAR - 1}.htm" not in joined
    assert f"fomchistorical{LAST_HISTORICAL_YEAR + 1}.htm" not in joined


def test_starts_at_the_earliest_year_and_ends_with_the_calendars_page() -> None:
    urls = list(iter_index_urls())

    assert urls[0] == f"{BASE}/monetarypolicy/fomchistorical{FIRST_HISTORICAL_YEAR}.htm"
    assert urls[-1] == CALENDARS_URL


def test_calendars_page_is_yielded_exactly_once() -> None:
    """It has no {year} in it. Yielding it inside the loop would emit the same
    string 85 times."""
    assert list(iter_index_urls()).count(CALENDARS_URL) == 1


def test_index_urls_are_unique() -> None:
    urls = list(iter_index_urls())

    assert len(urls) == len(set(urls))


def test_index_urls_are_absolute_and_https() -> None:
    assert all(u.startswith("https://") for u in iter_index_urls())


def test_a_fresh_call_restarts_the_sequence() -> None:
    """Generators are single-use. Each call must return a new one, or the
    second consumer silently gets nothing."""
    first = list(iter_index_urls())
    second = list(iter_index_urls())

    assert first == second
    assert first


# --------------------------------------------------------------------------
# classify_url -- real URLs from every era
# --------------------------------------------------------------------------

CASES: list[tuple[str, FedInfoType, str]] = [
    # Minutes, across all five URL schemes.
    ("/monetarypolicy/fomcminutes20090128.htm", FedInfoType.MINUTES, "2009-01-28"),
    ("/fomc/minutes/19960131.htm", FedInfoType.MINUTES, "1996-01-31"),
    ("/fomc/MINUTES/1993/19930203min.htm", FedInfoType.MINUTES, "1993-02-03"),
    ("/monetarypolicy/files/FOMChistmin19750415.pdf", FedInfoType.MINUTES, "1975-04-15"),
    ("/monetarypolicy/files/fomchminec19420302.pdf", FedInfoType.MINUTES, "1942-03-02"),
    # Same series, but with a part number wedged in before the date.
    ("/monetarypolicy/files/FOMChminec119401101.pdf", FedInfoType.MINUTES, "1940-11-01"),
    ("/monetarypolicy/files/FOMChminec219410317.pdf", FedInfoType.MINUTES, "1941-03-17"),
    # Statements, three paths across three eras.
    ("/fomc/19960131DEFAULT.htm", FedInfoType.STATEMENT, "1996-01-31"),
    ("/newsevents/press/monetary/20060131a.htm", FedInfoType.STATEMENT, "2006-01-31"),
    ("/newsevents/pressreleases/monetary20200315a.htm", FedInfoType.STATEMENT, "2020-03-15"),
    # Transcripts.
    ("/monetarypolicy/files/FOMC19760120meeting.pdf", FedInfoType.TRANSCRIPT, "1976-01-20"),
    ("/monetarypolicy/files/FOMC19950113confcall.pdf", FedInfoType.TRANSCRIPT, "1995-01-13"),
    # Deliberation records. mod and moa are different documents.
    (
        "/monetarypolicy/files/fomcmod19750121.pdf",
        FedInfoType.MEMORANDA_OF_DISCUSSION,
        "1975-01-21",
    ),
    ("/monetarypolicy/files/fomcmoa19750121.pdf", FedInfoType.MINUTES_OF_ACTIONS, "1975-01-21"),
    (
        "/monetarypolicy/files/fomcropa19750415.pdf",
        FedInfoType.RECORD_OF_POLICY_ACTIONS,
        "1975-04-15",
    ),
    # Briefing books: Greenbook/Bluebook pre-2010, Tealbook after.
    (
        "/monetarypolicy/files/FOMC19750415greenbook19750409.pdf",
        FedInfoType.BRIEFING_BOOK,
        "1975-04-15",
    ),
    (
        "/monetarypolicy/files/FOMC19750415bluebook19750411.pdf",
        FedInfoType.BRIEFING_BOOK,
        "1975-04-15",
    ),
    ("/monetarypolicy/files/fomc19950201gbpt119950126.pdf", FedInfoType.BRIEFING_BOOK, "1995-02-01"),
    ("/monetarypolicy/files/fomc19950201gbsup19950127.pdf", FedInfoType.BRIEFING_BOOK, "1995-02-01"),
    ("/monetarypolicy/files/FOMC20150128tealbooka20150121.pdf", FedInfoType.TEAL_BOOK, "2015-01-28"),
    ("/monetarypolicy/files/FOMC20150128tealbookb20150122.pdf", FedInfoType.TEAL_BOOK, "2015-01-28"),
    # Beige Books: Redbook pre-1983, dated directory 1996-2010, flat after.
    (
        "/monetarypolicy/files/fomc19750121redbook19750115.pdf",
        FedInfoType.BEIGE_BOOK,
        "1975-01-21",
    ),
    ("/fomc/beigebook/1996/19961030/default.htm", FedInfoType.BEIGE_BOOK, "1996-10-30"),
    ("/fomc/beigebook/2008/20080305/fullreport20080305.pdf", FedInfoType.BEIGE_BOOK, "2008-03-05"),
    ("/monetarypolicy/beigebook202001.htm", FedInfoType.BEIGE_BOOK, "2020-01-01"),
    # Press conferences.
    ("/monetarypolicy/fomcpresconf20150128.htm", FedInfoType.PRESS_CONFERENCE, "2015-01-28"),
]


@pytest.mark.parametrize(("url", "expected_type", "expected_date"), CASES)
def test_classifies_real_urls(url: str, expected_type: FedInfoType, expected_date: str) -> None:
    result = classify_url(BASE + url)

    assert result is not None, "should have matched a rule"
    assert result.info_type == expected_type
    assert result.meeting_date == _date(expected_date)


@pytest.mark.parametrize(("url", "expected_type", "expected_date"), CASES)
def test_classification_is_case_insensitive(
    url: str, expected_type: FedInfoType, expected_date: str
) -> None:
    """The Fed's casing is inconsistent across eras, not just in one filename."""
    for variant in (url.upper(), url.lower()):
        result = classify_url(BASE + variant)
        assert result is not None, variant
        assert result.info_type == expected_type, variant
        assert result.meeting_date == _date(expected_date), variant


@pytest.mark.parametrize(("url", "expected_type", "expected_date"), CASES)
def test_classification_preserves_the_original_url(
    url: str, expected_type: FedInfoType, expected_date: str
) -> None:
    """Matching happens on a lowercased copy; the URL carried on the result
    must be the one you can actually fetch."""
    original = BASE + url
    result = classify_url(original)

    assert result is not None
    assert result.url == original


def test_greenbook_dates_use_the_meeting_not_the_document() -> None:
    """Briefing-book URLs carry two dates: the meeting, then when the book was
    written. The first is the one that joins to everything else."""
    result = classify_url(f"{BASE}/monetarypolicy/files/FOMC19750415greenbook19750409.pdf")

    assert result is not None
    assert result.meeting_date == _date("1975-04-15")


# --------------------------------------------------------------------------
# classify_url -- the three outcomes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "/default.htm",
        "/aboutthefed/bios/board/default.htm",
        "/aboutthefed/advisorydefault.htm",
        "/releases/g17/current/default.htm",
        "/monetarypolicy/fomc.htm",
        "/feeds/feeds.htm",
        "https://www.facebook.com/federalreserve",
        "#content",
        "",
    ],
)
def test_site_chrome_returns_none(url: str) -> None:
    """No date means not a document. This is the whole defence against the
    substring table's failure mode, where /aboutthefed/.../default.htm
    classified as a monetary policy statement."""
    assert classify_url(url) is None


def test_unmatched_but_dated_urls_are_unknown_not_none() -> None:
    """The distinction that matters: a URL carrying a date but matching no
    rule is a GAP IN _RULES, not site chrome. Collapsing the two is why the
    missing Tealbooks stayed invisible."""
    result = classify_url(f"{BASE}/monetarypolicy/files/FOMC19950201agenda.pdf")

    assert result is not None
    assert result.info_type is FedInfoType.UNKNOWN
    assert result.meeting_date == _date("1995-02-01")


def test_index_pages_are_not_documents() -> None:
    """Every URL from iter_index_urls() is an index. If one classified as a
    document, step 7 would ingest a table of contents as minutes."""
    for url in iter_index_urls():
        assert classify_url(url) is None, url


def test_impossible_dates_do_not_classify() -> None:
    """An 8-digit run is not necessarily a date. 20201345 has month 13."""
    assert classify_url(f"{BASE}/monetarypolicy/files/FOMC20201345meeting.pdf") is None


@pytest.mark.parametrize("year", ["1194", "0219", "2194", "9999"])
def test_implausible_years_are_rejected(year: str) -> None:
    """A date can parse cleanly and still be nonsense.

    FOMChminec119401101.pdf is part 1 of the 1940-11-01 minutes, but reading
    the first 8 digits gives 1194-01-10 -- a valid date in the twelfth
    century. The bounds check turns a silent corruption into a dropped URL.
    """
    assert classify_url(f"{BASE}/monetarypolicy/files/FOMC{year}0110meeting.pdf") is None


def test_part_numbers_do_not_shift_the_date() -> None:
    """The regression itself: 18 documents dated to the 12th century, and 218
    more dropped outright when the shifted digits failed to parse at all."""
    with_part = classify_url(f"{BASE}/monetarypolicy/files/FOMChminec119401101.pdf")
    without = classify_url(f"{BASE}/monetarypolicy/files/fomchminec19420302.pdf")

    assert with_part is not None and with_part.meeting_date == _date("1940-11-01")
    assert without is not None and without.meeting_date == _date("1942-03-02")


# --------------------------------------------------------------------------
# the rules table itself
# --------------------------------------------------------------------------


def test_every_rule_captures_a_date_group() -> None:
    """classify_url reads group(1) unconditionally. A rule without one raises
    IndexError at runtime, on whichever page first contains that URL shape."""
    for rule in _RULES:
        assert rule.pattern.groups >= 1, rule.pattern.pattern


def test_rule_patterns_are_lowercase() -> None:
    """Matching runs against a lowercased URL, so an uppercase character in a
    pattern makes that rule permanently unreachable -- silently."""
    for rule in _RULES:
        pattern = rule.pattern.pattern
        assert pattern == pattern.lower(), pattern


def test_rules_are_immutable() -> None:
    """A tuple of frozen dataclasses. Nothing should be able to reorder the
    table at runtime, because order is part of the logic."""
    assert isinstance(_RULES, tuple)

    with pytest.raises((AttributeError, TypeError)):
        _RULES[0].info_type = FedInfoType.UNKNOWN  # type: ignore[misc]


@pytest.mark.parametrize(("url", "expected_type", "expected_date"), CASES)
def test_at_most_one_rule_type_matches_each_url(
    url: str, expected_type: FedInfoType, expected_date: str
) -> None:
    """The invariant that makes rule ORDER irrelevant rather than tacitly
    load-bearing. If two rules with different types can match one URL, the
    table is order-dependent and a future reorder changes the corpus.
    """
    lowered = (BASE + url).lower()
    types = {r.info_type for r in _RULES if r.pattern.search(lowered)}

    assert len(types) == 1, f"{url} matches {types}"


def test_no_match_predates_its_rules_since_year() -> None:
    """`since` encodes when a document type began to exist. A match before it
    means the pattern is catching something else -- the check that found
    19 'statements' on the 1965 page."""
    for url, _, expected_date in CASES:
        result = classify_url(BASE + url)
        assert result is not None
        if result.rule is not None and result.rule.since is not None:
            assert result.meeting_date.year >= result.rule.since, url


def test_statement_rule_requires_the_fomc_prefix() -> None:
    """Pinning the specific bug: default.htm is the Fed's index filename
    sitewide, so an unanchored rule swallowed ~20 nav links per page."""
    statement_rules = [r for r in _RULES if "default" in r.pattern.pattern]

    assert statement_rules
    for rule in statement_rules:
        assert "/fomc/" in rule.pattern.pattern


# --------------------------------------------------------------------------
# parse_index
# --------------------------------------------------------------------------

PAGE = f"""
<html><body>
  <nav>
    <a href="/default.htm">Home</a>
    <a href="/aboutthefed/bios/board/default.htm">Board Members</a>
    <a href="https://www.facebook.com/federalreserve">Facebook</a>
  </nav>
  <div>
    <p><a href="/fomc/19950201default.htm">Statement</a></p>
    <p><a href="/fomc/MINUTES/1995/19950201min.htm">Minutes</a> (Released March 31, 1995)</p>
    <p><a href="{BASE}/monetarypolicy/files/FOMC19950201meeting.pdf">Transcript (10.12 MB PDF)</a></p>
    <p><a href="/monetarypolicy/files/fomc19950201gbpt119950126.pdf">1.82 MB PDF</a></p>
    <p><a href="/monetarypolicy/files/FOMC19950201agenda.pdf">Agenda (58 KB PDF)</a></p>
    <p><a href="/fomc/19950201default.htm">Statement</a></p>
  </div>
</body></html>
""".encode()


def test_parse_index_keeps_only_documents() -> None:
    docs = parse_index(PAGE)

    assert len(docs) == 4, [d.url for d in docs]
    assert {d.info_type for d in docs} == {
        FedInfoType.STATEMENT,
        FedInfoType.MINUTES,
        FedInfoType.TRANSCRIPT,
        FedInfoType.BRIEFING_BOOK,
    }


def test_parse_index_drops_the_agenda() -> None:
    """Agendas are date-bearing and deliberately excluded, so they must fall
    out as UNKNOWN rather than being collected."""
    assert not any("agenda" in d.url.lower() for d in parse_index(PAGE))


def test_parse_index_dedupes_repeated_links() -> None:
    """The statement is linked twice in PAGE, as it often is on real pages."""
    statements = [d for d in parse_index(PAGE) if d.info_type is FedInfoType.STATEMENT]

    assert len(statements) == 1


def test_parse_index_resolves_relative_hrefs() -> None:
    assert all(d.url.startswith(BASE) for d in parse_index(PAGE))


def test_parse_index_dates_from_the_url_not_the_prose() -> None:
    """The minutes link is followed by '(Released March 31, 1995)'. The
    meeting was February 1. Trusting the surrounding text would file the
    document under the wrong meeting -- silently, and only for minutes."""
    minutes = next(d for d in parse_index(PAGE) if d.info_type is FedInfoType.MINUTES)

    assert minutes.meeting_date == _date("1995-02-01")


def test_parse_index_sets_the_source() -> None:
    assert all(d.source is SourceSystem.FED_GOV for d in parse_index(PAGE))


def test_parse_index_derives_media_type_from_the_extension() -> None:
    """FedSourceDocument defaults media_type to application/pdf, but roughly
    half of these documents are HTML."""
    by_type = {d.url.rsplit(".", 1)[-1]: d.media_type for d in parse_index(PAGE)}

    assert by_type["pdf"] == "application/pdf"
    assert by_type["htm"] == "text/html"


def test_parse_index_strips_file_sizes_from_titles() -> None:
    transcript = next(d for d in parse_index(PAGE) if d.info_type is FedInfoType.TRANSCRIPT)

    assert transcript.title == "Transcript"


@pytest.mark.parametrize("label", ["HTML", "PDF", "pdf", "Accessible version"])
def test_parse_index_synthesises_a_title_when_the_text_is_only_a_format(label: str) -> None:
    """Where a page offers one document in two formats the anchor text is the
    format, not a title -- 11% of the corpus was stored as "HTML" or "PDF"."""
    page = f'<a href="/monetarypolicy/fomcminutes20150128.htm">{label}</a>'.encode()

    (doc,) = parse_index(page)

    assert doc.title == "Minutes 2015-01-28"


def test_parse_index_synthesises_a_title_when_the_text_is_only_a_size() -> None:
    """One real anchor reads '1.82 MB PDF' and nothing else. Stripping the
    size leaves an empty string, which is not a usable title."""
    book = next(d for d in parse_index(PAGE) if d.info_type is FedInfoType.BRIEFING_BOOK)

    assert book.title == "Briefing Book 1995-02-01"


def test_parse_index_on_an_empty_page() -> None:
    assert parse_index(b"<html></html>") == []


def test_parse_index_is_pure() -> None:
    """Same bytes in, same documents out. No hidden state between calls."""
    assert parse_index(PAGE) == parse_index(PAGE)


# --------------------------------------------------------------------------
# the dataclasses
# --------------------------------------------------------------------------


def test_classified_is_frozen() -> None:
    result = Classified(FedInfoType.MINUTES, _date("1995-02-01"), "https://x")

    with pytest.raises(AttributeError):
        result.info_type = FedInfoType.UNKNOWN  # type: ignore[misc]


def test_rule_provenance_is_not_part_of_equality() -> None:
    """`rule` is carried for diagnostics -- which pattern fired -- so two
    results describing the same document stay equal regardless."""
    rule = UrlRule(re.compile(r"(\d{8})"), FedInfoType.MINUTES)
    bare = Classified(FedInfoType.MINUTES, _date("1995-02-01"), "https://x")
    with_rule = Classified(FedInfoType.MINUTES, _date("1995-02-01"), "https://x", rule)

    assert bare == with_rule
