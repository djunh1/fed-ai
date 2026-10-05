from zenml import pipeline

from steps.etl import build_fed_manifest, extract_fed_text


@pipeline
def fed_data_etl(
    sources: list[str],
    cache_dir: str | None = None,
    force: bool = False,
    incremental: bool = False,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int | None = None,
) -> str:
    """Enumerate the FOMC corpus, then extract text for a bounded slice of it.

    One pipeline, two corpora. `sources` selects which, so fed.gov and FRASER
    run as separate invocations against separate configs -- separate
    schedules, separate failure domains, no shared blast radius -- while
    writing to the same two collections, distinguished by `source`. They are
    never merged: the sources differ in granularity (every document per
    meeting vs one representative file), so there is no safe dedupe key.

    The two steps are chained by data dependency: extract_fed_text consumes
    the manifest step's URL list, so ZenML orders them without an explicit
    `after=`. Both are idempotent, so re-running is the resume path.
    """
    document_urls = build_fed_manifest(
        sources=sources, cache_dir=cache_dir, force=force, incremental=incremental
    )
    last_step = extract_fed_text(
        document_urls=document_urls,
        sources=sources,
        year_from=year_from,
        year_to=year_to,
        cache_dir=cache_dir,
        limit=limit,
    )

    return last_step.invocation_id
