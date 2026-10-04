from zenml import pipeline

from steps.etl import build_fed_manifest, extract_fed_text


@pipeline
def fed_data_etl(
    cache_dir: str | None = None,
    force: bool = False,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int | None = None,
) -> str:
    """Enumerate the FOMC corpus, then extract text for a bounded slice of it.

    The two steps are chained by data dependency: extract_fed_text consumes
    the manifest step's URL list, so ZenML orders them without an explicit
    `after=`. Both are idempotent, so re-running is the resume path.
    """
    document_urls = build_fed_manifest(cache_dir=cache_dir, force=force)
    last_step = extract_fed_text(
        document_urls=document_urls,
        year_from=year_from,
        year_to=year_to,
        cache_dir=cache_dir,
        limit=limit,
    )

    return last_step.invocation_id
