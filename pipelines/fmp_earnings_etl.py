from zenml import pipeline

from steps.etl import build_fmp_universe, extract_fmp_earnings


@pipeline
def fmp_earnings_etl(
    min_price: float = 5.0,
    min_market_cap: float = 300_000_000,
    min_volume: int = 200_000,
    exchanges: list[str] = ["NYSE", "NASDAQ", "AMEX"],  # noqa: B006 (ZenML params must be JSON)
    data_dir: str = "data/fmp_earnings",
    incremental: bool = False,
    stale_after_days: int = 75,
    limit: int | None = None,
    max_failure_pct: float = 5.0,
) -> str:
    """Screen the universe, then fetch earnings into a parquet snapshot. No Mongo (ADR 0001)."""
    symbols = build_fmp_universe(
        min_price=min_price, min_market_cap=min_market_cap, min_volume=min_volume, exchanges=exchanges
    )
    last_step = extract_fmp_earnings(
        symbols=symbols,
        data_dir=data_dir,
        incremental=incremental,
        stale_after_days=stale_after_days,
        limit=limit,
        max_failure_pct=max_failure_pct,
    )
    return last_step.invocation_id
