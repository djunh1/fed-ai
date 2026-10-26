from zenml import pipeline

from steps.etl import build_fmp_universe, enrich_fmp_price_action, extract_fmp_earnings


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
    symbols: list[str] | None = None,
    enriched_dir: str = "data/fmp_enriched",
    price_action_from: str = "2016-01-01",
    price_action_to: str | None = None,
) -> str:
    """Screen the universe, fetch earnings, then add price action. Parquet only (ADR 0001, 0008)."""
    universe = build_fmp_universe(
        min_price=min_price,
        min_market_cap=min_market_cap,
        min_volume=min_volume,
        exchanges=exchanges,
        symbols=symbols,
    )
    snapshot_path = extract_fmp_earnings(
        symbols=universe,
        data_dir=data_dir,
        incremental=incremental,
        stale_after_days=stale_after_days,
        limit=limit,
        max_failure_pct=max_failure_pct,
    )
    last_step = enrich_fmp_price_action(
        snapshot_path=snapshot_path,
        enriched_dir=enriched_dir,
        incremental=incremental,
        price_action_from=price_action_from,
        price_action_to=price_action_to,
        limit=limit,
        max_failure_pct=max_failure_pct,
    )
    return last_step.invocation_id
