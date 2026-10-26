from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step

from llm_engineering.application.api.fmp_api import FmpEarningsApi


@step
def build_fmp_universe(
    min_price: float,
    min_market_cap: float,
    min_volume: int,
    exchanges: list[str],
    symbols: list[str] | None = None,
) -> Annotated[list[str], ArtifactConfig(name="fmp_universe", tags=["fmp", "universe"])]:
    """Screened symbols, largest market cap first. One FMP call (ADR 0006).

    `symbols`, if set, replaces the screen (QA runs, re-running named tickers).
    """
    if symbols:
        chosen = list(dict.fromkeys(s.strip().upper() for s in symbols))
        get_step_context().add_output_metadata(
            output_name="fmp_universe",
            metadata={"query": {"source": "config symbols"}, "retrieved": {"symbols": len(chosen), "top_10": chosen[:10]}},
        )
        return chosen

    stats: dict = {}
    symbols = FmpEarningsApi().universe(
        min_price=min_price,
        min_market_cap=min_market_cap,
        min_volume=min_volume,
        exchanges=tuple(exchanges),
        stats=stats,
    )

    get_step_context().add_output_metadata(
        output_name="fmp_universe",
        metadata={
            "query": {
                "source": "screener",
                "min_price": min_price,
                "min_market_cap": min_market_cap,
                "min_volume": min_volume,
                "exchanges": exchanges,
            },
            "retrieved": {**stats, "top_10": symbols[:10]},
        },
    )
    return symbols
