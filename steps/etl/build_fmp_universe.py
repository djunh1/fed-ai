from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.api.fmp_api import FmpEarningsApi


@step
def build_fmp_universe(
    min_price: float,
    min_market_cap: float,
    min_volume: int,
    exchanges: list[str],
) -> Annotated[list[str], "symbols"]:
    """Screened symbols, largest market cap first. One FMP call (ADR 0006)."""
    stats: dict = {}
    symbols = FmpEarningsApi().universe(
        min_price=min_price,
        min_market_cap=min_market_cap,
        min_volume=min_volume,
        exchanges=tuple(exchanges),
        stats=stats,
    )

    get_step_context().add_output_metadata(
        output_name="symbols",
        metadata={
            "query": {
                "min_price": min_price,
                "min_market_cap": min_market_cap,
                "min_volume": min_volume,
                "exchanges": exchanges,
            },
            "retrieved": {**stats, "top_10": symbols[:10]},
        },
    )
    return symbols
