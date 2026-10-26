import pandas as pd

from tools.viewer.server import MAX_LIMIT, query


def _frame():
    return pd.DataFrame(
        {
            "ticker": pd.array(["AAPL", "AAL", "MSFT", "MS", None], dtype="string"),
            "eps": [1.0, None, 3.0, 2.0, 5.0],
        }
    )


def test_no_filter_returns_all():
    page, total = query(_frame())
    assert total == 5 and len(page) == 5


def test_prefix_is_case_insensitive():
    page, total = query(_frame(), tickers="aa")
    assert total == 2
    assert set(page["ticker"]) == {"AAPL", "AAL"}


def test_several_tickers_by_comma_or_space():
    _, total = query(_frame(), tickers="aapl, ms")
    assert total == 3
    _, total = query(_frame(), tickers="AAPL MSFT")
    assert total == 2


def test_null_ticker_never_matches():
    _, total = query(_frame(), tickers="A")
    assert total == 2


def test_sort_puts_nulls_last_both_ways():
    page, _ = query(_frame(), sort="eps", desc=True)
    assert list(page["eps"][:4]) == [5.0, 3.0, 2.0, 1.0]
    page, _ = query(_frame(), sort="eps", desc=False)
    assert list(page["eps"][:4]) == [1.0, 2.0, 3.0, 5.0]
    assert pd.isna(page["eps"].iloc[-1])


def test_unknown_sort_column_ignored():
    page, _ = query(_frame(), sort="nope")
    assert list(page["ticker"][:2]) == ["AAPL", "AAL"]


def test_paging_and_total_independent():
    page, total = query(_frame(), offset=3, limit=10)
    assert total == 5 and len(page) == 2


def test_limit_clamped():
    _, _ = query(_frame(), limit=10**9)
    page, _ = query(pd.DataFrame({"ticker": ["A"] * (MAX_LIMIT + 5)}), limit=10**9)
    assert len(page) == MAX_LIMIT


def test_newest_snapshot_by_name_ignores_tmp_and_strays(tmp_path):
    from tools.viewer.server import newest_snapshot

    for name in [
        "earnings_20260101_000000.parquet",
        "earnings_20260301_000000.parquet",
        "earnings_20260401_000000.parquet.tmp",
        "earnings_20260501_000000_copy.parquet",
        "enriched_20260601_000000.parquet",
    ]:
        (tmp_path / name).touch()
    assert newest_snapshot(tmp_path, "earnings_").name == "earnings_20260301_000000.parquet"
    assert newest_snapshot(tmp_path / "missing", "earnings_") is None


def _earnings():
    return pd.DataFrame(
        {
            "ticker": pd.array(["AAPL", "AAPL", "JPM"], dtype="string"),
            "date": pd.to_datetime(["2015-01-27", "2024-05-02", "2024-04-12"]),
            "time": pd.array(["amc", "amc", None], dtype="string"),
            "eps_actual": [3.06, 1.53, 4.44],
        }
    )


def test_with_prices_left_joins_on_key_and_orders_columns():
    from tools.viewer.server import KEY, PRICE_COLUMNS, with_prices

    enriched = pd.DataFrame(
        {
            "ticker": pd.array(["AAPL", "JPM"], dtype="string"),
            "date": pd.to_datetime(["2024-05-02", "2024-04-12"]),
            "time": pd.array(["amc", None], dtype="string"),
            "date_0d": pd.to_datetime(["2024-05-03", "2024-04-15"]),
            "open_0d": [186.65, 191.0],
            "10_d": [1.0, 2.0],
            "20_d": [3.0, 4.0],
            "60_d": [5.0, 6.0],
            "volume": [1.0, 2.0],
        }
    )
    out = with_prices(_earnings(), enriched)
    assert list(out.columns) == KEY + PRICE_COLUMNS + ["eps_actual"]
    assert len(out) == 3
    assert pd.isna(out.loc[0, "date_0d"])  # pre-2016, not enriched
    assert out.loc[1, "date_0d"] == pd.Timestamp("2024-05-03")
    assert out.loc[2, "10_d"] == 2.0  # null time still joins
    assert out["day_0_check"].isna().all()  # older enriched file without the column


def test_with_prices_without_enriched_adds_null_columns():
    from tools.viewer.server import PRICE_COLUMNS, with_prices

    out = with_prices(_earnings(), None)
    assert out[PRICE_COLUMNS].isna().all().all()
    assert len(out) == 3
