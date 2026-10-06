# fed-ai

Ingests the public record of FOMC policy deliberation — minutes, statements,
transcripts, memoranda, briefing books and Beige Books — as the raw corpus for a
downstream RAG and fine-tuning pipeline.

Two sources, ingested separately and never merged:

| Source | Documents | Notes |
|---|---|---|
| federalreserve.gov | ~5,200 | complete modern corpus, 1936→present, no API key |
| FRASER (St. Louis Fed) | 581 | reaches back to 1933, pre-extracted text, needs a key |

Rows in Mongo carry a `source` field, so precedence between the two is a
retrieval-time decision rather than something baked into ingestion.

## Local services

Start everything:

```bash
uv run poe local-docker-infrastructure-up   # Mongo, mongo-express, Qdrant
uv run poe local-zenml-server-up            # ZenML dashboard
```

| Service | URL | Credentials |
|---|---|---|
| ZenML dashboard | http://127.0.0.1:8237 | `default` user, no password |
| mongo-express | http://localhost:8081 | none (`ME_CONFIG_BASICAUTH: "false"`) |
| MongoDB | `mongodb://llm_engineering:llm_engineering@127.0.0.1:27017` | `llm_engineering` / `llm_engineering` |
| Qdrant | http://localhost:6333/dashboard | none |

Collections written by the ETL, both browsable in mongo-express:

- **`fed_documents`** — one row per document URL: source, type, meeting date, title, media type.
- **`fed_text`** — one row per extracted document: text, char count, per-page char counts.

Split in two because `NoSQLBaseDocument` has no upsert — both steps only ever
insert — and because it keeps multi-megabyte transcripts out of the metadata
collection.

Stop with `uv run poe local-docker-infrastructure-down`.

## Running the ETL

```bash
uv run poe run-fed-data-etl          # fed.gov, year range from the config
uv run poe run-fed-data-etl-update   # monthly update: 2 fetches instead of 86
uv run poe run-fraser-data-etl       # FRASER, needs FED_API_KEY
```

Bound the expensive half with `year_from` / `year_to` in
`configs/fed_data_etl.yaml`. The full fed.gov backfill is 15–40 GB and 1–2
hours; a single year is seconds once `temp/http_cache` is warm.

Every run is idempotent — resumption is just re-running. Both steps skip what
is already stored, so a run killed at document 3,000 picks up at 3,000.

## FMP earnings to parquet

A separate pipeline, `fmp_earnings_etl`, collects historical earnings
announcements for screened US stocks from Financial Modeling Prep and writes
one complete parquet snapshot per run. It does not use Mongo. The consumer is
`~/dev/sip_news_data_platform`, which reads the newest file.

```bash
uv run poe run-fmp-earnings-etl          # full run, ~2,000 symbols, ~17 min
uv run poe run-fmp-earnings-etl-update   # refresh new, stale and unresolved symbols only
```

Needs `FMP_API_KEY` in `.env` (Starter plan or above).

- **Output:** `data/fmp_earnings/earnings_<UTC YYYYMMDD_HHMMSS>.parquet`, one row
  per (ticker, date), the sister project's 15 columns plus `ingested_at`.
  The newest file by name is the current dataset; failed runs write nothing.
- **Universe:** the FMP screener with the sister project's floors (price $5,
  market cap $300M, volume 200k). Change them per run in
  `configs/fmp_earnings_etl.yaml`.
- **Smoke run:** set `limit: 5` in a copy of the config; it takes the largest
  market caps first.
- **Run stats:** each step attaches metadata to its output. In the dashboard,
  open the run, click an output node, then the Metadata tab. From Python:
  `uv run python temp/debug_fmp_metadata.py`.

Design decisions are in `docs/adr/` (local-only).

## Looking at the data

```bash
uv run python temp/debug_corpus.py          # or F5 → "Debug: browse the corpus"
```

Read-only. Also importable:

```python
from temp.debug_corpus import *
summary()                  # counts by source and type
coverage()                 # extraction progress by decade
show(year=2015)            # list the manifest rows
read("minutes", 1933)      # print a document, wrapped for reading
quality()                  # scan for nav chrome and OCR damage
```

`temp/reset_corpus.py` clears the collections when a classification rule or an
extraction container changes and the old rows need rebuilding. It asks first.

`.vscode/launch.json` has a debug config per module.

## FRASER API key

Free, instant, no account:

```bash
curl --data '{"email": "you@example.com", "description": "fed-ai"}' \
  https://fraser.stlouisfed.org/api/api_key
```

Put it in `.env` as `FED_API_KEY`. Only `/api/*` is metered, at 30 req/min; the
file hosts serving PDFs and extracted text are unauthenticated, so document
fetches never send the key.

## Tests

```bash
uv run pytest tests/ -q
```
