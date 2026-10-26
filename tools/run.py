from datetime import datetime as dt
from pathlib import Path

import click
from loguru import logger

from llm_engineering import settings
from pipelines import fed_data_etl, fmp_earnings_etl, feature_engineering_fed


@click.command(
    help="""
LLM Fed project CLI v0.0.1.

Main entry point for pipeline execution. Runs a pipeline with the orchestrator
stack component configured in your active ZenML stack.

Examples:

  \b
  # Run the ETL pipeline with the default config
  python -m tools.run --run-etl

  \b
  # Run it without caching, over a different year range
  python -m tools.run --run-etl --no-cache --etl-config-filename fed_data_etl.yaml

  \b
  # FMP earnings to parquet (incremental: add --etl-config-filename fmp_earnings_etl_incremental.yaml)
  python -m tools.run --run-fmp-earnings-etl --no-cache

  \b
  # Push local .env settings into the ZenML secret store
  python -m tools.run --export-settings
"""
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--run-etl",
    is_flag=True,
    default=False,
    help="Whether to run the fed.gov ETL pipeline.",
)
@click.option(
    "--run-fmp-earnings-etl",
    is_flag=True,
    default=False,
    help="Whether to run the FMP earnings ETL pipeline.",
)
@click.option(
    "--run-feature-engineering",
    is_flag=True,
    default=False,
    help="Whether to run the Fed feature engineering pipeline.",
)
@click.option(
    "--etl-config-filename",
    default=None,
    help="Config in configs/. Defaults to the chosen pipeline's own.",
)
@click.option(
    "--export-settings",
    is_flag=True,
    default=False,
    help="Whether to export your settings to ZenML or not.",
)
def main(
    no_cache: bool = False,
    run_etl: bool = False,
    run_fmp_earnings_etl: bool = False,
    etl_config_filename: str | None = None,
    run_feature_engineering: bool = False,
    export_settings: bool = False,
) -> None:
    assert run_etl or run_fmp_earnings_etl or run_feature_engineering or export_settings, "Please specify an action to run."
    assert sum([run_etl, run_fmp_earnings_etl, run_feature_engineering]) <= 1, "Run one pipeline at a time."

    if export_settings:
        logger.info("Exporting settings to ZenML secrets.")
        settings.export()

    pipeline_args = {
        "enable_cache": not no_cache,
    }
    root_dir = Path(__file__).resolve().parent.parent

    if run_etl:
        _run(fed_data_etl, "fed_data_etl", etl_config_filename or "fed_data_etl.yaml", root_dir, pipeline_args)

    if run_fmp_earnings_etl:
        _run(fmp_earnings_etl, "fmp_earnings_etl", etl_config_filename or "fmp_earnings_etl.yaml", root_dir, pipeline_args)

    if run_feature_engineering:
        _run(feature_engineering_fed, "feature_engineering_fed", etl_config_filename or "feature_engineering_fed.yaml", root_dir, pipeline_args)


def _run(pipeline, name: str, config_filename: str, root_dir: Path, pipeline_args: dict) -> None:
    config_path = root_dir / "configs" / config_filename
    assert config_path.exists(), f"Config file not found: {config_path}"
    run_name = f"{name}_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    pipeline.with_options(**pipeline_args, config_path=config_path, run_name=run_name)()


if __name__ == "__main__":
    main()
