from datetime import datetime as dt
from pathlib import Path

import click
from loguru import logger

from llm_engineering import settings
from pipelines import fed_data_etl


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
    "--etl-config-filename",
    default="fed_data_etl.yaml",
    help="Filename of the ETL config file in configs/.",
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
    etl_config_filename: str = "fed_data_etl.yaml",
    export_settings: bool = False,
) -> None:
    assert run_etl or export_settings, "Please specify an action to run."

    if export_settings:
        logger.info("Exporting settings to ZenML secrets.")
        settings.export()

    pipeline_args = {
        "enable_cache": not no_cache,
    }
    root_dir = Path(__file__).resolve().parent.parent

    if run_etl:
        run_args_etl = {}
        pipeline_args["config_path"] = root_dir / "configs" / etl_config_filename
        assert pipeline_args[
            "config_path"
        ].exists(), f"Config file not found: {pipeline_args['config_path']}"
        pipeline_args["run_name"] = (
            f"fed_data_etl_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        fed_data_etl.with_options(**pipeline_args)(**run_args_etl)


if __name__ == "__main__":
    main()
