"""MSA filtering command using HHfilter."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from atomworks.enums import MSAFileExtension
from atomworks.ml.preprocessing.msa.filtering import HHFilterConfig, MSAFilterConfig, filter_msas

from .common import enable_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def filter(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Source directory containing MSA files to filter",
    ),
    output_dir: Path = typer.Argument(
        None,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        help="Destination directory for filtered files. If not provided uses the input paths with the specified output extension.",
    ),
    input_extension: str = typer.Option(
        MSAFileExtension.A3M_GZ.value,
        "--input-extension",
        "-i",
        help="File extension for input MSA files (e.g., .a3m, .a3m.gz, .afa, .afa.gz)",
    ),
    output_extension: str = typer.Option(
        MSAFileExtension.A3M_GZ.value,
        "--output-extension",
        "-o",
        help="File extension for output MSA files (e.g., .a3m, .a3m.gz, .afa, .afa.gz)",
    ),
    max_sequences: int = typer.Option(
        10_000,
        "--max-sequences",
        "--maxseq",
        help="Maximum number of sequences to keep in each MSA",
    ),
    max_identity: float = typer.Option(
        90.0,
        "--max-identity",
        "--id",
        help="Maximum pairwise sequence identity (%)",
    ),
    min_coverage: float = typer.Option(
        50.0,
        "--min-coverage",
        "--cov",
        help="Minimum coverage with query (%)",
    ),
    num_workers: int | None = typer.Option(
        None,
        "--num-workers",
        "-j",
        help="Number of parallel workers (defaults to min(CPU_COUNT, 16))",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Filter MSA files using HHfilter to reduce sequence count and redundancy.

    This command applies HHfilter to MSA files to reduce their size by:
    1. Limiting the maximum number of sequences
    2. Filtering by maximum pairwise sequence identity
    3. Filtering by minimum coverage with the query sequence

    Can be applied to organized MSA files or any directory of MSA files.
    Automatic compression/decompression is applied based on the input and output file extensions.

    Examples:
        # Filter files in a separate output directory
        atomworks msa filter ./msas ./filtered_msas --max-sequences 1000

        # Convert uncompressed MSA files to compressed ones while filtering
        atomworks msa filter ./msas ./filtered_msas --input-extension .a3m --output-extension .a3m.gz --max-sequences 1000
    """
    hhfilter_config = HHFilterConfig(
        max_sequences=max_sequences,
        max_identity_percent=max_identity,
        min_coverage_percent=min_coverage,
    )

    config = MSAFilterConfig(
        input_extension=input_extension,
        output_extension=output_extension,
        hhfilter=hhfilter_config,
        num_workers=num_workers,
    )

    # Display configuration
    typer.secho("MSA Filtering Configuration:", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Input Directory: {input_dir}")
    typer.echo(f"  Input File Extension: {config.input_extension}")
    typer.echo(f"  Output File Extension: {config.output_extension}")
    typer.echo(f"  Number of Workers: {config.num_workers}")
    typer.secho("  HHfilter Configuration:", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"    Max Sequences: {config.hhfilter.max_sequences}")
    typer.echo(f"    Max Identity %: {config.hhfilter.max_identity_percent}")
    typer.echo(f"    Min Coverage %: {config.hhfilter.min_coverage_percent}")

    enable_logging(verbose)

    # Run the filtering process
    try:
        typer.echo("Starting MSA filtering process...")
        logger.info(f"Beginning MSA filtering from {input_dir}")
        filter_msas(input_dir=input_dir, output_dir=output_dir, config=config)
        logger.info("MSA filtering completed successfully")
        typer.secho("MSA filtering completed successfully!", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error during MSA filtering: {e!s}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
