"""Command line interface for managing a local CCD mirror."""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
from pathlib import Path

import typer


def _run_rsync_list(remote_path: str) -> tuple[bool, str]:
    """Try to list a remote rsync path and return success and output/error.

    Args:
        remote_path: The remote rsync path to list.

    Returns:
        A tuple of (success, output_or_error).
    """
    try:
        completed = subprocess.run(
            ["rsync", "--list-only", remote_path],
            check=False,
            capture_output=True,
            text=True,
        )
        success = completed.returncode == 0
        output = completed.stdout if success else completed.stderr
        return success, output
    except FileNotFoundError:
        return False, "rsync executable not found. Please install rsync."


def _rsync_sync(remote_path: str, dest_path: Path) -> None:
    """Synchronize CCD CIF files from rsync remote to local destination.

    Mirrors the shell behavior: recursive, times preserved where possible, include only directories and .cif files.
    """
    cmd = [
        "rsync",
        "-rltvz",
        "--stats",
        "--no-perms",
        "--chmod=ug=rwX,o=rX",
        "--delete",
        "--omit-dir-times",
        "--include=*/",
        "--include",
        "*.cif",
        "--exclude",
        "*",
        remote_path,
        str(dest_path),
    ]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"rsync failed with exit code {completed.returncode}")


def _download_components_cif_gz(dest_path: Path) -> Path:
    """Download components.cif.gz to dest and return path.

    Uses urllib to avoid external wget dependency.
    """
    import urllib.request

    url = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
    out_path = dest_path / "components.cif.gz"
    with urllib.request.urlopen(url) as response, open(out_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    return out_path


def _gunzip_inplace(gz_path: Path) -> Path:
    """Gunzip file in place, returning the decompressed file path."""
    out_path = gz_path.with_suffix("")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink(missing_ok=True)
    return out_path


app = typer.Typer(help="PDBeChem CCD utilities")


@app.command("sync")
def sync_ccd(
    destination_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
        help="Destination directory to mirror CCD into.",
    ),
    remote: str = typer.Option(
        "rsync://ftp.ebi.ac.uk/pub/databases/msd/pdbechem_v2/ccd/",
        "--remote",
        help="Rsync remote path for CCD.",
    ),
    write_env_hint: bool = typer.Option(
        True,
        "--write-env-hint/--no-write-env-hint",
        help="Print a hint for setting CCD_MIRROR_PATH in your environment.",
    ),
) -> None:
    """Mirror the PDBeChem CCD to a local directory.

    This replicates the behavior of scripts/setup_ccd_mirror.sh using Python-only dependencies.
    """
    typer.echo("Setting up CCD mirror...")
    destination_path.mkdir(parents=True, exist_ok=True)

    typer.echo("Testing rsync connection to EBI server...")
    ok, output = _run_rsync_list(remote)
    if not ok:
        typer.secho("Connection test failed", fg=typer.colors.RED)
        typer.echo("Error output:")
        typer.echo(output)
        raise typer.Exit(code=1)
    typer.secho("Connection test successful", fg=typer.colors.GREEN)

    typer.echo("Syncing files from PDBeChem CCD (this may take a few minutes)...")
    _rsync_sync(remote, destination_path)
    typer.secho("Sync completed successfully!", fg=typer.colors.GREEN)

    # Download and uncompress components.cif.gz
    gz_path = _download_components_cif_gz(destination_path)
    _gunzip_inplace(gz_path)

    # Write README
    readme_path = destination_path / "README"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("# CCD Mirror Information\n\n")
        from datetime import datetime

        fh.write(f"Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Sync command: atomworks sync_ccd {destination_path}\n")
        fh.write(f"Sync user: {os.getenv('USER') or os.getenv('USERNAME') or 'unknown'}\n")

    if write_env_hint:
        typer.echo("")
        typer.echo(f"Set 'CCD_MIRROR_PATH={destination_path}' in your .env file")
        typer.echo(f"  ... or run 'export CCD_MIRROR_PATH={destination_path}' in your shell")
        typer.echo(f"  ... or add 'export CCD_MIRROR_PATH={destination_path}' to your shell profile.")
        typer.echo("")
