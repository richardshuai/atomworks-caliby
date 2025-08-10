from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import typer
from tqdm import tqdm

from .pdb import _collect_pdb_ids, _pdb_id_to_relpath, _rsync_fetch_specific, _run_rsync_list

TEST_PACK_URL = "https://files.ipd.uw.edu/pub/atomworks/test_pack_latest.tar.gz"
"""The URL for the latest AtomWorks test pack. Should be untared in `tests/data/shared`."""

app = typer.Typer(help="Setup utilities for AtomWorks")


def _download_file(url: str, dest_path: Path) -> None:
    """Download a file from URL to dest_path, showing a progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
        total_str = response.headers.get("Content-Length") or response.headers.get("content-length")
        total = int(total_str) if total_str and total_str.isdigit() else None

        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading test pack",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            chunk_size = 1024 * 64  # 64 KiB
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))


def _extract_tar_gz(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz archive into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir, filter="data")


def _find_missing_mmCIFs(pdb_ids: Iterable[str], mirror_root: Path) -> list[str]:
    """Return pdb_ids whose corresponding mmCIF files are missing under mirror_root."""
    missing: list[str] = []
    for pid in pdb_ids:
        rel = _pdb_id_to_relpath(pid)
        cif_path = mirror_root / rel
        if not cif_path.is_file():
            missing.append(pid)
    return missing


@app.command("tests")
def setup_tests(
    tests_shared_dir: Path = typer.Option(
        Path("tests/data/shared"), "--tests-shared-dir", help="Where to extract the test pack."
    ),
    pdb_mirror_path: Path | None = typer.Option(
        None,
        "--pdb-mirror-path",
        help="Root of the RCSB divided mmCIF mirror. Defaults to $PDB_MIRROR_PATH if unset.",
    ),
    remote: str = typer.Option(
        "rsync.wwpdb.org::ftp/data/structures/divided/mmCIF/",
        "--remote",
        help="Rsync remote base path for divided mmCIF tree.",
    ),
    port: int = typer.Option(33444, "--port", help="Rsync server port for RCSB."),
    keep_archive: bool = typer.Option(False, "--keep-archive", help="Keep downloaded test pack archive."),
) -> None:
    """Download the test pack and ensure required PDB mmCIFs are present in the mirror.

    Steps:
    1) Download and extract the AtomWorks test pack into `tests/data/shared` (by default).
    2) Read `test_pdb_ids.txt` from the test pack.
    3) Download any missing PDB mmCIFs listed there into the given PDB mirror path using rsync.

    Example:
        atomworks setup tests --pdb-mirror-path /data/rcsb/mmcif
    """
    typer.echo("Setting up AtomWorks test environment...")

    # Resolve PDB mirror path
    if pdb_mirror_path is None:
        env_path = os.getenv("PDB_MIRROR_PATH")
        if env_path:
            pdb_mirror_path = Path(env_path)
        else:
            typer.secho(
                "No PDB mirror path provided and $PDB_MIRROR_PATH is not set. "
                "Use --pdb-mirror-path or set the environment variable.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
    pdb_mirror_path.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Using PDB mirror path: {pdb_mirror_path}")

    # Download and extract test pack
    tests_shared_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "test_pack_latest.tar.gz"
        typer.echo(f"Downloading test pack from {TEST_PACK_URL} ...")
        _download_file(TEST_PACK_URL, archive_path)
        typer.secho("Download complete", fg=typer.colors.GREEN)

        typer.echo(f"Extracting test pack into {tests_shared_dir} ...")
        _extract_tar_gz(archive_path, tests_shared_dir)
        typer.secho("Extraction complete", fg=typer.colors.GREEN)

        if keep_archive:
            keep_path = tests_shared_dir / archive_path.name
            keep_path.write_bytes(archive_path.read_bytes())

    # Read PDB IDs from the test pack
    ids_file = tests_shared_dir / "test_pdb_ids.txt"
    if not ids_file.is_file():
        typer.secho(f"Missing PDB id list: {ids_file}", fg=typer.colors.RED)
        raise typer.Exit(code=3)
    typer.echo(f"Reading PDB ids from {ids_file}")
    ids = _collect_pdb_ids(None, ids_file)
    typer.echo(f"Found {len(ids)} unique PDB ids in the test pack list")

    # Determine missing mmCIFs
    typer.echo(f"Checking for missing PDB mmCIFs at {pdb_mirror_path} ...")
    missing = _find_missing_mmCIFs(ids, pdb_mirror_path)
    if missing:
        typer.echo(f"{len(missing)} PDB ids are missing from the mirror; testing rsync connectivity...")
        ok, output = _run_rsync_list(remote, port)
        if not ok:
            typer.secho("Connection test failed", fg=typer.colors.RED)
            typer.echo("Error output:")
            typer.echo(output)
            raise typer.Exit(code=4)
        typer.secho("Connection test successful", fg=typer.colors.GREEN)

        typer.echo("Fetching missing PDB mmCIFs via rsync...")
        _rsync_fetch_specific(remote, pdb_mirror_path, missing, port)
        typer.secho("PDB sync complete", fg=typer.colors.GREEN)
    else:
        typer.secho("All required PDB mmCIFs are already present", fg=typer.colors.GREEN)

    typer.secho("Test setup completed successfully!", fg=typer.colors.GREEN)
