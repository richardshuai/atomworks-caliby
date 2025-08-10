from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

import typer


PDB_ID_REGEX = re.compile(r"^[0-9a-zA-Z]{4}$")


def _normalize_pdb_id(pdb_id: str) -> str:
    """Return a normalized, lower-case 4-char PDB id or raise ValueError."""
    pdb_id = pdb_id.strip().lower()
    if not PDB_ID_REGEX.match(pdb_id):
        raise ValueError(f"Invalid PDB id: {pdb_id}")
    return pdb_id


def _pdb_id_to_relpath(pdb_id: str) -> Path:
    """Map a PDB id to its relative mmCIF path under the divided layout.

    Example: '1a0i' -> 'a0/1a0i.cif.gz'
    """
    pid = _normalize_pdb_id(pdb_id)
    subdir = pid[1:3]
    return Path(subdir) / f"{pid}.cif.gz"


def _run_rsync_list(remote_path: str, port: int | None) -> tuple[bool, str]:
    """Try to list a remote rsync path and return success and output/error."""
    cmd = ["rsync", "--list-only"]
    if port is not None:
        cmd.extend(["--port", str(port)])
    cmd.append(remote_path)
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        success = completed.returncode == 0
        output = completed.stdout if success else completed.stderr
        return success, output
    except FileNotFoundError:
        return False, "rsync executable not found. Please install rsync."


def _rsync_sync_full(remote_path: str, dest_path: Path, port: int | None) -> None:
    """Perform a full mirror of the mmCIF divided tree into dest_path."""
    cmd = [
        "rsync",
        "-rltvz",
        "--stats",
        "--no-perms",
        "--chmod=ug=rwX,o=rX",
        "--delete",
        "--omit-dir-times",
    ]
    if port is not None:
        cmd.extend(["--port", str(port)])
    cmd.extend([remote_path, str(dest_path)])
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"rsync full sync failed with exit code {completed.returncode}")


def _rsync_fetch_specific(remote_base: str, dest_path: Path, pdb_ids: Iterable[str], port: int | None) -> None:
    """Fetch only specific PDB ids by rsync-ing their individual files.

    Creates subdirectories under dest_path as needed and transfers each file.
    """
    for pdb_id in pdb_ids:
        rel = _pdb_id_to_relpath(pdb_id)
        src = f"{remote_base}{rel.as_posix()}"
        dest_subdir = dest_path / rel.parent
        dest_subdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync",
            "-rltvz",
            "--stats",
            "--no-perms",
            "--chmod=ug=rwX,o=rX",
        ]
        if port is not None:
            cmd.extend(["--port", str(port)])
        cmd.extend([src, str(dest_subdir)])
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"rsync for {pdb_id} failed with exit code {completed.returncode}")


def _collect_pdb_ids(pdb_ids: list[str] | None, pdb_ids_file: Path | None) -> list[str]:
    """Combine ids from CLI list and an optional file; return normalized unique ids."""
    collected: list[str] = []
    if pdb_ids:
        collected.extend(pdb_ids)
    if pdb_ids_file:
        with open(pdb_ids_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                collected.append(line)
    normalized: list[str] = []
    seen: set[str] = set()
    for pid in collected:
        try:
            norm = _normalize_pdb_id(pid)
        except ValueError:
            continue
        if norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


app = typer.Typer(help="RCSB PDB mmCIF mirror utilities")


@app.command("sync")
def sync_pdb(
    destination_path: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
        help="Destination directory to mirror PDB mmCIFs into.",
    ),
    remote: str = typer.Option(
        "rsync.wwpdb.org::ftp/data/structures/divided/mmCIF/",
        "--remote",
        help="Rsync remote base path for divided mmCIF tree.",
    ),
    port: int = typer.Option(33444, "--port", help="Rsync server port for RCSB."),
    pdb_id: list[str] = typer.Option(
        None,
        "--pdb-id",
        help="PDB id to sync (may be used multiple times). If omitted and no file is provided, full sync is done.",
    ),
    pdb_ids_file: Path | None = typer.Option(
        None,
        "--pdb-ids-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a file containing one PDB id per line.",
    ),
    write_env_hint: bool = typer.Option(
        True,
        "--write-env-hint/--no-write-env-hint",
        help="Print a hint for setting PDB_MIRROR_PATH in your environment.",
    ),
) -> None:
    """Mirror the RCSB PDB mmCIF archive fully or for specific PDB ids.

    If no ids are provided via --pdb-id or --pdb-ids-file, a full mirror is performed.
    As of August 2025, a full RCSB PDB mirror requires about 100 GB of disk space.
    """
    typer.echo("Setting up RCSB PDB mirror...")
    destination_path.mkdir(parents=True, exist_ok=True)

    ids = _collect_pdb_ids(pdb_id, pdb_ids_file)

    # Test rsync connection
    typer.echo("Testing rsync connection to RCSB server...")
    ok, output = _run_rsync_list(remote, port)
    if not ok:
        typer.secho("Connection test failed", fg=typer.colors.RED)
        typer.echo("Error output:")
        typer.echo(output)
        raise typer.Exit(code=1)
    typer.secho("Connection test successful", fg=typer.colors.GREEN)

    if ids:
        typer.echo(f"Syncing {len(ids)} selected PDB ids...")
        _rsync_fetch_specific(remote, destination_path, ids, port)
    else:
        typer.echo("Syncing full RCSB PDB mmCIF tree (this may take a while and requires about 100 GB of disk space)...")
        _rsync_sync_full(remote, destination_path, port)
    typer.secho("Sync completed successfully!", fg=typer.colors.GREEN)

    # Write README
    readme_path = destination_path / "README"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("# RCSB PDB Mirror Information\n\n")
        from datetime import datetime

        fh.write(f"Last sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        cmd_repr = ["atomworks", "sync_pdb", str(destination_path)]
        if ids:
            cmd_repr.extend(sum((["--pdb-id", i] for i in ids), []))
        fh.write(f"Sync command: {' '.join(cmd_repr)}\n")
        fh.write(f"Sync user: {os.getenv('USER') or os.getenv('USERNAME') or 'unknown'}\n")

    if write_env_hint:
        typer.echo("")
        typer.echo(f"Set 'PDB_MIRROR_PATH={destination_path}' in your .env file")
        typer.echo(f"  ... or run 'export PDB_MIRROR_PATH={destination_path}' in your shell")
        typer.echo(f"  ... or add 'export PDB_MIRROR_PATH={destination_path}' to your shell profile.")
        typer.echo("")


