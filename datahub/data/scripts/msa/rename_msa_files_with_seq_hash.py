"""
Script to process and rename sequence files with hashed filenames, using SHA-256 cryptographic hashes of the one-letter sequences.

This script reads sequence files (.a3m.gz, .a3m, or .afa) from a source directory, computes a hashed filename for each sequence, and saves the files in a structured format in the destination directory.

Usage:
    python rename_msa_files_with_seq_hash.py --src_dir <source_directory> --dest_dir <destination_directory> --file_extension <.a3m.gz or .afa> --num_workers <number_of_workers>

Example (takes about ~1 hour for 100k protein MSA files; much less time for RNA files):
    python rename_msa_files_with_seq_hash.py --src_dir /projects/ml/TrRosetta/PDB-2021AUG02/a3m --dest_dir /projects/ml/RF2_allatom/data_preprocessing/msa/protein --file_extension .a3m.gz --num_workers 12
    python rename_msa_files_with_seq_hash.py --src_dir /projects/ml/nucleic/torch --dest_dir /projects/ml/RF2_allatom/data_preprocessing/msa/rna --file_extension .afa --num_workers 12
"""

import gzip
import logging
import shutil
from multiprocessing import Pool, cpu_count
from os import PathLike
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from rf2aa.data_new.utils import hash_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_file(args) -> None:
    """
    Process a single file (a3m or afa), create a hashed filename for the sequence,
    and store it in a structured format in the destination directory.

    If a file was an uncompressed a3m file, it will be compressed to a3m.gz.

    Args:
        args (Tuple): A tuple containing the file path, destination directory, and file extension.
    """
    file, dest_dir, file_extension = args

    if ".a3m.gz" in file_extension:
        # Read the file using gzip
        with gzip.open(file, "rt") as f:
            lines: List[str] = f.readlines()
    elif ".afa" in file_extension or ".a3m" in file_extension:
        # Read the file directly
        with file.open("rt") as f:
            lines: List[str] = f.readlines()
    else:
        raise ValueError("Invalid file extension. Must be one of '.a3m.gz', '.a3m', or '.afa'.")

    if len(lines) < 2:
        logger.warning(f"Skipping file due to insufficient lines: {file}")
        return

    # The first line is the sequence ID, the second line is the sequence (for both a3m and afa files)
    sequence: str = lines[1].strip().replace("\n", "")
    hashed_filename: str = hash_sequence(sequence)

    # Find or create a subfolder based on the first 3 characters of the hashed filename
    subfolder: Path = dest_dir / hashed_filename[:3]
    try:
        subfolder.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # For multiprocessing, we must handle race conditions when creating directories
        pass

    # Create a new filename based on the hashed sequence and the file extension (converting a3m to a3m.gz, and eschewing any other suffixes)
    new_file_extension = ".a3m.gz" if ".a3m" in file_extension else ".afa"
    new_file_path: Path = subfolder / f"{hashed_filename}{new_file_extension}"

    # Check if the file already exists before copying
    if not new_file_path.exists():
        # If the file was an a3m and not a3m.gz, we need to compress it
        if ".a3m.gz" not in file_extension and ".a3m.gz" in new_file_extension:
            with open(file, "rb") as f_in:
                with gzip.open(new_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy(file, new_file_path)


def process_files(
    src_dir: PathLike, dest_dir: PathLike, file_extension: str = ".a3m.gz", num_workers: Optional[int] = None
) -> None:
    """
    Process all files in the source directory, create hashed filenames for the sequences,
    and store them in a structured format in the destination directory.

    Args:
        src_dir (PathLike): Path to the source directory containing files (including subdirectories).
        dest_dir (PathLike): Path to the destination directory to store processed files.
        file_extension (str): The file extension to process (either ".a3m.gz" or ".afa").
        num_workers (Optional[int]): Number of worker processes to use for concurrent processing. Defaults to None (which uses the number of CPU cores).
    """

    # Ensure the file extension contains ".a3m" or ".afa"
    if ".a3m" not in file_extension and ".afa" not in file_extension:
        raise ValueError("Invalid file extension. Must contain '.a3m' or '.afa'.")

    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    if not dest_path.exists():
        dest_path.mkdir(parents=True)

    # Get all files with the specified extension in the source directory and its subdirectories
    logger.info(f"Searching for files with extension {file_extension} in {src_dir}")
    files = [f for f in src_path.rglob(f"*{file_extension}") if "".join(f.suffixes) == file_extension]

    if not files:
        raise FileNotFoundError(f"No files with extension {file_extension} found in {src_dir}")
    logger.info(f"Found {len(files)} files with extension {file_extension} in {src_dir}")
    logger.info(f"Saving processed files to {dest_dir}")

    # Number of workers and chunksize
    if num_workers is None:
        num_workers = min(cpu_count(), 16)
    chunksize = min(100, len(files) // num_workers)

    # Arguments for process_single_file function
    args = [(file, dest_path, file_extension) for file in files]

    # Create a pool of workers
    with Pool(processes=num_workers) as pool:
        _ = list(
            tqdm(
                pool.imap(process_single_file, args, chunksize=chunksize),
                total=len(args),
                desc="Processing files",
                unit="file",
            )
        )
        logger.info("Processing complete.")


if __name__ == "__main__":
    # fire.Fire(process_files)
    process_files(
        src_dir="/projects/ml/RF2_allatom/msas_past_20200430",
        dest_dir="/projects/ml/RF2_allatom/data_preprocessing/msa/protein",
        file_extension=".msa0.a3m",
        num_workers=4,
    )
