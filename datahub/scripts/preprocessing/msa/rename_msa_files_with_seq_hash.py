"""Script to process and rename sequence files with hashed filenames, using SHA-256 cryptographic hashes of the one-letter sequences.

This script reads sequence files (.a3m.gz, .a3m, or .afa) from a source directory, computes a hashed filename for each sequence, and saves the files in a structured format.
Such a process is necessary to use a directory of MSA's during training or inference for our internal structure-based models.

Optionally, we:
    - Shard the files into subfolders in the destination directory based on the first 3 characters of the hashed filename
    - Use multiprocessing to process files concurrently
    - Copy rather than move the files to the destination directory
    - Compress a3m files to a3m.gz if they are not already compressed

Usage:
    python rename_msa_files_with_seq_hash.py --src_dir <source_directory> --dest_dir <destination_directory> --file_extension <.a3m.gz or .a3m or .afa> --num_workers <number_of_workers>

Example:
    python rename_msa_files_with_seq_hash.py --src_dir /projects/ml/TrRosetta/PDB-2021AUG02/a3m --dest_dir /projects/ml/RF2_allatom/data_preprocessing/msa/protein --file_extension .a3m.gz --num_workers 12
    python rename_msa_files_with_seq_hash.py --src_dir /projects/ml/nucleic/torch --dest_dir /projects/ml/RF2_allatom/data_preprocessing/msa/rna --file_extension .afa --num_workers 12
"""

import gzip
import logging
import shutil
from multiprocessing import Pool, cpu_count
from os import PathLike
from pathlib import Path
from typing import List

import fire
from tqdm import tqdm

from datahub.utils.io import get_sharded_file_path
from datahub.utils.misc import hash_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rename_and_move_single_file(args: tuple) -> None:
    """Process a single MSA file by renaming it with a hashed filename and saving it to the destination directory.

    This function is called by the multiprocessing pool

    Args:
        args (Tuple): Tuple of (file, dest_dir, file_extension, depth, copy, compress)
    """
    file, dest_dir, file_extension, depth, copy, compress = args

    # Default depth to 0 if None (no nesting)
    effective_depth = depth if depth is not None else 0

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

    # Decide final extension
    if ".afa" in file_extension:
        new_file_extension = ".afa"
    else:
        # If compress is True and original extension is .a3m => .a3m.gz
        if compress and ".a3m" in file_extension and ".a3m.gz" not in file_extension:
            new_file_extension = ".a3m.gz"
        else:
            # Keep original if not compressing
            new_file_extension = file.suffix if file.suffix in [".gz", ".afa"] else file_extension

    # Build the sharded file path
    sharded_file_path = get_sharded_file_path(
        base_dir=dest_dir, file_hash=hashed_filename, extension=new_file_extension, depth=effective_depth
    )

    try:
        sharded_file_path.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # For multiprocessing, we must handle race conditions when creating directories
        pass

    # Check if the file already exists before copying
    if not sharded_file_path.exists():
        # If the file was an a3m and not a3m.gz, we need to compress it
        if ".a3m.gz" not in file_extension and ".a3m.gz" in new_file_extension:
            with open(file, "rb") as f_in:
                with gzip.open(sharded_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            if not copy:
                file.unlink()  # remove the original file
        else:
            if copy:
                shutil.copy(file, sharded_file_path)
            else:
                shutil.move(file, sharded_file_path)


def rename_and_move_files(
    src_dir: PathLike,
    dest_dir: PathLike,
    file_extension: str = ".a3m.gz",
    num_workers: int | None = None,
    depth: int | None = None,
    copy: bool = True,
    compress: bool = True,
) -> None:
    """Renames and nests MSA files in a destination directory based on their sequence.

    Formats the files in the directory in a manner appropriate for training or inference with our structure-based models.

    We:
        - Rename the files with the hash of the sequence (the first row of the MSA)
        - Optionally compress .a3m files to .a3m.gz
        - Optionally shard the files in subfolders based on the first characters of their hash

    Args:
        src_dir (PathLike): Path to the directory containing source files (including subdirectories)
        dest_dir (PathLike): Path to the directory where processed files will be saved
        file_extension (str): The file extension to search for in the `src_dir`. Must contain '.a3m' or '.afa'
        num_workers (Optional[int]): Number of worker processes. Defaults to min(CPU count, 16)
        depth (Optional[int]): Number of subfolders to nest (shard) under dest_dir. If None => 0 (no nesting)
        copy (bool): Whether to copy (True) or move (False) the files to the destination directory
        compress (bool): Whether to compress .a3m files to .a3m.gz
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    if ".a3m" not in file_extension and ".afa" not in file_extension:
        raise ValueError("file_extension must contain '.a3m' or '.afa'.")

    logger.info(f"Searching for files with extension {file_extension} in {src_dir}")
    files = [f for f in src_path.rglob(f"*{file_extension}") if "".join(f.suffixes) == file_extension]
    if not files:
        raise FileNotFoundError(f"No files with extension {file_extension} found in {src_dir}")

    logger.info(f"Found {len(files)} files with extension {file_extension}")
    logger.info(f"Saving processed files to {dest_dir}")

    if num_workers is None:
        num_workers = min(cpu_count(), 16)
    chunksize = max(1, min(100, len(files) // max(1, num_workers)))

    args_list = []
    for file in files:
        entry = (file, dest_path, file_extension, depth, copy, compress)
        args_list.append(entry)

    with Pool(processes=num_workers) as pool:
        list(
            tqdm(
                pool.imap(rename_and_move_single_file, args_list, chunksize=chunksize),
                total=len(args_list),
                desc="Processing files",
                unit="file",
            )
        )

    logger.info("Processing complete.")


if __name__ == "__main__":
    fire.Fire(rename_and_move_files)
