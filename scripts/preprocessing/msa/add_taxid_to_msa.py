"""
Script for adding taxids to an MSA file that has UniRef100 IDs already.
Useful for outputs of MMSeqs2 pipeline which only outputs UniRef100 IDs.

NOTE: To reduce overhead, run this on an entire directory of MSA files, and not during
inference time since the script involves loading a large dictionary into memory. A dictionary
definitely isn't the best way to do this, but this step shouldn't be a bottleneck.

Example usage:
    python add_taxid_to_msa.py --msa_dir /net/scratch/mkazman/msa/example_msas/
"""

import gzip
import logging
import pickle
from pathlib import Path

import fire
from tqdm import tqdm

from atomworks.ml.transforms.msa._msa_loading_utils import extract_tax_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_one_file(msa_file: Path, uniref_id_to_tax_id: dict) -> None:
    """
    Parse one MSA file and add taxids to it in place. Will skip individual lines if the taxid is already
    present or there is no UniRef100 ID.

    NOTE: datahub.transforms.msa._msa_loading_utils.extract_tax_id expects the taxid to be in the header
    in the format "TaxId=9606", so that's how we add it.

    Args:
        msa_file (Path): The path to the MSA file.
        uniref_id_to_tax_id (dict): A dictionary mapping Uniref100 IDs to taxids.
    Returns:
        None
    """

    if ".a3m" not in msa_file.name:
        raise NotImplementedError("Only .a3m files are supported right now.")

    # handle gzipped files as well
    if str(msa_file).endswith(".gz"):
        open_func = gzip.open
        write_mode = "wt"
        read_mode = "rt"
    else:
        open_func = open
        write_mode = "w"
        read_mode = "r"

    new_lines = []
    with open_func(msa_file, read_mode) as f:
        for line in f:
            if line.startswith(">UniRef100"):  # header of uniprot hit
                uniref_id = line.split("\t")[0].split("_")[1]
                if extract_tax_id(line, "") == "":  # doesn't already have taxid
                    taxid = uniref_id_to_tax_id.get(uniref_id, None)
                    if taxid:
                        # write taxid to the end of the line
                        line = f"{line.strip()}\tTaxId={taxid}\n"
                        new_lines.append(line)
                        continue  # already added line, so skip to next iteration
            new_lines.append(line)

    with open_func(msa_file, write_mode) as f:  # overwrite the file
        f.writelines(new_lines)


def _parse_nested_msa_dir(
    msa_dir: Path,
    uniref_id_to_tax_id_path: Path = Path("/projects/ml/mmseqs_gpu/uniref_to_taxid_2025_01_23.pkl"),
    file_ext: str = ".a3m.gz",
) -> None:
    """
    Loads uniref_id to tax_id dictionary and parse all .a3m files in a nested directory structure.

    Args:
        msa_dir (Path): The directory containing the MSA files.
        uniref_id_to_tax_id_path (Path): The path to the uniref_id to tax_id dictionary.
        file_ext (str): The extension of the MSA files.
    Returns:
        None
    """
    msa_dir = Path(msa_dir)
    # use pickle to load the dictionary
    logger.info(f"Loading uniref_id to tax_id dictionary from {uniref_id_to_tax_id_path}")
    with open(uniref_id_to_tax_id_path, "rb") as f:
        uniref_id_to_tax_id = pickle.load(f)
    logger.info(f"Finished loading {len(uniref_id_to_tax_id)} uniref_id to tax_id mappings.")

    logger.info(f"Parsing all .a3m files in {msa_dir}")
    msa_files = list(msa_dir.glob(f"**/*{file_ext}"))
    for msa_file in tqdm(msa_files, desc="Adding tax IDs to MSA files"):
        try:
            _parse_one_file(msa_file, uniref_id_to_tax_id)
        except gzip.BadGzipFile:
            logger.error(f"Corrupted file: {msa_file}. Skipping...")


if __name__ == "__main__":
    fire.Fire(_parse_nested_msa_dir)
