"""
We use a taxid map to get taxid from Uniref100 IDs that are used in the MSA files.

This script turns the idmapping_selected.tab file from the Uniref100 database into a
pickle file that can be used to map taxids to Uniref100 IDs.

Download the idmapping_selected.tab file from: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/

Example usage:
    python setup_taxid_map.py --idmapping_file /projects/ml/mmseqs_gpu/uniref100_idmapping_selected.tab --output_file /projects/ml/mmseqs_gpu/uniref_to_taxid_2025_01_23.pkl
"""

import pickle
from pathlib import Path

import fire
from tqdm import tqdm


def _process_idmapping(idmapping_file: Path) -> dict:
    """
    Process the idmapping_selected.tab file and return a dictionary mapping taxids to Uniref100 IDs.
    Assumes column 0 is the Uniref100 ID and column 12 is the taxid.

    Args:
        idmapping_file: Path to the idmapping_selected.tab file.
    Returns:
        A dictionary mapping Uniref100 IDs to taxids. e.g. {"Q6GZX4": 654924, ...}
    """
    uniref_id_to_tax_id = {}
    with open(idmapping_file) as f:
        for line in tqdm(f, desc="Processing idmapping_selected.tab. Generally 250M+ lines"):
            parts = line.strip().split("\t")
            if len(parts) < 13:
                raise ValueError(f"Expected at least 13 columns in idmapping_selected.tab, got {len(parts)}")
            uniref_id_to_tax_id[parts[0]] = int(parts[12])
    return uniref_id_to_tax_id


def process_and_dump_idmapping(idmapping_file: Path, output_file: Path) -> None:
    """
    Process the idmapping_selected.tab file and dump the resulting dictionary to a pickle file.

    Args:
        idmapping_file: Path to the idmapping_selected.tab file.
        output_file: Path to the output pickle file.
    """
    uniref_id_to_tax_id = _process_idmapping(idmapping_file)

    with open(output_file, "wb") as f:
        pickle.dump(uniref_id_to_tax_id, f)


if __name__ == "__main__":
    fire.Fire(process_and_dump_idmapping)
