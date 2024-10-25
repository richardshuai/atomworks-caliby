"""
This script creates protein and nucleic acid FASTA files from a pandas DataFrame. 
We later use the FASTA file as input to `MMseqs` for clustering.

The files created are:
-   protein_sequeces.fasta (All proteins, both L- and D-peptides, of all lengths)
-   nucleic_acid_sequences.fasta (All nucleic acids)

Small molecules are clustered according to CCD `res_name` identity, which is handled directly through the pandas dataframe when clustering.
Note that for clustering, we use the CANONICAL processed sequence, which is the sequence after processing and then mapping non-standard to standard amino acids, where possible (per AF-3).

Example usage:
python create_fasta_files_from_df.py \
    --pn_units_df /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_07_01/pn_units_df.parquet \
    --output_dir /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_07_01
"""

import logging
from os import PathLike
from pathlib import Path

import fire
import pandas as pd
from cifutils.enums import ChainTypeInfo

from datahub.utils.misc import hash_sequence

logger = logging.getLogger(__name__)


def wrap_sequence(sequence: str, line_length: int = 80) -> str:
    """
    Wrap a sequence string to a specified line length.

    Args:
        sequence (str): The sequence string to wrap.
        line_length (int): The maximum line length. Default is 80.

    Returns:
        str: The wrapped sequence string.
    """
    return "\n".join(sequence[i : i + line_length] for i in range(0, len(sequence), line_length))


def create_fasta_files(pn_units_df: PathLike | str | pd.DataFrame, output_dir: PathLike | str) -> None:
    """
    Create separate FASTA files for proteins and nucleic acids from sequences stored in a Parquet file.

    Args:
        pn_units_df (pd.DataFrame | PathLike | str): Dataframe, as a path Parquet or object directly, containing a column with the sequences.
        output_dir (PathLike | str): Directory where the output FASTA files will be saved.
    """
    # Load the pn_unit_df, if it is not already a DataFrame
    if not isinstance(pn_units_df, pd.DataFrame):
        df = pd.read_parquet(pn_units_df)
    else:
        df = pn_units_df

    # Remove rows where the sequence is not given
    df = df[df["q_pn_unit_processed_entity_canonical_sequence"].notnull()]

    # Remove rows where the sequence is all unknown ("X")
    df = df[df["q_pn_unit_processed_entity_canonical_sequence"].apply(lambda x: not all(char == "X" for char in x))]

    # Build protein DataFrame
    proteins_df = df[df["q_pn_unit_type"].isin([chain_type.value for chain_type in ChainTypeInfo.PROTEINS])]

    # Build nucleic acid DataFrame
    nucleic_acids_df = df[df["q_pn_unit_type"].isin([chain_type.value for chain_type in ChainTypeInfo.NUCLEIC_ACIDS])]
    assert len(proteins_df) + len(nucleic_acids_df) == len(df)

    # Create output directory if it does not exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write protein sequences to FASTA file, de-duplicating as we go
    seen_protein_hashes = set()
    with open(output_dir / "protein_sequences.fasta", "w") as protein_fasta_file:
        for sequence in proteins_df["q_pn_unit_processed_entity_canonical_sequence"]:
            sequence_hash = hash_sequence(sequence)

            # Skip if we have already seen this sequence
            if sequence_hash in seen_protein_hashes:
                continue

            wrapped_sequence = wrap_sequence(sequence)
            protein_fasta_file.write(f">{sequence_hash}\n{wrapped_sequence}\n")
            seen_protein_hashes.add(sequence_hash)

    # Write nucleic acid sequences to FASTA file, de-duplicating as we go
    seen_nucleic_acid_hashes = set()
    with open(output_dir / "nucleic_acid_sequences.fasta", "w") as nucleic_acid_fasta_file:
        for sequence in nucleic_acids_df["q_pn_unit_processed_entity_canonical_sequence"]:
            sequence_hash = hash_sequence(sequence)

            # Skip if we have already seen this sequence
            if sequence_hash in seen_nucleic_acid_hashes:
                continue

            wrapped_sequence = wrap_sequence(sequence)
            nucleic_acid_fasta_file.write(f">{sequence_hash}\n{wrapped_sequence}\n")
            seen_nucleic_acid_hashes.add(sequence_hash)

    logger.info(f"FASTA files saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(create_fasta_files)
