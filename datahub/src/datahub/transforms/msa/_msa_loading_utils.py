import re
import string
from os import PathLike

import numpy as np

from datahub.utils.io import open_file


def extract_tax_id(line: str, unknown_tax_id: str = "") -> str:
    """Extract taxonomy ID from the header line"""
    # ...extract the TaxID from the header line
    # (Example line: ">UniRef100_A0A183IZU9 Kinesin-like protein n=1 Tax=Soboliphyme baturini TaxID=241478 RepID=A0A183IZU9_9BILA")
    match = re.search(r"TaxID=(\d+)", line)
    if match:
        return match.group(1)
    return unknown_tax_id  # (unknown tax ID, which must be handled correctly when pairing downstream)


def parse_msa(
    filename: PathLike, maxseq: int = 10000, query_tax_id: str = "query"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Routes to the appropriate MSA parser based on the file extension."""
    if filename.name.endswith((".a3m", ".a3m.gz")):
        return parse_a3m(filename, maxseq, query_tax_id)
    elif filename.name.endswith((".afa", ".afa.gz", ".fasta", ".fasta.gz")):
        return parse_fasta(filename, maxseq, query_tax_id)
    else:
        raise ValueError(f"Unsupported MSA file extension: {filename.name}")


def parse_fasta(filename: PathLike, maxseq: int = 10000, query_tax_id: str = "query") -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a FASTA (.afa or .fasta) file and returns sequences as a numpy array, along with insertion positions and taxonomy IDs.

    NOTE: As written, we do not handle insertions; we set the insertion array to all zeros. Currently, RNA MSAs do not have insertions.
    If that changes in the future, we will need to update this function.

    TODO: Update this function to handle insertions. Note that in FASTA files, insertions must be handled differently than in A3M files.
    For FASTA, we would need to remove all gaps from the query sequence, and consider non-gap characters in those columns as insertions.

    Args:
        filename (PathLike): The path to the FASTA file (can be gzipped).
        maxseq (int): The maximum number of sequences to read from the file (for processing speed).
        query_tax_id (str): The taxonomy ID for the query sequence.

    Returns:
        msa (np.ndarray): Array of shape (N, L) where N is the number of sequences and L is the length of sequences.
        ins (np.ndarray): Array of shape (N, L) where N is the number of sequences and L is the length of sequences.
        tax_ids (np.ndarray): Array of shape (N,) containing the taxonomy IDs for each sequence in the MSA.

    References:
        - UniProt FASTA Header Documentation (https://www.uniprot.org/help/fasta-headers)
    """
    msa = []
    ins = []
    tax_ids = []

    fstream = open_file(filename)

    for index, line in enumerate(fstream):
        # Extract taxonomy ID from the header line, but don't process like the rest of the MSA
        if line[0] == ">":
            if index == 0:
                # ...force the query sequence to have the query tax ID
                tax_ids.append(query_tax_id)  # query sequence
            else:
                # ...extract the TaxID from the header line
                tax_ids.append(extract_tax_id(line))

            # ...don't process the header line any further
            continue

        # ...remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # ...append to MSA (no lowercase letters in FASTA files that we need to remove)
        msa.append(line)

        # ...get the sequence length
        L = len(msa[-1])

        # HACK: There are never insertions in RNA MSAs, so we set the insertion array to all zeros
        i = np.zeros((L))
        ins.append(i)

        # ...break if we've reached the maximum number of sequences
        if len(msa) >= maxseq:
            break

    # ...convert lists to numpy arrays for return
    msa_array = np.array([list(seq) for seq in msa], dtype="S")
    ins_array = np.array(ins)
    tax_ids_array = np.array(tax_ids)

    return msa_array, ins_array, tax_ids_array


def parse_a3m(
    filename: PathLike, maxseq: int = 10000, query_tax_id: str = "query"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads an A3M file and returns sequences as a numpy array, along with insertion positions and taxonomy IDs.
    A3M files differ from A2M files in that dots (".") are discarded for compactness; thus, lines may be different lengths (but we still have the same number of aligned columns).

    While parsing:
        - Keep track of number of insertions to the LEFT of each position.
        - Keep track of taxonomy IDs to support MSA pairing downstream.

    NOTE: Files must contain only ASCII characters; we do not handle Unicode characters.

    Parameters:
        filename (str or Path): The path to the A3M file, can be gzipped.
        maxseq (int): The maximum number of sequences to read from the file (for processing speed). Passed from the associated Transform.
        query_tax_id (str): The taxonomy ID for the query sequence.

    Returns:
        msa (np.ndarray):
            Array of shape (N, L), where N is the number of sequences (up to maxseq) and L is the length of the aligned columns.
            Each element is a byte string representing the amino acid or nucleotide at that position.
        ins (np.ndarray):
            Array of shape (N, L), where N is the number of sequences (up to maxseq) and L is the length of the aligned columns.
            Tracks the number of insertions (relative to the query sequence) before (to the LEFT of) an aligned column. If there's an
            insertion before a position, the value will be > 0; otherwise it will be 0. For the query sequence, this will be all zeros.
        tax_ids (np.ndarray):
            Array of shape (N,) containing the taxonomy IDs for each sequence in the MSA.

    References:
        - A3M Format Documentation (https://yanglab.qd.sdu.edu.cn/trRosetta/msa_format.html#a3m)
    """
    msa = []
    ins = []
    tax_ids = []

    # ...create a translation table to remove lowercase letters
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # ...open the file
    fstream = open_file(filename)

    for index, line in enumerate(fstream):
        # Extract taxonomy ID from the header line, but don't process like the rest of the MSA
        if line[0] == ">":
            if index == 0:
                # ...force the query sequence to have the query tax ID
                tax_ids.append(query_tax_id)  # query sequence
            else:
                # ...extract the TaxID from the header line
                tax_ids.append(extract_tax_id(line))

            # ...don't process the header line any further
            continue

        # ...remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # ...remove lowercase letters and append to MSA
        # (lowercase letters represent insertion positions between alignment columns)
        msa.append(line.translate(table))

        # ...get the sequence length
        # (since we removed lowercase letters, and we're using a3m without dot representations, all sequences should be the same length)
        L = len(msa[-1])

        # (0 - match or gap; 1 - insertion)
        a = np.array([0 if c.isupper() or c == "-" else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # ...get the positions of insertions
            pos = np.where(a == 1)[0]

            # ...shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # ...get position of insertions in cleaned sequence and their length
            pos, num = np.unique(a, return_counts=True)

            # ...append to the matrix of insertions
            # (num represents the number of insertions to the LEFT of the index specified by pos)
            i[pos] = num

        ins.append(i)

        # ...break if we've reached the maximum number of sequences
        if len(msa) >= maxseq:
            break

    fstream.close()

    # ...convert lists to numpy arrays for return
    msa_array = np.array([list(seq) for seq in msa], dtype="S")
    ins_array = np.array(ins)
    tax_ids_array = np.array(tax_ids)

    return msa_array, ins_array, tax_ids_array
