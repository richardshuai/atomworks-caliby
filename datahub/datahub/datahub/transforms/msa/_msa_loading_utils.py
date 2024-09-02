import gzip
import re
import string
from os import PathLike
from pathlib import Path

import numpy as np


def parse_fasta(filename: PathLike, maxseq: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a FASTA file and returns the sequences along with insertion arrays (defaults to zeros).
    Currently, used to parse RNA MSA files, which are stored in FASTA format.

    NOTE: As written, we do not handle insertions; we set the insertion array to all zeros. This is because our
    current RNA MSAs do not have any insertions. If that changes in the future, we will need to update this function.

    TODO: Update this function to correctly incorporate maxseq.
    TODO: Update this function to handle insertions.
    TODO: Update this function to parse tax IDs from the FASTA headers.

    Parameters:
    - filename (PathLike): The path to the FASTA file.
    - maxseq (int): The maximum number of sequences to read from the file (for processing speed). Not currently used.

    Returns:
    - msa (np.ndarray): Array of shape (N, L) where N is the number of sequences and L is the length of sequences.
    - ins (np.ndarray): Array of shape (N, L) where N is the number of sequences and L is the length of sequences.
      Tracks the number of insertions before (to the LEFT of) an aligned column. For the current function, all zeros.
    """
    msa = []
    ins = []

    filename = Path(filename)  # Ensure filename is converted to Path
    assert filename.exists(), f"FASTA file {filename} does not exist"

    with filename.open("r") as fstream:
        for line in fstream:
            # ...skip the labels
            if line[0] == ">":
                continue

            # ...remove right whitespaces
            line = line.rstrip()

            if len(line) == 0:
                continue

            # ...remove lowercase letters and append to MSA
            msa.append(line)

            # ...get the sequence length
            L = len(msa[-1])

            # NOTE: There are never insertions in RNA MSAs
            i = np.zeros((L))
            ins.append(i)

    msa_array = np.array([list(seq) for seq in msa], dtype="S")
    return msa_array, np.array(ins)


def parse_a3m(
    filename: PathLike, maxseq: int = 10000, paired: bool = False, query_tax_id: str = "query"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads an A3M file and returns sequences as a numpy array, along with insertion positions and taxonomy IDs.
    A3M files differ from A2M files in that dots (".") are discarded for compactness; thus, lines may be different lengths (but we still have the same number of aligned columns).

    While parsing:
    - Keep track of number of insertions to the LEFT of each position.
    - Keep track of taxonomy IDs to support MSA pairing downstream.

    NOTE: Files must contain only ASCII characters; we do not handle Unicode characters for memory efficiency.

    Parameters:
        filename (str or Path): The path to the A3M file, can be gzipped.
        maxseq (int): The maximum number of sequences to read from the file (for processing speed). Passed from the associated Transform.
        paired (bool): Whether the MSA is paired, which impacts how tax_ids are parsed.
        query_tax_id (str): The taxonomy ID for the query sequence.

    Returns:
    msa (np.ndarray):
        Array of shape (N, L), where N is the number of sequences (up to maxseq) and L is the length of the aligned columns.
        Each element is a byte string representing the amino acid or nucleotide at that position.
    ins (np.ndarray):
        Array of shape (N, L), where N is the number of sequences (up to maxseq) and L is the length of the aligned columns.
        Tracks the number of insertions before (to the LEFT of) an aligned column.
        If there's an insertion before a position, the value will be > 0; otherwise it will be 0.
        For the query sequence, this will be all zeros.
    tax_ids (np.ndarray):
        Array of shape (N,) containing the taxonomy IDs for each sequence in the MSA.
        If paired is True, the taxonomy IDs are extracted directly from the fasta headers.
        If paired is False, the taxonomy IDs are extracted from the fasta headers using regex.
    """
    msa = []
    ins = []
    tax_ids = []

    # ...create a translation table to remove lowercase letters
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    filename = Path(filename)

    # ...assert that the file exists
    assert filename.exists(), f"MSA a3m file {filename} but does not exist"

    if filename.suffix == ".gz":
        fstream = gzip.open(filename, "rt")
    else:
        fstream = filename.open("r")

    for index, line in enumerate(fstream):
        # ...skip labels
        if line[0] == ">":
            if paired:  # (paired MSAs only have a TAXID in the fasta header)
                tax_ids.append(line[1:].strip())
            else:  # (unpaired MSAs have all the metadata so use regex to pull out TAXID)
                # Example line: ">UniRef100_A0A183IZU9 Kinesin-like protein n=1 Tax=Soboliphyme baturini TaxID=241478 RepID=A0A183IZU9_9BILA"
                match = re.search(r"TaxID=(\d+)", line)
                if match:
                    tax_ids.append(match.group(1))
                elif index == 0:
                    # (query sequence only has the identifying line ">{template_msa_lookup_id}", e.g., ">076568")
                    tax_ids.append(query_tax_id)  # query sequence
                else:
                    tax_ids.append("")  # (unknown tax ID; append empty string (must be handled correctly when pairing)
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

        if len(msa) >= maxseq:
            break

    fstream.close()

    # ...convert lists to numpy arrays for return
    msa_array = np.array([list(seq) for seq in msa], dtype="S")
    ins_array = np.array(ins)
    tax_ids_array = np.array(tax_ids)

    return msa_array, ins_array, tax_ids_array
