"""Shared test utils and fixtures for all tests"""

import os
from pathlib import Path
from biotite.structure import AtomArray
import numpy as np
import biotite.structure as struc
from cifutils.cifutils_biotite.cifutils_biotite import CIFParser as CIFParserBiotite
from cifutils.cifutils_legacy.cifutils_legacy import CIFParser as CIFParserLegacy

TEST_DATA_DIR = Path(os.path.dirname(__file__)) / "data"
CIF_PARSER_BIOTITE = CIFParserBiotite()
CIF_PARSER_LEGACY = CIFParserLegacy()


def get_digs_path(pdbid: str) -> str:
    pdbid = pdbid.lower()
    filename = f"/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz"
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename


def _get_atom_array_stats(arr: AtomArray) -> str:
    msg = f"AtomArray: {len(arr)} atoms, {struc.get_residue_count(arr)} residues, {struc.get_chain_count(arr)} chains\n"
    msg += f"\t... unique chain ids: {np.unique(arr.chain_id)}\n"
    msg += f"\t... unique residue ids: {np.unique(arr.res_id)}\n"
    msg += f"\t... unique atom types: {np.unique(arr.atom_name)}\n"
    msg += f"\t... unique elements: {np.unique(arr.element)}\n"
    return msg


def assert_same_atom_array(
    arr1: AtomArray,
    arr2: AtomArray,
    annotations_to_compare: list[str] = ["chain_id", "res_name", "res_id", "atom_name", "element"],
    max_print_length: int = 10,
):
    if not len(arr1) == len(arr2):
        msg = "AtomArrays are not the same length.\n"
        msg += f"\tarr1: {_get_atom_array_stats(arr1)}\n"
        msg += f"\tarr2: {_get_atom_array_stats(arr2)}\n"
        raise ValueError(msg)

    for annotation in annotations_to_compare:
        annot1 = arr1.get_annotation(annotation)
        annot2 = arr2.get_annotation(annotation)
        mismatch_mask = annot1 != annot2
        if np.any(mismatch_mask):
            msg = f"AtomArrays are not equivalent in `{annotation}`\n"
            arr1_mismatch = arr1[mismatch_mask][:max_print_length]  # max len to reduce length of print output
            arr2_mismatch = arr2[mismatch_mask][:max_print_length]
            msg += f"\tarr1: \n{arr1_mismatch}\n"
            msg += f"\tarr2: \n{arr2_mismatch}\n"
            raise ValueError(msg)

    if not np.allclose(arr1.coord, arr2.coord, equal_nan=True):
        mismatch_mask = np.any(~np.isclose(arr1.coord, arr2.coord, equal_nan=True), axis=1)
        arr1_mismatch = arr1[mismatch_mask][:max_print_length]
        arr2_mismatch = arr2[mismatch_mask][:max_print_length]
        msg = "AtomArrays are not equivalent in coordinates\n"
        msg += f"\tarr1: \n{arr1_mismatch}\n"
        msg += f"\tarr2: \n{arr2_mismatch}\n"
        raise ValueError(msg)
