from itertools import cycle

import biotite.structure as struc
import numpy as np
from cifutils.constants import (
    AA_LIKE_CHEM_TYPES,
    DNA_LIKE_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
)

from datahub.transforms.atom_array import (
    add_global_token_id_annotation,
)
from datahub.utils.token import get_token_starts

# fmt: off
AF3_TOKENS = (
    # 20 AA + 1 unknown AA
    list(STANDARD_AA) + ["UNK"]
    +
    # 4 RNA + 1 unknown RNA
    list(STANDARD_RNA) + ["X"]
    +
    # 4 DNA + 1 unknown DNA
    list(STANDARD_DNA) + ["DX"]
    # 1 gap
    + ["<G>"]
)
"""Sequence tokens in AF3"""
# fmt: on


# Build up chemcomp type mappings
ccd = struc.info.ccd.get_ccd()
_all_res_names = ccd["chem_comp"]["id"].as_array()
_all_res_chemtypes = np.char.upper(ccd["chem_comp"]["type"].as_array())

_is_rna_like = np.isin(_all_res_chemtypes, list(RNA_LIKE_CHEM_TYPES))
_is_dna_like = np.isin(_all_res_chemtypes, list(DNA_LIKE_CHEM_TYPES))
_is_aa_like = np.isin(_all_res_chemtypes, list(AA_LIKE_CHEM_TYPES))

res_name_to_token = dict(zip(_all_res_names[_is_rna_like], cycle(["X"])))
res_name_to_token |= dict(zip(_all_res_names[_is_dna_like], cycle(["DX"])))
res_name_to_token |= dict(zip(AF3_TOKENS, AF3_TOKENS))
to_af3_token = np.vectorize(lambda res_name: res_name_to_token.get(res_name, "UNK"))

token_to_int = {token: i for i, token in enumerate(AF3_TOKENS)}
to_af3_int = np.vectorize(lambda x: token_to_int.get(x, token_to_int["UNK"]))


def encode_atom_names_like_af3(atom_names: np.ndarray) -> np.ndarray:
    """Encodes atom names like AF3"""
    # Ensure uppercase
    atom_names = np.char.upper(atom_names)
    # Turn into 4 character ASCII string (this truncates longer atom names)
    atom_names = atom_names.astype("|S4")
    # Pad to 4 char string with " " (ord(" ") = 32)
    atom_names = np.char.ljust(atom_names, width=4, fillchar=" ")
    # Interpret ASCII bytes to uint8
    atom_names = atom_names.view(np.uint8)
    # Reshape to (N, 4) and subtract 32 to get back to range [0, 64]
    return atom_names.reshape(-1, 4) - 32


# map atom to token index
def get_atom_to_token_idx(atom_array):
    add_global_token_id_annotation(atom_array)
    return atom_array.get_annotation("token_id")


def get_token_to_atom_idx(atom_array):
    return get_token_starts(atom_array)
