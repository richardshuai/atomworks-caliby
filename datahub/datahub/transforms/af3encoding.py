from itertools import cycle

import biotite.structure as struc
import numpy as np
from cifutils.utils.selection_utils import annot_start_stop_idxs

from datahub.encoding_definitions import AA_LIKE_CHEM_TYPES, DNA_LIKE_CHEM_TYPES, RNA_LIKE_CHEM_TYPES
from datahub.transforms.atom_array import (
    add_global_token_id_annotation,
    get_within_entity_idx,
    get_within_group_res_idx,
)
from datahub.transforms.atomize import atomize_residues
from datahub.utils.token import get_token_starts

STANDARD_AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
"""The 20 standard amino acids"""

STANDARD_RNA = ["A", "C", "G", "U"]
"""The 4 standard RNA nucleotides"""

STANDARD_DNA = ["DA", "DC", "DG", "DT"]
"""The 4 standard DNA nucleotides"""

AF3_TOKENS = (
    # 20 AA + 1 unknown AA
    STANDARD_AA
    + ["UNK"]
    +
    # 4 RNA + 1 unknown RNA
    STANDARD_RNA
    + ["X"]
    +
    # 4 DNA + 1 unknown DNA
    STANDARD_DNA
    + ["DX"]
    +
    # 1 gap
    ["<G>"]
)
"""Sequence tokens in AF3"""


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


def encode_atom_names_like_af3(atom_names):
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


atom_array = atomize_residues(
    atom_array,
    atomize_by_default=True,
    res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
    move_atomized_part_to_end=False,
    validate_atomize=False,
)

# SET 1 (token level)
# ... get token-level array
token_starts = get_token_starts(atom_array)
token_level_array = atom_array[token_starts]
# identifier tokens
residue_index = get_within_group_res_idx(token_level_array, group_by="chain_iid")
token_index = np.arange(len(token_starts))
asym_name, asym_id = np.unique(token_level_array.chain_iid, return_inverse=True)
entity_name, entity_id = np.unique(token_level_array.chain_entity, return_inverse=True)
# TODO: Sym ID is not quite right yet ...
sym_name, sym_id = get_within_entity_idx(token_level_array, level="chain")
# sequence tokens
restype = to_af3_int(token_level_array.res_name)
# molecule type
is_protein = np.isin(token_level_array.res_name, _all_res_names[_is_aa_like])
is_rna = np.isin(token_level_array.res_name, _all_res_names[_is_rna_like])
is_dna = np.isin(token_level_array.res_name, _all_res_names[_is_dna_like])
is_ligand = ~(is_protein | is_rna | is_dna)

# SET 2 (atom level)
# TODO: ref_pos (get reference conformers for each CCD code or SMILES
# TODO: ref_mask
# Encode element
ref_element = atom_array.element.astype(int)
# Charge
ref_charge = atom_array.charge
# atom name
ref_atom_name_chars = encode_atom_names_like_af3(atom_array.atom_name)
# ref_space_uid
ref_space_uid = add_global_token_id_annotation(atom_array).token_id

# SET 3 (MSA level)
# -> Nate

# SET 4 (template level)

# SET 5 (bond level)
# TODO:


annot_start_stop_idxs(atom_array, ["chain_iid"], add_exclusive_stop=False)


# 1. Crop full atom array
#  - We need to get conformers for things that:
#     - is_polymer & token_in_crop
#     - (~is_polymer) & token_in_crop
#  (Decision: treat covalently bonded ligands differently?)
#    i.e. get a conformer for the covalently bonded ligand or
#         only get a conformer for the residues

# 2. Generate the conformers (residue level)
#  - Conformer pos & conformer mask

# 3. Encode sequence, element, atom_name, res_iid
# (DONE)

# def get_af3_reference()
