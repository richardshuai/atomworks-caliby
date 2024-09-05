from datahub.encoding_definitions import TokenEncoding, AA_LIKE_CHEM_TYPES, DNA_LIKE_CHEM_TYPES, RNA_LIKE_CHEM_TYPES
import biotite.structure as struc
from itertools import cycle
import numpy as np 
from datahub.utils.token import get_token_starts
from datahub.transforms.atomize import atomize_residues
from toolz.curried import pipe, curry, compose
from typing import Sequence
"""Sequence tokens in AF3
Note that tokens for `atomized` residues are 'repeated' 
"""
STANDARD_AA = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
"""The 20 standard amino acids"""

STANDARD_RNA = ['A', 'C', 'G', 'U']
"""The 4 standard RNA nucleotides"""

STANDARD_DNA = ['DA', 'DC', 'DG', 'DT']
"""The 4 standard DNA nucleotides"""

AF3_TOKENS = (
    # 20 AA + 1 unknown AA
    STANDARD_AA + ["UNK"] +
    # 4 RNA + 1 unknown RNA
    STANDARD_RNA + ["X"] +
    # 4 DNA + 1 unknown DNA
    STANDARD_DNA + ["DX"] +
    # 1 gap
    ["<G>"]
)

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


atom_array = atomize_residues(
    atom_array,
    atomize_by_default= True,
    res_names_to_ignore = STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
    move_atomized_part_to_end=False,
    validate_atomize=False
)

# Encode sequence
restype = to_af3_int(atom_array.res_name[get_token_starts(atom_array)])
# Encode element
ref_element = atom_array.element.astype(int)
# Charge
ref_charge = atom_array.charge
# atom name
atom_array.atom_name.astype("|S4").view(np.uint8).reshape(-1, 4)
atom_array.res_name[atom_array.atomize | ]

a = np.array(AF3_TOKENS)

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

encode_atom_names_like_af3(atom_array.atom_name)




atom_array.res_name.view(np.int32).reshape(-1, 5).shape



def tokenize_af3_sequence(sequence: Sequence[str]) -> Sequence[str]:


# 1. Crop full atom array
#  - We need to get conformers for things that:
#     - is_polymer & token_in_crop
#     - (~is_polymer) & token_in_crop
#  (Decision: treat covalently bonded ligands differently?)
#    i.e. get a conformer for the covalently bonded ligand or
#         only get a conformer for the residues

# 2. Generate the conformers (residue level)
#  - Conformer pos & conformer mask

# 3. Encode sequence, element, atom_name, res_iid
# (DONE)

def get_af3_reference()