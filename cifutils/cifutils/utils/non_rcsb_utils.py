"""
Utility functions for handling non-RCSB CIF files.
Such files do not follow the standard CIF format and thus may require special handling.
"""

__all__ = ["load_monomer_sequence_information_from_atom_array", "infer_chain_info_from_atom_array"]

from biotite.structure import AtomArray
import biotite.structure as struc
from cifutils.common import deduplicate_iterator
from cifutils.utils.residue_utils import get_chem_comp_type
from cifutils.utils.sequence_utils import get_1_from_3_letter_code
from cifutils.enums import ChainType
from cifutils.constants import (
    AA_LIKE_CHEM_TYPES,
    POLYPEPTIDE_L_CHEM_TYPES,
    POLYPEPTIDE_D_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
    DNA_LIKE_CHEM_TYPES,
)
import logging
import numpy as np
from biotite.structure.io.pdbx import CIFCategory

logger = logging.getLogger("cifutils")


def load_monomer_sequence_information_from_atom_array(chain_info_dict: dict, atom_array: AtomArray) -> dict:
    """
    Load monomer sequence information into a chain_info_dict using the AtomArray as the ground-truth for
    both polymers and non-polymers.

    Assumes that there are no fully unresolved residues in the AtomArray; otherwise, the sequence will be incomplete.
    """
    for chain_id in np.unique(atom_array.chain_id):
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]
        residue_id_list, residue_name_list = struc.get_residues(chain_atom_array)

        if chain_id not in chain_info_dict:
            chain_info_dict[chain_id] = {}
        chain_info_dict[chain_id]["residue_name_list"] = list(residue_name_list)
        chain_info_dict[chain_id]["residue_id_list"] = list(residue_id_list)

    return chain_info_dict


def infer_processed_entity_sequences_from_atom_array(chain_info_dict: dict, atom_array: AtomArray) -> dict:
    """
    Infer processed entity sequences from an AtomArray.
    """
    for chain_id in np.unique(atom_array.chain_id):
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]
        _, residue_name_list = struc.get_residues(chain_atom_array)
        chain_type = ChainType(chain_atom_array.chain_type[0])

        if chain_id not in chain_info_dict:
            chain_info_dict[chain_id] = {}

        # Create the processed single-letter sequence representations
        processed_entity_non_canonical_sequence = [
            get_1_from_3_letter_code(residue, chain_type, use_closest_canonical=False) for residue in residue_name_list
        ]
        processed_entity_canonical_sequence = [
            get_1_from_3_letter_code(residue, chain_type, use_closest_canonical=True) for residue in residue_name_list
        ]
        chain_info_dict[chain_id]["processed_entity_non_canonical_sequence"] = "".join(
            processed_entity_non_canonical_sequence
        )
        chain_info_dict[chain_id]["processed_entity_canonical_sequence"] = "".join(processed_entity_canonical_sequence)

    return chain_info_dict


def infer_chain_info_from_atom_array(atom_array: AtomArray) -> dict:
    """
    Infer chain type information from an AtomArray, in the event that the CIF file does not contain the necessary information.
    Such situations may arise when the CIF file is not from the RCSB PDB database (e.g., distillation).

    TODO: Tests for this function
    TODO: Re-write this function to use the ChainType IntEnum
    """
    logger.info(
        "Could not read ChainType from CIF file, inferring from AtomArray (ensure this is the correct behavior)!"
    )
    chain_info_dict = {}

    # Loop through chains
    for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
        # ...get a list of the residue names
        _, res_names = struc.get_residues(atom_array[atom_array.chain_id == chain_id])

        chain_type_counts = {
            key: 0
            for key in [
                "aa_like",
                "polypeptide(D)",
                "polypeptide(L)",
                "polydeoxyribonucleotide",
                "polyribonucleotide",
                "non-polymer",
            ]
        }
        # ...infer the chain type based on each residue
        for res_name in res_names:
            chem_comp = get_chem_comp_type(res_name)
            # Increment the count for the chain type
            if chem_comp in AA_LIKE_CHEM_TYPES:
                chain_type_counts["aa_like"] += 1
            if chem_comp in POLYPEPTIDE_D_CHEM_TYPES:
                chain_type_counts["polypeptide(D)"] += 1
            elif chem_comp in POLYPEPTIDE_L_CHEM_TYPES:
                chain_type_counts["polypeptide(L)"] += 1
            elif chem_comp in RNA_LIKE_CHEM_TYPES:
                chain_type_counts["polyribonucleotide"] += 1
            elif chem_comp in DNA_LIKE_CHEM_TYPES:
                chain_type_counts["polydeoxyribonucleotide"] += 1
            else:
                chain_type_counts["non-polymer"] += 1

        # Default to polymer
        is_polymer = True

        # ...if we have both RNA and DNA, set the chain type to RNA/DNA hybrid
        if chain_type_counts["polyribonucleotide"] > 0 and chain_type_counts["polydeoxyribonucleotide"] > 0:
            chain_type = "polydeoxyribonucleotide/polyribonucleotide hybrid"
        # ...if we have more L-polypeptides than D-polypeptides, set to L-polypeptide
        elif chain_type_counts["polypeptide(L)"] > chain_type_counts["polypeptide(D)"]:
            chain_type = "polypeptide(L)"
        # ...if we have more D-polypeptides than L-polypeptides, set to D-polypeptide
        elif chain_type_counts["polypeptide(L)"] > chain_type_counts["polypeptide(D)"]:
            chain_type = "polypeptide(D)"
        # ...if we only have "aa_like", default to "polypeptide(L)"
        elif (
            chain_type_counts["aa_like"] > 0
            and chain_type_counts["polypeptide(L)"] == 0
            and chain_type_counts["polypeptide(D)"] == 0
        ):
            chain_type = "polypeptide(L)"
        # ...if we have RNA, set to polyribonucleotide
        elif chain_type_counts["polyribonucleotide"] > 0:
            chain_type = "polyribonucleotide"
        # ...if we have DNA, set to polydeoxyribonucleotide
        elif chain_type_counts["polydeoxyribonucleotide"] > 0:
            chain_type = "polydeoxyribonucleotide"
        # ...otherwise set to non-polymer
        elif chain_type_counts["non-polymer"] > 0:
            chain_type = "non-polymer"
            is_polymer = False
        else:
            raise ValueError(f"Could not infer chain type for chain {chain_id}")

        chain_info_dict[chain_id] = {"is_polymer": is_polymer, "type": chain_type}

    return chain_info_dict


def get_identity_op_expr_category() -> CIFCategory:
    return CIFCategory.deserialize(
        """_pdbx_struct_oper_list.id                   1 
        _pdbx_struct_oper_list.type                 'identity operation' 
        _pdbx_struct_oper_list.name                 1_555 
        _pdbx_struct_oper_list.symmetry_operation   x,y,z 
        _pdbx_struct_oper_list.matrix[1][1]         1.0000000000 
        _pdbx_struct_oper_list.matrix[1][2]         0.0000000000 
        _pdbx_struct_oper_list.matrix[1][3]         0.0000000000 
        _pdbx_struct_oper_list.vector[1]            0.0000000000 
        _pdbx_struct_oper_list.matrix[2][1]         0.0000000000 
        _pdbx_struct_oper_list.matrix[2][2]         1.0000000000 
        _pdbx_struct_oper_list.matrix[2][3]         0.0000000000 
        _pdbx_struct_oper_list.vector[2]            0.0000000000 
        _pdbx_struct_oper_list.matrix[3][1]         0.0000000000 
        _pdbx_struct_oper_list.matrix[3][2]         0.0000000000 
        _pdbx_struct_oper_list.matrix[3][3]         1.0000000000 
        _pdbx_struct_oper_list.vector[3]            0.0000000000 """
    )


def get_identity_assembly_gen_category(chain_ids: list[str]) -> CIFCategory:
    return CIFCategory.deserialize(
        f"""_pdbx_struct_assembly_gen.assembly_id 1
        _pdbx_struct_assembly_gen.oper_expression 1
        _pdbx_struct_assembly_gen.asym_id_list {",".join(chain_ids)}
        """
    )
