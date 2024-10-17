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
    Infer chain type information from an AtomArray, in the event that the PDB or CIF file does not contain the necessary information.
    Such situations may arise when the CIF file is not from the RCSB PDB database (e.g., distillation).

    WARNING: Use this function as a "last resort" when the chain type information is not explicitly provided in the CIF file;
    otherwise, use `get_chain_info_from_category`.

    Args:
        atom_array (AtomArray): The AtomArray object to infer chain information from.

    Returns:
        dict: A dictionary containing the inferred chain information (chain type and whether the chain is a polymer).
    """
    logger.info(
        "Could not read ChainType from CIF file, inferring from AtomArray (ensure this is the correct behavior)!"
    )
    chain_info_dict = {}

    # Loop through chains
    for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]

        # ...get a list of the residue names
        _, res_names = struc.get_residues(chain_atom_array)

        chain_type_counts = {
            key: 0
            for key in [
                "aa_like",
                ChainType.POLYPEPTIDE_D,
                ChainType.POLYPEPTIDE_L,
                ChainType.DNA,
                ChainType.RNA,
                ChainType.NON_POLYMER,
            ]
        }

        # ...infer the chain type based on each residue
        for res_name in res_names:
            chem_comp = get_chem_comp_type(res_name)
            # Increment the count for the appropriate chain type category

            # (All amino acid-like chem types are considered "aa_like")
            if chem_comp in AA_LIKE_CHEM_TYPES:
                chain_type_counts["aa_like"] += 1
                # (We further differentiate between L- and D-polypeptides)
                if chem_comp in POLYPEPTIDE_D_CHEM_TYPES:
                    chain_type_counts[ChainType.POLYPEPTIDE_D] += 1
                elif chem_comp in POLYPEPTIDE_L_CHEM_TYPES:
                    chain_type_counts[ChainType.POLYPEPTIDE_L] += 1
            # (We differentiate between RNA and DNA)
            elif chem_comp in RNA_LIKE_CHEM_TYPES:
                chain_type_counts[ChainType.RNA] += 1
            elif chem_comp in DNA_LIKE_CHEM_TYPES:
                chain_type_counts[ChainType.DNA] += 1
            # (All other chem types are considered non-polymer)
            else:
                chain_type_counts[ChainType.NON_POLYMER] += 1

        # Default to polymer
        is_polymer = True

        # WARNING: The following logic is heuristic, and may fail in cases of multiple residues types within a chain.

        # ...if we have both RNA and DNA, set the chain type to RNA/DNA hybrid
        if chain_type_counts[ChainType.RNA] > 0 and chain_type_counts[ChainType.DNA] > 0:
            inferred_chain_type_str = str(ChainType.DNA_RNA_HYBRID)

        # ...if we have proteins, set to either L- or D-polypeptide, depending on the counts
        elif chain_type_counts[ChainType.POLYPEPTIDE_L] > 0 or chain_type_counts[ChainType.POLYPEPTIDE_D] > 0:
            # ...if we have equal or more L-polypeptides than D-polypeptides in the chain, set to L-polypeptide
            if chain_type_counts[ChainType.POLYPEPTIDE_L] >= chain_type_counts[ChainType.POLYPEPTIDE_D]:
                inferred_chain_type_str = str(ChainType.POLYPEPTIDE_L)

            # ...if we have more D-polypeptides than L-polypeptides, set to D-polypeptide
            elif chain_type_counts[ChainType.POLYPEPTIDE_L] < chain_type_counts[ChainType.POLYPEPTIDE_D]:
                inferred_chain_type_str = str(ChainType.POLYPEPTIDE_D)

        # ...if we only have "aa_like", default to "polypeptide(L)"
        elif (
            chain_type_counts["aa_like"] > 0
            and chain_type_counts[ChainType.POLYPEPTIDE_L] == 0
            and chain_type_counts[ChainType.POLYPEPTIDE_D] == 0
        ):
            inferred_chain_type_str = str(ChainType.POLYPEPTIDE_L)

        # ...if we have RNA, set to polyribonucleotide
        elif chain_type_counts[ChainType.RNA] > 0:
            inferred_chain_type_str = str(ChainType.RNA)
        # ...if we have DNA, set to polydeoxyribonucleotide
        elif chain_type_counts[ChainType.DNA] > 0:
            inferred_chain_type_str = str(ChainType.DNA)
        # ...otherwise set to non-polymer, if we have non-polymer residues
        elif chain_type_counts[ChainType.NON_POLYMER] > 0:
            inferred_chain_type_str = str(ChainType.NON_POLYMER)
            is_polymer = False
        else:
            raise ValueError(f"Could not infer chain type for chain {chain_id}")

        # ...if the all atoms are "hetatm", override the chain type to non-polymer
        if np.all(chain_atom_array.hetero):
            inferred_chain_type_str = str(ChainType.NON_POLYMER)
            is_polymer = False

        chain_info_dict[chain_id] = {"is_polymer": is_polymer, "type": inferred_chain_type_str}

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
