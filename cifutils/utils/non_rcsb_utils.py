"""
Utility functions for handling non-RCSB CIF files.
Such files do not follow the standard CIF format and thus may require special handling.
"""

from biotite.structure import AtomArray
import biotite.structure as struc
from cifutils.common import deduplicate_iterator
from cifutils.utils.residue_utils import get_chem_comp_type
from cifutils.constants import (
    AA_LIKE_CHEM_TYPES,
    POLYPEPTIDE_L_CHEM_TYPES,
    POLYPEPTIDE_D_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
    DNA_LIKE_CHEM_TYPES,
)
import logging

logger = logging.getLogger(__name__)


def load_monomer_sequence_information_from_atom_array(chain_info_dict: dict, atom_array: AtomArray) -> dict:
    """
    Load monomer sequence information into a chain_info_dict using the AtomArray as the ground-truth for
    both polymers and non-polymers.

    Assumes that there are no fully unresolved residues in the AtomArray; otherwise, the sequence will be incomplete.
    """
    for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]
        residue_id_list, residue_name_list = struc.get_residues(chain_atom_array)

        if chain_id not in chain_info_dict:
            chain_info_dict[chain_id] = {}
        chain_info_dict[chain_id]["residue_name_list"] = list(residue_name_list)
        chain_info_dict[chain_id]["residue_id_list"] = list(residue_id_list)

    return chain_info_dict


def infer_chain_info_from_atom_array(atom_array: AtomArray) -> dict:
    """
    Infer chain type information from an AtomArray, in the event that the CIF file does not contain the necessary information.
    Such situations may arise when the CIF file is not from the RCSB PDB database (e.g., distillation).

    TODO: Tests for this function
    TODO: Re-write this function to use the ChainType IntEnum
    """
    logger.warning(
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
