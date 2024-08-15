"""
Utility functions to handle creation and manipulation of residues.
"""
import numpy as np
from biotite.structure import AtomArray, Atom
import biotite.structure as struc
from cifutils.cifutils_biotite.common import exists
from biotite.structure.io.pdbx import CIFBlock
import logging
from cifutils.cifutils_biotite.utils.atom_matching_utils import get_matching_atom
from cifutils.cifutils_biotite.utils.cifutils_biotite_utils import (
    fix_bonded_atom_charges,
    get_bond_type_from_order_and_is_aromatic,
)
from cifutils.cifutils_biotite.transforms.categories import category_to_df
from functools import lru_cache

logger = logging.getLogger(__name__)

def cached_residue_utils_factory(known_residues: list[str], data_by_residue: callable) -> tuple[callable, callable]:
    """ 
    Factory function to build cached helper functions for building residue atoms.
    We must invoke closure since dictionaries are not hashable and cannot be used as keys in lru_cache.

    Args:
        known_residues (list): A list of valid residue names.
        data_by_residue (callable): A function that returns CCD data for a given residue name.
    """
    known_residues_set = set(known_residues)

    @lru_cache(maxsize=None)
    def build_residue_atoms(residue_name: str, add_hydrogens: bool) -> list[Atom]:
        """
        Build a list of atoms for a given residue name from CCD data.

        Args:
            residue_name (str): The name of the residue.
            add_hydrogens (bool): Whether to add hydrogens to the residue.

        Returns:
            list[Atom]: A list of Atom objects initialized with zero coordinates.
        """
        if residue_name not in known_residues_set:
            raise ValueError(f"Residue {residue_name} not found in precompiled CCD data.")

        ccd_atoms = data_by_residue(residue_name)["atoms"]
        atom_list = [
            struc.Atom(
                [0.0, 0.0, 0.0],
                res_name=residue_name,
                atom_name=atom_name,
                element=atom_data["element"],
                charge=atom_data["charge"],
                leaving_atom_flag=atom_data["leaving_atom_flag"],
                leaving_group=atom_data["leaving_group"],
                is_metal=atom_data["is_metal"],
                hyb=atom_data["hyb"],
                nhyd=atom_data["nhyd"],
                hvydeg=atom_data["hvydeg"],
                align=atom_data["align"],
            )
            for atom_name, atom_data in ccd_atoms.items()
        ]

        # Remove hydrogens, if necessary
        if not add_hydrogens:
            atom_list = [atom for atom in atom_list if atom.element != 1]

        return atom_list
    
    return build_residue_atoms