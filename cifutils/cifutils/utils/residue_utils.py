"""
Utility functions to handle creation and manipulation of residues.
"""

__all__ = [
    "get_chem_comp_type",
    "build_chem_comp_atom_list",
    "add_missing_atoms_as_unresolved",
]

import numpy as np
from biotite.structure import Atom
import biotite.structure as struc
import logging
from cifutils.common import (
    exists,
)
import os
from functools import cache
from cifutils.constants import CCD_MIRROR_PATH
from cifutils.common import not_isin
from cifutils.constants import HYDROGEN_LIKE_SYMBOLS
from cifutils.utils.ccd import get_ccd_component

logger = logging.getLogger("cifutils")


@cache
def _chem_comp_type_dict() -> dict[str, str]:
    """
    Get a dictionary of all residue names and their corresponding chemical component types.
    """
    ccd = struc.info.ccd.get_ccd()  # NOTE: biotite caches this internally
    chem_comp_ids = np.char.upper(ccd["chem_comp"]["id"].as_array())
    chem_comp_types = np.char.upper(ccd["chem_comp"]["type"].as_array())
    return dict(zip(chem_comp_ids, chem_comp_types))


def get_chem_comp_type(ccd_code: str, strict: bool = False) -> str:
    """
    Get the chemical component type for a CCD code from the Chemical Component Dictionary (CCD).
    Can be combined with CHEM_TYPES from `cifutils_biotite.constants` to determine if a component is a
    protein, nucleic acid, or carbohydrate.

    Args:
        ccd_code (str): The CCD code for the component. E.g. `ALA` for alanine, `NAP` for N-acetyl-D-glucosamine.

    Example:
        >>> get_chem_comp_type("ALA")
        'L-PEPTIDE LINKING'
    """
    chem_comp_type = _chem_comp_type_dict().get(ccd_code, None)

    # ... handle unknown chemical component types
    if not exists(chem_comp_type):
        if strict:
            # ... raise an error if we want to fail loudly
            raise ValueError(f"Chemical component type for `{ccd_code=}` not found in CCD.")

        # ... otherwise set chemical component type to "other" - the equivalent of unknown.
        logger.info(f"Chemical component type for `{ccd_code=}` not found in CCD. Using 'other'.")
        chem_comp_type = "OTHER"

    return chem_comp_type


@cache
def build_chem_comp_atom_list(
    ccd_code: str, keep_hydrogens: bool, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH
) -> list[Atom]:
    """
    Build a list of atoms for a given chemical component code (CCD code) from CCD data.

    Args:
        ccd_code (str): The CCD code for the residue. E.g. `ALA` for alanine, `NAP` for N-acetyl-D-glucosamine.
        keep_hydrogens (bool): Whether to add hydrogens to the residue.

    Returns:
        list[Atom]: A list of Atom objects initialized with zero coordinates.
    """

    ccd_atoms = get_ccd_component(ccd_code, ccd_mirror_path, coords=None, add_properties=False)
    if not keep_hydrogens:
        ccd_atoms = ccd_atoms[not_isin(ccd_atoms.element, HYDROGEN_LIKE_SYMBOLS)]

    return [atom for atom in ccd_atoms]
