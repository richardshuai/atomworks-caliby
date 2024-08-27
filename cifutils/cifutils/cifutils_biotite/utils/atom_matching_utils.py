"""
Utility functions to support proper matching and resolution of residue atoms in a structure.
"""

from cifutils.cifutils_biotite.utils.io_utils import logger
import numpy as np
from biotite.structure.atoms import AtomArray
from cifutils.cifutils_biotite.common import exists
import biotite.structure as struc
from functools import cache
from biotite.structure import Atom


def get_matching_atom(res: AtomArray, atom_name: str, try_alt_atom_id: bool = True) -> Atom:
    """Selects a `single` atom from a residue that matches the given atom name or alternative atom id.
    If none or more are found it raises an error.
    """
    # If the atom name exists, simply select it
    res_name = res.res_name[0]
    atom = res[res.atom_name == atom_name]

    if len(atom) == 0 and try_alt_atom_id:
        # if the atom name does not exist, try the alternative atom id
        # as the alternative atom id's are sometimes used and try to
        # match the alternative atom id
        std_alt_map = get_std_alt_atom_id_conversion(res_name)
        if atom_name in std_alt_map["std_to_alt"]:
            # ... try looking for the atom by its standard name
            atom = res[res.atom_name == std_alt_map["std_to_alt"][atom_name]]
        elif atom_name in std_alt_map["alt_to_std"]:
            # ... otherwise try looking for the atom by its alternative name
            atom = res[res.atom_name == std_alt_map["alt_to_std"][atom_name]]
        else:
            # ... set atom to be empty to trigger error
            atom = []

    if len(atom) != 1:
        # Check if we found a matching atom, otherwise error
        msg = f"Found {len(atom)} matching atoms for {atom_name} in {res_name}:\n{res}\n\n"
        raise ValueError(msg)

    return atom


@cache
def get_std_alt_atom_id_conversion(res_name: str) -> dict:
    std_atom_ids = struc.info.ccd.get_from_ccd("chem_comp_atom", res_name, "atom_id")
    alt_atom_ids = struc.info.ccd.get_from_ccd("chem_comp_atom", res_name, "alt_atom_id")

    assert exists(std_atom_ids) and (
        len(std_atom_ids) > 0
    ), f"{res_name} info does not exist in biotite's CCD. Try to update it to fix this assertion."
    assert len(std_atom_ids) == len(
        alt_atom_ids
    ), f"{res_name} has {len(std_atom_ids)} standard atom ids and {len(alt_atom_ids)} alternative atom ids"

    mapping = {"std_to_alt": dict(zip(std_atom_ids, alt_atom_ids)), "alt_to_std": dict(zip(alt_atom_ids, std_atom_ids))}

    return mapping


def standardize_heavy_atom_ids(atom_array: AtomArray) -> np.ndarray:
    _found_alt_atom_ids = 0
    atom_name_all = []
    for res in struc.residue_iter(atom_array):
        res_name = res.res_name

        # NOTE: We do not rename any H atoms, as we only care about
        #  covalent bonds in the struct_conn category later and so
        #  we will never have to match up H's.
        is_heavy = res.element != 1  # 1 is hydrogen, deuterium, tritium here
        is_heavy &= ~np.isin(res.element, ["H", "D", "H2", "T", "1"])

        atom_name = res.atom_name

        # Check if an atom array uses standard atom ids
        try:
            mapping = get_std_alt_atom_id_conversion(res_name[0])
        except AssertionError as e:
            # deal with residues which do not yet exist in biotite's CCD
            # skip, but warn
            logger.info(
                f"{e.__class__.__name__}: {e}. Trying to continue processing, but consider updating biotite's CCD."
            )
            atom_name_all.append(atom_name)
            continue

        std_atoms = np.array(list(mapping["std_to_alt"].keys()))
        if not np.all(np.isin(atom_name[is_heavy], std_atoms)):
            _found_alt_atom_ids += 1
            # Convert to standard atom ids
            atom_name_renamed = np.array(
                [mapping["alt_to_std"].get(atom_id, atom_id) for atom_id in atom_name[is_heavy]]
            )

            # Ensure that renaming created no dupliates
            if len(np.unique(atom_name_renamed)) != len(atom_name_renamed):
                # if updates resulted in non-unique atom names, warn the user and
                # proceed with old atom names
                logger.error(
                    "Duplicate atom names found after renaming. This is likely because a mix of "
                    "standard and alternative atom ids was used in the input residue. Trying to "
                    "proceed without renaming."
                )
            else:
                # if updates are unique, rename and proceed
                atom_name[is_heavy] = atom_name_renamed

        atom_name_all.append(atom_name)

    if _found_alt_atom_ids > 0:
        logger.debug(f"Found {_found_alt_atom_ids} alternative atom ids.")

    return np.concatenate(atom_name_all)
