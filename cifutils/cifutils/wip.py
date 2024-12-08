from cifutils.utils.selection_utils import get_residue_starts
from cifutils.constants import UNKNOWN_LIGAND, CCD_MIRROR_PATH
import cifutils.transforms.atom_array as ta
from cifutils.utils.io_utils import load_any
from cifutils.utils.bond_utils import get_inferred_polymer_bonds, get_struct_conn_bonds
from biotite.database.rcsb import fetch
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack, BondList
import os
from cifutils.utils.ccd import check_ccd_codes_are_available, get_ccd_component
import biotite.structure as struc
from typing import Sequence
import logging

logger = logging.getLogger(__file__)
atom_array = load_any(fetch("6lyz", "cif"), model=1)

try:
    from biotite.structure import concatenate
    logger.info("Biotite updated. Please remove the below definition.")
except:
    # TODO: Replace through biotite import after upgrade to 1.0.2
    def concatenate(atoms):
        """
        Concatenate multiple :class:`AtomArray` or :class:`AtomArrayStack` objects into
        a single :class:`AtomArray` or :class:`AtomArrayStack`, respectively.
        Parameters
        ----------
        atoms : iterable object of AtomArray or AtomArrayStack
            The atoms to be concatenated.
            :class:`AtomArray` cannot be mixed with :class:`AtomArrayStack`.
        Returns
        -------
        concatenated_atoms : AtomArray or AtomArrayStack
            The concatenated atoms, i.e. its ``array_length()`` is the sum of the
            ``array_length()`` of the input ``atoms``.
        Notes
        -----
        The following rules apply:
        - Only the annotation categories that exist in all elements are transferred.
        - The box of the first element that has a box is transferred, if any.
        - The bonds of all elements are concatenated, if any element has associated bonds.
        For elements without a :class:`BondList` an empty :class:`BondList` is assumed.
        Examples
        --------
        >>> atoms1 = array([
        ...     Atom([1,2,3], res_id=1, atom_name="N"),
        ...     Atom([4,5,6], res_id=1, atom_name="CA"),
        ...     Atom([7,8,9], res_id=1, atom_name="C")
        ... ])
        >>> atoms2 = array([
        ...     Atom([1,2,3], res_id=2, atom_name="N"),
        ...     Atom([4,5,6], res_id=2, atom_name="CA"),
        ...     Atom([7,8,9], res_id=2, atom_name="C")
        ... ])
        >>> print(concatenate([atoms1, atoms2]))
                    1      N                1.000    2.000    3.000
                    1      CA               4.000    5.000    6.000
                    1      C                7.000    8.000    9.000
                    2      N                1.000    2.000    3.000
                    2      CA               4.000    5.000    6.000
                    2      C                7.000    8.000    9.000
        """
        # Ensure that the atoms can be iterated over multiple times
        if not isinstance(atoms, Sequence):
            atoms = list(atoms)

        length = 0
        depth = None
        element_type = None
        common_categories = set(atoms[0].get_annotation_categories())
        box = None
        has_bonds = False
        for element in atoms:
            if element_type is None:
                element_type = type(element)
            else:
                if not isinstance(element, element_type):
                    raise TypeError(
                        f"Cannot concatenate '{type(element).__name__}' "
                        f"with '{element_type.__name__}'"
                    )
            length += element.array_length()
            if isinstance(element, AtomArrayStack):
                if depth is None:
                    depth = element.stack_depth()
                else:
                    if element.stack_depth() != depth:
                        raise IndexError("The stack depths are not equal")
            common_categories &= set(element.get_annotation_categories())
            if element.box is not None and box is None:
                box = element.box
            if element.bonds is not None:
                has_bonds = True

        if element_type == AtomArray:
            concat_atoms = AtomArray(length)
        elif element_type == AtomArrayStack:
            concat_atoms = AtomArrayStack(depth, length)
        concat_atoms.coord = np.concatenate([element.coord for element in atoms], axis=-2)
        for category in common_categories:
            concat_atoms.set_annotation(
                category,
                np.concatenate(
                    [element.get_annotation(category) for element in atoms], axis=0
                ),
            )
        concat_atoms.box = box
        if has_bonds:
            # Concatenate bonds of all elements
            concat_atoms.bonds = BondList.concatenate(
                [
                    element.bonds
                    if element.bonds is not None
                    else BondList(element.array_length())
                    for element in atoms
                ]
            )

        return concat_atoms


def get_empty_ccd_template(
    ccd_code: str,
    ccd_mirror_path: os.PathLike,
    chain_id: str,
    res_id: int,
    remove_hydrogens: bool = True,
    **annotations: int | float | str,
) -> AtomArray:
    template_cc = get_ccd_component(ccd_code, ccd_mirror_path, coords=None)

    if remove_hydrogens:
        template_cc = ta.remove_hydrogens(template_cc)

    # set default annotations
    for annot, value in annotations.items():
        if value is not None:
            template_cc.set_annotation(annot, np.full(len(template_cc), value))

    return template_cc


def match_residue_to_template(
    real: AtomArray,
    has_bfactor: bool,
    use_ccd_charges: bool,
    keep_hydrogens: bool,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH
):
    ccd_code = real.res_name[0]

    # ... get empty template (no occupation, nan coordinates)
    template = get_empty_ccd_template(
        ccd_code,
        ccd_mirror_path=ccd_mirror_path,
        chain_id=real.chain_id[0],
        res_id=real.res_id[0],
        occupancy=0.0,
        bfactor=np.nan if has_bfactor else None,
        keep_hydrogens=keep_hydrogens,
    )

    # ... determine whether to use the standard or alternative atom naming
    match_by = "atom_name"
    if "alt_atom_id" in template.get_annotation_categories():
        n_matches_std = np.sum(np.isin(real.atom_name, template.atom_name))
        n_matches_alt = np.sum(np.isin(real.atom_name, template.alt_atom_id))
        match_by = "alt_atom_id"
    # ... and record what we used to match
    template.set_annotation("uses_alt_atom_id", [(match_by == "alt_atom_id")] * len(template))

    # ... match the atoms that exist in the chemical component
    for atom in real:
        # match_by: 'atom_name' or 'alt_atom_id'
        match = np.where(template.get_annotation(match_by) == atom.atom_name)[0]

        if len(match) == 0:
            # ... drop atoms that are not in the template (!This should not happen, except for UNK/X/DX!)
            logger.warning(f"{ccd_code}: Atom {atom} not found in template {template}. Will be dropped.")
            continue
        elif len(match) > 1:
            # ... drop atoms that are duplicated in the template (!This should not happen!)
            logger.warning(
                f"{ccd_code}: Atom {atom} found multiple times in template {template}. Only first will be matched."
            )
            continue

        # ... copy over the annotations
        template.coord[match] = atom.coord
        template.occupancy[match] = atom._annot.get("occupancy", 1.0)
        template.ins_code[match] = atom._annot.get("ins_code", template.ins_code[match])
        if has_bfactor:
            template.bfactor[match] = atom._annot.get("bfactor", np.nan)
        if not use_ccd_charges:
            template.charge[match] = atom._annot.get("charge", template.charge[match])
    
    # ... return matched array
    return template


def match_array_to_template(
    atom_array: AtomArray,
    use_ccd_charges: bool = True,
    keep_hydrogens: bool = False,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> list[AtomArray]:
    
    has_bfactor = "bfactor" in atom_array.get_annotation_categories()    

    # Initialize return variable
    template_list: list[AtomArray] = []

    _cc_start_stop_idxs = get_residue_starts(atom_array, add_exclusive_stop=True)
    for cc_start, cc_stop in zip(_cc_start_stop_idxs[:-1], _cc_start_stop_idxs[1:]):
        ccd_code = atom_array.res_name[cc_start]
        real = atom_array[cc_start:cc_stop]

        # ... if the chemical component is not in the CCD or unknown
        #     we cannot get a template and therefore just copy over from the
        #     atom array
        if (ccd_code is UNKNOWN_LIGAND) or not check_ccd_codes_are_available([ccd_code], ccd_mirror_path):
            n_atoms = real.array_length()
            real.set_annotation("stereo", np.full(n_atoms, fill_value="N", dtype="<U1")) 
            real.set_annotation("is_leaving_atom", np.zeros(n_atoms, dtype=bool))
            real.set_annotation("is_backbone_atom", np.zeros(n_atoms, dtype=bool))
            real.set_annotation("is_n_terminal_atom", np.zeros(n_atoms, dtype=bool))
            real.set_annotation("is_c_terminal_atom", np.zeros(n_atoms, dtype=bool))
            real.set_annotation("uses_alt_atom_id", np.zeros(n_atoms, dtype=bool))
            template_list.extend(real)
            continue

        # ... otherwise, we can get a template from the CCD
        template = match_residue_to_template(
            real=real,
            has_bfactor=has_bfactor,
            use_ccd_charges=use_ccd_charges,
            keep_hydrogens=keep_hydrogens
        )

        template_list.append(template)

    return template_list


def get_leaving_atoms(
        atom_array: AtomArray,
        atom_idxs_to_check: np.ndarray
) -> np.ndarray:
    
    atom_array.set_annotation("idx", np.arange(atom_array.array_length()))
    is_atom_leaving = np.zeros(atom_array.array_length(), dtype=bool) 
    
    for idx in atom_idxs_to_check:
        # TODO: Maybe pre-compute these?
        # ... get all atoms bonded to this idx
        bonded_atoms = atom_array.bonds.get_bonded_atoms(idx)

        # ... filter down to atoms that are in the same residue
        #     as the atom to check
        in_same_res = atom_array.res_id[bonded_atoms] == atom_array.res_id[idx]
        #     and atoms that are flagged as possible leaving groups
        can_leave = atom_array.is_leaving_atom[bonded_atoms]
        maybe_leaving_atoms = bonded_atoms[in_same_res & can_leave]

        if len(maybe_leaving_atoms) == 0:
            logger.info(
                f"Atom {atom_array[idx]} is involved in an inter-residue bond, "
                "but appears to not have a leaving group."
            )
        
        # ... perform a bond-graph walk on the residue, starting from 
        #     any of the `maybe_leaving_atoms` and find groups attached
        #     to each 'maybe_leaving_atom'
        for leaving_atom_idx in maybe_leaving_atoms:



def add_missing_atoms(
        atom_array: AtomArray
) -> AtomArray:
    # TODO(smathis)

    # ... match all residues to a CCD template 
    #     (unless no CCD template esits, in which case we copy over)
    #     this also creates the intra-residue bonds from the CCD
    matched_templates = match_array_to_template(atom_array)

    # ... concatenate individual residues to an AtomArray
    atoms = struc.concatenate(matched_templates)
    
    # ... create inter-residue polymer bonds
    polymer_bonds = get_inferred_polymer_bonds(atoms)
    atoms.bonds = atoms.bonds.merge(polymer_bonds)

    # ... create any remaining inter-residue bonds that
    #     are specified in struct_conn
    struct_conn_bonds = get_struct_conn_bonds()
    atoms.bonds = atoms.bonds.merge(struct_conn_bonds)

    # ... check which atoms to inspect for leaving groups:
    atom_idxs_with_inter_bonds = np.unique(polymer_bonds[:, :2].flatten())

    # ... remove leaving group atoms
    is_atom_leaving = get_leaving_atoms(atom_array, atom_idxs_to_check=atom_idxs_with_inter_bonds)

    # ... fix charges of bonded atoms

    return atoms

