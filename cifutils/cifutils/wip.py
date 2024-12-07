# Add missing atoms:

from cifutils.utils.selection_utils import get_residue_starts
from cifutils.constants import UNKNOWN_LIGAND, CCD_MIRROR_PATH
import cifutils.transforms.atom_array as ta
from cifutils.utils.io_utils import load_any
from biotite.database.rcsb import fetch
import os
from cifutils.utils.ccd import check_ccd_codes_are_available

atom_array = load_any(fetch("6lyz", "cif"), model=1)


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


def add_missing_atoms(
    atom_array: AtomArray,
    use_ccd_charges: bool = True,
    keep_hydrogens: bool = True,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> list[AtomArray]:
    chem_comp_list = []
    has_bfactor = "bfactor" in atom_array.get_annotation_categories()

    _cc_start_stop_idxs = get_residue_starts(atom_array, add_exclusive_stop=True)
    for cc_start, cc_stop in zip(_cc_start_stop_idxs[:-1], _cc_start_stop_idxs[1:]):
        ccd_code = atom_array.res_name[cc_start]

        # 1. If the chemical component is not in the CCD or unknown
        #    we cannot get a template and therefore just copy over from the
        #    atom array
        if (ccd_code is UNKNOWN_LIGAND) or not check_ccd_codes_are_available([ccd_code], ccd_mirror_path):
            chem_comp_list.append(atom_array[cc_start:cc_stop])
            continue

        # 2a. Otherwise, we can get a template from the CCD
        template_cc = get_empty_ccd_template(
            ccd_code,
            ccd_mirror_path=ccd_mirror_path,
            chain_id=atom_array.chain_id[cc_start],
            res_id=atom_array.res_id[cc_start],
            occupancy=0.0,
            bfactor=np.nan if has_bfactor else None,
            keep_hydrogens=keep_hydrogens,
        )

        # 2b. ... and match the atoms that exist in the chemical component
        real_cc = atom_array[cc_start:cc_stop]
        for atom in real_cc:
            match = np.where(template_cc.atom_name == atom.atom_name)[0]

            if len(match) == 0:
                # ... drop atoms that are not in the template (!This should not happen, except for UNK/X/DX!)
                logger.warning(f"{ccd_code}: Atom {atom} not found in template {template_cc}. Will be dropped.")
                continue
            elif len(match) > 1:
                # ... drop atoms that are duplicated in the template (!This should not happen!)
                logger.warning(
                    f"{ccd_code}: Atom {atom} found multiple times in template {template_cc}. Only first will be matched."
                )
                continue

            # ... copy over the annotations
            template_cc.coord[match] = atom.coord
            template_cc.occupancy[match] = atom._annot.get("occupancy", 1.0)
            template_cc.ins_code[match] = atom._annot.get("ins_code", template_cc.ins_code[match])
            if has_bfactor:
                template_cc.bfactor[match] = atom._annot.get("bfactor", np.nan)
            if not use_ccd_charges:
                template_cc.charge[match] = atom._annot.get("charge", template_cc.charge[match])

        chem_comp_list.append(template_cc)

    return chem_comp_list


add_missing_atoms(atom_array)