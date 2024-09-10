import logging
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from datahub.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from datahub.transforms.atom_array import add_global_token_id_annotation
from datahub.transforms.base import Transform
from datahub.transforms.rdkit_utils import atom_array_from_rdkit, res_name_to_rdkit_with_conformers

logger = logging.getLogger("datahub")


def _get_rdkit_mols_with_conformers(
    res_stochiometry: dict[str, int], timeout_seconds: float = 10.0, **generate_conformers_kwargs
) -> dict[str, Chem.Mol]:
    """
    Generate RDKit molecules with conformers for each residue (given the counts in `res_stochiometry`).

    Args:
        res_stochiometry (dict[str, int]): A dictionary mapping residue names to their count.
        timeout_seconds (float, optional): Maximum time allowed for conformer generation per residue.
            Defaults to 10.0 seconds.
        **generate_conformers_kwargs: Additional keyword arguments to pass to the
            generate_conformers function.

    Returns:
        dict[str, Chem.Mol]: A dictionary mapping residue names to RDKit molecules with generated conformers.

    Note:
        This function uses the res_name_to_rdkit_with_conformers function to generate conformers
        for each residue. If conformer generation fails or times out for a residue, it falls back
        to using the idealized conformer from the CCD entry if available.

    Reference:
        - https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """
    ref_mols = {}
    for res_name, count in res_stochiometry.items():
        mol = res_name_to_rdkit_with_conformers(
            res_name=res_name, n_conformers=count, timeout_seconds=timeout_seconds, **generate_conformers_kwargs
        )
        ref_mols[res_name] = mol

    return ref_mols


def _encode_atom_names_like_af3(atom_names: np.ndarray) -> np.ndarray:
    """Encodes atom names like AF3.

    This generates the `ref_atom_name_chars` feature used in AF3.
        One-hot encoding of the unique atom names in the reference conformer.
        Each character is encoded as ord(c) - 32, and names are padded to
        length 4.

    Reference:
        - https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """
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


def get_af3_reference_molecule_features(
    atom_array: AtomArray, conformer_generation_timeout: float = 10.0, **generate_conformers_kwargs
) -> dict[str, Any]:
    """
    Get AF3 reference features for each residue in the atom array.

    Args:
        - atom_array (AtomArray): The input atom array.
        - conformer_generation_timeout (float, optional): Maximum time allowed for conformer generation per residue.
            Defaults to 10.0 seconds.
        - **generate_conformers_kwargs: Additional keyword arguments to pass to the generate_conformers function.

    Returns:
        dict[str, Any]: A dictionary containing the generated reference features.

    This function generates the following reference features for AF3:
    - ref_pos: [N_atoms, 3] Atom positions in the reference conformer, with a random rotation and
      translation applied. Atom positions are given in Å.
    - ref_mask: [N_atoms] Mask indicating which atom slots are used in the reference conformer.
    - ref_element: [N_atoms] One-hot encoding of the element atomic number for each atom in the
      reference conformer, up to atomic number 128.
    - ref_charge: [N_atoms] Charge for each atom in the reference conformer.
    - ref_atom_name_chars: [N_atoms, 4, 64] One-hot encoding of the unique atom names in the reference conformer.
      Each character is encoded as ord(c) - 32, and names are padded to length 4.
    - ref_space_uid: [N_atoms] Numerical encoding of the chain id and residue index associated with
      this reference conformer. Each (chain id, residue index) tuple is assigned an integer on first appearance.

    Reference:
        - Section 2.8 of the AF3 supplementary information
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """
    # Generate reference conformers for each residue (if cropped, each residue that has tokens in the crop)
    # ... get residue-level stochiometry
    _, _res_names = struc.get_residues(atom_array)
    res_stochiometry = dict(zip(*np.unique(_res_names, return_counts=True)))

    # ... get reference molecules with conformers for each residue
    ref_mols = _get_rdkit_mols_with_conformers(
        res_stochiometry=res_stochiometry, timeout_seconds=conformer_generation_timeout, **generate_conformers_kwargs
    )

    # ... get reference positions for each residue
    ref_pos = np.zeros((len(atom_array), 3), dtype=np.float32)
    ref_mask = np.zeros((len(atom_array), 3), dtype=bool)

    # Fill `ref_pos` and `ref_mask` arrays
    # ... helper variables to iterate over residues
    _res_start_ends = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]
    # ... and to keep track of the next conformer to use for each residue type
    _next_conf_idx = {res_name: 0 for res_name in ref_mols}

    # ... iterate over all residues in the atom array and fill the `ref_pos` and `ref_mask` arrays
    #     using the next reference conformer for each residue type
    for _res_start, _res_end in zip(_res_starts, _res_ends):
        res_name = atom_array.res_name[_res_start]

        # ... turn conformer into an atom array
        conf_arr = atom_array_from_rdkit(
            ref_mols[res_name],
            conformer_id=_next_conf_idx[res_name],
            remove_hydrogens=True,
        )

        # ... match the atoms that are in present in the residue and set the reference position and mask
        for _atom_idx, atom_name in enumerate(atom_array.atom_name[_res_start:_res_end]):
            matching_atom_idx = np.where(conf_arr.atom_name == atom_name)[0]
            if len(matching_atom_idx) == 0:
                logger.warning(f"Atom {atom_name} not found in conformer for residue {res_name}.")
                continue
            matching_atom_idx = matching_atom_idx[0]
            coords = conf_arr.coord[matching_atom_idx]
            ref_pos[_res_start + _atom_idx] = coords
            ref_mask[_res_start + _atom_idx] = np.isfinite(coords).all()

        _next_conf_idx[res_name] += 1

    # Generate remaining reference features
    # ... element
    ref_element = atom_array.element.astype(int)
    # ... charge
    ref_charge = atom_array.charge
    # ... atom name
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_array.atom_name)
    # ... space uid (type conversion needed for some older torch versions)
    ref_space_uid = add_global_token_id_annotation(atom_array).token_id.astype(np.int64)

    return {
        "ref_pos": ref_pos,  # (n_atoms, 3)
        "ref_mask": ref_mask,  # (n_atoms)
        "ref_element": ref_element,  # (n_atoms)
        "ref_charge": ref_charge,  # (n_atoms)
        "ref_atom_name_chars": ref_atom_name_chars,  # (n_atoms, 4)
        "ref_space_uid": ref_space_uid,  # (n_atoms)
    }


class GetAF3ReferenceMoleculeFeatures(Transform):
    """
    Generate AF3 reference molecule features for each residue in the atom array.

    This transform adds the following features to the data dictionary under the 'feats' key:
        - ref_pos: [N_atoms, 3] Atom positions in the reference conformer, with a random rotation and
          translation applied. Atom positions are given in Å.
        - ref_mask: [N_atoms] Mask indicating which atom slots are used in the reference conformer.
        - ref_element: [N_atoms] One-hot encoding of the element atomic number for each atom in the
          reference conformer, up to atomic number 128.
        - ref_charge: [N_atoms] Charge for each atom in the reference conformer.
        - ref_atom_name_chars: [N_atoms, 4, 64] One-hot encoding of the unique atom names in the reference conformer.
          Each character is encoded as ord(c) - 32, and names are padded to length 4.
        - ref_space_uid: [N_atoms] Numerical encoding of the chain id and residue index associated with
          this reference conformer. Each (chain id, residue index) tuple is assigned an integer on first appearance.

    Note: This transform should be applied after cropping.

    Reference:
        - Section 2.8 of the AF3 supplementary information
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """

    def __init__(self, conformer_generation_timeout: float = 10.0, **generate_conformers_kwargs):
        self.conformer_generation_timeout = conformer_generation_timeout
        self.generate_conformers_kwargs = generate_conformers_kwargs

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name", "element", "charge", "atom_name"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Generate reference features
        reference_features = get_af3_reference_molecule_features(
            atom_array,
            conformer_generation_timeout=self.conformer_generation_timeout,
            **self.generate_conformers_kwargs,
        )

        # Add reference features to the 'feats' dictionary
        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(reference_features)

        return data
