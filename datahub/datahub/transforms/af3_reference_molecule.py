import logging
from typing import Any

import numpy as np
import toolz
from biotite.structure import AtomArray
from cifutils.utils.selection_utils import get_residue_starts
from rdkit import Chem

from datahub.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from datahub.transforms.base import Transform
from datahub.transforms.rdkit_utils import atom_array_from_rdkit, find_automorphisms, res_name_to_rdkit_with_conformers

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


def _map_reference_conformer_to_residue(
    res_name: str, atom_names: np.ndarray, conformer: AtomArray, automorphs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Maps the coordinate and automorphism information from a reference conformer to a
    given residue, dropping all atoms that are not in the residue.

    Args:
        - res_name (str): The name of the residue to map to.
        - atom_names (np.ndarray): Array of atom names in the residue to map to.
        - conformer (AtomArray): The reference conformer.
        - automorphs (np.ndarray): Array of automorphisms for the conformer.

    Returns:
        - ref_pos (np.ndarray): Reference positions for atoms in the residue.
        - ref_mask (np.ndarray): Mask indicating valid reference positions.
        - automorphs (np.ndarray): Filtered and adjusted automorphisms for the residue.
    """

    # ... mark the atoms that are in the residue (keep) and where
    keep = np.zeros(len(conformer), dtype=bool)  # [n_atoms_in_conformer]
    to_within_res_idx = -np.ones(len(conformer), dtype=int)  # [n_atoms_in_conformer]

    for i, atom_name in enumerate(atom_names):
        matching_atom_idx = np.where(conformer.atom_name == atom_name)[0]
        if len(matching_atom_idx) == 0:
            logger.warning(f"Atom {atom_name} not found in conformer for residue {res_name} with {atom_names=}.")
            continue
        matching_atom_idx = matching_atom_idx[0]
        keep[matching_atom_idx] = True
        to_within_res_idx[matching_atom_idx] = i

    # ... fill the reference positions
    coord = conformer.coord[keep][to_within_res_idx[keep]]
    ref_pos = coord  # [n_atoms_in_res, 3]
    ref_mask = np.isfinite(coord).all(axis=-1)  # [n_atoms_in_res]

    # ... filter the automorphs to only keep the ones that are relevant
    # ... 1. change the 'in-conformer' index to the 'in-residue' index,
    #        dropping any atoms that are not in the residue
    automorphs = to_within_res_idx[automorphs][:, keep, :]  # [n_automorphs, n_atoms_in_res, 2]
    # ... 2. drop any automorphs that would include atoms not in the residue
    #        (-1 got assigned to atoms not in residue)
    _has_all_atoms_in_residue = (automorphs >= 0).all(axis=(-1, -2))  # [n_automorphs]
    automorphs = automorphs[_has_all_atoms_in_residue]
    # ... 3. drop any automorphs that are duplicates when only considering
    #        the atoms that are in the residue
    _, _is_first_unique = np.unique(automorphs, axis=0, return_index=True)
    _is_first_unique = np.sort(_is_first_unique)
    automorphs = automorphs[_is_first_unique]

    return ref_pos, ref_mask, automorphs  # [n_atoms_in_res, 3], [n_atoms_in_res], [n_automorphs, n_atoms_in_res, 2]


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
    - ref_automorphs: dict(int, torch.Tensor): A dictionary mapping the `ref_space_uid` to the automorphisms
        of the reference conformer.

    Reference:
        - Section 2.8 of the AF3 supplementary information
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """
    # Generate reference conformers for each residue (if cropped, each residue that has tokens in the crop)
    # ... get residue-level stochiometry
    _res_start_ends = get_residue_starts(atom_array, add_exclusive_stop=True)
    _res_starts, _res_ends = _res_start_ends[:-1], _res_start_ends[1:]
    _res_names = atom_array.res_name[_res_starts]
    res_stochiometry = dict(zip(*np.unique(_res_names, return_counts=True)))

    # ... get reference molecules with conformers for each residue
    ref_mols = _get_rdkit_mols_with_conformers(
        res_stochiometry=res_stochiometry, timeout_seconds=conformer_generation_timeout, **generate_conformers_kwargs
    )

    # ... get automorphisms for each molecule
    ref_mol_automorphs = toolz.valmap(find_automorphisms, ref_mols)
    _max_automorphs = max(map(len, ref_mol_automorphs.values()))

    # ... get reference positions for each residue
    ref_pos = np.zeros((len(atom_array), 3), dtype=np.float32)
    ref_mask = np.zeros(len(atom_array), dtype=bool)
    ref_automorphs = np.zeros((_max_automorphs, len(atom_array), 2), dtype=int)
    ref_automorphs_mask = np.zeros((_max_automorphs, len(atom_array)), dtype=bool)

    # Fill `ref_pos` and `ref_mask` arrays
    # ... helper variable to keep track of the next conformer to use for each residue type
    _next_conf_idx = {res_name: 0 for res_name in ref_mols}

    # ... iterate over all residues in the atom array and fill the `ref_pos` and `ref_mask` arrays
    #     using the next reference conformer for each residue type
    max_automorphs = 1
    for res_start, res_end in zip(_res_starts, _res_ends):
        res_name = atom_array.res_name[res_start]

        # ... turn conformer into an atom array
        conformer = atom_array_from_rdkit(
            ref_mols[res_name],
            conformer_id=_next_conf_idx[res_name],
            remove_hydrogens=True,
        )

        # ... map the reference conformer information to the given residue
        _ref_pos, _ref_mask, _ref_automorphs = _map_reference_conformer_to_residue(
            res_name=res_name,
            atom_names=atom_array.atom_name[res_start:res_end],
            conformer=conformer,
            automorphs=ref_mol_automorphs[res_name],
        )

        # ... fill the reference features for this residue
        ref_pos[res_start:res_end] = _ref_pos
        ref_mask[res_start:res_end] = _ref_mask
        ref_automorphs[: len(_ref_automorphs), res_start:res_end] = _ref_automorphs
        ref_automorphs_mask[: len(_ref_automorphs), res_start:res_end] = True
        max_automorphs = max(max_automorphs, len(_ref_automorphs))

        # ... update to the next conformer index
        _next_conf_idx[res_name] += 1

    # ... resize the reference automorphism arrays to the maximum number of automorphisms
    ref_automorphs = ref_automorphs[:max_automorphs]
    ref_automorphs_mask = ref_automorphs_mask[:max_automorphs]

    # Generate remaining reference features
    # ... element
    ref_element = atom_array.element.astype(int)
    # ... charge
    ref_charge = atom_array.charge
    # ... atom name
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_array.atom_name)
    # ... space uid (type conversion needed for some older torch versions)
    ref_space_uid = atom_array.token_id.astype(np.int64)
    return {
        "ref_pos": ref_pos,  # (n_atoms, 3)
        "ref_mask": ref_mask,  # (n_atoms)
        "ref_element": ref_element,  # (n_atoms)
        "ref_charge": ref_charge,  # (n_atoms)
        "ref_atom_name_chars": ref_atom_name_chars,  # (n_atoms, 4)
        "ref_space_uid": ref_space_uid,  # (n_atoms)
        "ref_automorphs": ref_automorphs,  # (max_automorphs, n_atoms, 2), residue-local indices
        "ref_automorphs_mask": ref_automorphs_mask,  # (max_automorphs, n_atoms)
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

    requires_previous_transforms = ["AddGlobalTokenIdAnnotation"]

    def __init__(self, conformer_generation_timeout: float = 10.0, **generate_conformers_kwargs):
        self.conformer_generation_timeout = conformer_generation_timeout
        self.generate_conformers_kwargs = generate_conformers_kwargs

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name", "element", "charge", "atom_name", "token_id"])
        atom_array = data["atom_array"]
        assert (
            len(atom_array[atom_array.atom_name == "OXT"]) == 0
        ), "OXT atoms should be removed before applying this transform."

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
