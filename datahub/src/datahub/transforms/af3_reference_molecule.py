import logging
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import toolz
import torch
from biotite.structure import AtomArray
from cifutils.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER, UNKNOWN_LIGAND
from cifutils.tools.rdkit import atom_array_from_rdkit
from cifutils.utils.ccd import get_available_ccd_codes
from cifutils.utils.selection import get_residue_starts
from rdkit import Chem

from datahub.enums import GroundTruthConformerPolicy
from datahub.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from datahub.transforms.base import Transform
from datahub.transforms.rdkit_utils import (
    ccd_code_to_rdkit_with_conformers,
    find_automorphisms_with_rdkit,
    sample_rdkit_conformer_for_atom_array,
)
from datahub.utils.geometry import masked_center, random_rigid_augmentation

logger = logging.getLogger("datahub")

# UNL is a special CCD code for unknown ligands; we do not consider it "known" as it has no structure
KNOWN_CCD_CODES = get_available_ccd_codes() - {UNKNOWN_LIGAND}


def _get_rdkit_mols_with_conformers(
    res_stochiometry: dict[str, int],
    timeout: float | None = 10.0,
    timeout_strategy: Literal["signal", "subprocess"] = "subprocess",
    **generate_conformers_kwargs,
) -> dict[str, Chem.Mol]:
    """Generate RDKit molecules with conformers for each residue in bulk (given the counts in `res_stochiometry`).

    Args:
        res_stochiometry (dict[str, int]): A dictionary mapping residue names to their count.
        timeout (float | None): The timeout for the automorphism search. If None, no timeout is applied and
            the timeout strategy is ignored (no subprocesses will be spawned). Defaults to 10.0 seconds.
        timeout_strategy (Literal["signal", "subprocess"]): The strategy to use for the timeout.
            Defaults to "subprocess".
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
        if res_name not in KNOWN_CCD_CODES:
            ref_mols[res_name] = None  # placeholder so that the unknown CCD codes are still counted later on
            continue
        mol = ccd_code_to_rdkit_with_conformers(
            ccd_code=res_name, n_conformers=count, timeout=timeout, **generate_conformers_kwargs
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
    res_name: str, atom_names: np.ndarray, conformer: AtomArray, automorphs: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Maps the coordinate and automorphism information from a reference conformer to a
    given residue, dropping all atoms that are not in the residue.

    Args:
        - res_name (str): The name of the residue to map to.
        - atom_names (np.ndarray): Array of atom names in the residue to map to.
        - conformer (AtomArray): The reference conformer.
        - automorphs (np.ndarray | None): Array of automorphisms for the conformer. If not
            provided, no automorphisms are returned.

    Returns:
        - ref_pos (np.ndarray): Reference positions for atoms in the residue.
        - ref_mask (np.ndarray): Mask indicating valid reference positions.
        - automorphs (np.ndarray | None): Filtered and adjusted automorphisms for the residue, if provided.
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

    if automorphs is not None:
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
    atom_array: AtomArray,
    conformer_generation_timeout: float = 10.0,
    should_generate_automorphisms_with_rdkit: bool = True,
    apply_random_rotation_and_translation: bool = True,
    use_element_for_atom_names_of_atomized_tokens: bool = False,
    timeout_strategy: Literal["signal", "subprocess"] = "subprocess",
    **generate_conformers_kwargs,
) -> dict[str, Any]:
    """Get AF3 reference features for each residue in the atom array.

    Args:
        - atom_array (AtomArray): The input atom array.
        - conformer_generation_timeout (float, optional): Maximum time allowed for conformer generation per residue.
            Defaults to 10.0 seconds. If None, no timeout is applied and the timeout strategy is ignored (no subprocesses will be spawned).
        - should_generate_automorphisms_with_rdkit (bool, optional): Whether to generate automorphisms using RDKit. For example,
            we may want to generate automorphisms directly with networkx instead. Defaults to True.
        - apply_random_rotation_and_translation (bool, optional): Whether to apply a random rotation and translation to each conformer (AF-3-style)
        - timeout_strategy (Literal["signal", "subprocess"]): The strategy to use for the timeout.
            Defaults to "subprocess".
        - **generate_conformers_kwargs: Additional keyword arguments to pass to the generate_conformers function.

    Returns:
        dict[str, Any]: A dictionary containing the generated reference features.

    This function generates the following reference features for AF3:
        - ref_pos: [N_atoms, 3] Atom positions in the reference conformer, with a random rotation and
            translation applied. Atom positions are given in Å.
        - ref_mask: [N_atoms] Mask indicating which atom slots are used in the reference conformer.
        - ref_element: [N_atoms, 128] One-hot encoding of the element atomic number for each atom in the
            reference conformer, up to atomic number 128.
        - ref_charge: [N_atoms] Charge for each atom in the reference conformer.
        - ref_atom_name_chars: [N_atoms, 4, 64] One-hot encoding of the unique atom names in the reference conformer.
            Each character is encoded as ord(c) - 32, and names are padded to length 4.
        - ref_space_uid: [N_atoms] Numerical encoding of the chain id and residue index associated with
            this reference conformer. Each (chain id, residue index) tuple is assigned an integer on first appearance.
        - ref_automorphs: A dictionary mapping the `ref_space_uid` to the automorphisms
            of the reference conformer.
        - ref_pos_is_ground_truth (optional): [N_atoms] Whether the reference conformer is the ground-truth conformer.
            Determined by the `ground_truth_conformer_policy` annotation.

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
    _has_explicit_hydrogens = "H" in atom_array.element

    # ... get reference molecules with conformers for each residue
    # (We do not generate conformers for unknown CCD codes here, as we will do that later)
    ref_mols = _get_rdkit_mols_with_conformers(
        res_stochiometry=res_stochiometry,
        hydrogen_policy="auto" if _has_explicit_hydrogens else "remove",
        timeout=conformer_generation_timeout,
        timeout_strategy=timeout_strategy,
        **generate_conformers_kwargs,
    )

    # ... generate conformers for CCD codes that are unknown (e.g., any custom small molecules & UNL)
    unknown_ccd_conformers = defaultdict(list)
    if not all(res_name in KNOWN_CCD_CODES for res_name in res_stochiometry):
        res_indices_with_unknown = np.where(~np.isin(_res_names, list(KNOWN_CCD_CODES)))[0]
        for res_index in res_indices_with_unknown:
            res_name = _res_names[res_index]

            unknown_ccd_conformers[res_name].append(
                sample_rdkit_conformer_for_atom_array(
                    atom_array[_res_starts[res_index] : _res_ends[res_index]],
                    timeout=conformer_generation_timeout,
                    timeout_strategy=timeout_strategy,
                    **generate_conformers_kwargs,
                )
            )

    # ... initialize automorphism-related variables (which we may or may not be needed)
    ref_mol_automorphs = None
    ref_automorphs = None
    ref_automorphs_mask = None

    if should_generate_automorphisms_with_rdkit:
        # ... get automorphisms
        ref_mol_automorphs = toolz.valmap(find_automorphisms_with_rdkit, ref_mols)
        _max_automorphs = max(map(len, ref_mol_automorphs.values()))
        # ...initialize tensors to store automorphisms and masks
        ref_automorphs = np.zeros((_max_automorphs, len(atom_array), 2), dtype=int)
        ref_automorphs_mask = np.zeros((_max_automorphs, len(atom_array)), dtype=bool)

    # ... initialize reference features
    ref_pos = np.zeros((len(atom_array), 3), dtype=np.float32)
    ref_mask = np.zeros(len(atom_array), dtype=bool)
    ref_pos_is_ground_truth = np.zeros(len(atom_array), dtype=bool)

    # Fill `ref_pos` and `ref_mask` arrays
    # ... helper variable to keep track of the next conformer to use for each residue type
    _next_conf_idx = {res_name: 0 for res_name in ref_mols}

    # ... iterate over all residues in the atom array and fill the `ref_pos` and `ref_mask` arrays using the next reference conformer for each residue type
    # We also check the `ground_truth_conformer_policy` annotation to see if we should use the ground-truth conformer
    max_automorphs = 1
    for res_start, res_end in zip(_res_starts, _res_ends):
        res_name = atom_array.res_name[res_start]

        # ... turn conformer into an atom array
        if res_name not in KNOWN_CCD_CODES:
            # (conformers for unknown CCD codes are already atom arrays, since we generated them directly)
            conformer = unknown_ccd_conformers[res_name][_next_conf_idx[res_name]]
        else:
            conformer = atom_array_from_rdkit(
                ref_mols[res_name],
                conformer_id=_next_conf_idx[res_name],
                remove_hydrogens=True,
            )

        if "ground_truth_conformer_policy" in atom_array.get_annotation_categories():
            # We replace the generated conformer with the ground-truth conformer if either:
            # (a) the ground-truth conformer policy is set to "replace" for all atoms in the residue
            # (b) the current conformer is all 0's/NaN's (i.e., the conformer generation failed), and the policy is set to "fallback" for all atoms in the residue
            if np.all(
                atom_array.ground_truth_conformer_policy[res_start:res_end] == GroundTruthConformerPolicy.REPLACE
            ) or (
                np.all(np.nan_to_num(conformer.coord) == 0)
                and np.all(
                    atom_array.ground_truth_conformer_policy[res_start:res_end] == GroundTruthConformerPolicy.FALLBACK
                )
            ):
                # NOTE: Inefficient since we generate with RDKit, and then discard, the conformer; however, this replacement-based approach is more interpretable and thus preferred
                if np.isnan(atom_array.coord[res_start:res_end]).any():
                    logger.warning(
                        "Ground-truth conformer requested, but NaNs found in the atom array. Conformer will not be replaced with ground truth."
                    )
                else:
                    # ... use the ground-truth AtomArray (e.g., during inference if we provide a SDF, or if we want to leak ligand geometry)
                    conformer = atom_array[res_start:res_end]
                    # (Center around the origin to avoid leaking 1D information)
                    conformer.coord = masked_center(conformer.coord)
                    ref_pos_is_ground_truth[res_start:res_end] = True

        # ... map the reference conformer information to the given residue
        _ref_pos, _ref_mask, _ref_automorphs = _map_reference_conformer_to_residue(
            res_name=res_name,
            atom_names=atom_array.atom_name[res_start:res_end],
            conformer=conformer,
            automorphs=ref_mol_automorphs[res_name] if ref_mol_automorphs else None,
        )

        # ... apply a random rotation and translation to the reference conformer, if requested
        if apply_random_rotation_and_translation:
            # TODO: Implement more elegantly directly in numpy
            _ref_pos = random_rigid_augmentation(torch.from_numpy(_ref_pos[np.newaxis, :]), batch_size=1).numpy()

        # ... fill the reference features for this residue
        ref_pos[res_start:res_end] = _ref_pos
        ref_mask[res_start:res_end] = _ref_mask

        # ... fill the automorphisms for this residue, generating automorphisms from RDKit
        if _ref_automorphs is not None:
            ref_automorphs[: len(_ref_automorphs), res_start:res_end] = _ref_automorphs
            ref_automorphs_mask[: len(_ref_automorphs), res_start:res_end] = True
            max_automorphs = max(max_automorphs, len(_ref_automorphs))

        # ... update to the next conformer index
        _next_conf_idx[res_name] += 1

    # ... resize the reference automorphism arrays to the maximum number of automorphisms
    if ref_automorphs is not None:
        ref_automorphs = ref_automorphs[:max_automorphs]
        ref_automorphs_mask = ref_automorphs_mask[:max_automorphs]

    # Generate remaining reference features
    # ... element
    ref_element = (
        atom_array.atomic_number
        if "atomic_number" in atom_array.get_annotation_categories()
        else np.vectorize(ELEMENT_NAME_TO_ATOMIC_NUMBER.get)(atom_array.element)
    )
    # ... charge
    ref_charge = atom_array.charge

    # ... atom name
    ref_atom_name_chars = _encode_atom_names_like_af3(atom_array.atom_name)

    if use_element_for_atom_names_of_atomized_tokens:
        assert (
            "atomize" in atom_array.get_annotation_categories()
        ), "Atomize annotation is required when using element for atom names of atomized tokens."
        ref_atom_name_chars[atom_array.atomize] = _encode_atom_names_like_af3(atom_array.element[atom_array.atomize])

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
        "ref_pos_is_ground_truth": ref_pos_is_ground_truth,  # (n_atoms)
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

    Optionally, the following features can be added as well:
        - ref_automorphs: [N_automorphs, N_atoms, 2] Automorphisms of the reference conformer.
          Each automorphism is a mapping from one atom to another. The first column is the source atom index,
          and the second column is the target atom index. The automorphisms are given in residue-local indices.
        - ref_automorphs_mask: [N_automorphs, N_atoms] Mask indicating which atom slots are used in the automorphisms.

    Note: This transform should be applied after cropping.

    Reference:
        - Section 2.8 of the AF3 supplementary information
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """

    requires_previous_transforms = ["AddGlobalTokenIdAnnotation"]

    def __init__(
        self,
        conformer_generation_timeout: float = 10.0,
        should_generate_automorphisms_with_rdkit: bool = True,
        use_element_for_atom_names_of_atomized_tokens: bool = False,
        apply_random_rotation_and_translation: bool = True,
        **generate_conformers_kwargs,
    ):
        self.conformer_generation_timeout = conformer_generation_timeout
        self.should_generate_automorphisms_with_rdkit = should_generate_automorphisms_with_rdkit
        self.use_element_for_atom_names_of_atomized_tokens = use_element_for_atom_names_of_atomized_tokens
        self.apply_random_rotation_and_translation = apply_random_rotation_and_translation
        self.generate_conformers_kwargs = generate_conformers_kwargs

        if self.use_element_for_atom_names_of_atomized_tokens:
            logger.warning("Using element type for atom names of atomized tokens.")

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name", "element", "charge", "atom_name", "token_id"])

        if self.use_element_for_atom_names_of_atomized_tokens:
            check_atom_array_annotation(data, ["atomize"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        # Generate reference features
        reference_features = get_af3_reference_molecule_features(
            atom_array,
            conformer_generation_timeout=self.conformer_generation_timeout,
            should_generate_automorphisms_with_rdkit=self.should_generate_automorphisms_with_rdkit,
            use_element_for_atom_names_of_atomized_tokens=self.use_element_for_atom_names_of_atomized_tokens,
            apply_random_rotation_and_translation=self.apply_random_rotation_and_translation,
            **self.generate_conformers_kwargs,
        )

        # Add reference features to the 'feats' dictionary
        if "feats" not in data:
            data["feats"] = {}
        data["feats"].update(reference_features)

        return data


def random_replace_non_polymer_conformer_with_ground_truth(atom_array: AtomArray, per_pn_unit_probability: float = 1.0):
    """With some probability, flags non-polymer residues to use the ground-truth coordinates as the reference conformer.

    Adds the `ground_truth_conformer_policy` annotation to the AtomArray if it does not already exist.
    This annotation indicates if/how residues should use the ground-truth coordinates (i.e., the coordinates from the original structure) as the reference conformer.

    Possible values are (as defined in the GroundTruthConformerPolicy enum):
        -  REPLACE: Use the ground-truth coordinates as the reference conformer.
        -  FALLBACK: Use the ground-truth coordinates only if our standard conformer generation pipeline fails (e.g., we cannot generate a conformer with RDKit,
            and the molecule is either not in the CCD or the CCD entry is invalid).
        -  IGNORE: Do not use the ground-truth coordinates as the reference conformer, under any circumstances.

    Args:
        atom_array (AtomArray): The input atom array.
        per_pn_unit_probability (float, optional): The probability that a non-polymer pn_unit (chain) will use the ground-truth conformer. Defaults to 1.0.

    Returns:
        AtomArray: The input atom array with the `ground_truth_conformer_policy` annotation added.
    """
    # ... add the annotation if it does not already exist, defaulting to all False
    if "ground_truth_conformer_policy" not in atom_array.get_annotation_categories():
        atom_array.set_annotation(
            "ground_truth_conformer_policy", np.full(len(atom_array), GroundTruthConformerPolicy.IGNORE, dtype=np.int8)
        )

    # ... loop over all ligands and set the `ground_truth_conformer_policy` annotation to "replace" with some probability
    non_polymer_pn_unit_iids = np.unique(atom_array.pn_unit_iid[~atom_array.is_polymer])
    for pn_unit_iid in non_polymer_pn_unit_iids:
        mask = atom_array.pn_unit_iid == pn_unit_iid

        # (With some probability, use the ground-truth conformer)
        if np.random.rand() < per_pn_unit_probability:
            atom_array.ground_truth_conformer_policy[mask] = GroundTruthConformerPolicy.REPLACE

    return atom_array


class RandomReplaceNonPolymerConformerWithGroundTruth(Transform):
    """With some probability, flags non-polymer residues to use the ground-truth coordinates as the reference conformer."""

    incompatible_previous_transforms = ["GetAF3ReferenceMoleculeFeatures"]

    def __init__(self, per_pn_unit_probability: float = 1.0):
        self.per_pn_unit_probability = per_pn_unit_probability

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict) -> dict:
        data["atom_array"] = random_replace_non_polymer_conformer_with_ground_truth(
            data["atom_array"], self.per_pn_unit_probability
        )
        return data
