from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from assertpy import assert_that
from biotite.structure import AtomArray
from cifutils.constants import AF3_EXCLUDED_LIGANDS

from datahub.common import exists
from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING
from datahub.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddProteinTerminiAnnotation,
    AddWithinPolyResIdxAnnotation,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveTerminalOxygen,
    RemoveUnsupportedChainTypes,
    SortLikeRF2AA,
)
from datahub.transforms.atom_frames import AddAtomFrames
from datahub.transforms.atomize import AtomizeResidues, FlagNonPolymersForAtomization
from datahub.transforms.base import Compose, ConvertToTorch, RandomRoute
from datahub.transforms.bonds import (
    AddRF2AABondFeaturesMatrix,
    AddRF2AATraversalDistanceMatrix,
    AddTokenBondAdjacency,
)
from datahub.transforms.chirals import AddRF2AAChiralFeatures
from datahub.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from datahub.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from datahub.transforms.encoding import EncodeAtomArray, atom_array_from_encoding
from datahub.transforms.feature_aggregation import AggregateFeaturesLikeRF2AA
from datahub.transforms.msa._msa_featurizing_utils import encode_msa_like_RF2AA
from datahub.transforms.msa.msa import (
    EncodeMSA,
    FeaturizeMSALikeRF2AA,
    FillFullMSAFromEncoded,
    LoadPolymerMSAs,
    PairAndMergePolymerMSAs,
)
from datahub.transforms.openbabel_utils import (
    AddOpenBabelMoleculesForAtomizedMolecules,
    GetChiralCentersFromOpenBabel,
)
from datahub.transforms.symmetry import (
    AddPostCropMoleculeEntityToFreeFloatingLigands,
    CreateSymmetryCopyAxisLikeRF2AA,
)
from datahub.transforms.template import AddRFTemplates, FeaturizeRFTemplatesForRF2AA
from datahub.utils.numpy import get_connected_components_from_adjacency


# Helper functions
def _is_atom(seq):
    return seq > 32


def _assert_shape(t: torch.Tensor | np.ndarray, s: tuple[int, ...]):
    assert_that(tuple(t.shape)).is_equal_to(s)


def _is_symmetric(array: torch.Tensor | np.ndarray) -> bool:
    # Check that matrix is symmetric
    return np.array_equal(array, array.T, equal_nan=True)


def _is_block_diagonal_with_full_blocks(array: torch.Tensor | np.ndarray) -> bool:
    # NOTE: This only
    # Check that matrix is 2D
    assert_that(len(array.shape)).is_equal_to(2)

    # Check that matrix is square
    n_rows, n_cols = array.shape
    assert_that(n_rows).is_equal_to(n_cols)

    # Get occupied entries
    occupied = np.asarray(array != 0)

    # Check 1: A necessary condition for a full block diagonal structure is that
    #  the matrix retains it's occupancy pattern when squared.
    if not np.allclose(occupied, np.linalg.matrix_power(occupied, 2)):
        return False

    # Check 2: Explicitly go over the matrix and check that it is block diagonal
    start = 0
    end = np.where(occupied[0])[0][-1] + 1
    is_block_diagonal = True
    for row in occupied:
        if row[start] == 0:
            # case: jump to new block - increment block start & end
            start = end
            end = np.where(row)[0][-1] + 1

        # validate block structure
        is_block_diagonal &= (row[start:end] != 0).all()
        is_block_diagonal &= (start == 0) or (row[:start] == 0).all()
        is_block_diagonal &= (end == n_cols) or (row[end:] == 0).all()

        if not is_block_diagonal:
            # ... early stop
            return False

    # Create a mask for all occupied entries:
    return is_block_diagonal


def _are_all_blocks_the_same_size(array: torch.Tensor | np.ndarray) -> bool:
    assert _is_block_diagonal_with_full_blocks(array)
    block_diffs = np.unique(np.where(array.diff())[1]) + 1
    block_starts_ends = np.concatenate([[0], block_diffs, [array.shape[0]]])
    block_sizes = block_starts_ends[1:] - block_starts_ends[:-1]
    return np.all(block_sizes == block_sizes[0])


class RF2AAInputs(NamedTuple):
    """A named tuple containing the inputs to the RF2AA model."""

    seq: np.ndarray
    msa: np.ndarray
    msa_masked: np.ndarray
    msa_full: np.ndarray
    mask_msa: np.ndarray
    xyz: np.ndarray  # `true_crds` in rf2aa code
    mask: np.ndarray  # `mask_crds` in rf2aa code
    idx_pdb: np.ndarray
    xyz_t: np.ndarray
    t1d: np.ndarray
    mask_t: np.ndarray
    xyz_prev: np.ndarray
    mask_prev: np.ndarray
    same_chain: np.ndarray
    unclamp: np.ndarray
    negative: np.ndarray
    atom_frames: np.ndarray
    bond_feats: np.ndarray
    dist_matrix: np.ndarray
    chirals: np.ndarray
    ch_label: np.ndarray
    symmgp: str
    task: str
    example_id: str  # `item` in rf2aa code

    @classmethod
    def from_dict(cls, data: dict) -> "RF2AAInputs":
        return cls(**{key: data[key] for key in cls._fields})

    def to_atom_array(self, symm_copy: int = 0) -> AtomArray:
        """Decode the inputs into an AtomArray for the given `symm_copy`."""
        is_batched = self.xyz.ndim == 5

        seq = self.msa[0, 0, 0] if is_batched else self.msa[0, 0]
        token_is_atom = _is_atom(seq).unsqueeze(1).expand((len(seq), 36))

        # Get the symmetric copy (i) for the polymer, but the first automorph for the ligand
        atomized = token_is_atom[:, 0]
        xyz = self.xyz[0, symm_copy] if is_batched else self.xyz[symm_copy]
        mask = self.mask[0, symm_copy] if is_batched else self.mask[symm_copy]
        if atomized.any():
            xyz[atomized] = self.xyz[0, symm_copy, atomized] if is_batched else self.xyz[symm_copy, atomized]
            mask[atomized] = self.mask[0, 0, atomized] if is_batched else self.mask[0, atomized]

        molecule_entity = self.ch_label[0] if is_batched else self.ch_label

        chain_id = np.empty(len(seq))
        same_chain = self.same_chain[0] if is_batched else self.same_chain
        for i, idxs in enumerate(get_connected_components_from_adjacency(same_chain.numpy())):
            chain_id[idxs] = i

        return atom_array_from_encoding(
            encoded_coord=xyz,
            encoded_mask=mask,
            encoded_seq=seq,
            chain_id=chain_id,
            chain_entity=molecule_entity,
            encoding=RF2AA_ATOM36_ENCODING,
            token_is_atom=token_is_atom,
        )

    def num_res(self) -> int:
        is_batched = self.xyz.ndim == 5
        msa = self.msa[0] if is_batched else self.msa
        return (~_is_atom(msa[0, 0])).sum().item()

    def num_atoms(self) -> int:
        is_batched = self.xyz.ndim == 5
        msa = self.msa[0] if is_batched else self.msa
        return (_is_atom(msa[0, 0])).sum().item()


def build_rf2aa_transform_pipeline(
    protein_msa_dir: PathLike | str,
    rna_msa_dir: PathLike | str,
    # Recycles parameters
    n_recycles: int = 5,  # Paper: 5
    # Cropping parameters
    crop_size: int = 256,  # Paper: 256
    crop_center_cutoff_distance: float = 15.0,
    crop_spatial_probability: float = 0.5,
    crop_contiguous_probability: float = 0.5,
    # Filtering parameters
    unresolved_ligand_atom_limit: int | float | None = 0.1,
    filter_to_annotated_pn_units: bool = True,
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    # Atomization parameters
    res_names_to_atomize: list[str] = None,
    # MSA parameters
    max_msa_sequences: int = 10_000,  # Paper: 10_000
    dense_msa: bool = True,
    n_msa_cluster_representatives: int = 256,  # Paper model: 256
    msa_n_extra_rows: int = 1024,  # Paper mode: 1024
    msa_mask_probability: float = 0.15,
    msa_mask_behavior_probs: dict[str, float] = {
        "replace_with_random_aa": 0.1,
        "replace_with_msa_profile": 0.1,
        "do_not_replace": 0.1,
    },
    order_independent_atom_frame_prioritization: bool = True,
    polymer_token_indices: torch.Tensor = torch.arange(32),
    # Template parameters
    n_template: int = 5,
    pick_top_templates: bool = False,
    template_max_seq_similarity: float = 60.0,
    template_min_seq_similarity: float = 10.0,
    template_min_length: int = 10,
    # Symmetry resolution parameters
    max_automorphs: int = 1_000,
    max_isomorphs: int = 1_000,
    # Miscellaneous parameters
    use_negative_interface_examples: bool = False,
    unclamp_loss_probability: float = 0.1,
    black_hole_init: bool = True,
    black_hole_init_noise_scale: float = 5.0,  # Angstroms (Paper: 5.0)
    # Filter parameters:
    max_allowed_num_atoms: int = 150_000,
    # Cache params:
    msa_cache_dir: PathLike | str | None = "/projects/ml/RF2_allatom/cache/msa",
) -> Compose:
    """
    Creates a transformation pipeline for the RF2AA model, applying a series of transformations to the input data.

    Args:
        - protein_msa_dir (PathLike | str): Directory to look for protein MSA files.
        - rna_msa_dir (PathLike | str): Directory to look for RNA MSA files.
        - n_recycles (int, optional): Number of recycles for the MSA featurization. Defaults to 5.
        - crop_size (int, optional): Size of the crop for spatial and contiguous cropping (in number of tokens).
            Defaults to 384.
        - crop_center_cutoff_distance (float, optional): Cutoff distance for the center of the crop (in Angstroms).
            Defaults to 15.0.
        - crop_spatial_probability (float, optional): Probability of performing spatial cropping. Defaults to 0.5.
        - crop_contiguous_probability (float, optional): Probability of performing contiguous cropping. Defaults to 0.5.
        - unresolved_ligand_atom_limit (int | float, optional): Limit for above which a ligand is considered unresolved.
            many unresolved atoms has its atoms removed. If None, all atoms are kept, if between 0 and 1, the number of
            atoms is capped at that percentage of the crop size. If an integer >= 1, the number of unresolved atoms is
            capped at that number. Defaults to 0.1.
        - filter_to_annotated_pn_units (bool, optional): Whether to filter to annotated pn units.
            This saves a time and memory for large structures. Defaults to True.
        - res_names_to_atomize (list[str], optional): List of residue names to *always* atomize. Note that RF2AA already
            atomizes all residues that are not in the encoding (i.e. that are not standard AA, RNA, DNA or special masks).
            Therefore only specify this if you want to deterministically atomize certain standard tokens. Defaults to None.
        - max_msa_sequences (int, optional): Maximum number of MSA sequences to load. Defaults to 10,000.
        - dense_msa (bool, optional): Whether to use dense MSA pairing. Defaults to True.
        - n_msa_cluster_representatives (int, optional): Number of MSA cluster representatives to select. Defaults to 100.
        - msa_n_extra_rows (int, optional): Number of extra rows for MSA. Defaults to 100.
        - msa_mask_probability (float, optional): Probability of masking MSA sequences according to `msa_mask_behavior_probs`.
            Defaults to 0.15.
        - msa_mask_behavior_probs (dict[str, float], optional): Probabilities for different MSA mask behaviors.
            Defaults to {"replace_with_random_aa": 0.1, "replace_with_msa_profile": 0.1, "do_not_replace": 0.1},
            which is the BERT style masking.
        - order_independent_atom_frame_prioritization (bool, optional): Whether to prioritize order-independent atom frames.
            Defaults to True.
        - n_template (int, optional): Number of templates to use. Defaults to 5.
        - pick_top_templates (bool, optional): Whether to pick the top templates if there are more than `n_template`. If
            False, the templates are selected randomly among all templates. Defaults to False.
        - template_max_seq_similarity (float, optional): Maximum sequence similarity cutoff for templates.
            Defaults to 60.0.
        - template_min_seq_similarity (float, optional): Minimum sequence similarity cutoff for templates.
            Defaults to 10.0.
        - template_min_length (int, optional): Minimum length cutoff for templates. Defaults to 10.
        - max_automorphs (int, optional): Maximum number of automorphs after which to cap small molecule ligand
            symmetry resolution. Defaults to 1,000.
        - max_isomorphs (int, optional): Maximum number of polymer isomorphs after which to cap symmetry resolution.
            Defaults to 1,000.
        - use_negative_interface_examples (bool, optional): Whether to use negative interface examples. Defaults to False.
        - unclamp_loss_probability (float, optional): Probability of unclamping the loss during training. Defaults to 0.1.
        - black_hole_init (bool, optional): Whether to use black hole initialization. Defaults to True.
        - black_hole_init_noise_scale (float, optional): Noise scale for black hole initialization. Defaults to 5.0.

    For more details on the parameters, see the RF2AA paper and the documentation for the respective Transforms.

    Returns:
        Compose: A composed transformation pipeline.
    """
    assert np.isclose(
        crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
    ), "Crop probabilities must sum to 1.0"
    assert crop_size > 0, "Crop size must be greater than 0"
    assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"

    if unresolved_ligand_atom_limit is None:
        unresolved_ligand_atom_limit = 1_000_000
    elif unresolved_ligand_atom_limit < 1:
        unresolved_ligand_atom_limit = np.ceil(crop_size * unresolved_ligand_atom_limit)

    encoding = RF2AA_ATOM36_ENCODING
    protein_msa_dir = Path(protein_msa_dir)
    rna_msa_dir = Path(rna_msa_dir)

    transforms = [
        # ============================================
        # 1. Prepare the structure
        # ============================================
        # ...remove hydrogens for efficiency
        RemoveHydrogens(),  # * (already cached from the parser)
        RemoveTerminalOxygen(),  # RF2AA does not encode terminal oxygen for AA residues.
        # ...remove unsupported chain types
        RemoveUnsupportedChainTypes(),  # e.g., DNA_RNA_HYBRID, POLYPEPTIDE_D, etc.
        # RaiseIfTooManyAtoms(max_atoms=max_allowed_num_atoms),
        HandleUndesiredResTokens(undesired_res_names),  # e.g., non-standard residues
        # ...filtering
        # RemoveUnresolvedLigandAtomsIfTooMany(
        #     unresolved_ligand_atom_limit=unresolved_ligand_atom_limit
        # ),  # Crop size * 10%
        # ...add an annotation that is a unique atom ID across the entire structure, and won't change as we crop or reorder the AtomArray
        AddGlobalAtomIdAnnotation(),
        # ...add additional annotations that we'll use later
        AddProteinTerminiAnnotation(),  # e.g., N-terminus, C-terminus
        AddWithinPolyResIdxAnnotation(),  # add annotation relevant for matching MSA and template info
        # ============================================
        # 2. Perform relevant atomizations to arrive at final tokens
        # ============================================
        # ...sample residues to atomize (in RF2AA, with some probability, we atomize protein residues randomly)
        # TODO: SampleResiduesToAtomize
        # ...handle covalent modifications by atomizing and attaching the bonded residue to the non-polymer
        FlagAndReassignCovalentModifications(),
        # ...flag non-polymers for atomization (in case there are polymer tokens outside of a polymer)
        FlagNonPolymersForAtomization(),
        # ...atomize
        AtomizeResidues(
            atomize_by_default=True,
            res_names_to_atomize=res_names_to_atomize,
            res_names_to_ignore=encoding.tokens,
            move_atomized_part_to_end=True,
        ),
        # ... sort poly then non-poly
        SortLikeRF2AA(),
        # ... add global and token IDs
        AddGlobalTokenIdAnnotation(),
        # ============================================
        # 3. Extract openbabel molecules for atomized residues and ligands
        # ============================================
        AddOpenBabelMoleculesForAtomizedMolecules(),
        # ... get chiral centers from openbabel molecules
        GetChiralCentersFromOpenBabel(),
    ]

    contiguous_crop_transform = CropContiguousLikeAF3(crop_size=crop_size, keep_uncropped_atom_array=True)
    spatial_crop_transform = CropSpatialLikeAF3(
        crop_size=crop_size, crop_center_cutoff_distance=crop_center_cutoff_distance, keep_uncropped_atom_array=True
    )
    if crop_contiguous_probability > 0 and crop_spatial_probability > 0:
        transforms += [
            # ...crop around our query pn_unit(s) early, since we don't need the full structure moving forward
            RandomRoute(
                transforms=[
                    contiguous_crop_transform,
                    spatial_crop_transform,
                ],
                probs=[crop_contiguous_probability, crop_spatial_probability],
            ),
        ]
    elif crop_contiguous_probability > 0:
        transforms.append(contiguous_crop_transform)
    elif crop_spatial_probability > 0:
        transforms.append(spatial_crop_transform)

    transforms += [
        AddPostCropMoleculeEntityToFreeFloatingLigands(),
        # ============================================
        # 4. Encode the structure
        # ============================================
        # ...encode the AtomArray (note that we've already atomized)
        EncodeAtomArray(encoding),
        # ============================================
        # 5. Load and pair MSAs
        # ============================================
        LoadPolymerMSAs(
            protein_msa_dir=protein_msa_dir,
            rna_msa_dir=rna_msa_dir,
            max_msa_sequences=max_msa_sequences,  # maximum number of sequences to load (we later subsample further)
            msa_cache_dir=Path(msa_cache_dir) if exists(msa_cache_dir) else None,
        ),
        PairAndMergePolymerMSAs(
            unpaired_padding="-",
            dense=dense_msa,
        ),
        EncodeMSA(encoding_function=encode_msa_like_RF2AA),
        FillFullMSAFromEncoded(pad_token=encoding.token_to_idx["UNK"]),
        # ============================================
        # 5. Load and featurize templates (proteins only)
        # ============================================
        AddRFTemplates(
            max_n_template=n_template,
            pick_top=pick_top_templates,
            max_seq_similarity=template_max_seq_similarity,
            min_seq_similarity=template_min_seq_similarity,
            min_template_length=template_min_length,
        ),
        # ============================================
        # 6. Add misc. features (chirals, bond features, etc.)
        # ============================================
        # ...chirals
        AddRF2AAChiralFeatures(),
        # ...bonds
        AddTokenBondAdjacency(),
        AddRF2AABondFeaturesMatrix(),
        AddRF2AATraversalDistanceMatrix(),
        # ...atom frames
        AddAtomFrames(order_independent_atom_frame_prioritization=order_independent_atom_frame_prioritization),
        # ============================================
        # 7. Convert to torch and featurize
        # ============================================
        ConvertToTorch(
            keys=[
                "polymer_msas_by_chain_id",
                "encoded",
                "full_msa_details",
                "rf2aa_bond_features_matrix",
                "rf2aa_traversal_distance_matrix",
                "rf2aa_atom_frames",
            ]
        ),
        FeaturizeMSALikeRF2AA(
            n_recycles=n_recycles,
            n_msa_cluster_representatives=n_msa_cluster_representatives,  # Paper model: 256
            n_extra_rows=msa_n_extra_rows,  # Paper mode: 1024
            mask_behavior_probs=msa_mask_behavior_probs,
            mask_probability=msa_mask_probability,
            encoding=encoding,
            polymer_token_indices=polymer_token_indices,
        ),
        FeaturizeRFTemplatesForRF2AA(
            n_template=n_template, mask_token_idx=encoding.token_to_idx["<M>"], encoding=encoding
        ),
        # ============================================
        # 8. Create symmetry copies (isomorphic chain swaps for polys, automorphisms for small molecules)
        # ============================================
        CreateSymmetryCopyAxisLikeRF2AA(
            encoding=encoding,
            max_automorphs=max_automorphs,
            max_isomorphisms=max_isomorphs,
        ),
        # ============================================
        # 9. Aggregate features into final format for RF2AA and remove unused features
        # ============================================
        AggregateFeaturesLikeRF2AA(
            encoding=encoding,
            use_negative_interface_examples=use_negative_interface_examples,
            unclamp_loss_probability=unclamp_loss_probability,
            black_hole_init=black_hole_init,
            black_hole_init_noise_scale=black_hole_init_noise_scale,
        ),
    ]

    return Compose(transforms, track_rng_state=True)


def assert_satisfies_rf2aa_assumptions(sample: dict[str, Any]):
    """
    Asserts that the given sample satisfies the assumptions required for a
    successful forward and backward pass through RF2AA.
    """
    # Find out if there is a batch dimension:
    if sample["seq"].ndim == 3:
        # ... we have a batch dimension -- remove it
        sample = {k: v[0] for k, v in sample.items()}
    else:
        assert sample["seq"].ndim == 2, f"seq must have 2 or 3 dimensions, but has {sample['seq'].ndim}."

    # Extract the data
    seq = sample["seq"]
    msa = sample["msa"]
    msa_masked = sample["msa_masked"]
    msa_full = sample["msa_full"]
    mask_msa = sample["mask_msa"]
    true_crds = sample["xyz"]
    mask_crds = sample["mask"]
    idx_pdb = sample["idx_pdb"]
    xyz_t = sample["xyz_t"]
    t1d = sample["t1d"]
    mask_t = sample["mask_t"]
    xyz_prev = sample["xyz_prev"]
    mask_prev = sample["mask_prev"]
    same_chain = sample["same_chain"]
    unclamp = sample["unclamp"]
    negative = sample["negative"]
    atom_frames = sample["atom_frames"]
    bond_feats = sample["bond_feats"]
    dist_matrix = sample["dist_matrix"]
    chirals = sample["chirals"]
    ch_label = sample["ch_label"]
    symmgp = sample["symmgp"]
    task = sample["task"]
    item = sample["example_id"]

    # Check basic types
    assert isinstance(unclamp.item(), bool)
    assert isinstance(negative.item(), bool)
    assert symmgp == "C1", f"{item}: Got unexpected symmgp: {symmgp}"
    assert isinstance(task, str)
    assert isinstance(item, str)

    # Check basic shapes
    NTOTAL = 36  # ... number of atoms per token

    n_recycles, N, L = msa.shape[:3]
    num_atoms = (_is_atom(seq[0]).sum()).item()
    _assert_shape(seq, (n_recycles, L))
    _assert_shape(msa, (n_recycles, N, L))
    _assert_shape(msa_masked, (n_recycles, N, L, 164))
    N_full = msa_full.shape[1]
    assert N_full > 0, f"{item}: N_full is {N_full}. But at least the query sequence should be present."
    _assert_shape(msa_full, (n_recycles, N_full, L, 83))
    _assert_shape(mask_msa, (n_recycles, N, L))
    N_symm = true_crds.shape[0]
    _assert_shape(true_crds, (N_symm, L, NTOTAL, 3))
    _assert_shape(mask_crds, (N_symm, L, NTOTAL))
    _assert_shape(idx_pdb, (L,))
    N_templ = xyz_t.shape[0]
    _assert_shape(xyz_t, (N_templ, L, NTOTAL, 3))
    _assert_shape(t1d, (N_templ, L, 80))
    _assert_shape(mask_t, (N_templ, L, NTOTAL))
    _assert_shape(xyz_prev, (L, NTOTAL, 3))
    _assert_shape(mask_prev, (L, NTOTAL))
    _assert_shape(same_chain, (L, L))
    _assert_shape(atom_frames, (num_atoms, 3, 2))
    _assert_shape(bond_feats, (L, L))
    _assert_shape(dist_matrix, (L, L))
    n_chirals = chirals.shape[0]
    _assert_shape(chirals, (n_chirals, 5))
    _assert_shape(ch_label, (L,))
    assert symmgp == "C1", f"{symmgp}"

    # Assert that the masking works correctly
    assert not true_crds[mask_crds].isnan().any()
    assert not xyz_t[mask_t].isnan().any()
    assert not xyz_prev[mask_prev].isnan().any()

    # Check 2D matrices are symmetric
    assert _is_symmetric(same_chain), f"{item}: same_chain is not symmetric"
    assert _is_symmetric(bond_feats), f"{item}: bond_feats is not symmetric"
    assert _is_symmetric(dist_matrix), f"{item}: dist_matrix is not symmetric"

    # Assert that the correspondence between chains is the same in ch_label and same_chain is valid
    ch_label_diffs = np.where(ch_label.diff())[0]
    same_chain_diffs = np.unique(np.where(same_chain.diff())[1])
    assert np.all(
        np.isin(ch_label_diffs, same_chain_diffs)
    ), f"{item}: ch_label_diffs: {ch_label_diffs}, same_chain_diffs: {same_chain_diffs}"

    # Assert that there are polymer tokens in the example:
    num_res_tokens = ((~_is_atom(seq[0])).sum()).item()
    assert (
        num_res_tokens > 0
    ), f"{item}: num_res_tokens: {num_res_tokens}. No polymer tokens at all. This would lead RF2AA to crash."
    assert num_res_tokens + num_atoms == L, f"{item}: num_res_tokens: {num_res_tokens}, num_atoms: {num_atoms}, L: {L}"

    if num_atoms > 0:
        # NOTE: According to Rohith this assumption is not needed.
        # # Assert that `ch_label` is contiguous in the non-poly sector:
        # for label in np.unique(ch_label[num_res_tokens:]):
        #     assert (
        #         np.diff(np.where(ch_label[num_res_tokens:] == label)[0]).max() == 1
        #     ), f"{item}: `ch_label` is not contiguous for label {label} in the non-poly sector."

        # Assert that `same_chain` is block diagonal in the non-poly sector:
        assert _is_block_diagonal_with_full_blocks(
            same_chain[num_res_tokens:, num_res_tokens:]
        ), f"{item}: non-poly sector of `same_chain` is not block diagonal"

        # Assert that in the non-poly sector,
        for label in np.unique(ch_label[num_res_tokens:]):
            # ...all blocks where `ch_label` is the same are the same size:
            idxs = np.where(ch_label[num_res_tokens:] == label)[0]
            # NOTE: This will currently fail for cropped covalent modifications!
            assert _are_all_blocks_the_same_size(
                same_chain[num_res_tokens:, num_res_tokens:][np.ix_(idxs, idxs)]
            ), f"{item}: `same_chain` block {label} is not the same size"

            # ... ensure there is no entirely unresolved `ch_label` segment in
            #     the non-poly sector:
            assert (
                mask_crds[0, idxs + num_res_tokens, :]
            ).any(), f"{item}: Chain with `chain_label` {label} is entirely unresolved in the non-poly sector."

    # Assert that there are no masks in `msa`:
    assert not (msa == 21).any(), f"{item}: There are masks in the ground truth `msa`."

    # Assert that there are no entirely unresolved chains in the poly sector:
    for label in np.unique(ch_label[:num_res_tokens]):
        idxs = np.where(ch_label[:num_res_tokens] == label)[0]
        assert (
            mask_crds[0, idxs, :]
        ).any(), f"{item}: Chain with `chain_label` {label} is entirely unresolved in the poly sector."

    # Ensure there is at least one resolved coordinate for each symmetry copy:
    #  mask_crds: (N_symm, L, NTOTAL)
    assert mask_crds.any(
        dim=(1, 2)
    ).all(), f"{item}: There are no resolved coordinates for at least one symmetry copy (neither poly nor non-poly)."

    # Ensure there is at least one resolved coordinate for each symmetry copy in the poly sector (excluding padding):
    N_symm_poly = mask_crds[:, :num_res_tokens].any(dim=(1, 2)).max().item()
    assert (
        N_symm_poly > 0
    ), f"{item}: There are no resolved coordinates for the poly sector of at least one symmetry copy (excluding padding)."

    # If the given symmetry copy has a poly swap, check that the N-CA-C of at least one residue is resolved, which
    #  is needed to construct the poly frames.
    symm_copies_has_at_least_one_resolved_N_CA_C = (mask_crds[:N_symm_poly, :num_res_tokens, :3].sum(dim=2) == 3).any(
        dim=1
    )
    problems = np.where(~symm_copies_has_at_least_one_resolved_N_CA_C)[0]
    assert (
        len(problems) == 0
    ), f"{item}: The following symmetry copies have no resolved N-CA-C coordinates for the poly sector: {problems}"
