import logging
from typing import Any

import numpy as np
import torch
from biotite.structure import AtomArray

from datahub.transforms._checks import check_atom_array_annotation, check_contains_keys
from datahub.transforms.base import Transform

logger = logging.getLogger("datahub")


def featurize_atom_level_embeddings(
    atom_array: AtomArray,
    cached_residue_level_data: dict,
    residue_conformer_indices: dict[int, np.ndarray],
    ignore_res_names: list[str] | None = None,
    global_std: torch.Tensor | np.ndarray | None = None,
    threshold: float = 1e3,
) -> dict[str, np.ndarray]:
    """Return atom-level embeddings and mask for each atom in the atom_array.

    For each atom, looks up its embedding by residue name and uses the selected conformer indices
    to concatenate embeddings along the batch dimension.

    If the global mean and standard deviation are provided, the embeddings are normalized to have zero mean and unit variance.

    Args:
        atom_array: AtomArray to featurize.
        cached_residue_level_data: Dict of cached data by residue name.
        residue_conformer_indices: Dict mapping global residue ID to selected conformer indices.
        ignore_res_names: List of residue names to ignore. If None, no residues are ignored.
        global_std: Global standard deviation of the embeddings (e.g., across all conformers of all residues). If None, no normalization is performed.
        threshold: Maximum absolute value for descriptors. If any descriptor exceeds this threshold, the entire residue is ignored.

    Returns:
        dict: {'atom_level_embedding': (n_conformers, L, D), 'has_atom_level_embedding': (L,), 'mean_atom_level_embedding': (L, D)}
    """
    L = len(atom_array)
    res_names = atom_array.res_name
    atom_names = atom_array.atom_name
    global_res_ids = atom_array.res_id_global

    # Infer dimensions from first available descriptors
    try:
        first_descriptors = next(
            res_data["descriptors"]
            for res_data in cached_residue_level_data.values()
            if res_data.get("descriptors") is not None
        )
        embedding_dim = first_descriptors.shape[-1]  # Last dimension is features
        # Get number of conformers from first residue instance
        n_conformers = len(next(iter(residue_conformer_indices.values()))) if residue_conformer_indices else 0
    except (StopIteration, ValueError):
        # No embeddings found - return empty tensors
        return {
            "atom_level_embedding": np.zeros((0, L, embedding_dim), dtype=np.float32),
            "has_atom_level_embedding": np.zeros(L, dtype=bool),
            "mean_atom_level_embedding": np.zeros((L, embedding_dim), dtype=np.float32),
        }

    # Initialize embeddings with shape (n_conformers, L, embedding_dim)
    embeddings = np.full((n_conformers, L, embedding_dim), np.nan, dtype=np.float32)
    has_embedding = np.zeros(L, dtype=bool)

    for i, (res_name, atom_name, global_res_id) in enumerate(zip(res_names, atom_names, global_res_ids)):
        # (Skip checks)
        if ignore_res_names is not None and res_name in ignore_res_names:
            continue
        if res_name not in cached_residue_level_data or global_res_id not in residue_conformer_indices:
            continue

        res_data = cached_residue_level_data[res_name]
        if res_data.get("descriptors") is None or res_data.get("atom_names") is None:
            # ... no descriptors or atom names
            continue

        try:
            atom_idx = list(res_data["atom_names"]).index(atom_name)
        except ValueError:
            # ... atom name not found in atom names list
            continue

        conformer_indices = residue_conformer_indices[global_res_id]
        selected_descriptors = res_data["descriptors"][conformer_indices, atom_idx, :]

        # Check if any descriptor exceeds the threshold (diverged - likely a bad reference conformer)
        if np.any(np.abs(selected_descriptors) > threshold):
            continue

        if global_std is not None:
            global_std = global_std.numpy() if isinstance(global_std, torch.Tensor) else global_std

            # Normalize the descriptors to have unit variance
            # (Following Stable Diffusion's methodology, we don't zero-center the embeddings by subtracting the mean)
            assert (
                global_std.ndim == 1 and global_std.shape[0] == selected_descriptors.shape[-1]
            ), "global_std must have shape (embedding_dim,)"
            selected_descriptors = selected_descriptors / global_std

        # Pad or truncate to match n_conformers
        n_selected = len(conformer_indices)
        if n_selected <= n_conformers:
            embeddings[:n_selected, i, :] = selected_descriptors
        else:
            embeddings[:, i, :] = selected_descriptors[:n_conformers]

        has_embedding[i] = True

    return {
        "atom_level_embedding": np.nan_to_num(embeddings),  # (n_conformers, L, D)
        "has_atom_level_embedding": has_embedding,  # (L,)
        "mean_atom_level_embedding": np.nan_to_num(embeddings.mean(axis=0)),  # (L, D)
    }


class FeaturizeAtomLevelEmbeddings(Transform):
    """Featurizes atom-level embeddings from cached data and adds them to the "feats" key.

    Uses cached residue-level data and conformer indices to create atom-level embeddings
    with a batch dimension for multiple conformers per residue.

    See `featurize_atom_level_embeddings` for details.

    Args:
        ignore_res_names (list[str] | None): List of residue names to ignore. If None, no residues are ignored.
    """

    requires_previous_transforms = [
        "LoadCachedResidueLevelData",
        "RandomSubsampleCachedConformers",
        "AddGlobalResIdAnnotation",
    ]

    def __init__(
        self, ignore_res_names: list[str] | None = None, mask_rdkit_conformers: bool = False, threshold: float = 1e3
    ):
        self.ignore_res_names = ignore_res_names
        self.mask_rdkit_conformers = mask_rdkit_conformers
        self.threshold = threshold

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["res_id_global"])
        check_contains_keys(data, ["cached_residue_level_data", "residue_conformer_indices"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]
        assert "residues" in data["cached_residue_level_data"], "cached_residue_level_data must contain 'residues' key"

        cached_residue_level_data = data["cached_residue_level_data"]["residues"]
        residue_conformer_indices = data["residue_conformer_indices"]

        std = None
        if data["cached_residue_level_data"].get("metadata"):
            std = data["cached_residue_level_data"]["metadata"].get("std")

        result = featurize_atom_level_embeddings(
            atom_array,
            cached_residue_level_data,
            residue_conformer_indices,
            ignore_res_names=self.ignore_res_names,
            global_std=std,
            threshold=self.threshold,
        )

        feats = data.setdefault("feats", {})
        feats.update(result)

        # (Optional) Mask the RDKit conformers where the atom level embedding IS present
        if self.mask_rdkit_conformers:
            assert "ref_pos" in feats
            mask = feats["has_atom_level_embedding"]
            feats["ref_pos"][mask] = 0.0
            feats["ref_mask"][mask] = 0.0

        assert all(
            key in feats for key in ("atom_level_embedding", "has_atom_level_embedding", "mean_atom_level_embedding")
        )

        return data
