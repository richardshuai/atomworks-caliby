"""Transforms on MSAs"""

from __future__ import annotations

import logging
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from biotite.structure import AtomArray
from cifutils.enums import ChainType

from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING, TokenEncoding
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.atom_array import (
    AddWithinPolyResIdxAnnotation,
)
from datahub.transforms.atomize import AtomizeResidues
from datahub.transforms.base import ConvertToTorch, Transform
from datahub.transforms.msa._msa_featurizing_utils import (
    assign_extra_rows_to_cluster_representatives,
    build_indices_should_be_counted_masks,
    build_msa_index_can_be_masked,
    mask_msa_like_bert,
    summarize_clusters,
    uniformly_select_msa_cluster_representatives,
)
from datahub.transforms.msa._msa_pairing_utils import join_multiple_msas_by_tax_id
from datahub.utils.misc import cache_to_disk_as_pickle, grouped_count, hash_sequence, parse_a3m, parse_fasta
from datahub.utils.token import apply_token_wise, get_token_count, get_token_starts

logger = logging.getLogger(__name__)


class PairAndMergePolymerMSAs(Transform):
    """
    Pairs and merges multiple polymer MSAs by tax_id.
    Ensures that the query sequence is always the first sequence in the MSA.
    Stores results in "merged_polymer_msa" in the data dictionary, with keys:
    - msa: The merged MSA.
    - ins: The merged insertion array.
    - msa_is_padded_mask: A mask indicating whether a given position in the MSA is padded due to unpaired sequences (1) or not (0).
    - tax_ids: The merged taxonomic IDs.
    - any_paired: A boolean array indicating whether a sequence is paired with any other sequence.
    - all_paired: A boolean array indicating whether a sequence is paired with all other sequences.

    Unpaired sequences can be handled in two ways:
    - Dense pairing: Unpaired sequences are densely packed at the bottom of the MSA (AF-3 style).
    - Sparse pairing: Unpaired sequences are block-diagonally added to the bottom of the MSA (AF-Multimer style).
    """

    requires_previous_transforms = ["LoadPolymerMSAs"]

    def __init__(
        self,
        unpaired_padding: str = "-",
        dense: bool = False,
    ):
        self.unpaired_padding = unpaired_padding
        self.dense = dense

    def check_input(self, data: dict):
        check_contains_keys(data, ["polymer_msas_by_chain_id"])

    def forward(self, data: dict) -> dict:
        # If we have no polymer MSAs, we can skip this step
        if len(data["polymer_msas_by_chain_id"]) == 0:
            return data

        atom_array = data["atom_array"]

        # Create a map from unique entity IDs to constituent chain IDs
        # We directly generate the mapping from the AtomArray since the `rcsb_entity` may be inaccurate post-processing
        # We need entity-level information to pair chains that belong to separate entities; otherwise, we simply concatenate the MSAs
        chain_entity_to_chain_ids = {}
        chain_id_to_chain_entity = {}

        for chain_entity in np.unique(atom_array.chain_entity):
            # Get unique chain IDs corresponding to the current chain entity
            chain_ids = np.unique(atom_array.chain_id[atom_array.chain_entity == chain_entity])
            chain_entity_to_chain_ids[chain_entity] = chain_ids

            # Map each chain ID back to its chain entity
            for chain_id in chain_ids:
                chain_id_to_chain_entity[chain_id] = chain_entity

        # Loop through entities to:
        # (1) Create a list of MSAs to pair, choosing one chain per entity ID (as they are all the same, by definition)
        # (2) Keep track of the number of residues for each entity ID
        msa_list = []
        num_residues_by_chain_entity = {}
        for chain_entity in chain_entity_to_chain_ids.keys():
            first_chain_id = chain_entity_to_chain_ids[chain_entity][0]
            if first_chain_id in data["polymer_msas_by_chain_id"]:
                msa = data["polymer_msas_by_chain_id"][first_chain_id]
                msa_list.append(msa)
                num_residues_by_chain_entity[chain_entity] = msa["msa"].shape[1]

        # Create masks for each entity to index into the merged MSA
        entity_masks = {}
        chain_entity_array = np.concatenate(
            [
                [chain_entity] * num_residues_by_chain_entity[chain_entity]
                for chain_entity in chain_entity_to_chain_ids.keys()
                if chain_entity in num_residues_by_chain_entity
            ]
        )
        for chain_entity in chain_entity_to_chain_ids.keys():
            entity_masks[chain_entity] = chain_entity_array == chain_entity

        if len(msa_list) > 1:
            # Heteromeric complex - pair and merge the MSAs
            merged_polymer_msas = join_multiple_msas_by_tax_id(
                msa_list, unpaired_padding=self.unpaired_padding, dense=self.dense
            )
        else:
            # Homomeric complex - no need to pair, we will concatenate the MSAs later
            merged_polymer_msas = msa_list[0]
            # We consider homomers to be unpaired
            merged_polymer_msas["all_paired"] = np.zeros(merged_polymer_msas["msa"].shape[0], dtype=bool)
            merged_polymer_msas["any_paired"] = np.zeros(merged_polymer_msas["msa"].shape[0], dtype=bool)

        # Distribute entity-level MSAs to chain-level MSAs by pointing each chain_id to the MSA for its corresponding chain_entity
        polymer_msas_by_chain_entity = {}
        for chain_entity in chain_entity_to_chain_ids.keys():
            entity_mask = entity_masks[chain_entity]
            msa = merged_polymer_msas["msa"][:, entity_mask]
            ins = merged_polymer_msas["ins"][:, entity_mask]
            msa_is_padded_mask = merged_polymer_msas["msa_is_padded_mask"][:, entity_mask]

            polymer_msas_by_chain_entity[chain_entity] = {
                "msa": msa,
                "ins": ins,
                "msa_is_padded_mask": msa_is_padded_mask,
                "tax_ids": merged_polymer_msas["tax_ids"],  # Common across entities (sequence dimension)
                "any_paired": merged_polymer_msas["any_paired"],  # Common across entities (sequence dimension)
                "all_paired": merged_polymer_msas["all_paired"],  # Common across entities (sequence dimension)
            }

        for chain_id in data["polymer_msas_by_chain_id"].keys():
            chain_entity = chain_id_to_chain_entity[chain_id]
            # NOTE: We deep copy as a precaution, since if multiple chains point to the same dictionary object, we may modify the dictionary in place
            data["polymer_msas_by_chain_id"][chain_id] = deepcopy(polymer_msas_by_chain_entity[chain_entity])

        return data


class LoadPolymerMSAs(Transform):
    """
    Load MSAs for polymer chains within the example as numpy arrays of ASCII strings.
    For the MSAs that we find, store the MSA, insertions, and tax IDs in the data dictionary indexed by chain_id (e.g., "A").
    For polymers where we don't have, or can't locate, an MSA (e.g., short peptides, DNA, RNA/DNA hybrids), create a dummy MSA with only the query sequence.
    """

    protein_msa_dir: Path
    rna_msa_dir: Path
    max_msa_sequences: int
    msa_cache_dir: Path

    def __init__(
        self,
        protein_msa_dir: PathLike = "/projects/ml/RF2_allatom/data_preprocessing/msa/protein",
        rna_msa_dir: PathLike = "/projects/ml/RF2_allatom/data_preprocessing/msa/rna",
        max_msa_sequences: int = 10000,  # NOTE: Only applies to loading; we further sub-sample MSA downstream (e.g., for the standard or extra MSA stack)
        msa_cache_dir: PathLike | None = None,
    ):
        self.protein_msa_dir = Path(protein_msa_dir)
        self.rna_msa_dir = Path(rna_msa_dir)
        self.max_msa_sequences = max_msa_sequences

        # Apply decorator to cache the MSA data (NOTE: `None` turns off caching)
        self._load_msa_data = cache_to_disk_as_pickle(msa_cache_dir)(self._load_msa_data)

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "chain_info"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "chain_id"])

    def _load_msa_data(self, chain_type: ChainType, sequence: str, query_chain_msa_tax_id: str):
        """
        Load the MSA data based on chain type and file availability.

        Args:
            chain_type (ChainType): The type of the chain (Protein or RNA).
            sequence (str): The polymer one-letter sequence.
            query_chain_msa_tax_id (str): The tax ID for the query chain. Defaults to "query" to ensure the query sequence is paired with itself.

        Returns:
            A tuple of numpy arrays containing msa, ins, and tax_ids.
        """

        if chain_type.is_protein():
            # ...hash the sequence so we can look up the MSA file
            sequence_hash = hash_sequence(sequence)

            # (For proteins, we store information as a3m files, which are gzipped)
            protein_msa_file = (self.protein_msa_dir / sequence_hash[:3] / sequence_hash).with_suffix(".a3m.gz")
            if protein_msa_file.exists():
                msa, ins, tax_ids = parse_a3m(
                    protein_msa_file, query_tax_id=query_chain_msa_tax_id, maxseq=self.max_msa_sequences
                )
                return msa, ins, tax_ids

        elif chain_type == ChainType.RNA:
            # ...handle legacy behavior where RNA MSAs use T instead of U
            sequence = sequence.replace("U", "T")

            # ...hash the sequence so we can look up the MSA file
            sequence_hash = hash_sequence(sequence)

            # (For RNA, we store information as fasta files, with no compression)
            rna_msa_file = (self.rna_msa_dir / sequence_hash[:3] / sequence_hash).with_suffix(".afa")
            if rna_msa_file.exists():
                msa, ins = parse_fasta(rna_msa_file, maxseq=self.max_msa_sequences)

                # ...handle legacy behavior by converting all T to U in the MSA (see above)
                msa = np.char.replace(msa, "T", "U")

                # ...load dummy tax IDs (as we don't have them for RNA in the current directory)
                # TODO: Use the updated RNA directory that contains tax IDs, where we have them
                tax_ids = np.full(len(msa), "", dtype="<U10")
                tax_ids[0] = query_chain_msa_tax_id
                return msa, ins, tax_ids

        # Fallback - E.g., DNA, or RNA/DNA hybrid - load in single-sequence mode
        return None, None, None

    def _get_polymer_msas_by_chain_id(self, data: dict, polymer_chain_ids: np.array, chain_info: dict) -> dict:
        """
        Retrieves MSAs for each polymer chain ID (e.g., "A").
        If we can't find an MSA for a given chain ID, return a length-1 MSA containing only the query sequence.

        Parameters:
        - data (dict): Data containing atom array and chain information.
        - polymer_chain_ids (np.array): List of unique chain IDs for polymers.
        - chain_info (dict): Dictionary containing chain information for the polymers (e.g., type).

        Returns:
        - msas_by_chain_id (dict): Dictionary with chain IDs as keys and corresponding MSA data as values. Values are:
            msa (np.array): The MSA as a 2D np.array of ASCII int8 byte character strings
            msa_is_padded_mask (np.array): A mask indicating whether a given position in the MSA is padded (0) or not (1); defaults 1 for all positions
            ins (np.array): The insertion array for the MSA, indicating number of insertion to the LEFT of a given index
            tax_ids (np.array): The taxonomic IDs for each sequence in the MSA
        """
        msas_by_chain_id = {}
        for chain_id in polymer_chain_ids:
            # NOTE: If we re-generate MSAs in the future, we may opt to use the canonical sequence instead of the non-canonical sequence
            sequence = chain_info[chain_id]["processed_entity_non_canonical_sequence"]
            chain_type = ChainType.from_string(chain_info[chain_id]["type"])

            # Set the query chain tax_id to "query" to avoid pairing issues downstream (we force all query sequences to be paired with themselves)
            # Subsequent occurrences of the query sequence will not have the "query" tax ID, and will be paired appropriately
            query_chain_msa_tax_id = "query"

            # Load the MSA file from the correct directory (protein or RNA)
            msa, ins, tax_ids = self._load_msa_data(chain_type, sequence, query_chain_msa_tax_id)

            # If we found an MSA, store it in the dictionary; otherwise, create a dummy MSA with a single sequence (single-sequence mode)
            if msa is not None:
                assert (
                    msa[0] == sequence
                ), f"MSA sequence does not match the sequence from the parser for {data['pdb_id']} chain {chain_id}"

                # Convert the MSA into an np.array with the same shape as the insertion array
                msa_2d = np.array([list(seq) for seq in msa], dtype="S")

                msas_by_chain_id[chain_id] = {
                    "msa": msa_2d,
                    "msa_is_padded_mask": np.zeros(msa_2d.shape, dtype=bool),  # 1 = padded, 0 = not padded
                    "ins": ins,
                    "tax_ids": tax_ids,
                }
            else:
                # If we don't find a match, we return a dummy MSA with only the query sequence, converting the sequence to an np.array
                msas_by_chain_id[chain_id] = {
                    "msa": np.array([list(sequence)], dtype="S"),
                    "msa_is_padded_mask": np.zeros((1, len(sequence)), dtype=bool),  # 1 = padded, 0 = not padded
                    "ins": np.zeros((1, len(sequence)), dtype=np.uint8),
                    "tax_ids": np.array([query_chain_msa_tax_id]),
                }

        return msas_by_chain_id

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        chain_info = data["chain_info"]
        polymer_chain_ids = np.unique(atom_array.chain_id[np.isin(atom_array.chain_type, ChainType.get_polymers())])

        polymer_msas_by_chain_id = self._get_polymer_msas_by_chain_id(data, polymer_chain_ids, chain_info)

        # Add the MSAs to the data dictionary
        data["polymer_msas_by_chain_id"] = polymer_msas_by_chain_id

        return data


class EncodeMSA(Transform):
    """
    Encode a MSA from characters to token indices using the provided encoding function.

    Attributes:
        encoding_function (Callable): A function that takes an MSA and a chain type, and returns the encoded MSA, using the appropriate TokenEncoding.

    Example:
        If we have the following MSA:
        ```
        [
            ["A", "R", "C", "X"],  # X is an unknown amino acid
            ["A", "C", "B", "C"],  # B is an ambiguous amino acid
        ]
        ```
        Then the encoding function should return the following encoded MSA (using RF2AA encoding function):
        ```
        [
            [
                0,
                1,
                4,
                20,
            ],  # The RF2AA encoding maps A to 0, R to 1, C to 4, and ambiguous/unknown amino acids to 20
            [0, 4, 20, 4],
        ]
        ```
    """

    requires_previous_transforms = ["LoadPolymerMSAs"]

    def __init__(self, encoding_function: Callable):
        self.encoding_function = encoding_function

    def check_input(self, data: dict):
        check_contains_keys(data, ["polymer_msas_by_chain_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Loop through all of the polymer chain IDs in the atom array
        polymer_chain_ids = np.unique(atom_array.chain_id[atom_array.is_polymer])
        for chain_id in polymer_chain_ids:
            msa = data["polymer_msas_by_chain_id"][chain_id]["msa"]
            chain_type = ChainType.from_string(data["chain_info"][chain_id]["type"])

            # Encode the MSA (to tokens integers), based on the provided encoding function
            encoded_msa = self.encoding_function(msa=msa, chain_type=chain_type)  # [n_rows, n_res_in_chain] (int)

            # Set the encoded MSA in the output in-place
            data["polymer_msas_by_chain_id"][chain_id]["encoded_msa"] = encoded_msa

        return data


class FillFullMSAFromEncoded(Transform):
    """
    Fills in the full MSA from the encoded MSA, using the atom array to determine the order of the tokens.
    Starts by creating full np.arrays with default values (padding tokens), and then fills in the encoded MSA by looping over chain instances.

    This function requires that all MSAs have the same number of rows, but does not require them to necessarily be paired.

    Specifically:
    - If we cropped or otherwise removed residues, we drop them from the MSA (via indexing) to ensure the MSA is consistent with the atom array.
    - If we atomized residues, we drop the atomized pieces from the MSA, and include only the encoded atomized tokens.

    Attributes:
        pad_token (str): The token used for padding in the MSA. The pad token should match the padding token used when padding unpaired MSA sequences.

    Returns:
        The full MSA, with padding, as a 2D np.array of integers, stored in `data["encoded"]["msa"]`.
        Additionally, we store the following details in `data["full_msa_details"]`:
        - token_idx_has_msa (np.array): A mask indicating whether a given token has an MSA (1) or not (0).
        - msa_is_padded_mask (np.array): A mask indicating whether a given position in the MSA is padded (1) or not (0).
        - msa_raw_ins (np.array): The raw insertion counts for the MSA, before encoding.

    Example:
        If the atom array token order is:
        ```
        [
            Chain A, Residue 1 (A),
            Chain A, Residue 2 (R) [atomized, covalent modification],
            Chain A, Residue 3 (C),
            Chain B, Residue 1 (glycan) [atomized, non-polymer]
        ]
        ```
        And the MSA for Chain A is:
        ```
        [["A", "R", "C"], ["A", "R", "D"]]
        ```
        Then the expected `data["encoded"]["msa"]` would be:
        ```
        [
            [ "A", "R_1", "R_2", "C", "B_1", "B_2" ],
            [ "A", <PAD>, <PAD>, "D", <PAD>, <PAD> ]
        ]
        ```
        Where "R_1", "R_2" are the atomized tokens for the residue, and "B_1", "B_2" are the atomized tokens for the glycan.
        NOTE: Amino acids are represented as letters for clarity; in reality, they would be tokens (integers).

        The expected `data["full_msa_details"]` would be:
        ```
        {
            "token_idx_has_msa": [1, 0, 0, 1, 0, 0],
            "msa_is_padded_mask": [[0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1]],
            "msa_raw_ins": [...],  # Not shown
        }
        ```
    """

    requires_previous_transforms = ["EncodeMSA", AtomizeResidues, AddWithinPolyResIdxAnnotation]

    def __init__(self, pad_token: str):
        self.PAD_TOKEN = pad_token

    def check_input(self, data: dict):
        check_contains_keys(data, ["polymer_msas_by_chain_id", "encoded"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # If we have no polymer MSAs (either full or single-sequence), and therefore NO POLYMERS...
        if len(data["polymer_msas_by_chain_id"]) == 0:
            # ... we set `full_encoded_msa` to be the query sequence (expanded to 2D)
            full_encoded_msa = np.expand_dims(data["encoded"]["seq"], axis=0)  # [1, n_tokens_across_chains] (int)
            num_tokens_in_example = full_encoded_msa.shape[1]
            data["full_msa_details"] = {
                # ... we set `token_idx_has_msa` to all zeros
                "token_idx_has_msa": np.zeros(num_tokens_in_example, dtype=bool),
                # ... we set `msa_is_padded_mask` to all zeros
                "msa_is_padded_mask": np.zeros((1, num_tokens_in_example), dtype=bool),
                # ... we set `msa_raw_ins` to all zeros
                "msa_raw_ins": np.zeros((1, num_tokens_in_example), dtype=int),
            }
            data["encoded"]["msa"] = full_encoded_msa
            return data

        # Get the n_rows
        first_chain_id = next(iter(data["polymer_msas_by_chain_id"]))
        first_encoded_msa = data["polymer_msas_by_chain_id"][first_chain_id]["encoded_msa"]
        n_rows = first_encoded_msa.shape[0]

        # Check that the given padding token matches the padding token used when padding unpaired MSA sequences, if applicable
        existing_pad_tokens = data["polymer_msas_by_chain_id"][first_chain_id]["msa_is_padded_mask"] * first_encoded_msa
        if np.any(existing_pad_tokens):
            token = existing_pad_tokens.flat[np.flatnonzero(existing_pad_tokens)[0]]
            assert (
                token == self.PAD_TOKEN
            ), f"Given padding token {self.PAD_TOKEN} does not match existing padding token {token}"

        # ...set up empty encoded msa (purely padded) for encoded msa
        token_count = get_token_count(atom_array)
        full_encoded_msa = np.full(
            (n_rows, token_count), self.PAD_TOKEN, dtype=int
        )  # [n_rows, n_tokens_across_chains] (int)
        full_msa_is_padded_mask = np.ones(
            (n_rows, token_count), dtype=bool
        )  # [n_rows, n_tokens_across_chains] (bool) 1 = padded, 0 = not padded
        full_msa_ins = np.zeros((n_rows, token_count), dtype=int)  # [n_rows, n_tokens_across_chains] (int)

        # ...create a mask indicating whether any atom in each token is atomized
        is_token_atomized = apply_token_wise(atom_array, atom_array.atomize, np.any)  # [n_tokens_across_chains] (bool)

        # ...loop through all `chain_iids` (polymer/non-polymer) and populate relevant columns in encoding
        token_idx_has_msa = np.zeros(get_token_count(atom_array), dtype=bool)  # [n_tokens_across_chains] (bool)
        for chain_iid in np.unique(atom_array.chain_iid):
            is_atom_in_chain = atom_array.chain_iid == chain_iid  # [n_atoms_total] (bool)

            chain_instance_atom_array = atom_array[is_atom_in_chain]
            chain_id = chain_instance_atom_array.chain_id[0]

            # ...check if we have an MSA for this chain
            if chain_id in data["polymer_msas_by_chain_id"]:
                # ...if so, get the encoded MSA and the mask
                chain_encoded_msa = data["polymer_msas_by_chain_id"][chain_id][
                    "encoded_msa"
                ]  # [n_rows, n_res_in_chain] (int)
                msa_is_padded_mask = data["polymer_msas_by_chain_id"][chain_id][
                    "msa_is_padded_mask"
                ]  # [n_rows, n_res_in_chain] (bool)
                msa_ins = data["polymer_msas_by_chain_id"][chain_id]["ins"]  # [n_rows, n_res_in_chain] (int)

                # ...create a global mask to indicate whether any atom in each token is in this chain
                global_is_token_in_chain = apply_token_wise(
                    atom_array, is_atom_in_chain, np.any
                )  # [n_tokens_across_chains] (bool)

                # ...index into the MSA, dropping the atomized pieces, and any residues that may have been cropped or otherwise removed
                non_atomized_atoms = chain_instance_atom_array[~chain_instance_atom_array.atomize]
                if len(non_atomized_atoms) == 0:
                    # ... skip if there are no non-atomized atoms in this chain
                    continue

                within_poly_res_idx = non_atomized_atoms[
                    get_token_starts(non_atomized_atoms)
                ].within_poly_res_idx  # [n_non_atomized_res_in_chain] (int)
                subselected_encoded_msa = chain_encoded_msa[
                    :, within_poly_res_idx
                ]  # [n_rows, n_non_atomized_res_in_chain] (int)
                subselected_msa_is_padded_mask = msa_is_padded_mask[
                    :, within_poly_res_idx
                ]  # [n_rows, n_non_atomized_res_in_chain] (bool)
                subselected_msa_ins = msa_ins[:, within_poly_res_idx]  # [n_rows, n_non_atomized_res_in_chain] (int)

                # ...set all non-atomized tokens in this chain (e.g., full residues) to the subselected MSA
                mask = global_is_token_in_chain & (~is_token_atomized)  # [n_tokens_across_chains] (bool)
                full_encoded_msa[:, mask] = subselected_encoded_msa  # [n_rows, n_tokens_across_chains] (int)
                full_msa_is_padded_mask[:, mask] = (
                    subselected_msa_is_padded_mask  # [n_rows, n_tokens_across_chains] (bool)
                )
                full_msa_ins[:, mask] = subselected_msa_ins  # [n_rows, n_tokens_across_chains] (int)
                token_idx_has_msa[mask] = True  # [n_tokens_across_chains] (bool)

        # ...for the first row, set the tokens directly from the output of the `Atomize` transform (i.e., the atomized tokens, and anything without an MSA)
        full_encoded_msa[0] = data["encoded"]["seq"]  # [n_tokens_across_chains] (int)
        full_msa_is_padded_mask[0] = False  # [n_tokens_across_chains] (bool)

        data["encoded"]["msa"] = full_encoded_msa  # [n_rows, n_tokens_across_chains] (int)
        data["full_msa_details"] = {
            "token_idx_has_msa": token_idx_has_msa,  # [n_tokens_across_chains] (bool)
            "msa_is_padded_mask": full_msa_is_padded_mask,  # [n_rows, n_tokens_across_chains] (bool)
            # The insertions are not yet encoded (still raw counts), so we store them separately
            "msa_raw_ins": full_msa_ins,  # [n_rows, n_tokens_across_chains] (int)
        }

        return data


class FeaturizeMSALikeRF2AA(Transform):
    """
    Featurizes the MSA in the style of RF2AA, returning one featurized set of outputs for each recycle.

    Initialization arguments:
        encoding (TokenEncoding): The encoding object to use for the MSA.
        n_recycles (int): The number of recycles to perform. We will generate a unique featurized MSA for each recycle.
        n_msa_cluster_representatives (int): The number of MSA cluster representatives to select. The remaining sequences (up to `n_extra_rows`) will be used as extra MSA.
        n_extra_rows (int): The number of extra MSA rows to use. If there are fewer than `n_extra_rows` remaining sequences, we will use all of them.
        mask_behavior_probs (dict): A dictionary containing the probabilities for each BERT-style mask behavior. The keys are:
            - "replace_with_random_aa": The probability of replacing a masked index with a uniformly random amino acid.
            - "replace_with_msa_profile": The probability of replacing a masked index with an amino acid sampled from the MSA profile.
            - "do_not_replace": The probability of keeping the original amino acid at a masked index.
            - The final probability, "replace_with_mask_token", is implicitly 1 - sum(probs.values()).
        mask_probability (float): The probability of masking a given token in the MSA.
        polymer_token_indices (torch.Tensor, optional): Tensor of token indices that correspond to polymer residues. If not provided, we assume all tokens are polymer residues. Used for optimization.

    For each recycle, performs four primary steps:
        (1) Select cluster representatives from the full MSA
        (2) Mask the cluster representatives using a BERT-style mask
        (3) Assign the remaining sequences to the cluster representatives
        (4) Summarize the clusters into profiles and mean insertions at each position

    Outputs:
        For each recycle, we store the following in `data["msa_features_per_recycle_dict"]` (inspired by AF-2 Supplement, Table 1):
        - "first_row_of_msa": The first row of the MSA, which is the query sequence.
        - "cluster_representatives_msa_masked": The (masked) MSA for the cluster representatives.
        - "cluster_representatives_has_insertion": A mask indicating whether a given position in the cluster representatives MSA has an insertion.
        - "cluster_representatives_insertion_value": The raw insertion value at each position in the cluster representatives MSA, transformed to [0,1]
        - "cluster_insertion_mean": The mean insertion value at each position in the cluster representatives MSA, transformed to [0,1]
        - "cluster_profile": The MSA profile for the cluster representatives.
        - "extra_msa": The MSA for the extra sequences.
        - "extra_msa_has_insertion": A mask indicating whether a given position in the extra MSA has an insertion.
        - "extra_msa_insertion_value": The raw insertion value at each position in the extra MSA, transformed to [0,1]
    """

    requires_previous_transforms = ["FillFullMSAFromEncoded", "EncodeMSA", ConvertToTorch]

    def __init__(
        self,
        *,
        encoding: TokenEncoding = RF2AA_ATOM36_ENCODING,
        n_recycles: int,
        n_msa_cluster_representatives: int,
        n_extra_rows: int,
        mask_behavior_probs: dict,
        mask_probability: float,
        polymer_token_indices: torch.Tensor | None = None,
        eps: float = 1e-6,
    ):
        self.encoding = encoding
        self.n_recycles = n_recycles
        self.n_msa_cluster_representatives = n_msa_cluster_representatives
        self.n_extra_rows = n_extra_rows
        self.mask_behavior_probs = mask_behavior_probs
        self.mask_probability = mask_probability
        self.polymer_token_indices = polymer_token_indices
        self.eps = eps

    def check_input(self, data: dict):
        check_contains_keys(data, ["encoded", "full_msa_details"])

    def forward(self, data: dict) -> dict:
        encoded_msa = data["encoded"]["msa"]  # [n_rows, n_tokens_across_chains] (int)
        token_idx_has_msa = data["full_msa_details"]["token_idx_has_msa"]  # [n_tokens_across_chains] (bool)
        msa_is_padded_mask = data["full_msa_details"]["msa_is_padded_mask"]  # [n_rows, n_tokens_across_chains] (bool)
        msa_raw_ins = data["full_msa_details"]["msa_raw_ins"]  # [n_rows, n_tokens_across_chains] (int)

        # Select either the first `n_msa_cluster_representatives` rows or all rows, whichever is smaller
        n_rows, n_seq = encoded_msa.shape
        n_msa_cluster_representatives = min(self.n_msa_cluster_representatives, n_rows)

        # Compute the raw MSA profile, which is required for the BERT-style masking of the MSA, where with a 10% probability, we replace an amino acid with an amino
        # sampled from the MSA profile at a given position
        full_msa_profile = grouped_count(
            encoded_msa,
            mask=~msa_is_padded_mask,  # ... ignore padding when computing the profile
            groups=[
                torch.zeros(n_rows, dtype=torch.long),  # ... assign all sequences to the same group
                torch.arange(n_seq),  # ... assign each seq position to a different group
            ],
            n_tokens=self.encoding.n_tokens,  # ... return a float tensor
            dtype=torch.float,  # ... return a float tensor
        ).squeeze()  # [n_tokens_across_chains, n_tokens] (float)
        full_msa_profile /= (
            full_msa_profile.sum(dim=-1, keepdim=True) + self.eps
        )  # [n_tokens_across_chains, n_tokens] (float)

        # Generate a unique MSA (both cluster representative MSA and extra MSA) for each recycle
        msa_features_per_recycle_dict = {
            "first_row_of_msa": [],  # [n_tokens_across_chains] (int)
            "cluster_representatives_msa_ground_truth": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (int)
            "cluster_representatives_msa_masked": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (int)
            "cluster_representatives_has_insertion": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (bool)
            "cluster_representatives_insertion_value": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)
            "cluster_insertion_mean": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)
            "cluster_profile": [],  # [n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] (float)
            "extra_msa": [],  # [n_not_selected_rows, n_tokens_across_chains] (int)
            "extra_msa_has_insertion": [],  # [n_not_selected_rows, n_tokens_across_chains] (bool)
            "extra_msa_insertion_value": [],  # [n_not_selected_rows, n_tokens_across_chains] (float)
            "bert_mask_position": [],  # [n_msa_cluster_representatives, n_tokens_across_chains] (bool)
        }
        for _ in range(self.n_recycles):
            # ============================================================
            # (1) SELECT CLUSTER REPRESENTATIVES FROM THE FULL MSA
            # ============================================================

            # Select the MSA cluster representatives using the preferred sampling strategy
            selected_indices, not_selected_indices = uniformly_select_msa_cluster_representatives(
                n_rows, n_msa_cluster_representatives
            )

            # ============================================================
            # (2) MASK THE CLUSTER REPRESENTATIVES WITH BERT-STYLE MASK
            # ============================================================

            # Mask the MSA, using the BERT-style approach from AF2
            # We only apply the mask to the cluster representatives, not the extra MSA
            index_can_be_masked = build_msa_index_can_be_masked(
                msa_is_padded_mask=msa_is_padded_mask,
                token_idx_has_msa=token_idx_has_msa,
                encoded_msa=encoded_msa,
                encoding=self.encoding,
            )  # [n_rows, n_tokens_across_chains] (bool)
            mask_position = torch.zeros_like(encoded_msa, dtype=torch.bool)  # [n_rows, n_tokens_across_chains] (int)
            partial_masked_msa, partial_mask_position = mask_msa_like_bert(
                encoding=self.encoding,
                mask_behavior_probs=self.mask_behavior_probs,
                mask_probability=self.mask_probability,
                full_msa_profile=full_msa_profile,
                encoded_msa=encoded_msa[selected_indices],
                index_can_be_masked=index_can_be_masked[selected_indices],
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (int)

            # Clone the encoded MSA to avoid modifying the original...
            encoded_and_masked_msa = encoded_msa.detach().clone()

            # ...and update the masked positions
            encoded_and_masked_msa[selected_indices] = partial_masked_msa
            mask_position[selected_indices] = partial_mask_position

            # ============================================================
            # (3) ASSIGN THE EXTRA SEQUENCES TO THE CLUSTER REPRESENTATIVES
            # ============================================================

            # Define the tokens to ignore when clustering
            # NOTE: We would also need to ignore the gap token, if present in the encoding; we could consider having an encoding function to return "special" tokens to ignore.
            tokens_to_ignore = torch.tensor([self.encoding.token_to_idx[token_name] for token_name in ["<M>", "UNK"]])

            index_should_be_counted_mask = build_indices_should_be_counted_masks(
                encoded_msa=encoded_and_masked_msa,
                mask_position=mask_position,
                tokens_to_ignore=tokens_to_ignore,
                token_idx_has_msa=token_idx_has_msa,
            )  # [n_rows, n_tokens_across_chains] (bool)

            assignments = assign_extra_rows_to_cluster_representatives(
                cluster_representatives_msa=encoded_and_masked_msa[selected_indices],
                clust_reps_should_be_counted_mask=index_should_be_counted_mask[selected_indices],
                extra_msa=encoded_and_masked_msa[not_selected_indices],
                extra_msa_should_be_counted_mask=index_should_be_counted_mask[not_selected_indices],
            )  # [n_not_selected_rows] (int)

            # ============================================================
            # (4) SUMMARIZE THE CLUSTERS INTO PROFILES AND MEAN INSERTIONS
            # ============================================================
            msa_cluster_profiles = torch.zeros(
                encoded_and_masked_msa[selected_indices].shape + (self.encoding.n_tokens,), dtype=torch.float
            )  # [n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] (float)
            msa_cluster_mean_ins = torch.zeros_like(
                encoded_and_masked_msa[selected_indices], dtype=torch.float
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)

            # Summarize the clusters into profiles and mean insertions at each position. We can perform two optimizations here:
            # (1) We only need to summarize where we have MSA's; we will handle the atomized residues (and positions without MSAs) later
            # (2) We only need to worry about polymer tokens; all atom tokens for indices with MSAs will be zero by definition

            # ...determine the token indices we should be considering
            polymer_token_indices = (
                torch.arange(self.encoding.n_tokens, dtype=torch.long)
                if self.polymer_token_indices is None
                else self.polymer_token_indices
            )

            # ...compute the profiles and mean insertions for the cluster representatives
            msa_cluster_profiles_with_msas_poly_tokens, msa_cluster_mean_ins_with_msas = summarize_clusters(
                encoded_msa=encoded_and_masked_msa[
                    :, token_idx_has_msa
                ],  # Optimization 1: Only consider tokens with MSAs
                msa_raw_ins=msa_raw_ins[:, token_idx_has_msa],
                mask_position=mask_position[:, token_idx_has_msa],
                assignments=assignments,
                selected_indices=selected_indices,
                not_selected_indices=not_selected_indices,
                msa_is_padded_mask=msa_is_padded_mask[:, token_idx_has_msa],
                n_tokens=polymer_token_indices.shape[
                    0
                ],  # Optimization 2: Only consider non-atom tokens. NOTE: Hyper-specific to RF2AA; should be generalized in the future
                eps=self.eps,
            )  # [n_msa_cluster_representatives, n_tokens_with_msas, n_polymer_tokens] (float), [n_msa_cluster_representatives, n_tokens_with_msas] (float)

            # ...if we used a subset of the tokens, we need to map the profiles (but not insertions, since those don't have a token dimensino) back to the full token set, padding with zeros
            if polymer_token_indices.shape[0] < self.encoding.n_tokens:
                msa_cluster_profiles_with_msas = torch.zeros(
                    (tuple(msa_cluster_profiles_with_msas_poly_tokens.shape[:-1]) + (self.encoding.n_tokens,)),
                    dtype=torch.float,
                )  # [n_msa_cluster_representatives, n_tokens_with_msas, n_tokens] (float)
                msa_cluster_profiles_with_msas[:, :, polymer_token_indices] = msa_cluster_profiles_with_msas_poly_tokens
            else:
                msa_cluster_profiles_with_msas = msa_cluster_profiles_with_msas_poly_tokens

            # ...fill in the profiles and mean insertions for the cluster representatives
            msa_cluster_profiles[:, token_idx_has_msa] = msa_cluster_profiles_with_msas
            msa_cluster_mean_ins[:, token_idx_has_msa] = msa_cluster_mean_ins_with_msas
            del msa_cluster_profiles_with_msas, msa_cluster_mean_ins_with_msas

            # Now, handle the atomized residues and positions without MSAs:
            # (a) For insertions, they should be zeros everywhere by definition, since we have no MSA (which is handled by the initialization)
            # (b) For profiles, they should be 1 for the index of the amino acid in the query sequence, and 0 elsewhere (e.g., one-hot encoding of the query sequence)
            query_sequence_no_msa_profile = (
                torch.nn.functional.one_hot(encoded_and_masked_msa[0, ~token_idx_has_msa], self.encoding.n_tokens)
                .unsqueeze(0)
                .float()
            )  # [1, n_tokens_without_msas, n_tokens] (float)
            non_query_no_msa_profile = torch.zeros(
                ((n_msa_cluster_representatives - 1,) + tuple(query_sequence_no_msa_profile.shape[1:])),
                dtype=torch.float,
            )  # [n_msa_cluster_representatives - 1, n_tokens_with_msas, n_tokens] (float)
            msa_cluster_profiles_without_msas = torch.cat(
                [query_sequence_no_msa_profile, non_query_no_msa_profile], dim=0
            )  # [n_msa_cluster_representatives, n_tokens_without_msas, n_tokens] (float)

            msa_cluster_profiles[:, ~token_idx_has_msa] = msa_cluster_profiles_without_msas
            del msa_cluster_profiles_without_msas, query_sequence_no_msa_profile, non_query_no_msa_profile

            # Subselect the extra MSA rows
            # From AF2 Supplement, section 1.2.7:
            #   (...)
            #   4. The MSA sequences that have not been selected as cluster centres
            #   at step 1 are used to randomly sample N_{extra_seq} sequences
            #   without replacement. If there are less than N_{extra_seq} remaining
            #   sequences available, all of them are used.
            #   (...)
            if not_selected_indices.shape[0] > self.n_extra_rows:
                shuffled_indices = torch.randperm(not_selected_indices.shape[0])
                not_selected_indices = not_selected_indices[
                    shuffled_indices[: self.n_extra_rows]
                ]  # [n_extra_rows] (int)

            def transform_ins_counts(ins: torch.Tensor) -> torch.Tensor:
                """Transforms insertion counts into the range [0,1] using the function given in the AF2 Supplement"""
                return 2 / torch.pi * torch.arctan(ins / 3)

            # Sequence
            msa_features_per_recycle_dict["first_row_of_msa"].append(
                encoded_and_masked_msa[0]
            )  # [n_tokens_across_chains] (int)

            # ...without masks (ground truth for masked token recovery)
            msa_features_per_recycle_dict["cluster_representatives_msa_ground_truth"].append(
                encoded_msa[selected_indices]
            )

            # +------- Information about the msa cluster representatives (NOT the clusters themselves) -------+
            # ...with masks
            msa_features_per_recycle_dict["cluster_representatives_msa_masked"].append(
                encoded_and_masked_msa[selected_indices]
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (int)

            # ...insertions
            msa_features_per_recycle_dict["cluster_representatives_has_insertion"].append(
                msa_raw_ins[selected_indices] > 0
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (bool)
            msa_features_per_recycle_dict["cluster_representatives_insertion_value"].append(
                transform_ins_counts(msa_raw_ins[selected_indices])
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)

            # +------- Aggregated information about the msa clusters (e.g, profiles, insertions) -------+
            msa_features_per_recycle_dict["cluster_insertion_mean"].append(
                transform_ins_counts(msa_cluster_mean_ins)
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (float)
            msa_features_per_recycle_dict["cluster_profile"].append(
                msa_cluster_profiles
            )  # [n_msa_cluster_representatives, n_tokens_across_chains, n_tokens] (float)

            # +------- Information about the extra MSA -------+
            extra_msa = encoded_and_masked_msa[not_selected_indices]
            if extra_msa.shape[0] > 0:
                # ...replace the first row of the extra MSA with the (masked) query sequence (a RF2AA novelty)
                extra_msa[0] = encoded_and_masked_msa[0]
            else:
                # ...if there's no extra MSA, we need to create a dummy row with the query sequence (a RF2AA novelty)
                extra_msa = encoded_and_masked_msa[0].unsqueeze(0)

            msa_features_per_recycle_dict["extra_msa"].append(extra_msa)  # [n_extra_rows, n_tokens_across_chains] (int)

            # ...insertions
            msa_features_per_recycle_dict["extra_msa_has_insertion"].append(
                msa_raw_ins[not_selected_indices] > 0
                if not_selected_indices.shape[0] > 0
                else torch.zeros_like(extra_msa, dtype=torch.bool)
            )  # [n_extra_rows, n_tokens_across_chains] (bool)
            msa_features_per_recycle_dict["extra_msa_insertion_value"].append(
                transform_ins_counts(msa_raw_ins[not_selected_indices])
                if not_selected_indices.shape[0] > 0
                else torch.zeros_like(extra_msa, dtype=torch.float)
            )  # [n_extra_rows, n_tokens_across_chains] (float)

            # +------- Mask information -------+
            msa_features_per_recycle_dict["bert_mask_position"].append(
                mask_position[selected_indices]
            )  # [n_msa_cluster_representatives, n_tokens_across_chains] (bool)

        data["features_per_recycle_dict"] = msa_features_per_recycle_dict
        return data
