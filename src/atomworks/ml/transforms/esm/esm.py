import logging
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from biotite.structure import AtomArray
from esm import pretrained

from atomworks.io.enums import ChainType
from atomworks.ml.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from atomworks.ml.transforms.atom_array import (
    AddWithinPolyResIdxAnnotation,
)
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.esm._esm_generation import ESM_2_EMBED_DIM, generate_esm_embedding
from atomworks.ml.utils.io import get_sharded_file_path
from atomworks.ml.utils.misc import hash_sequence
from atomworks.ml.utils.token import apply_token_wise, get_token_count, get_token_starts

logger = logging.getLogger(__name__)


class LoadPolymerESMs(Transform):
    """
    Load ESM embeddings for all polymer chains in the AtomArray.

    For the embeddings that are found, store the ESM embedding in `polymer_esms_by_chain_id`
    indexed by chain_id (e.g., "A"). For embeddings that are not found, create a padded
    embedding and record the padding mask.

    Args:
        esm_embedding_dirs (list[dict]): The directories containing the ESM embeddings and
            their associated file types, as a list of dictionaries. If multiple
            directories are provided, all of them will be searched. Keys in the dictionary
            are:
                - dir (str): The directory where the ESM embedding files are stored.
                - extension (str): The file extension of the embedding files (e.g., ".pt").
                - directory_depth (int, optional): The directory nesting depth, i.e., the embedding file
                  might be stored at `dir/d8/07/d8074f77ba.pt`. Must be sharded
                  by the first two characters of the sequence hash. Defaults to 0 (flat directory).
            Note:
                (a) The files must be named using the SHA-256 hash of the sequence (see `hash_sequence` in
                    `utils/misc`).
                (b) Order matters - directories will be searched in the order provided, and the first match will be returned.
        geenrate_on_the_fly (bool, optional): Whether to generate the ESM embeddings on-the-fly if they are not found in the disk. Defaults to False.
        esm_model_name (str): The name of the ESM model to use. Must be one of the models supported by the `esm` package,
            the precomputed embeddings use esm2_t36_3B_UR50D. Defaults to "esm2_t36_3B_UR50D".
        esm_cache_dir (dict, optional): The directory to cache the on-the-fly generated ESM data. the dictionary format is the same as the esm_embedding_dirs.
            If None, caching is turned off.
        use_gpu (bool, optional): Whether to use the GPU for ESM embedding generation where none ESM embedding is saved in the disk. Defaults to False.
        esm_batch_size (int, optional): The batch size for ESM embedding generation. Defaults to 4096

    The `polymer_esms_by_chain_id` dictionary contains the following keys:
        - esm_embedding: The ESM embedding as a numpy array of shape (sequence_length, embedding_dim).
        - esm_is_padded_mask: A mask indicating whether the embedding is padded (True) or not (False).
    """

    esm_embedding_dirs: list[dict]
    generate_on_the_fly: bool
    esm_model_name: str
    esm_cache_dir: dict
    embedding_dim: int
    esm_batch_size: int

    def __init__(
        self,
        esm_embedding_dirs: list[dict] = [
            {
                "dir": "/projects/ml/RF2_allatom/PDB_ESM_embedding_all",
                "extension": ".pt",
                "directory_depth": 2,
            },
            {
                "dir": "/projects/ml/RF2_allatom/PDB_ESM_embedding",
                "extension": ".pt",
                "directory_depth": 2,
            },
        ],
        generate_on_the_fly: bool = False,
        esm_model_name: str = "esm2_t36_3B_UR50D",
        esm_cache_dir: PathLike | None = None,
        use_gpu: bool = True,
        esm_batch_size: int = 4096,
    ):
        self.esm_embedding_dirs = esm_embedding_dirs
        self.generate_on_the_fly = generate_on_the_fly
        # ...load the ESM model
        if generate_on_the_fly:
            model, alphabet = pretrained.load_model_and_alphabet(esm_model_name)
            if torch.cuda.is_available() and use_gpu:
                model = model.cuda()
            model.eval()
            self.model = {"model": model, "alphabet": alphabet}

            self.esm_to_inference = {}  # dictionary to store the ESM embeddings that are not found in the disk
            self.esm_cache_dir = esm_cache_dir
            self.esm_batch_size = esm_batch_size
        else:
            self.model = None
            self.esm_to_inference = None
            self.esm_cache_dir = None
            self.esm_batch_size = None

        self.embedding_dim = ESM_2_EMBED_DIM[esm_model_name]

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["atom_array", "chain_info"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "chain_id"])

    def _load_esm_data(self, chain_type: ChainType, sequence: str) -> dict:
        """
        Load the ESM embedding data based on chain type and file availability.

        Args:
            chain_type (ChainType): The type of the chain (Protein or RNA).
            sequence (str): The polymer one-letter sequence.

        Returns:
            A dictionary containing the ESM embedding.
        """
        esm_embedding = None
        find_esm_flag = False

        if chain_type.is_protein():
            # ...hash the sequence so we can look up the ESM embedding file
            sequence_hash = hash_sequence(sequence)

            # ...loop through all ESM embedding directories, checking if the requested file exists
            for esm_dir in self.esm_embedding_dirs:
                # ...build the path to the ESM embedding file based on the directory depth
                depth = esm_dir.get("directory_depth", 0)
                esm_file = get_sharded_file_path(Path(esm_dir["dir"]), sequence_hash, esm_dir["extension"], depth)

                # ...load the ESM embedding file if it exists
                if esm_file.exists():
                    # ...load the ESM embedding (assuming it's a PyTorch tensor saved with torch.save)
                    esm_embedding = torch.load(esm_file)
                    find_esm_flag = True
                    # (Don't look at the other ESM directories if we found the file in this one)
                    break

            # also check if we already have the ESM embedding in the cache directory
            if self.esm_cache_dir is not None:
                esm_file = get_sharded_file_path(
                    self.esm_cache_dir["dir"],
                    sequence_hash,
                    self.esm_cache_dir["extension"],
                    self.esm_cache_dir["directory_depth"],
                )
                if esm_file.exists():
                    esm_embedding = torch.load(esm_file)
                    find_esm_flag = True
            # Convert to numpy array if necessary
            if isinstance(esm_embedding, torch.Tensor):
                esm_embedding = esm_embedding.numpy()

            # ...if we didn't find the ESM embedding, add the sequence to the dictionary to generate the ESM embedding
            if esm_embedding is None and self.generate_on_the_fly:
                self.esm_to_inference[sequence_hash] = sequence
            elif esm_embedding is None:
                # if we don't have the ESM embedding and we don't want to generate it, we set the sequence_hash to None
                # we will do zero-filled embedding in the next step
                sequence_hash = None
                find_esm_flag = True

        else:
            # Currently, ESM embeddings are typically for proteins. If you have RNA embeddings,
            # you can handle them here.
            # ...handle RNA ESM embeddings if available
            sequence_hash = None
            find_esm_flag = True  # we don't have ESM embeddings for RNA/DNA, so we set the flag to True

        return {"sequence_hash": sequence_hash, "esm_embedding": esm_embedding, "find_esm_flag": find_esm_flag}

    def _get_polymer_esms_by_chain_id(self, polymer_chain_ids: np.array, chain_info: dict) -> dict:
        """
        Retrieves ESM embeddings for each polymer chain ID (e.g., "A").
        If we can't find an embedding for a given chain ID, create a zero-filled embedding
        and set the esm_is_padded_mask accordingly.

        Args:
            polymer_chain_ids (np.array): List of unique chain IDs for polymers.
            chain_info (dict): Dictionary containing chain information for the polymers (e.g., type).
                Must have chain IDs as keys, with "type" and "processed_entity_non_canonical_sequence" as sub-keys.

        Returns:
            esms_by_chain_id (dict): Dictionary with chain IDs as keys and corresponding ESM embedding data as values.
        """
        esms_by_chain_id = {}
        for chain_id in polymer_chain_ids:
            sequence = chain_info[chain_id]["processed_entity_non_canonical_sequence"]
            chain_type = chain_info[chain_id]["chain_type"]

            # ...load the ESM embedding file from the correct directory
            sequence_length = len(sequence)
            esm_data = self._load_esm_data(chain_type, sequence)

            # the protein ESM generated on-the-fly
            if not esm_data["find_esm_flag"]:
                esms_by_chain_id[chain_id] = {"sequence_hash": esm_data["sequence_hash"]}

            # the protein ESM found in the disk
            elif esm_data["esm_embedding"] is not None:
                esm_is_padded_mask = np.zeros(sequence_length, dtype=bool)
                esms_by_chain_id[chain_id] = {
                    "sequence_hash": esm_data["sequence_hash"],
                    "esm_embedding": esm_data["esm_embedding"],
                    "esm_is_padded_mask": esm_is_padded_mask,
                }
            # for the RNA/DNA sequences and for the protein ESM not found in the disk and we don't want to generate it on-the-fly
            else:
                esm_embedding = np.zeros((sequence_length, self.embedding_dim), dtype=np.float32)
                esm_is_padded_mask = np.ones(sequence_length, dtype=bool)  # True indicates padded

                esms_by_chain_id[chain_id] = {
                    "sequence_hash": None,
                    "esm_embedding": esm_embedding,
                    "esm_is_padded_mask": esm_is_padded_mask,
                }

                logger.warning(f"ESM embedding not found for chain ID {chain_id}. Using zero-filled embedding.")

        # ...loop through all the sequences that are not found in the disk and generate the ESM embeddings
        if self.generate_on_the_fly:
            esm_embeddings = generate_esm_embedding(
                self.esm_to_inference, self.model, toks_per_batch=self.esm_batch_size
            )
            for chain_id in polymer_chain_ids:
                if esms_by_chain_id[chain_id]["sequence_hash"] in esm_embeddings:
                    logger.warning(f"ESM embedding not found for chain ID {chain_id}. Generated on-the-fly.")
                    sequence_hash = esms_by_chain_id[chain_id]["sequence_hash"]

                    esm_embedding = esm_embeddings[sequence_hash]
                    sequence_length = len(esm_embedding)
                    esm_is_padded_mask = np.zeros(sequence_length, dtype=bool)
                    esms_by_chain_id[chain_id] = {
                        "sequence_hash": sequence_hash,
                        "esm_embedding": esm_embedding,
                        "esm_is_padded_mask": esm_is_padded_mask,
                    }

            # ...save the ESM embedding to the cache directory
            for sequence_hash, esm_embedding in esm_embeddings.items():
                if self.esm_cache_dir is not None:
                    esm_file = get_sharded_file_path(
                        self.esm_cache_dir["dir"],
                        sequence_hash,
                        self.esm_cache_dir["extension"],
                        self.esm_cache_dir["directory_depth"],
                    )
                    torch.save(esm_embedding, esm_file)

        return esms_by_chain_id

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        chain_info = data["chain_info"]
        polymer_chain_ids = np.unique(atom_array.chain_id[np.isin(atom_array.chain_type, ChainType.get_polymers())])

        polymer_esms_by_chain_id = self._get_polymer_esms_by_chain_id(polymer_chain_ids, chain_info)

        # Add the ESM embeddings to the data dictionary
        data["polymer_esms_by_chain_id"] = polymer_esms_by_chain_id

        return data


class FillFullESMFromEncoded(Transform):
    """
    Fills in the full ESM from the encoded MSA, using the atom array to determine the order of the tokens.
    Starts by creating full np.arrays with default values (padding tokens), and then fills in the encoded ESM embeddings by looping over chain instances.

    Returns:
        The ESM, with padding, as a 2D np.array of float, stored in `data["esm"]`.
        Additionally, we store the following details in `data["full_esm_details"]`:
            - esm_is_padded_mask (np.array): A mask indicating whether a given position in the ESM is padded (1) or not (0).
    """

    requires_previous_transforms = [AddWithinPolyResIdxAnnotation]

    def __init__(self, embedding_dim: int = 2560, pad_float: float = 0.0):
        self.EMBEDDING_DIM = embedding_dim
        self.PAD_FLOAT = pad_float

    def check_input(self, data: dict) -> None:
        check_contains_keys(data, ["polymer_esms_by_chain_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # ...set up empty encoded esm (purely padded) for encoded esm
        token_count = get_token_count(atom_array)
        full_encoded_esm = np.full(
            (token_count, self.EMBEDDING_DIM), self.PAD_FLOAT, dtype=float
        )  # [1, n_tokens_across_chains]
        full_esm_is_padded_mask = np.ones(
            (token_count), dtype=bool
        )  # [1, n_tokens_across_chains] (bool) 1 = padded, 0 = not padded

        # ...create a mask indicating whether any atom in each token is atomized
        is_token_atomized = apply_token_wise(atom_array, atom_array.atomize, np.any)  # [n_tokens_across_chains] (bool)

        # ...loop through all `chain_iids` (polymer/non-polymer) and populate relevant columns in encoding
        token_idx_has_esm = np.zeros(get_token_count(atom_array), dtype=bool)  # [n_tokens_across_chains] (bool)
        for chain_iid in np.unique(atom_array.chain_iid):
            is_atom_in_chain = atom_array.chain_iid == chain_iid  # [n_atoms_total] (bool)

            chain_instance_atom_array = atom_array[is_atom_in_chain]
            chain_id = chain_instance_atom_array.chain_id[0]

            # ...check if we have ESM embedding for this chain
            if chain_id in data["polymer_esms_by_chain_id"]:
                # ...if so, get the encoded MSA and the mask
                chain_encoded_esm = data["polymer_esms_by_chain_id"][chain_id][
                    "esm_embedding"
                ]  # [n_res_in_chain, embedding_dim]
                esm_is_padded_mask = data["polymer_esms_by_chain_id"][chain_id][
                    "esm_is_padded_mask"
                ]  # [n_res_in_chain] (bool)
                # ...create a global mask to indicate whether any atom in each token is in this chain
                global_is_token_in_chain = apply_token_wise(
                    atom_array, is_atom_in_chain, np.any
                )  # [n_tokens_across_chains] (bool)

                # ...index into the esm, dropping the atomized pieces, and any residues that may have been cropped or otherwise removed
                non_atomized_atoms = chain_instance_atom_array[~chain_instance_atom_array.atomize]
                if len(non_atomized_atoms) == 0:
                    # ... skip if there are no non-atomized atoms in this chain
                    continue

                within_poly_res_idx = non_atomized_atoms[
                    get_token_starts(non_atomized_atoms)
                ].within_poly_res_idx  # [n_non_atomized_res_in_chain] (int)
                subselected_encoded_esm = chain_encoded_esm[
                    within_poly_res_idx, :
                ]  # [n_non_atomized_res_in_chain, esm_embed_size] (float)
                subselected_esm_is_padded_mask = esm_is_padded_mask[
                    within_poly_res_idx
                ]  # [n_non_atomized_res_in_chain] (bool)

                # ...set all non-atomized tokens in this chain (e.g., full residues) to the subselected esm
                mask = global_is_token_in_chain & (~is_token_atomized)  # [n_tokens_across_chains] (bool)
                full_encoded_esm[mask, :] = subselected_encoded_esm  # [n_tokens_across_chains] (int)
                full_esm_is_padded_mask[mask] = subselected_esm_is_padded_mask  # [n_tokens_across_chains] (bool)
                token_idx_has_esm[mask] = True  # [n_tokens_across_chains] (bool)

        data["esm"] = full_encoded_esm  # [n_tokens_across_chains] (float)
        data["full_esm_details"] = {
            "token_idx_has_esm": token_idx_has_esm,  # [n_tokens_across_chains] (bool)
            "esm_is_padded_mask": full_esm_is_padded_mask,  # [n_tokens_across_chains] (bool)
        }

        return data
