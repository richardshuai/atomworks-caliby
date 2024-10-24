"""Transforms for adding and featurizing templates."""

import logging
import os
from dataclasses import dataclass
from functools import cache
from typing import Any

import biotite.structure as struc
import numpy as np
import pandas as pd
import torch
from assertpy import assert_that
from biotite.structure import AtomArray
from cifutils.enums import ChainType
from rf2aa.chemical import ChemicalData

from datahub.encoding_definitions import (
    LEGACY_RF2_ATOM14_ENCODING,
    RF2AA_ATOM36_ENCODING,
    TokenEncoding,
)
from datahub.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from datahub.transforms.atom_array import (
    AddWithinPolyResIdxAnnotation,
    chain_instance_iter,
)
from datahub.transforms.base import Transform
from datahub.transforms.encoding import atom_array_from_encoding, atom_array_to_encoding
from datahub.utils.numpy import select_data_by_id
from datahub.utils.token import get_token_count, get_token_starts

try:
    from rf2aa.chemical import ChemicalData

    chemdata = ChemicalData()
except Exception:
    from rf2aa.chemical import initialize_chemdata

    initialize_chemdata()
    chemdata = ChemicalData()

logger = logging.getLogger(__name__)


@dataclass
class RF2AATemplate:
    """
    Data class for holding template information in the RF, RF2 & RF2AA format.

    NOTE:
     - RF templates only exist for proteins
     - This is a helper class to cast the templates into a more `readable` format and also
       to provide an interface layer that allows us to deal with templates as atom_arrays, if
       we ever re-create templates or add templates for non-proteins
     - RF-style templates already come encoded in atom14 representation (RFAtom14, not AF2Atom14)

    Keys:
    - xyz: Tensor([1, n_templates x n_atoms_per_template, 14, 3]), raw coordinates of all templates
    - mask: Tensor([1, n_templates x n_atom_per_template, 14]), mask of all templates
    - qmap: Tensor([1, n_templates x n_atom_per_template, 2]), alignment mapping of all templates
        - index 0: which index in the query protein this template index matches to
        - index 1: which template index this matches to
    - f0d: Tensor([1, n_templates, 8?]), [0,:,4] holds sequence identity info
    - f1d: Tensor([1, n_templates x n_atoms_per_template, 3]), something in there may be related to template confidence, gaps?
    - seq: Tensor([1, 100677]) (tensor, encoded with Chemdata.aa2num encoding)
    - ids: list[tuple[str]]  # Holds the f"{pdb_id}_{chain_id}" of the template
    - label: list[str]  # holds the lookup_id for this template
    """

    xyz: torch.Tensor  # [1, n_templates x n_atoms_per_template, 14, 3]
    mask: torch.Tensor  # [1, n_templates x n_atom_per_template, 14]
    qmap: torch.Tensor  # [1, n_templates x n_atom_per_template, 2]
    f0d: torch.Tensor  # [1, n_templates, 8?]
    f1d: torch.Tensor  # [1, n_templates x n_atoms_per_template, 3]
    seq: torch.Tensor  # [1, n_templates x n_atoms_per_template]
    ids: list[tuple[str]]  # Holds the f"{pdb_id}_{chain_id}" of the template
    label: list[str]  # holds the lookup_id for this template

    def __post_init__(self):
        self.ids = np.array(self.ids).flatten().squeeze()  # Flatten the list of tuples into an array
        # Convert all tensors to numpy
        self.xyz = self.xyz.numpy()
        self.mask = self.mask.numpy()
        self.qmap = self.qmap.numpy()
        self.f0d = self.f0d.numpy()
        self.f1d = self.f1d.numpy()
        self.seq = self.seq.numpy()
        self.label = np.array(self.label)

    @property
    def lookup_id(self) -> str:
        return self.label[0]

    @property
    def n_templates(self) -> int:
        return self.f0d.shape[1]

    @property
    def seq_similarity_to_query(self) -> np.ndarray:
        return self.f0d[0, :, 4]

    @property
    def alignment_confidence(self) -> np.ndarray:
        return self.f1d[0, :, 2]

    @property
    def pdb_ids(self) -> np.ndarray:
        return np.array([i.split("_")[0] for i in self.ids])

    @property
    def chain_ids(self) -> np.ndarray:
        return np.array([i.split("_")[1] for i in self.ids])

    @property
    def n_res_per_template(self) -> np.ndarray:
        return np.unique(self.qmap[:, :, 1], return_counts=True)[1]

    @property
    def max_aligned_query_res_idx(self) -> np.ndarray:
        aligned_query_res_idxs = self.qmap[0, :, 0]
        new_template_start_idxs = np.cumsum(self.n_res_per_template)[:-1]
        groups = np.split(aligned_query_res_idxs, new_template_start_idxs)
        # get max in each group (= template)
        return np.array([np.max(g) for g in groups])

    @property
    def template_ids(self) -> list[str]:
        return np.array(self.ids)

    def subset(self, template_idxs: list[int]) -> "RF2AATemplate":
        """
        Subset the template to only include the template indices specified in `template_idxs`.
        """
        assert np.unique(template_idxs).size == len(template_idxs), "`template_idxs` must be unique"

        # Subset the data
        template_atom_idxs = np.where(np.isin(self.qmap[0, :, 1], template_idxs))[0]
        self.xyz = self.xyz[:, template_atom_idxs]
        self.mask = self.mask[:, template_atom_idxs]
        self.qmap = self.qmap[:, template_atom_idxs]

        # Update internal template index to be from 0 to n_templates
        n_res_per_template = np.unique(self.qmap[:, :, 1], return_counts=True)[1]
        self.qmap[0, :, 1] = np.repeat(np.arange(len(template_idxs)), n_res_per_template)

        self.f0d = self.f0d[:, template_idxs]
        self.f1d = self.f1d[:, template_atom_idxs]
        self.seq = self.seq[:, template_atom_idxs]
        self.ids = self.ids[template_idxs]
        return self

    def to_atom_array(self, template_idx: int) -> AtomArray:
        assert_that(template_idx).is_instance_of(int).is_between(0, self.n_templates - 1)

        # Get pdb_id and chain_id
        template_id = self.ids[template_idx]
        pdb_id, chain_id = template_id.split("_")

        # Get indices to select the residues for the template
        template_res_idxs = np.where(self.qmap[0, :, 1] == template_idx)[0]

        # Select the template data
        # ... coordinate info
        atom14_coords = self.xyz[0, template_res_idxs, :, :]
        # ... occupancy info
        atom14_mask = self.mask[0, template_res_idxs, :]
        # ... sequence info
        seq_tokenized = self.seq[0, template_res_idxs]

        # NOTE: There was a bug in the original code that saved the RF2 templates: Tryptophan (AA17) was using
        #  a wrong atom name ordering. This was fixed in the public version of the code:
        #  https://github.com/baker-laboratory/RoseTTAFold-All-Atom/blob/c1fd92455be2a4133ad147242fc91cea35477282/rf2aa/chemical.py#L2068C1-L2070C285
        #  and we include this fix here:
        # Create atom array
        atom_array = atom_array_from_encoding(
            atom14_coords,
            atom14_mask,
            seq_tokenized,
            encoding=LEGACY_RF2_ATOM14_ENCODING,
        )
        n_atom = len(atom_array)

        # ... repeat chain id for each atom in the residue
        atom_array.chain_id = np.repeat(np.array(chain_id), n_atom)

        # ... append custom annotation for which residue in the query protein this template
        #  residue aligns to (indexing starts with 0 at query sequence start)
        aligned_query_res_idx = self.qmap[0, template_res_idxs, 0]
        atom_array.set_annotation("aligned_query_res_idx", struc.spread_residue_wise(atom_array, aligned_query_res_idx))

        # ... append custom annotation for alignment confidence
        alignment_confidence = self.f1d[0, template_res_idxs, 2]
        atom_array.set_annotation("alignment_confidence", struc.spread_residue_wise(atom_array, alignment_confidence))

        return atom_array


def blank_rf2aa_template_features(
    n_template: int,
    n_token: int,
    encoding: TokenEncoding,
    mask_token_idx: int,
    init_coords: torch.Tensor | float = chemdata.INIT_CRDS,
) -> torch.Tensor:
    """
    Generates blank template features for RF2AA.

    Args:
        n_template (int): Number of templates.
        n_token (int): Number of tokens in the structure.
        encoding (TokenEncoding): Encoding object containing token and atom information.
        mask_token_idx (int, optional): Index of the mask token. Defaults to 20.
        init_coords (torch.Tensor | float, optional): Initial coordinates for the atoms. Defaults to chemdata.INIT_CRDS.

    Returns:
        tuple: A tuple containing the following elements:
            - xyz (torch.Tensor): Tensor of shape (n_template, n_token, encoding.n_atoms_per_token, 3) containing the coordinates of the atoms.
            - t1d (torch.Tensor): Tensor of shape (n_template, n_token, encoding.n_tokens) containing the 1D template features.
            - mask (torch.Tensor): Tensor of shape (n_template, n_token, encoding.n_atoms_per_token) containing the mask information.
            - template_origin (np.ndarray): Array of shape (n_template,) containing the origin of the templates.
    """
    # TODO: Fix fill value
    # Initialize blank template features
    xyz = torch.full((n_template, n_token, encoding.n_atoms_per_token, 3), fill_value=float("nan"))
    mask = torch.zeros((n_template, n_token, encoding.n_atoms_per_token), dtype=torch.bool)
    t1d = torch.zeros((n_template, n_token, encoding.n_tokens))
    template_origin = np.full(n_template, "")

    # Fill in the initial coordinates and mask values
    xyz[:, :] = init_coords

    t1d[..., mask_token_idx] = 1.0  # Set the mask token to 1.0
    # NOTE: In RF2AA the last dim of t1d is the `confidence`. We set it here just
    #  for code clarity.
    _confidence = torch.zeros((n_template, n_token))
    t1d[..., -1] = _confidence

    return xyz, t1d, mask, template_origin


@cache
def _lazy_load_template_lookup_dict() -> dict[str, int]:
    template_msa_lookup_df = pd.read_csv("/projects/ml/TrRosetta/PDB-2021AUG02/list_v02.csv")
    template_msa_lookup_df["HASH"] = template_msa_lookup_df["HASH"].apply(lambda x: f"{x:06d}")
    pdb_chain_id_to_hash_dict = dict(
        zip(template_msa_lookup_df["CHAINID"].tolist(), template_msa_lookup_df["HASH"].tolist())
    )
    return pdb_chain_id_to_hash_dict


def _get_rf_template_id(pdb_id: str, chain_id: str, chain_type: ChainType) -> str:
    """
    Retrieves the template lookup ID for a given PDB and chain ID combination.
    (NOTE: This is the `chid_to_hash` ID used for MSAs & Templates used in the original RF2AA)

    Parameters:
    - pdb_id (str): The PDB ID of the protein structure. E.g., "1A2K".
    - chain_id (str): The chain ID within the PDB structure. E.g., "A". Notably, no transformation ID.

    Returns:
    - str: The template lookup ID corresponding to the combined PDB and chain ID.
    """
    combined_id = f"{pdb_id}_{chain_id}"
    if chain_type == ChainType.POLYPEPTIDE_L:
        # For polypeptide(L) chains, we lookup the identified based on the mapping stored on disk
        # If we don't find a match, we append "_single_sequence" to the combined ID to ensure we won't find any MSAs
        return _lazy_load_template_lookup_dict().get(combined_id)
    elif chain_type == ChainType.RNA or chain_type == ChainType.DNA:
        # For nucleic acids, we use `{pdb_id}_{chain_id}` as the identifier
        return combined_id


def _load_rf_template(rf_template_id: str | None) -> torch.Tensor | None:
    if rf_template_id is None:
        # ... skip if no template ID (e.g. no matching template ID found in the lookup dict)
        return None

    path_to_template = f"/projects/ml/TrRosetta/PDB-2021AUG02/torch/hhr/{rf_template_id[:3]}/{rf_template_id}.pt"
    if not os.path.exists(path_to_template):
        # ... skip if template file does not exist
        return None

    return torch.load(path_to_template)


class AddRFTemplates(Transform):
    """
    Adds RF templates to the data.

    The templates are added to the data under the key `template`.

    Output features:
        - template (dict): A dictionary with chain IDs as keys and a list of templates for that chain as values.
            Each template is a dictionary with the following keys:
                - id (str): The template ID.
                - pdb_id (str): The PDB ID of the template.
                - chain_id (str): The chain ID of the template.
                - template_lookup_id (str): The lookup ID for the template - this is the `chid_to_hash` ID
                    used for MSAs & Templates used in the original RF2AA which is used to retrieve the template
                    from disk.
                - seq_similarity (float): The sequence similarity of the template to the query.
                - atom_array (AtomArray): The atom array of the template.
                - n_res (int): The number of residues in the template.
    """

    def __init__(
        self,
        max_n_template: int = 1,
        pick_top: bool = True,
        min_seq_similarity: float = 0.0,
        max_seq_similarity: float = 100.0,
        min_template_length: int = 0,
        filter_by_query_length: bool = False,
    ):
        """
        Initialize the AddRFTemplates transform.

        Args:
            max_n_template (int): Maximum number of templates to add. If more `max_n_template` is larger than the
                number of available templates for a chain, all templates are added. Default is 1.
            pick_top (bool): Whether to pick the top templates based on sequence similarity if there are more than
                `max_n_template` templates available. Default is True.
            min_seq_similarity (float): Minimum sequence similarity for templates to be included. Default is 0.0.
            max_seq_similarity (float): Maximum sequence similarity for templates to be included. Default is 100.0.
            min_template_length (int): Minimum length of the template to be included. Default is 0.
            filter_by_query_length (bool): Whether to filter templates by query length. Default is False.

        Raises:
            AssertionError: If `min_seq_similarity` or `max_seq_similarity` are not between 0.0 and 100.0.
            AssertionError: If `n_template` is not a positive integer.
            AssertionError: If `min_template_length` is not a non-negative integer.
        """
        assert_that(min_seq_similarity).is_between(0.0, 100.0)
        assert_that(max_seq_similarity).is_between(0.0, 100.0)
        assert_that(max_n_template).is_instance_of(int).is_greater_than(0)
        assert_that(min_template_length).is_instance_of(int).is_greater_than_or_equal_to(0)

        self.n_template = max_n_template
        self.pick_top = pick_top
        self.min_seq_similarity = min_seq_similarity
        self.max_seq_similarity = max_seq_similarity
        self.min_template_length = min_template_length
        self.filter_by_query_length = filter_by_query_length

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "chain_info", "pdb_id"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        pdb_id = data["pdb_id"]
        chain_info = data["chain_info"]

        # Load template information
        # NOTE: Currently templates only exist for proteins
        templates = {}
        for chain_id in chain_info:
            # get chain_type and convert to Enum
            chain_type = chain_info[chain_id]["type"]
            chain_type = ChainType.from_string(chain_type)
            rf_template_id = _get_rf_template_id(pdb_id, chain_id, chain_type)
            rf_template = _load_rf_template(rf_template_id)

            if rf_template is None:
                logger.debug(f"No RF template found for {pdb_id}_{chain_id}.")
                # early exit if no templates
                continue

            # NOTE: Could be made a lazy-load for each template only if it is selected
            #  if worker memory or speed becomes a bottleneck
            chain_templates = RF2AATemplate(**rf_template)
            is_valid = np.ones(chain_templates.n_templates, dtype=bool)

            # TODO: Revisit filtering logic once `cropping` is implemented to enable crop
            #  dependent filtering below (currently the below operates on the full query seq)
            if self.max_seq_similarity <= 100.0:
                # filter out templates with sequence similarity higher than cutoff
                is_valid &= chain_templates.seq_similarity_to_query <= self.max_seq_similarity

            if self.min_seq_similarity > 0.0:
                # filter out templates with sequence similarity lower than cutoff
                is_valid &= chain_templates.seq_similarity_to_query >= self.min_seq_similarity

            if self.min_template_length > 0:
                # filter out templates with fewer residues than cutoff
                is_valid &= chain_templates.n_res_per_template >= self.min_template_length

            # TODO: Possibly filter by deposition date. This will require a query to the PDB
            #  to get the deposition date of each template

            if not np.any(is_valid):
                # early exit if no valid templates after filter criteria
                continue

            # pick `n_template` (or fewer if fewer exist) valid templates
            valid_template_idxs = np.where(is_valid)[0]
            if not self.pick_top:
                valid_template_idxs = np.random.permutation(valid_template_idxs)

            # Add templates to template dict
            chain_templates = chain_templates.subset(valid_template_idxs[: self.n_template])
            templates[chain_id] = [
                {
                    "id": chain_templates.ids[i],
                    "pdb_id": chain_templates.pdb_ids[i],
                    "chain_id": chain_templates.chain_ids[i],
                    "template_lookup_id": chain_templates.lookup_id,
                    "seq_similarity": chain_templates.seq_similarity_to_query[i],
                    "atom_array": chain_templates.to_atom_array(i),
                    "n_res": chain_templates.n_res_per_template[i],
                }
                for i in range(chain_templates.n_templates)
            ]
            logger.debug(f"Added {len(templates[chain_id])} templates for chain {chain_id}: {chain_templates.ids}.")

        data["template"] = templates
        return data


class FeaturizeRFTemplatesForRF2AA(Transform):
    """
    A transform that featurizes RFTemplates templates for RF2AA.

    This class takes the templates added by the `AddRFTemplates` transform and featurizes them
    for use in the RF2AA model. The templates are added to the data under the key `template`.

    Attributes:
        - n_template (int): The number of templates to use.
        - mask_token_idx (int): The index of the mask token. Defaults to 21.
        - init_coords (torch.Tensor | float): The initial coordinates for the templates. Defaults to `chemdata.INIT_CRDS`.
        - encoding (TokenEncoding): The encoding to use for the templates. Defaults to `RF2AA_ATOM36_ENCODING`.

    Methods:
        check_input(data: dict[str, Any]) -> None:
            Checks the input data for the required keys and types.

        forward(data: dict[str, Any]) -> dict[str, Any]:
            Featurizes the templates and adds them to the data.

    Raises:
        AssertionError: If `n_template` is not a positive integer.
        AssertionError: If `encoding` is not an instance of `TokenEncoding`.
        AssertionError: If `init_coords` is a tensor and its dimensions do not match the expected shape.
    """

    requires_previous_transforms = [AddRFTemplates, AddWithinPolyResIdxAnnotation]

    def __init__(
        self,
        n_template: int,
        mask_token_idx: int = 21,  # NOTE: This is the mask token `MSK` index in the original RF2AA code
        init_coords: torch.Tensor | float = chemdata.INIT_CRDS,
        encoding: TokenEncoding = RF2AA_ATOM36_ENCODING,
    ):
        """
        Initializes the FeaturizeRFTemplatesForRF2AA transform.

        Args:
            - n_template (int): The number of templates to use. Must be a positive integer.
            - mask_token_idx (int, optional): The index of the mask token. Defaults to 21.
            - init_coords (torch.Tensor or float, optional): The initial coordinates for the templates.
                If a tensor, its dimensions must match the expected shape. Defaults to `chemdata.INIT_CRDS`.
            - encoding (TokenEncoding, optional): The encoding to use for the templates.
                Must be an instance of `TokenEncoding`. Defaults to `RF2AA_ATOM36_ENCODING`.

        Raises:
            AssertionError: If `n_template` is not a positive integer.
            AssertionError: If `encoding` is not an instance of `TokenEncoding`.
            AssertionError: If `init_coords` is a tensor and its dimensions do not match the expected shape.
        """
        assert_that(n_template).is_instance_of(int).is_greater_than(0)
        assert_that(encoding).is_instance_of(TokenEncoding)
        self.n_template = n_template
        self.mask_token_idx = mask_token_idx
        self.init_coords = init_coords
        self.encoding = encoding

        if isinstance(init_coords, torch.Tensor):
            n_dim = init_coords.shape[-1]
            assert_that(n_dim).is_equal_to(3)

            if init_coords.ndim >= 2:
                n_token = init_coords.shape[-2]
                assert_that(n_token).is_equal_to(encoding.n_atoms_per_token)

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["template", "atom_array"])
        check_is_instance(data, "template", dict)
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "within_poly_res_idx"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        # Extract data
        atom_array = data["atom_array"]
        templates_by_chain = data["template"]

        # Initialize empty template features (= all padded) to fill later
        xyz, t1d, mask, _ = blank_rf2aa_template_features(
            n_template=self.n_template,
            n_token=get_token_count(atom_array),
            encoding=self.encoding,
            mask_token_idx=self.mask_token_idx,
            init_coords=self.init_coords,
        )

        # Get full atom array token starts (useful for going from atom-level > token-level annotations)
        _a_token_starts = get_token_starts(atom_array)  # [n_token] (int)

        # Fill the template features chain by chain and template by template ...
        for chain in chain_instance_iter(atom_array):
            # Check for allowable chain types
            if chain.chain_type[0] != ChainType.POLYPEPTIDE_L.value:
                # Only fill templates for proteins
                continue

            # Check for chains where templates exist
            chain_id = chain.chain_id[0]
            if chain_id not in templates_by_chain:
                # Early exit if there are no templates for this chain
                continue

            # Get chain token starts (useful for going from atom-level > token-level annotations)
            _c_token_starts = get_token_starts(chain)  # [n_token_in_chain] (int)
            # ... atomized tokens cannot be matched to templates
            if "atomize" in chain.get_annotation_categories():
                is_token_atomized = chain.atomize[_c_token_starts]  # [n_token_in_chain] (bool)
            else:
                is_token_atomized = np.zeros_like(_c_token_starts, dtype=bool)
            matchable_query_chain_tokens = _c_token_starts[~is_token_atomized]  # [n_matchable_token_in_chain] (int)

            # Featurize the templates and insert into the template features
            for tmpl_idx, tmpl_data in enumerate(templates_by_chain[chain_id]):
                template = tmpl_data["atom_array"]

                # Filter the template to only include tokens that are aligned to the query chain and that are not atomized
                # ... we use -1 as a placeholder query_res_idx for template tokens without alignment
                has_aligned_res_annotation = template.aligned_query_res_idx >= 0
                # ... find all template tokens that are aligned to the query chain
                has_match_in_query_chain = np.isin(
                    template.aligned_query_res_idx, chain.within_poly_res_idx[matchable_query_chain_tokens]
                )
                # ... check there is at least one template token that is aligned to the query chain
                if not np.any(has_match_in_query_chain & has_aligned_res_annotation):
                    # skip templates that do not have any aligned residues in the query
                    # (e.g. because query chain was cropped and crop does not overlap with template)
                    continue
                # ... subset the template to only the relevant tokens
                template = template[has_match_in_query_chain & has_aligned_res_annotation]

                # Get template token starts (useful for going from atom-level > token-level annotations)
                _t_token_starts = get_token_starts(template)

                # Annotate the global `token_id` for the template tokens which will be used to match
                #  the template tokens to the query chain to fill the template features
                template_token_id = select_data_by_id(
                    select_ids=template.aligned_query_res_idx[_t_token_starts],
                    data_ids=chain.within_poly_res_idx[matchable_query_chain_tokens],
                    data=chain.token_id[matchable_query_chain_tokens],
                    axis=0,
                )  # [n_token_in_template] (int)

                # Encode template
                template_encoded = atom_array_to_encoding(
                    template, self.encoding
                )  # [n_token_in_template, ...] (float/bool/int)

                # Match based on global token ids
                _is_matched_token = np.isin(atom_array.token_id[_a_token_starts], template_token_id)  # [n_token] (bool)
                token_ids_to_fill = atom_array.token_id[_a_token_starts][
                    _is_matched_token
                ]  # [n_matchable_token_in_template] (int)
                token_idxs_to_fill = np.where(_is_matched_token)[0]  # [n_matchable_token_in_template] (int)

                # Fill coordinates
                _tmpl_xyz = select_data_by_id(
                    select_ids=token_ids_to_fill,
                    data_ids=template_token_id,
                    data=template_encoded["xyz"],
                    axis=0,
                )
                xyz[tmpl_idx, token_idxs_to_fill] = torch.tensor(_tmpl_xyz)

                # Fill mask
                _tmpl_mask = select_data_by_id(
                    select_ids=token_ids_to_fill,
                    data_ids=template_token_id,
                    data=template_encoded["mask"],
                    axis=0,
                )
                mask[tmpl_idx, token_idxs_to_fill] = torch.tensor(_tmpl_mask)

                # Fill 1D template features
                _tmpl_seq = select_data_by_id(
                    select_ids=token_ids_to_fill,
                    data_ids=template_token_id,
                    data=template_encoded["seq"],
                    axis=0,
                )
                _tmpl_confidence = select_data_by_id(
                    select_ids=token_ids_to_fill,
                    data_ids=template_token_id,
                    data=template.alignment_confidence[_t_token_starts],
                    axis=0,
                )
                # ... set one-hot encoded sequence for tokens where template features can be filled
                t1d[tmpl_idx, token_idxs_to_fill, :-1] = torch.nn.functional.one_hot(
                    torch.tensor(_tmpl_seq), self.encoding.n_tokens - 1
                ).float()
                # ... set confidence for tokens where template features can be filled
                #     for this we extract the residue-wise alignment confidence
                t1d[tmpl_idx, token_idxs_to_fill, -1] = torch.tensor(_tmpl_confidence)

        # Save the template features
        data["template_feat"] = {
            "xyz": xyz,  # [n_template, n_res, n_atoms_per_token, 3] (float)
            "mask": mask,  # [n_template, n_res, n_atoms_per_token] (bool)
            "t1d": t1d,  # [n_tepmlate, n_res, n_tokens],  [0:n_tokens-1] = one-hot encoded sequence, [-1] = confidence
        }
        return data
