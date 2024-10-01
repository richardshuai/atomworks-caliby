"""
Includes tests to assert that the data loading pipeline outputs examples that satisfy the assumptions of the AF3 model.
"""


import pytest
import numpy as np
import torch
import random

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from tests.datasets.conftest import AF3_PDB_DATASET
from datahub.encoding_definitions import RF2_ATOM23_ENCODING, AF3SequenceEncoding

def test_satisfies_af3_dataloading_assumptions(pdb_dataset=AF3_PDB_DATASET):
    """
    Tests that the data loading pipeline outputs examples that satisfy the assumptions of the AF3 model.
    """
    NUM_RANDOM_EXAMPLES = 10

    # Set the seed for reproducibility
    seed = 42
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Select deterministic examples to profile
    # NOTE: TEST_FILTERS ensures we don't end up with any huge examples that would slow down the test
    deterministic_indices = np.random.choice(len(pdb_dataset), NUM_RANDOM_EXAMPLES, replace=False)

    # Create a Subset of the dataset with the selected indices
    subset = Subset(pdb_dataset, deterministic_indices)

    # Create a DataLoader for the subset
    data_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    for sample in tqdm(data_loader):
        example_id = sample[0]["example_id"]
        try:
            assert_satisfies_af3_assumptions(sample[0])
        except AssertionError as e:
            # Update message with sample index
            rng_state_dict = pdb_dataset.get_dataset_by_idx(
                pdb_dataset.id_to_idx(example_id)
            ).transform.latest_rng_state_dict
            raise AssertionError(f"Assertion failed for sample {example_id}." + "\n" + f"{rng_state_dict}") from e


def assert_satisfies_af3_assumptions(sample):
    """
    Asserts that the features satisfy the assumptions of the AF3 model.
    """ 
    n_tokens, n_atoms, n_sequences, n_templates = assert_input_feature_dimensions(sample["feats"])
    return True


def assert_input_feature_dimensions(feats):
    """
    Asserts that the input features have the correct dimensions for the AF3 model.
    """
    # Check the dimensions of the input features
    #assert "f" in feats
    # find I, L and N

    f = feats
    n_token = f["restype"].shape[0]
    n_atoms = f["token_to_atom_map"].shape[0]

    n_templates = f["template_restype"].shape[0]
    n_sequences = f["msa"].shape[0]
    assert f["residue_index"].shape == (n_token,)
    assert f["token_index"].shape == (n_token,)
    assert f["asym_id"].shape == (n_token,)
    assert f["entity_id"].shape == (n_token,)
    assert f["sym_id"].shape == (n_token,)
    assert f["is_protein"].shape == (n_token,)
    assert f["is_ligand"].shape == (n_token,)
    assert f["is_dna"].shape == (n_token,)
    assert f["is_rna"].shape == (n_token,)
    assert f["ref_pos"].shape == (n_atoms, 3)
    assert f["ref_mask"].shape == (n_atoms,)
    assert f["ref_element"].shape == (n_atoms,)
    assert f["ref_charge"].shape == (n_atoms,)
    assert f["ref_atom_name_chars"].shape == (n_atoms, 4)
    assert f["ref_space_uid"].shape == (n_atoms,)
    #TODO: why are these in the input encoding???
    assert f["ref_automorphs"].shape[1:] == (n_atoms,2)
    assert f["ref_automorphs_mask"].shape[1:] == (n_atoms,)

    # templates
    assert f["template_restype"].shape == (n_templates, n_token,)
    assert f["template_pseudo_beta_mask"].shape == (n_templates, n_token,)
    assert f["template_backbone_frame_mask"].shape == (n_templates, n_token,)
    assert f["template_distogram"].shape == (n_templates, n_token, n_token, 39)
    assert f["template_unit_vector"].shape == (n_templates, n_token, n_token, 3)

    # bond feats
    assert f["token_bonds"].shape == (n_token, n_token)

    # msa
    assert f["msa"].shape == (n_sequences, n_token, 32) 
    assert f["has_deletion"].shape == (n_sequences, n_token)
    assert f["deletion_value"].shape == (n_sequences, n_token)
    assert f["profile"].shape == (n_token, 32)
    assert f["deletion_mean"].shape == (n_token,)
    return n_token, n_atoms, n_templates, n_sequences

