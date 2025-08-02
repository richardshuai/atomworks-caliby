import pickle
from pathlib import Path

import numpy as np
import pytest
from assertpy import assert_that
from biotite.structure import AtomArrayStack
from cifutils.constants import ATOMIC_NUMBER_TO_ELEMENT

from datahub.utils.io import convert_af3_model_output_to_atom_array_stack

# NOTE: Not the "true" model outputs; slightly pre-processed for storage efficiency
TEST_PICKLED_AF3_MODEL_OUTPUTS = ["af3_model_outs_protein_dna.pkl", "af3_model_outs_protein_ligand.pkl"]

TEST_DATA_PATH = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.slow
@pytest.mark.parametrize("file_path", TEST_PICKLED_AF3_MODEL_OUTPUTS)
def test_convert_af3_model_output_to_atom_array_stack(file_path: str):
    full_path = TEST_DATA_PATH / file_path

    # Load the model outputs
    with open(full_path, "rb") as f:
        model_outputs = pickle.load(f)

    # Convert the model outputs to an AtomArrayStack
    atom_array_stack = convert_af3_model_output_to_atom_array_stack(
        atom_to_token_map=model_outputs["atom_to_token_map"],
        pn_unit_iids=model_outputs["chain_iids"],
        decoded_restypes=model_outputs["decoded_restypes"],
        xyz=model_outputs["xyz"],
        elements=model_outputs["elements"],
        token_is_atomized=model_outputs["token_is_atomized"],
    )

    # Smoke tests
    assert_that(atom_array_stack).is_instance_of(AtomArrayStack)
    assert_that(atom_array_stack[0]).is_length(len(model_outputs["xyz"]))

    # Assert that the AtomArray has the correct elements
    uppercase_elements = np.array(
        [ATOMIC_NUMBER_TO_ELEMENT[atomic_number] for atomic_number in model_outputs["elements"]]
    )
    assert_that(np.array_equal(atom_array_stack.element, uppercase_elements)).is_true()

    # Assert that the AtomArray has the correct coordinates for the first (and only) model
    assert_that(np.array_equal(atom_array_stack.coord[0], model_outputs["xyz"])).is_true()


if __name__ == "__main__":
    pytest.main(["-s", "-v", "-m not very_slow", __file__])
