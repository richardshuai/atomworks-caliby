import numpy as np
import pytest

from datahub.datasets.dataframe_parsers import PNUnitsDFParser, load_from_row
from datahub.preprocessing.constants import SUPPORTED_CHAIN_TYPES
from datahub.tests.conftest import CIF_PARSER, PN_UNITS_DF
from datahub.transforms.atom_array import (
    RemoveUnsupportedChainTypes,
)
from datahub.transforms.base import Compose, TransformPipelineError

UNSUPPORTED_CHAIN_TYPE_TEST_CASES = [
    "104D",  # DNA/RNA Hybrid
    "5X3O",  # polypeptide(D)
]


@pytest.mark.parametrize("pdb_id", UNSUPPORTED_CHAIN_TYPE_TEST_CASES)
def test_remove_unsupported_chain_types(pdb_id: str):
    rows = PN_UNITS_DF[
        (PN_UNITS_DF["pdb_id"] == pdb_id.lower()) & (PN_UNITS_DF["assembly_id"] == "1")
    ]  # We only need the first assembly for UNSUPPORTED_CHAIN_TYPE_TEST_CASES

    assert not rows.empty

    for _, row in rows.iterrows():
        data = load_from_row(row, PNUnitsDFParser(), cif_parser=CIF_PARSER)
        is_unsupported_type = row["q_pn_unit_type"] not in SUPPORTED_CHAIN_TYPES
        original_atom_array = data["atom_array"].copy()

        # Apply transforms
        # fmt: off
        pipeline = Compose([
            RemoveUnsupportedChainTypes(),
        ], track_rng_state=False)
        # fmt: on

        output = None
        if is_unsupported_type:
            with pytest.raises(TransformPipelineError):
                output = pipeline(data)
        else:
            output = pipeline(data)

        if output:
            atom_array = output["atom_array"]
            num_unsupported_atoms = len(original_atom_array) - len(atom_array)
            assert num_unsupported_atoms > 0, "There should be some atoms removed"
            chain_types = np.unique(atom_array.chain_type)
            assert np.all(np.isin(chain_types, SUPPORTED_CHAIN_TYPES)), "All remaining chain types should be supported"
