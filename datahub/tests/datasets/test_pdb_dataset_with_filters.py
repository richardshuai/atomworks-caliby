import logging

import numpy as np
import pytest

from datahub.datasets.dataframe_parsers import PNUnitsDFParser
from datahub.datasets.pdb_dataset import PDBDataset
from tests.conftest import PN_UNITS_DF
from tests.datasets.conftest import RF2AA_INTERFACES_DATASET, RF2AA_PN_UNITS_DATASET

# NOTE: See the conftest for the filters applied to PN_UNITS_DATASET, which are validated below


def test_filter_impact():
    # Check that the filter had an impact (rows were dropped)
    original_data_length = len(PN_UNITS_DF)
    filtered_data_length = len(RF2AA_PN_UNITS_DATASET.data)
    assert filtered_data_length < original_data_length, "Filter did not reduce the number of rows"


def test_deposition_date_filter():
    # Check that the deposition date filter was applied correctly
    filtered_data = RF2AA_PN_UNITS_DATASET.data
    assert (filtered_data["deposition_date"] < "2022-01-01").all(), "Deposition date filter did not work correctly"


def test_resolution_filter():
    # Check that the resolution filter was applied correctly
    filtered_data = RF2AA_PN_UNITS_DATASET.data
    assert (filtered_data["resolution"] <= 5.0).all(), "Resolution filter did not work correctly"


def test_method_filter():
    # Check that the method filter was applied correctly
    filtered_data = RF2AA_PN_UNITS_DATASET.data
    assert (
        filtered_data["method"].isin(["X-RAY_DIFFRACTION", "ELECTRON_MICROSCOPY"]).all()
    ), "Method filter did not work correctly"


def test_af3_excluded_ligands_filter():
    # Check that we don't have any Query PN Units that are AF-3 excluded ligands
    filtered_pn_units_data = RF2AA_PN_UNITS_DATASET.data
    filtered_interfaces_data = RF2AA_INTERFACES_DATASET.data

    # ...check query PN Units
    assert np.any(
        filtered_pn_units_data.example_id == "{['pdb', 'pn_units']}{2pno}{3}{['G_1']}"
    ), "Entry removed that contained valid PN Units"
    assert not np.any(
        filtered_pn_units_data.example_id == "{['pdb', 'pn_units']}{2pno}{3}{['DB_1']}"
    ), "Entry remained that contains AF-3 excluded ligands as query PN Units"

    # ...check interfaces
    assert not np.any(
        filtered_interfaces_data.pdb_id == "{['pdb', 'interfaces']}{2pno}{3}{['DB_1', 'G_1']}"
    ), "Entry remained that contains AF-3 excluded ligands as query PN Units"
    assert np.any(
        filtered_interfaces_data.example_id == "{['pdb', 'interfaces']}{2pno}{3}{['G_1', 'H_1']}"
    ), "Entry removed that contained valid PN Units"


def test_filter_no_impact(caplog):
    # Test for filters that do not remove any rows
    filters = ["resolution != -1"]
    with caplog.at_level(logging.WARNING):
        PDBDataset(
            name="test",
            dataset_path=PN_UNITS_DF.copy(),
            dataset_parser=PNUnitsDFParser(),
            filters=filters,
            transform=None,
        )
    assert "did not remove any rows" in caplog.text, "Warning for no impact filter not raised"


def test_filter_remove_all_rows():
    # Test for filters that remove all rows
    filters = ["resolution < 0.0"]
    with pytest.raises(ValueError, match="removed all rows"):
        PDBDataset(
            name="test",
            dataset_path=PN_UNITS_DF.copy(),
            dataset_parser=PNUnitsDFParser(),
            filters=filters,
            transform=None,
        )


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=WARNING", __file__])
