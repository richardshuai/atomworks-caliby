import logging

import pytest

from datahub.datasets.dataframe_parsers import PNUnitsDFParser
from datahub.datasets.pdb_dataset import PDBDataset
from datahub.tests.conftest import PN_UNITS_DATASET, PN_UNITS_DF

# NOTE: See the conftest for the filters applied to PN_UNITS_DATASET, which are validated below


def test_filter_impact():
    # Check that the filter had an impact (rows were dropped)
    original_data_length = len(PN_UNITS_DF)
    filtered_data_length = len(PN_UNITS_DATASET.data)
    assert filtered_data_length < original_data_length, "Filter did not reduce the number of rows"


def test_deposition_date_filter():
    # Check that the deposition date filter was applied correctly
    filtered_data = PN_UNITS_DATASET.data
    assert (filtered_data["deposition_date"] < "2022-01-01").all(), "Deposition date filter did not work correctly"


def test_resolution_filter():
    # Check that the resolution filter was applied correctly
    filtered_data = PN_UNITS_DATASET.data
    assert (filtered_data["resolution"] <= 5.0).all(), "Resolution filter did not work correctly"


def test_method_filter():
    # Check that the method filter was applied correctly
    filtered_data = PN_UNITS_DATASET.data
    assert (
        filtered_data["method"].isin(["X-RAY_DIFFRACTION", "ELECTRON_MICROSCOPY"]).all()
    ), "Method filter did not work correctly"


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
