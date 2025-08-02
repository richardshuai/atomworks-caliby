"""
Uses pytest to test the CIF IO functions.
Has a fixture to create a valid BindNoBindMeasurement with all fields or missing optional fields.
Tests that you can save and load and everything is preserved as expected. This is general (does not hardcode
all of the field names so you don't have to update the test when you update the fields)

Edge cases to test:
- numpy arrays in fields
- dictionary with various types of values. Do they get saved and reloaded with the correct type?
"""

import tempfile

import numpy as np
import pytest

import atomworks.ml.databases.data_source_utils as ds_utils
from atomworks.ml.databases.dataclasses import BindNoBindMeasurement, DataSource
from atomworks.ml.databases.io_utils import load_measurement_from_cif, save_measurement_to_cif


def _compare_fields(
    orig: BindNoBindMeasurement, loaded: BindNoBindMeasurement, compare_data_source_fields: bool = False
):
    """Compare all fields except atom_array"""
    fields_to_compare = list(BindNoBindMeasurement.get_field_info().keys())
    if not compare_data_source_fields:
        data_source_fields = list(DataSource.get_field_info().keys())
        data_source_fields.remove("data_source_id")  # already in bind no bind measurement
        fields_to_compare = [field for field in fields_to_compare if field not in data_source_fields]

    for field in fields_to_compare:
        if field == "atom_array":
            continue
        orig_val = getattr(orig, field)
        loaded_val = getattr(loaded, field)
        if isinstance(orig_val, np.ndarray):
            np.testing.assert_array_equal(orig_val, loaded_val)
        elif isinstance(orig_val, dict):
            # Compare dicts, allowing for np.array to list conversion
            for k, v in orig_val.items():
                if isinstance(v, np.ndarray):
                    np.testing.assert_array_equal(v, loaded_val[k])
                else:
                    assert loaded_val[k] == v
        else:
            assert loaded_val == orig_val, f"Field {field} mismatch: {loaded_val} != {orig_val}"


def test_save_and_load_roundtrip(
    bind_no_bind_measurement_fields_no_structure,
    atom_array,
    chemical_components,
    temp_data_source_db,
    data_source_fields,
):
    """Test that saving and loading a measurement preserves all fields and types."""

    measurement_fields = bind_no_bind_measurement_fields_no_structure | {"atom_array": atom_array}
    bind_no_bind_measurement = BindNoBindMeasurement(**measurement_fields)
    bind_no_bind_measurement_with_data_source = BindNoBindMeasurement(**measurement_fields | data_source_fields)

    # shouldn't work with no data source
    with tempfile.TemporaryDirectory() as tmpdir:
        cif_file = f"{tmpdir}/test_measurement.cif"
        with pytest.raises(ValueError):
            save_measurement_to_cif(bind_no_bind_measurement, cif_file)

    # now upload data source
    ds_utils.upload_data_source(DataSource(**data_source_fields))

    with tempfile.TemporaryDirectory() as tmpdir:
        cif_file = f"{tmpdir}/test_measurement.cif"
        save_measurement_to_cif(bind_no_bind_measurement, cif_file)
        loaded = load_measurement_from_cif(cif_file)

    _compare_fields(bind_no_bind_measurement, loaded)
    _compare_fields(bind_no_bind_measurement_with_data_source, loaded, compare_data_source_fields=True)

    # should work without an optional field
    measurement_fields.pop(list(BindNoBindMeasurement.get_optional_fields().keys())[0])
    bind_no_bind_measurement = BindNoBindMeasurement(**measurement_fields)
    with tempfile.TemporaryDirectory() as tmpdir:
        cif_file = f"{tmpdir}/test_measurement.cif"
        save_measurement_to_cif(bind_no_bind_measurement, cif_file)
        loaded = load_measurement_from_cif(cif_file)
    _compare_fields(bind_no_bind_measurement, loaded)
