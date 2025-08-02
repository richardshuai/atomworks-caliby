"""
Test for the BindNoBindMeasurement dataclass.

tries creating a BindNoBindMeasurement with all fields, missing optional fields, invalid fields, and missing required fields.

"""

import pytest

from datahub.databases.dataclasses import BindNoBindMeasurement


def test_bind_no_bind_measurement_dataclass(
    atom_array,
    chemical_components,
    bind_no_bind_measurement_fields_no_structure,
    bind_no_bind_measurement_data_source_fields,
):
    # Should be able to create BindNoBindMeasurement with atom array or chemical components
    bnb_fields = bind_no_bind_measurement_fields_no_structure | {"atom_array": atom_array}
    _ = BindNoBindMeasurement(**bnb_fields)

    bnb_fields["atom_array"] = chemical_components
    _ = BindNoBindMeasurement(**bnb_fields)

    # missing an optional field should still work
    bnb_fields.pop(list(BindNoBindMeasurement.get_optional_fields().keys())[0])
    _ = BindNoBindMeasurement(**bnb_fields)

    # invalid field should raise an error
    bnb_fields["invalid_field"] = 1
    with pytest.raises(TypeError):
        BindNoBindMeasurement(**bnb_fields)
    bnb_fields.pop("invalid_field")

    # missing a required field should raise an error
    bnb_fields.pop(list(BindNoBindMeasurement.get_required_fields().keys())[0])
    with pytest.raises(TypeError):
        BindNoBindMeasurement(**bnb_fields)

    # should be able to create one with data source fields
    bnb_data_source_fields = bind_no_bind_measurement_data_source_fields | {"atom_array": atom_array}
    _ = BindNoBindMeasurement(**bnb_data_source_fields)
