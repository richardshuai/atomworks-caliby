import pytest

import atomworks.ml.databases.data_source_utils as ds_utils
from atomworks.ml.databases.dataclasses import DataSource
from atomworks.ml.databases.enums import DataSourceType, ExperimentType, ProblemType


def make_example_datasource():
    return DataSource(
        author="testauthor",
        year=2024,
        data_source_tag="tag1",
        problem=ProblemType.PPI,
        data_source_type=DataSourceType.EXPERIMENT_IN_HOUSE,
        experiment_type=ExperimentType.SPR,
        experiment_metadata={"foo": "bar"},
        data_source_description="A test data source",
    )


def test_datasource_instantiation():
    ds = make_example_datasource()
    assert ds.data_source_id == "testauthor_2024_tag1"
    assert ds.author == "testauthor"
    assert ds.problem == ProblemType.PPI


def test_upload_and_get_data_source(temp_data_source_db):
    ds = make_example_datasource()
    # Should create the DB file if it doesn't exist
    ds_utils.upload_data_source(ds)
    # Now retrieve it
    loaded = ds_utils.get_data_source(ds.data_source_id)
    assert loaded is not None
    assert loaded.data_source_id == ds.data_source_id
    assert loaded.author == ds.author
    assert loaded.problem == ds.problem


def test_upload_duplicate_raises(temp_data_source_db):
    ds = make_example_datasource()
    ds_utils.upload_data_source(ds)
    with pytest.raises(ValueError):
        ds_utils.upload_data_source(ds)


def test_update_data_source(temp_data_source_db):
    ds = make_example_datasource()
    ds_utils.upload_data_source(ds)
    # Change a field and update
    ds.data_source_description = "Updated description"
    ds_utils.update_data_source(ds)
    loaded = ds_utils.get_data_source(ds.data_source_id)
    assert loaded.data_source_description == "Updated description"


def test_update_nonexistent_raises(temp_data_source_db):
    ds = make_example_datasource()
    with pytest.raises(ValueError):
        ds_utils.update_data_source(ds)


def test_get_nonexistent_returns_none(temp_data_source_db):
    assert ds_utils.get_data_source("nonexistent_id") is None
