"""Tests for AseDBDataset using real LMDB data."""

from pathlib import Path

import pytest
from biotite.structure import AtomArray

from atomworks.ml.datasets.ase_dataset import AseDBDataset

# Get the test data path
TEST_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ml" / "ase_lmdb"
TEST_LMDB_PATH = TEST_DATA_DIR / "test_omol25.aselmdb"


@pytest.fixture
def omol25_lmdb_path():
    """Path to test LMDB database."""
    if not TEST_LMDB_PATH.exists():
        pytest.skip(f"Test LMDB file not found at {TEST_LMDB_PATH}")
    return str(TEST_LMDB_PATH)


def test_load_and_convert_to_atom_array(omol25_lmdb_path):
    """Test loading an entry from ASE LMDB and converting to AtomArray."""
    dataset = AseDBDataset(
        lmdb_path=omol25_lmdb_path,
        name="test_omol25",
    )

    # Load the first entry
    data = dataset[0]

    # Verify structure
    assert "atom_array" in data
    assert isinstance(data["atom_array"], AtomArray)
    assert len(data["atom_array"]) > 0
    assert hasattr(data["atom_array"], "coord")
    assert hasattr(data["atom_array"], "element")
