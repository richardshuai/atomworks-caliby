import pytest
import numpy as np
from biotite.database import rcsb
from biotite.structure import AtomArray, AtomArrayStack
from cifutils.utils.io_utils import load_any, get_structure, read_any, to_cif_string
import tempfile
from contextlib import nullcontext
import io


@pytest.mark.parametrize(
    "pdb_id, file_type, directory",
    [
        ("6lyz", "pdb", True),
        ("6lyz", "pdb", False),
        ("6lyz", "cif", True),
        ("6lyz", "cif", False),
        ("6lyz", "bcif", True),
        ("6lyz", "bcif", False),
    ],
)
def test_load_any(pdb_id, file_type, directory):
    with tempfile.TemporaryDirectory() if directory else nullcontext() as tmp_dir:
        # Test loading from a buffer or file
        loaded_structure = load_any(rcsb.fetch(pdb_id, file_type, tmp_dir), file_type=file_type)
        assert isinstance(loaded_structure, (AtomArray, AtomArrayStack))
        assert loaded_structure.array_length() > 0


@pytest.mark.parametrize(
    "extra_fields, assume_residues_all_resolved, include_bonds, model",
    [
        ([], False, True, None),
        (["b_factor", "occupancy"], False, True, None),
        ([], True, True, None),
        ([], False, False, None),
        ([], False, True, 1),
    ],
)
def test_get_structure_configurations(extra_fields, assume_residues_all_resolved, include_bonds, model):
    # Fetch 6lyz.cif as a buffer
    cif_buffer = rcsb.fetch("6lyz", "cif")

    # Read the buffer into a CIFFile object
    cif_file = read_any(cif_buffer, file_type="cif")

    # Get the structure with different configurations
    structure = get_structure(
        cif_file,
        extra_fields=extra_fields,
        assume_residues_all_resolved=assume_residues_all_resolved,
        include_bonds=include_bonds,
        model=model,
    )

    assert isinstance(structure, (AtomArray, AtomArrayStack))
    assert structure.array_length() > 0

    # Check if extra fields are present
    for field in extra_fields:
        assert field in structure.get_annotation_categories()

    # Check if bonds are included
    if include_bonds:
        assert structure.bonds is not None
    else:
        assert structure.bonds is None

    # Check if all occupancies are 1.0 when assume_residues_all_resolved is True
    if assume_residues_all_resolved:
        assert np.all(structure.occupancy == 1.0)

    # Check if the correct model is returned when specified
    if model is not None:
        assert isinstance(structure, AtomArray)
    else:
        assert isinstance(structure, AtomArrayStack)


def test_to_cif_string():
    cif_buffer = rcsb.fetch("6lyz", "cif")
    cif_structure = load_any(cif_buffer, file_type="cif", extra_fields=["b_factor", "occupancy", "charge"])
    cif_string = to_cif_string(cif_structure)
    assert isinstance(cif_string, str)
    assert len(cif_string) > 0

    cif_structure2 = load_any(
        io.StringIO(cif_string),
        file_type="cif",
        assume_residues_all_resolved=False,
        extra_fields=["b_factor", "occupancy", "charge"],
    )
    assert np.allclose(cif_structure.coord, cif_structure2.coord)
    assert np.all(cif_structure.atom_name == cif_structure2.atom_name)
    assert np.all(cif_structure.element == cif_structure2.element)
    assert np.all(cif_structure.charge == cif_structure2.charge)
    assert np.all(cif_structure.chain_id == cif_structure2.chain_id)
    assert np.all(cif_structure.res_name == cif_structure2.res_name)
    assert np.all(cif_structure.res_id == cif_structure2.res_id)
    assert np.all(cif_structure.b_factor == cif_structure2.b_factor)
    assert np.all(cif_structure.occupancy == cif_structure2.occupancy)

    # Test if we can write custom metadata
    metadata = {
        "test_category": {"test_col1": "data", "test_col2": "data2"},
        "test_category2": {"test_col1": np.arange(10), "test_col2": np.arange(10)},
        "test_category3": {"test_col1": [1, 3, 4], "test_col2": [2, 5, "a"]},
    }
    cif_string2 = to_cif_string(
        cif_structure,
        id="test_id",
        extra_categories=metadata,
    )

    metadata_serealized = (
        "_test_category.test_col1   data\n"
        "_test_category.test_col2   data2\n"
        "#\n"
        "loop_\n"
        "_test_category2.test_col1 \n"
        "_test_category2.test_col2 \n"
        "0 0 \n"
        "1 1 \n"
        "2 2 \n"
        "3 3 \n"
        "4 4 \n"
        "5 5 \n"
        "6 6 \n"
        "7 7 \n"
        "8 8 \n"
        "9 9 \n"
        "#\n"
        "loop_\n"
        "_test_category3.test_col1 \n"
        "_test_category3.test_col2 \n"
        "1 2 \n"
        "3 5 \n"
        "4 a \n"
        "#\n"
    )
    assert metadata_serealized in cif_string2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
