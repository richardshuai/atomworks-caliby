import pytest
import numpy as np
from biotite.database import rcsb
from biotite.structure import AtomArray, AtomArrayStack
from cifutils.utils.io_utils import load_any, get_structure, read_any
import tempfile
from contextlib import nullcontext


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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
