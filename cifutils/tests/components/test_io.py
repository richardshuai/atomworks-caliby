import io
import tempfile
from contextlib import nullcontext

import biotite.structure as struc
import numpy as np
import pytest
from biotite.database import rcsb
from biotite.structure import AtomArray, AtomArrayStack

from cifutils import parse
from cifutils.tools.inference import components_to_atom_array
from cifutils.utils.io_utils import get_structure, load_any, read_any, to_cif_file, to_cif_string, to_pdb_string


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
        assert isinstance(loaded_structure, AtomArray | AtomArrayStack)
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

    assert isinstance(structure, AtomArray | AtomArrayStack)
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

    with pytest.raises(ValueError, match="Ambiguous bond annotations detected"):
        cif_string = to_cif_string(cif_structure)

    # Make identifiers unique
    cif_structure.res_id = struc.spread_residue_wise(cif_structure, np.arange(struc.get_residue_count(cif_structure)))
    # ... drop HOH
    cif_structure = cif_structure[0, cif_structure.res_name != "HOH"]
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

    metadata_serialized = (
        "#\n"
        "_test_category.test_col1   data\n"
        "_test_category.test_col2   data2\n"
        "#\n"
        "loop_\n"
        "_test_category2.test_col1 \n"
        "_test_category2.test_col2 \n"
        "0 0\n"
        "1 1\n"
        "2 2\n"
        "3 3\n"
        "4 4\n"
        "5 5\n"
        "6 6\n"
        "7 7\n"
        "8 8\n"
        "9 9\n"
        "#\n"
        "loop_\n"
        "_test_category3.test_col1 \n"
        "_test_category3.test_col2 \n"
        "1 2\n"
        "3 5\n"
        "4 a\n"
        "#\n"
    )
    assert metadata_serialized in cif_string2, "Metadata not found in serialized CIF string."


def test_to_pdb_string():
    pdb_buffer = rcsb.fetch("6lyz", "pdb")
    pdb_structure = load_any(pdb_buffer, file_type="pdb", extra_fields=["b_factor", "occupancy", "charge"], model=1)
    n_atoms = pdb_structure.array_length()
    pdb_string = to_pdb_string(pdb_structure)
    assert isinstance(pdb_string, str)
    assert len(pdb_string) > 0

    # Test that we can load the pdb string back into an AtomArray
    pdb_buffer2 = io.StringIO(pdb_string)
    pdb_structure2 = load_any(pdb_buffer2, file_type="pdb", extra_fields=["b_factor", "occupancy", "charge"], model=1)
    assert pdb_structure2.array_length() == n_atoms
    assert np.allclose(pdb_structure.coord, pdb_structure2.coord)
    assert np.all(pdb_structure.atom_name == pdb_structure2.atom_name)
    assert np.all(pdb_structure.element == pdb_structure2.element)
    assert np.all(pdb_structure.charge == pdb_structure2.charge)
    assert np.all(pdb_structure.chain_id == pdb_structure2.chain_id)
    assert np.all(pdb_structure.res_name == pdb_structure2.res_name)
    assert np.all(pdb_structure.res_id == pdb_structure2.res_id)
    assert np.all(pdb_structure.b_factor == pdb_structure2.b_factor)
    assert np.all(pdb_structure.occupancy == pdb_structure2.occupancy)


def test_parse_with_no_resolved_atoms(tmpdir):
    # Spoof the input data using the inference pipeline
    smiles = "C[C@]12CC[C@@H](C[C@H]1CC[C@@H]3[C@@H]2C[C@H]([C@]4([C@@]3(CC[C@@H]4C5=CC(=O)OC5)O)C)O)O"
    inputs = [
        {
            "smiles": smiles,
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "A",
        }
    ]
    atom_array = components_to_atom_array(inputs)

    # Use the tmpdir fixture to create a temporary file path
    cif_path = tmpdir.join("test.cif")
    to_cif_file(atom_array, cif_path)

    # ... parse the atom array
    out = parse(
        cif_path,
        assume_residues_all_resolved=True,
    )

    # Smoke test
    assert out is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
