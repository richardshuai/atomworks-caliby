import pytest
from cifutils.cifutils_biotite import cifutils_biotite, cifutils_biotite_utils
from tests.conftest import get_digs_path
from biotite.structure.io import pdbx
import numpy as np

MULTIPLE_ASSEMBLY_TEST_CASES = [
    {"pdbid": "1A7J", "n_assemblies": 3},
    {"pdbid": "1KTD", "n_assemblies": 3},
]

ASSEMBLY_ATOM_COORDINATES_TEST_CASES = ["1A8O", "1RXZ", "4NDZ", "5XNL", "6DMG", "2E2H"]

cif_parser = cifutils_biotite.CIFParser()


@pytest.mark.parametrize("test_case", MULTIPLE_ASSEMBLY_TEST_CASES)
def test_assembly_counts(test_case: dict):
    # unpack test case
    pdbid = test_case["pdbid"]
    n_assemblies = test_case["n_assemblies"]

    # parse the file
    filename = get_digs_path(pdbid)

    # test the different build_assembly options
    out_no_assembly = cif_parser.parse(filename, build_assembly=None)
    assert len(out_no_assembly["assemblies"]) == 0

    out_first = cif_parser.parse(filename, build_assembly="first")
    assert len(out_first["assemblies"]) == 1

    out_all = cif_parser.parse(filename, build_assembly="all")
    assert len(out_all["assemblies"]) == n_assemblies


@pytest.mark.parametrize("pdb_id", ASSEMBLY_ATOM_COORDINATES_TEST_CASES)
def test_assembly_atom_coordinates(pdb_id: str):
    path = get_digs_path(pdb_id)

    # Biotite
    file = cifutils_biotite_utils.read_cif_file(path)
    biotite_assembly = pdbx.get_assembly(
        file,
        assembly_id="1",
        use_author_fields=False,
        altloc="occupancy",
        extra_fields=[
            "atom_id",
            "occupancy",
        ],
        model=1,
    )
    biotite_assembly = biotite_assembly[biotite_assembly.occupancy > 0]

    # Cifutils
    cifutils_assembly = cif_parser.parse(path, build_assembly="first")
    atom_array = cifutils_assembly["assemblies"]["1"]
    resolved_atoms = atom_array[atom_array.occupancy > 0]

    # Check that the number of atoms match
    assert len(biotite_assembly) == len(resolved_atoms), f"Number of atoms within assembly do not match for {pdb_id}"

    # Check that the atom locations match
    assert np.allclose(biotite_assembly.coord, resolved_atoms.coord), f"Atom coordinates do not match for {pdb_id}"

    # Check that the atom names match
    assert np.all(biotite_assembly.atom_name == resolved_atoms.atom_name), f"Atom names do not match for {pdb_id}"
