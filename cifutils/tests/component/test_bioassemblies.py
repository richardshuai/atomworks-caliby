import pytest
from cifutils.cifutils_biotite import cifutils_biotite_utils
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE, assert_same_atom_array
from biotite.structure.io import pdbx

MULTIPLE_ASSEMBLY_TEST_CASES = [
    {"pdbid": "1a7j", "n_assemblies": 3},
    {"pdbid": "5vos", "n_assemblies": 1},
]

ASSEMBLY_ATOM_COORDINATES_TEST_CASES = ["1A8O", "1RXZ", "4NDZ", "5XNL", "6DMG", "2E2H"]


@pytest.mark.parametrize("test_case", MULTIPLE_ASSEMBLY_TEST_CASES)
def test_assembly_counts(test_case: dict):
    # unpack test case
    pdbid = test_case["pdbid"]
    n_assemblies = test_case["n_assemblies"]

    # parse the file
    filename = get_digs_path(pdbid)

    # test the different build_assembly options
    out_no_assembly = CIF_PARSER_BIOTITE.parse(filename, build_assembly=None)
    assert len(out_no_assembly["assemblies"]) == 0

    out_first = CIF_PARSER_BIOTITE.parse(filename, build_assembly="first")
    assert len(out_first["assemblies"]) == 1

    out_all = CIF_PARSER_BIOTITE.parse(filename, build_assembly="all")
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
    cifutils_assembly = CIF_PARSER_BIOTITE.parse(
        path,
        build_assembly="first",
        fix_arginines=False,
        remove_crystallization_aids=False,
        remove_waters=False,
    )
    atom_array = cifutils_assembly["assemblies"]["1"][0]
    resolved_atoms = atom_array[atom_array.occupancy > 0]

    assert_same_atom_array(
        biotite_assembly,
        resolved_atoms,
        annotations_to_compare=["chain_id", "res_name", "atom_name"],
        # NOTE: We do not compare res_id as waters don't match in the res_id and elements as we turn elements into integers
    )