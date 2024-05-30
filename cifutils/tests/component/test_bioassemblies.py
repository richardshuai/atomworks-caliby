import pytest
from cifutils.cifutils_biotite import cifutils_biotite
from tests.conftest import get_digs_path

TEST_CASES = [
    {"pdbid": "1A7J", "n_assemblies": 3},
    {"pdbid": "1KTD", "n_assemblies": 3},
]

cif_parser_none = cifutils_biotite.CIFParser(build_assembly=None)
cif_parser_first = cifutils_biotite.CIFParser(build_assembly="first")
cif_parser_all = cifutils_biotite.CIFParser(build_assembly="all")


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_remove_crystallization_aids(test_case: dict):
    # unpack test case
    pdbid = test_case["pdbid"]
    n_assemblies = test_case["n_assemblies"]

    # parse the file
    filename = get_digs_path(pdbid)

    # test the different build_assembly options
    out_no_assembly = cif_parser_none.parse(filename)
    assert len(out_no_assembly["assemblies"]) == 0

    out_first = cif_parser_first.parse(filename)
    assert len(out_first["assemblies"]) == 1

    out_all = cif_parser_all.parse(filename)
    assert len(out_all["assemblies"]) == n_assemblies
