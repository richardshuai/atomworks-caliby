import pytest
from tests.conftest import get_digs_path
from cifutils.cifutils_biotite import cifutils_biotite, cifutils_biotite_utils

TEST_CASES = [
    "5xa9",
    "5xag",
    "5xaf",
    "5xa7",
    "3qh0",
    "7qh6",
    "7qh4",
    "4qht",
    "1qh8",
    "6vo3",
    "7jsx",
    "5jsi",
    "6mti",
    "4mtl",
    "1s3q",
    "8czo",
    "6xkk",
    "2xkq",
    "4xku",
    "6xkl",
    "6vec",
    "7qbf",
    "6s91",
    "6jy3",
    "6jy4",
    "5jyg",
    "7dwz",
    "1dwk",
    "5dws",
    "6dwj",
    "5dwy",
    "5cpz",
    "3h6o",
    "5xa9",
    "5xag",
    "5xaf",
    "5xa7",
    "3qh0",
    "7qh6",
    "7qh4",
    "4qht",
    "1qh8",
    "6vo3",
]

cif_parser = cifutils_biotite.CIFParser()
# Manual debug
pdb_id = TEST_CASES[0]
cif = cifutils_biotite_utils.read_cif_file(get_digs_path(pdb_id))
cifutils_biotite_utils.category_to_dict(cif.block, "pdbx_struct_assembly")
cifutils_biotite_utils.category_to_dict(cif.block, "pdbx_struct_assembly_gen")
cifutils_biotite_utils.category_to_dict(cif.block, "pdbx_struct_oper_list")


@pytest.mark.parametrize("assembly_id", TEST_CASES)
def test_deserialize_assembly(pdb_id: str):
    digs_path = get_digs_path(pdb_id)
    result = cif_parser.parse(digs_path, build_assembly="all")
    assert result is not None
