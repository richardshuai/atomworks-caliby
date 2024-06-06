import pytest
from cifutils.cifutils_biotite import cifutils_biotite
from tests.conftest import get_digs_path

TEST_CASES = [
    "6xa4",
    "1qh9",
    "6qhw",
    "5vo3",
    "5t4j",
    "3t44",
    "6t4v",
]

cif_parser = cifutils_biotite.CIFParser(add_missing_atoms=True)


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_atom_order(pdb_id: str):
    path = get_digs_path(pdb_id)
    result = cif_parser.parse(path, build_assembly="all")
    assert result is not None
