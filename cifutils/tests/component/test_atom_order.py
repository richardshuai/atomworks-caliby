import pytest
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE

TEST_CASES = [
    "6xa4",
    "1qh9",
    "6qhw",
    "5vo3",
    "5t4j",
    "3t44",
    "6t4v",
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_atom_order(pdb_id: str):
    path = get_digs_path(pdb_id)
    result = CIF_PARSER_BIOTITE.parse(path, add_missing_atoms=True, build_assembly=None)
    assert result is not None
