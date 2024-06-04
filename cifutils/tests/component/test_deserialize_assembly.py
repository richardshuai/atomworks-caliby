import pytest
from tests.conftest import get_digs_path
from cifutils.cifutils_biotite import cifutils_biotite

TEST_CASES = [
    # With the wrong version of biotite, these will lead to cif deserialization errors as the assembly category is represented slightly differently in these files
    "5xa9",
    "5xag",
    "5xaf",
    "5ocm",
]

cif_parser = cifutils_biotite.CIFParser()


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_deserialize_assembly(pdb_id: str):
    digs_path = get_digs_path(pdb_id)
    result = cif_parser.parse(digs_path, build_assembly="all")
    assert result is not None
