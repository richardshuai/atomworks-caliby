import pytest
from tests.conftest import CIF_PARSER_BIOTITE
from pathlib import Path

DIR = Path(__file__).parent.parent / "data"
CIF_PATHS = [DIR / "example_distillation_output.cif"]


@pytest.mark.parametrize("path", CIF_PATHS)
def test_load_with_all_resolved(path: str):
    result = CIF_PARSER_BIOTITE.parse(
        filename=path,
        assume_residues_all_resolved=True,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        residues_to_remove=[],
        build_assembly="all",
        patch_symmetry_centers=True,
        keep_hydrogens=False,
        fix_arginines=True,
    )
    # Check if processing runs through
    assert result is not None

    # Check if the extra metadata is present (from the custom `_extra_metadata` CIFCategory)
    assert result["metadata"]["extra_metadata"] is not None


if __name__ == "__main__":
    pytest.main([__file__])
