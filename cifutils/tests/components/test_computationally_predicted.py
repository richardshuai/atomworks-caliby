import pytest
from tests.conftest import CIF_PARSER_BIOTITE
from pathlib import Path
import numpy as np

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


def test_af2_predicted_pdb_example():
    result = CIF_PARSER_BIOTITE.parse(
        filename=DIR / "UniRef50_A0A0S8JQ92_AF2_predicted.pdb",
        remove_waters=True,
        residues_to_remove=[],
    )
    # Check if processing runs through
    assert result is not None


def test_bcif_example():
    result = CIF_PARSER_BIOTITE.parse(
        filename=DIR / "6lyz.bcif",
    )
    # Check if processing runs through
    assert result is not None


def test_pdb_with_same_chain_poly_non_poly():
    result = CIF_PARSER_BIOTITE.parse(
        filename=DIR / "1qfe.pdb",
        keep_hydrogens=False,
    )
    # Check if processing runs through
    assert result is not None

    # Check that we don't have any chains with polymeric and non-polymeric residues
    atom_array = result["assemblies"]["1"][0]
    polymer_chain_ids = np.unique(atom_array.chain_id[atom_array.is_polymer])
    non_polymer_chain_ids = np.unique(atom_array.chain_id[~atom_array.is_polymer])
    assert len(set(polymer_chain_ids).intersection(non_polymer_chain_ids)) == 0

    # Assert that all residues have coordinates
    assert np.all(np.isfinite(atom_array.coord))

    # Assert that all atoms are full occupancy
    assert np.all(atom_array.occupancy == 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
