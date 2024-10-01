import biotite.structure as struc
import numpy as np
import pytest
import torch
from cifutils.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER


from datahub.transforms.atom_array import add_global_token_id_annotation
from datahub.transforms.af3_reference_molecule import (
    _map_reference_conformer_to_residue,
    get_af3_reference_molecule_features,
)
from datahub.transforms.rdkit_utils import atom_array_to_rdkit, find_automorphisms
from datahub.transforms.symmetry import apply_automorphs


def test_contrieved_tyr():
    # Create general residue
    orig = struc.info.residue("TYR")
    orig = orig[orig.atom_name != "OXT"]
    orig = orig[orig.element != "H"]
    orig[np.array([5, 6])] = orig[np.array([6, 5])]  # swap two atoms
    orig = add_global_token_id_annotation(orig)
    # Create reference molecule
    conformer = struc.info.residue("TYR")
    automorphs = find_automorphisms(atom_array_to_rdkit(conformer, infer_hydrogens=False))

    # Map reference molecule to residue
    ref_pos, ref_mask, ref_automorphs = _map_reference_conformer_to_residue(
        res_name="TYR",
        atom_names=orig.atom_name,
        conformer=conformer,
        automorphs=automorphs,
    )

    # Check that the reference molecule is correctly mapped to the residue
    assert ref_pos.shape == (len(orig), 3), f"{ref_pos.shape=} should be ({len(orig)}, 3)"
    assert ref_mask.shape == (len(orig),), f"{ref_mask.shape=} should be ({len(orig)},)"
    assert ref_automorphs.shape == (2, len(orig), 2), f"{ref_automorphs.shape=} should be (2, {len(orig)}, 2)"

    assert np.allclose(ref_pos, orig.coord), f"{ref_pos=} should be {orig.coord=}"
    assert np.allclose(ref_mask, True)

    expected_automorphs = np.array(
        [
            [
                [0, 0],  # N
                [1, 1],  # CA
                [2, 2],  # C
                [3, 3],  # O
                [4, 4],  # CB
                [6, 6],  # CG
                [5, 5],  # CD1
                [7, 7],  # CD2
                [8, 8],  # CE1
                [9, 9],  # CE2
                [10, 10],  # CZ
                [11, 11],
            ],  # OH
            [
                [0, 0],  # N
                [1, 1],  # CA
                [2, 2],  # C
                [3, 3],  # O
                [4, 4],  # CB
                [6, 6],  # CG
                [5, 7],  # CD1 <> CD2
                [7, 5],  # CD2 <> CD1
                [8, 9],  # CE1 <> CE2
                [9, 8],  # CE2 <> CE1
                [10, 10],  # CZ
                [11, 11],
            ],  # OH
        ]
    )

    assert np.allclose(ref_automorphs, expected_automorphs), f"{ref_automorphs=} should be {expected_automorphs=}"

    permuted_coords = apply_automorphs(torch.tensor(orig.coord), ref_automorphs)
    assert (permuted_coords[0] == torch.tensor(orig.coord)).all()
    assert not (permuted_coords[1] == torch.tensor(orig.coord)).all()


@pytest.mark.parametrize(
    "res_name",
    [
        "TYR",
        "ALA",
        "GLY",
        "PHE",
        "PRO",
        "VAL",
        "CYS",
        "LEU",
        "MET",
        "ASP",
        "GLU",
        "LYS",
        "ARG",
        "SER",
        "THR",
        "ASN",
        "GLN",
        "HIS",
        "TRP",
        "UNK",
        "R2R",
    ],
)
def test_get_af3_reference_molecule_features_res(res_name):
    atom_array = struc.info.residue(res_name)
    atom_array = atom_array[atom_array.atom_name != "OXT"]
    atom_array = atom_array[atom_array.element != "H"]
    atom_array = add_global_token_id_annotation(atom_array)
    # ... turn element into atomic number
    atom_array.element = np.array([ELEMENT_NAME_TO_ATOMIC_NUMBER[el.capitalize()] for el in atom_array.element])
    n_atom = len(atom_array)
    features = get_af3_reference_molecule_features(atom_array)
    assert "ref_pos" in features
    assert "ref_mask" in features
    assert "ref_element" in features
    assert "ref_charge" in features
    assert "ref_atom_name_chars" in features

    # Check if we have the correct number of conformers
    if res_name in ["TYR", "PHE", "VAL", "LEU"]:
        assert (
            len(features["ref_automorphs"]) == 2
        ), f"Expected 2 conformers for {res_name}, but got {len(features['ref_automorphs'])}"
    elif res_name == "R2R":
        assert (
            len(features["ref_automorphs"]) == 1000
        ), f"Expected 1000 conformers for {res_name}, but got {len(features['ref_automorphs'])}"
    else:
        assert (
            len(features["ref_automorphs"]) == 1
        ), f"Expected 1 conformer for {res_name}, but got {len(features['ref_automorphs'])}"
    assert features["ref_pos"].shape == (n_atom, 3)


def test_get_af3_reference_molecule_features_chain():
    atom_array = struc.info.residue("ALA") + struc.info.residue("R2R") + struc.info.residue("TYR")
    atom_array = atom_array[atom_array.atom_name != "OXT"]
    atom_array = atom_array[atom_array.element != "H"]
    atom_array = add_global_token_id_annotation(atom_array)

    # ... turn element into atomic number
    atom_array.element = np.array([ELEMENT_NAME_TO_ATOMIC_NUMBER[el.capitalize()] for el in atom_array.element])
    n_atoms = len(atom_array)

    features = get_af3_reference_molecule_features(atom_array)
    assert "ref_pos" in features
    assert "ref_mask" in features
    assert "ref_element" in features
    assert "ref_charge" in features
    assert "ref_atom_name_chars" in features
    assert (
        len(features["ref_automorphs"]) == 1000
    ), f"Expected 1000 conformers but got {len(features['ref_automorphs'])}"

    assert features["ref_pos"].shape == (n_atoms, 3)
    assert features["ref_mask"].shape == (n_atoms,)
    assert features["ref_element"].shape == (n_atoms,)
    assert features["ref_charge"].shape == (n_atoms,)
    assert features["ref_atom_name_chars"].shape == (n_atoms, 4)
    assert features["ref_automorphs"].shape == (1000, n_atoms, 2)

    # Inspect automorphs visually:
    # import matplotlib.pyplot as plt
    # plt.matshow(features["ref_automorphs_mask"][:10])
    # plt.matshow(features["ref_automorphs"][:10, :, 0])
    # plt.matshow(features["ref_automorphs"][:10, :, 1])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
