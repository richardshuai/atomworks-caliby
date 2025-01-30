import biotite.structure as struc
import numpy as np
import pytest
import torch
from cifutils.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from cifutils.utils.selection import get_residue_starts

from datahub.transforms.af3_reference_molecule import (
    _encode_atom_names_like_af3,
    _map_reference_conformer_to_residue,
    get_af3_reference_molecule_features,
)
from datahub.transforms.atom_array import add_global_token_id_annotation
from datahub.transforms.atomize import atomize_by_ccd_name
from datahub.transforms.rdkit_utils import atom_array_to_rdkit, find_automorphisms_with_rdkit
from datahub.transforms.symmetry import apply_automorphs
from datahub.utils.rng import create_rng_state_from_seeds, rng_state


def test_contrived_tyr():
    # Create general residue
    orig = struc.info.residue("TYR")
    orig = orig[orig.atom_name != "OXT"]
    orig = orig[orig.element != "H"]
    orig[np.array([5, 6])] = orig[np.array([6, 5])]  # swap two atoms
    orig = add_global_token_id_annotation(orig)
    # Create reference molecule
    conformer = struc.info.residue("TYR")
    automorphs = find_automorphisms_with_rdkit(atom_array_to_rdkit(conformer, hydrogen_policy="keep"))

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

    n_atom = len(atom_array)

    # Check if we can compute features WITHOUT generating RDKit automorphisms (smoke test)
    features_no_automorphs = get_af3_reference_molecule_features(
        atom_array, should_generate_automorphisms_with_rdkit=False
    )
    assert "ref_pos" in features_no_automorphs
    assert "ref_mask" in features_no_automorphs
    assert "ref_element" in features_no_automorphs
    assert "ref_charge" in features_no_automorphs
    assert "ref_atom_name_chars" in features_no_automorphs

    seed = 42
    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
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
    # Add the necessary annotations from `parse`
    atom_array = atom_array[atom_array.atom_name != "OXT"]
    atom_array = atom_array[atom_array.element != "H"]
    atom_array = add_global_token_id_annotation(atom_array)

    # We atomize so that we can test using the element for atom names of atomized tokens
    atom_array = atomize_by_ccd_name(atom_array, res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA)

    n_atoms = len(atom_array)

    seed = 42
    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        features = get_af3_reference_molecule_features(atom_array, apply_random_rotation_and_translation=False)
        features_with_elements_for_atomized_atom_names = get_af3_reference_molecule_features(
            atom_array, apply_random_rotation_and_translation=False, use_element_for_atom_names_of_atomized_tokens=True
        )

    assert "ref_pos" in features
    assert "ref_mask" in features
    assert "ref_element" in features
    assert "ref_charge" in features
    assert "ref_atom_name_chars" in features
    assert (
        len(features["ref_automorphs"]) == 1000
    ), f"Expected 1000 conformers but got {len(features['ref_automorphs'])}"

    # ... check that the atom name features are the same for non-atomized tokens
    assert np.all(
        features["ref_atom_name_chars"][~atom_array.atomize]
        == features_with_elements_for_atomized_atom_names["ref_atom_name_chars"][~atom_array.atomize]
    )

    # ... check that the atom name features for atomized tokens are encoded correctly, when indicated
    encoded_elements = _encode_atom_names_like_af3(atom_array.element)
    assert np.all(
        features_with_elements_for_atomized_atom_names["ref_atom_name_chars"][atom_array.atomize]
        == encoded_elements[atom_array.atomize]
    )

    assert features["ref_pos"].shape == (n_atoms, 3)
    assert features["ref_mask"].shape == (n_atoms,)
    assert features["ref_element"].shape == (n_atoms,)
    assert features["ref_charge"].shape == (n_atoms,)
    assert features["ref_atom_name_chars"].shape == (n_atoms, 4)
    assert features["ref_automorphs"].shape == (1000, n_atoms, 2)

    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        features__with_random_rototranslation = get_af3_reference_molecule_features(
            atom_array, apply_random_rotation_and_translation=True
        )

        # Assert that the features are different
        assert not np.allclose(features["ref_pos"], features__with_random_rototranslation["ref_pos"])
        assert np.allclose(features["ref_mask"], features__with_random_rototranslation["ref_mask"])

    # Inspect automorphs visually:
    # import matplotlib.pyplot as plt
    # plt.matshow(features["ref_automorphs_mask"][:10])
    # plt.matshow(features["ref_automorphs"][:10, :, 0])
    # plt.matshow(features["ref_automorphs"][:10, :, 1])


def test_reference_conformer_generation_for_two_molecules_only_differing_by_transformation_id():
    # fmt: off
    atom_array = struc.array([
        struc.Atom(np.array([44.869,     8.188,    36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  atomic_number=7,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([45.024,     7.456,    34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([44.142,     6.714,    34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", atomic_number=8,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.669,     8.171,    36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.812,     8.982,    38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.152,     8.296,    39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.479,     9.3  ,    40.792 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="SD", atomic_number=16, charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.232,     8.184,    42.102 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CE", atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.46 ,     8.724,    36.151 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="C",  atomic_number=6,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.339,     9.907,    35.831 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O",  atomic_number=8,  charge=0,  transformation_id="1"),
        struc.Atom(np.array([58.656483, 34.763695, 36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  atomic_number=7,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.212917, 35.263927, 34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", atomic_number=6,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([60.296505, 34.87109 , 34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", atomic_number=8,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.271206, 33.732964, 36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", atomic_number=6,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([58.49736 , 33.451305, 38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", atomic_number=6,  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.42145 , 33.22273 , 39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", atomic_number=6,  charge=0,  transformation_id="2"),
    ])
    # fmt: on

    assert len(get_residue_starts(atom_array)) == 2
    atom_array.set_annotation("token_id", np.arange(len(atom_array)))
    features = get_af3_reference_molecule_features(atom_array)
    assert len(features) > 0, "Expected features to be non-empty"
