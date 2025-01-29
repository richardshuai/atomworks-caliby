import biotite.structure as struc
import numpy as np
import pytest
from assertpy import assert_that
from biotite.structure import AtomArray
from rdkit import Chem

from cifutils.tools.inference import components_to_atom_array
from cifutils.tools.rdkit import (
    atom_array_from_rdkit,
    atom_array_to_rdkit,
    ccd_code_to_rdkit,
    get_morgan_fingerprint_from_rdkit_mol,
    smiles_to_rdkit,
)
from cifutils.utils.io_utils import load_any

try:
    # Settings for debugging & interactive tests
    from rdkit.Chem.Draw import IPythonConsole

    IPythonConsole.kekulizeStructures = False
    IPythonConsole.drawOptions.addAtomIndices = True
    IPythonConsole.ipython_3d = True
    IPythonConsole.ipython_useSVG = True
    IPythonConsole.drawOptions.addStereoAnnotation = True
    IPythonConsole.molSize = 600, 300
except ImportError:
    pass

TEST_SMILES = [
    "C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)O)O)O)N",  # Adenosine
    "C1=CC=CC=C1",  # Benzene
    "c1cc(c[n+](c1)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)([O-])O[P@@](=O)(O)OC[C@@H]3[C@H]([C@H]([C@@H](O3)n4cnc5c4ncnc5N)O)O)O)O)C(=O)N",  # NAD
]
TEST_ATOM_ARRAYS = [struc.info.residue("ALA"), struc.info.residue("NAD")]


@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_smiles_to_rdkit_to_atom_array(smiles):
    mol = smiles_to_rdkit(smiles)

    # remove the inferred hydrogens again
    mol = Chem.RemoveHs(mol)

    atom_array = atom_array_from_rdkit(mol, set_coord_if_available=True, remove_hydrogens=True)

    # Add extra annotations
    atom_array.res_name = ["UNL"] * atom_array.array_length()
    atom_array.chain_id = ["A"] * atom_array.array_length()
    atom_array.set_annotation(
        "atom_name", atom_array.element.astype(object) + np.arange(atom_array.array_length()).astype(str)
    )

    assert isinstance(atom_array, AtomArray)
    assert atom_array.array_length() == mol.GetNumAtoms()


@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_smiles_to_atom_array_to_rdkit(smiles):
    inputs = []
    inputs.append(
        {
            "smiles": smiles,
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "A",
        }
    )
    atom_array = components_to_atom_array(inputs)
    mol = atom_array_to_rdkit(atom_array, set_coord=True, hydrogen_policy="keep")
    new_atom_array = atom_array_from_rdkit(mol, set_coord_if_available=True, remove_hydrogens=False)
    assert new_atom_array.array_length() == atom_array.array_length()
    for annotation in ["chain_id", "res_id", "res_name", "atom_name"]:
        assert np.array_equal(new_atom_array.get_annotation(annotation), atom_array.get_annotation(annotation))


@pytest.mark.parametrize("test_atom_array", TEST_ATOM_ARRAYS)
def test_atom_array_rdkit_interconversion(test_atom_array):
    test_atom_array.chain_id = ["A"] * test_atom_array.array_length()

    # Convert AtomArray to RDKit Mol
    mol = atom_array_to_rdkit(test_atom_array, set_coord=True, hydrogen_policy="keep")

    # Convert back to AtomArray
    new_atom_array = atom_array_from_rdkit(mol, set_coord_if_available=True, remove_hydrogens=False)

    # Check if the number of atoms is preserved
    assert new_atom_array.array_length() == test_atom_array.array_length()

    # Check if annotations are preserved
    for annotation in ["chain_id", "res_id", "res_name", "atom_name"]:
        assert np.array_equal(new_atom_array.get_annotation(annotation), test_atom_array.get_annotation(annotation))
    assert np.allclose(new_atom_array.coord, test_atom_array.coord)


def test_fixing_molecules():
    smi = "c1cc(c[n](c1)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)([O-])O[P@](=O)(O)OC[C@@H]3[C@H]([C@H]([C@H](O3)n4cnc5c4ncnc5N)OP(=O)(O)O)O)O)O)C(=O)N"
    smi_correct = "c1cc(c[n+](c1)[C@H]2[C@@H]([C@@H]([C@H](O2)CO[P@@](=O)([O-])O[P@](=O)(O)OC[C@@H]3[C@H]([C@H]([C@H](O3)n4cnc5c4ncnc5N)OP(=O)(O)O)O)O)O)C(=O)N"

    # Check that loading and sanitizing `smi` fails
    with pytest.raises(Chem.MolSanitizeException):
        smiles_to_rdkit(smi)

    mol = smiles_to_rdkit(smi, sanitize=False)  # noqa: F841
    mol_correct = smiles_to_rdkit(smi_correct)  # noqa: F841

    # TODO: Currently this cannot be fixed by our `fix_mol` function. Revisit this test once we implemented the remaining `TODO`s in `fix_mol`.
    # mol = fix_mol(
    #     mol,
    #     attempt_fix_by_normalizing_like_chembl=True,
    #     attempt_fix_by_normalizing_like_rdkit=True,
    #     attempt_fix_valence_by_changing_formal_charge=True,
    #     in_place=True,
    # )
    # assert Chem.MolToInchi(mol) == Chem.MolToInchi(mol_correct)


@pytest.fixture(scope="module")
def molecules():
    # Create RDKit molecules for some amino acids and small molecules
    mols = {
        "Leucine": ccd_code_to_rdkit("LEU", hydrogen_policy="remove"),
        "Isoleucine": ccd_code_to_rdkit("ILE", hydrogen_policy="remove"),
        "Glycine": ccd_code_to_rdkit("GLY", hydrogen_policy="remove"),
        "HEM": ccd_code_to_rdkit("HEM", hydrogen_policy="remove"),
        "NAG": ccd_code_to_rdkit("NAG", hydrogen_policy="remove"),
        "BMA": ccd_code_to_rdkit("BMA", hydrogen_policy="remove"),
        "CustomUNL": atom_array_to_rdkit(
            load_any("tests/data/test_unl_ligand_with_bonds.cif", model=1), set_coord=False, hydrogen_policy="keep"
        ),
    }
    return mols


def test_fingerprints(molecules):
    # Generate fingerprints for each molecule
    fingerprints = {name: get_morgan_fingerprint_from_rdkit_mol(mol) for name, mol in molecules.items()}

    # Calculate similarities and check if similar molecules have higher similarity scores
    sim_leu_ile = Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["Isoleucine"])
    sim_leu_gly = Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["Glycine"])
    sim_nag_bma = Chem.DataStructs.TanimotoSimilarity(fingerprints["NAG"], fingerprints["BMA"])
    sim_nag_hem = Chem.DataStructs.TanimotoSimilarity(fingerprints["NAG"], fingerprints["HEM"])
    # Assert that leucine is more similar to isoleucine than to glycine
    assert_that(sim_leu_ile).is_greater_than(sim_leu_gly).described_as(
        "Leucine should be more similar to Isoleucine than to Glycine"
    )

    # Asser that sugars (NAG and BMA) are more similar to each other than to HEM by at least a factor of 5
    assert_that(sim_nag_bma).is_greater_than(5 * sim_nag_hem).described_as(
        "Sugars should be more similar to each other than to HEM"
    )

    # Residues should have a similarity of 1.0 with themselves
    assert_that(Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["Leucine"])).is_equal_to(1.0)

    # Lycine and [NAG, BMA, HEM] should be less similar than 0.3 (very different)
    assert_that(Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["NAG"])).is_less_than(0.3)
    assert_that(Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["BMA"])).is_less_than(0.3)
    assert_that(Chem.DataStructs.TanimotoSimilarity(fingerprints["Leucine"], fingerprints["HEM"])).is_less_than(0.3)
    assert_that(Chem.DataStructs.TanimotoSimilarity(fingerprints["CustomUNL"], fingerprints["Leucine"])).is_less_than(
        0.3
    )


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
