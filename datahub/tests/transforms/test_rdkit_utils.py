import time

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray
from cifutils.tools.inference import components_to_atom_array
from cifutils.tools.rdkit import atom_array_from_rdkit, atom_array_to_rdkit, smiles_to_rdkit
from rdkit import Chem

from datahub.transforms.rdkit_utils import (
    ccd_code_to_rdkit_with_conformers,
    generate_conformers,
    sample_rdkit_conformer_for_atom_array,
)

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
    "C[C@]12CC[C@@H](C[C@H]1CC[C@@H]3[C@@H]2C[C@H]([C@]4([C@@]3(CC[C@@H]4C5=CC(=O)OC5)O)C)O)O",
]
TEST_ATOM_ARRAYS = [struc.info.residue("ALA"), struc.info.residue("NAD")]


@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_smiles_to_rdkit_with_conformer_to_atom_array(smiles):
    mol = smiles_to_rdkit(smiles)
    mol = generate_conformers(mol, seed=42, n_conformers=1, infer_hydrogens=True, optimize=True)

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
def test_smiles_to_atom_array_to_conformer(smiles):
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
    mol = atom_array_to_rdkit(atom_array, infer_hydrogens=True)
    mol = generate_conformers(mol, seed=42, n_conformers=1, infer_hydrogens=True, optimize=True)
    new_atom_array = atom_array_from_rdkit(mol, set_coord_if_available=True, remove_hydrogens=True)

    # Ensure we didn't drop any atoms
    assert len(atom_array) == len(new_atom_array)

    # Assert no NaN coordinates
    assert not np.any(np.isnan(new_atom_array.coord))


@pytest.mark.parametrize("smiles", TEST_SMILES)
def test_generate_conformers_for_smiles(smiles):
    mol = smiles_to_rdkit(smiles)

    # Generate new conformer
    mol_with_conf = generate_conformers(mol, seed=42, n_conformers=1, infer_hydrogens=True, optimize=True)

    assert mol_with_conf.GetNumConformers() == 1


@pytest.mark.parametrize("test_atom_array", TEST_ATOM_ARRAYS)
def test_sample_rdkit_conformer_consistency(test_atom_array):
    test_atom_array.chain_id = ["A"] * test_atom_array.array_length()
    test_atom_array = test_atom_array[test_atom_array.element != "H"]
    original_coords = test_atom_array.coord

    # Sample new conformer
    new_atom_array = sample_rdkit_conformer_for_atom_array(test_atom_array, seed=42)
    # ... superimpose to original
    new_atom_array, _ = struc.superimpose(mobile=new_atom_array, fixed=test_atom_array)

    # Check if the number of atoms and their order is preserved
    assert new_atom_array.array_length() == test_atom_array.array_length()
    assert np.array_equal(new_atom_array.atom_name, test_atom_array.atom_name)
    assert np.array_equal(new_atom_array.res_id, test_atom_array.res_id)
    assert np.array_equal(new_atom_array.res_name, test_atom_array.res_name)

    # Calculate RMSD
    rmsd = struc.rmsd(new_atom_array, test_atom_array)

    # Check if RMSD is different from original
    assert not np.allclose(original_coords, new_atom_array.coord)

    # WARNING: The RMSD threshold was tuned optically for the given test examples.
    #          This is not a strict requirement for conformation generation and
    #          may need to be adjusted when adding more test examples.
    assert rmsd < 4.0, f"RMSD is too high: {rmsd}. Test failed for {test_atom_array.res_name[0]}."


@pytest.mark.filterwarnings(
    "ignore: This process"
)  # Ignore RDKit warnings about fork in subprocess: popen_fork.py:66: DeprecationWarning: This process (pid=145252) is multi-threaded, use of fork() may lead to deadlocks in the child.
def test_conformer_generation_for_simple_molecules():
    start = time.time()
    mol = ccd_code_to_rdkit_with_conformers("ALA", n_conformers=3, timeout_seconds=2)
    end = time.time()
    assert mol.GetNumConformers() == 3
    _time_taken = end - start
    assert _time_taken < 3.0, f"Conformer generation took too long: {_time_taken} seconds, while timeout was 2 seconds."


@pytest.mark.filterwarnings(
    "ignore: This process"
)  # Ignore RDKit warnings about fork in subprocess: popen_fork.py:66: DeprecationWarning: This process (pid=145252) is multi-threaded, use of fork() may lead to deadlocks in the child.
def test_conformer_fallback_for_challenging_molecules():
    start = time.time()
    mol = ccd_code_to_rdkit_with_conformers("HEM", n_conformers=3, timeout_seconds=2)
    end = time.time()
    assert mol.GetNumConformers() == 3
    _time_taken = end - start
    assert _time_taken < 3.0, f"Conformer generation took too long: {_time_taken} seconds, while timeout was 2 seconds."


def test_conformer_generation_for_molecules_with_many_rotatable_bonds():
    # Test inspired by https://github.com/rdkit/rdkit/issues/1433#issuecomment-305097888
    mol = Chem.MolFromSmiles("CCCCCCCC[N+](CCCCCCCC)(CCCCCCCC)CCCCCCCC")
    mol_with_conf = generate_conformers(mol, seed=42, n_conformers=10, infer_hydrogens=True, optimize=True)
    assert mol_with_conf.GetNumConformers() == 10


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
