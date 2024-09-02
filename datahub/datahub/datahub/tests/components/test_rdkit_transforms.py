# TODO: Remove

import numpy as np
import pytest

# Settings for debugging & interactive tests
from rdkit.Chem.Draw import IPythonConsole

from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING
from datahub.tests.conftest import cached_parse
from datahub.transforms.atom_array import AddGlobalAtomIdAnnotation, FilterAndAnnotatePNUnits, RemoveHydrogens
from datahub.transforms.atomize import AtomizeResidues
from datahub.transforms.base import Compose
from datahub.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from datahub.transforms.rdkit_utils import (
    AddRDKitMoleculesForAtomizedMolecules,
    GenerateRDKitConformers,
    get_chiral_centers,
)

IPythonConsole.kekulizeStructures = False
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.ipython_3d = False
IPythonConsole.ipython_useSVG = True
IPythonConsole.drawOptions.addStereoAnnotation = True
IPythonConsole.molSize = 600, 300


TEST_CASES = [
    {"pdb_id": "5ocm"},
]


def test_add_rdkit_molecules_for_atomized_molecules(test_case):
    # Prepare input data
    data = cached_parse(test_case["pdb_id"])

    # Apply the transform
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            RemoveHydrogens(),
            FilterAndAnnotatePNUnits(),
            FlagAndReassignCovalentModifications(),
            AtomizeResidues(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
            AddRDKitMoleculesForAtomizedMolecules(),
        ]
    )
    result = pipe(data)

    # Check if the rdkit key is added to the data dictionary
    assert "rdkit" in result

    # Check if RDKit molecules are created for each unique pn_unit_iid of atomized residues
    unique_pn_unit_iids = np.unique(data["atom_array"].pn_unit_iid[data["atom_array"].atomize])
    assert len(result["rdkit"]) == len(unique_pn_unit_iids)

    # Check if each RDKit molecule has the correct number of atoms
    for pn_unit_iid, rdmol in result["rdkit"].items():
        pn_unit_mask = (data["atom_array"].pn_unit_iid == pn_unit_iid) & data["atom_array"].atomize
        expected_num_atoms = np.sum(pn_unit_mask)
        assert rdmol.GetNumAtoms() == expected_num_atoms


def test_generate_rdkit_conformers(atom_array_5ocm):
    # Prepare input data
    data = {
        "atom_array": atom_array_5ocm,
    }

    # Add atomize annotation (assuming all non-water residues are atomized for this test)
    data["atom_array"].set_annotation("atomize", data["atom_array"].res_name != "HOH")

    # First, add RDKit molecules
    add_rdkit_transform = AddRDKitMoleculesForAtomizedMolecules()
    data = add_rdkit_transform(data)

    # Apply the GenerateRDKitConformers transform
    n_conformers = 3
    transform = GenerateRDKitConformers(n_conformers=n_conformers)
    result = transform(data)

    # Check if the rdkit key is still in the data dictionary
    assert "rdkit" in result

    # Check if each RDKit molecule has the correct number of conformers
    for pn_unit_iid, rdmol in result["rdkit"].items():
        assert rdmol.GetNumConformers() == n_conformers


def test_get_chiral_centers(atom_array_5ocm):
    # Prepare input data
    data = {
        "atom_array": atom_array_5ocm,
    }

    # Add atomize annotation (assuming all non-water residues are atomized for this test)
    data["atom_array"].set_annotation("atomize", data["atom_array"].res_name != "HOH")

    # First, add RDKit molecules
    add_rdkit_transform = AddRDKitMoleculesForAtomizedMolecules()
    data = add_rdkit_transform(data)

    # Check if chiral centers are identified correctly
    for pn_unit_iid, rdmol in data["rdkit"].items():
        chiral_centers = get_chiral_centers(rdmol)

        # Check if chiral centers are returned as a list of dictionaries
        assert isinstance(chiral_centers, list)
        for center in chiral_centers:
            assert isinstance(center, dict)
            assert "chiral_center_idx" in center
            assert "bonded_explicit_atom_idxs" in center
            assert "chirality" in center

        # Check for known chiral centers in 5ocm (you may need to adjust these based on the actual structure)
        if rdmol.GetNumAtoms() > 10:  # Assuming this is a larger molecule like NAD
            assert len(chiral_centers) > 0, f"Expected chiral centers in molecule {pn_unit_iid}"


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
