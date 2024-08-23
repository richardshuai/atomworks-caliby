from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE
from cifutils.cifutils_biotite.constants import CRYSTALLIZATION_AIDS

import numpy as np
import pytest

# fmt: off
MOLECULE_ENTITY_TEST_CASES = [
    {
        # Protein-protein heteromer, with glycosylation
        "pdb_id": "1ivo",
        "chains_with_same_molecule_entity": [
            ["A", "E", "F", "G", "H", "I", "J"],
            ["B", "K", "L", "M"],
            ["C", "D"],
        ],
    },
    {
        # Protein-protein homomer, no transformations
        "pdb_id": "1mna",
        "chains_with_same_molecule_entity": [
            ["A", "B"],
        ],
    },
    {
        # Protein-protein homomer, with transformations
        "pdb_id": "1a8o",
        "chains_with_same_molecule_entity": [
            ["A"],
        ],
    },
    {
        # Protein-protein heteromer, with glycosylation, where two glycosylated chains have the same molecule entity (same bond connectivity, despite having multiple chains covalently bound)
        "pdb_id": "1hge",
        "chains_with_same_molecule_entity": [
            [
                "B", "N", "F", "X", "D", "S",
            ],  # Two equivalent glycosylated chains, each involving one protein and one glycan chain
            [
                "A", "J", "K", "G", "L", "C", "O", "P", "H", "Q", "E", "T", "U", "I", "V",
            ],  # Three equivalent glycosylated chains, each involving one protein and four glycan chains
            ["M", "R", "W"],  # Small molecules, all with the same entity ID
        ],
    },
]
# fmt: on

def validate_molecule_entity_annotations(parser_result: dict, test_case: dict):
    assembly_atom_array = parser_result["assemblies"]["1"][0] # Check the first model of the first assembly

    # Check that the number of molecule entitys is correct
    assert len(np.unique(assembly_atom_array.molecule_entity)) == len(test_case["chains_with_same_molecule_entity"])

    for chain_ids in test_case["chains_with_same_molecule_entity"]:
        chains_mask = np.isin(assembly_atom_array.chain_id, chain_ids)

        # Check that the ground truth chains with the same molecule entity match the computed molecule entitys 1hge
        assert len(np.unique(assembly_atom_array.molecule_entity[chains_mask])) == 1

        # Check that no other chains have the same molecule entity
        molecule_entity = assembly_atom_array.molecule_entity[chains_mask][0]
        all_chain_ids_with_chain_entity = np.unique(assembly_atom_array.chain_id[assembly_atom_array.molecule_entity == molecule_entity])
        assert set(all_chain_ids_with_chain_entity) == set(chain_ids)

@pytest.mark.parametrize("test_case", MOLECULE_ENTITY_TEST_CASES)
def test_add_molecule_entity_annotation(test_case: dict):
    path = get_digs_path(test_case["pdb_id"])
    result = CIF_PARSER_BIOTITE.parse(
        filename=path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        residues_to_remove=CRYSTALLIZATION_AIDS,
        patch_symmetry_centers=True,
        build_assembly="all",
        fix_arginines=True,
        convert_mse_to_met=True,
        keep_hydrogens=True,
        model=None,
    )
    assert result is not None
    validate_molecule_entity_annotations(result, test_case)

def test_add_molecule_entity_annotation_on_modified_pdb():
    """
    Tests on a custom-modified example of a molecule.
    This test loads the "1hge" PDB example (see test_add_molecule_entity_annotation)
    and manually adjusts the bond list to break the symmetry between the multiple copies of the same molecule.
    """
    pdb_id = "1hge"
    path = get_digs_path(pdb_id)
    result = CIF_PARSER_BIOTITE.parse(
        filename=path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        residues_to_remove=CRYSTALLIZATION_AIDS,
        patch_symmetry_centers=True,
        build_assembly="all",
        fix_arginines=True,
        convert_mse_to_met=True,
        keep_hydrogens=True,
        model=None,
    )
    atom_array = result["assemblies"]["1"][0] # First model

    # Manually adjust the atom array so that the polymer chain C is covalently bound to the glycan chain H at the second, rather than the first, residue
    # We thus break the symmetry between the multiple copies of the same molecule
    # We expect that the molecule entity for chain C will be different from the other chains with the same chain entity ID
    atom_a_mask = (
        (atom_array.chain_id == "C")
        & (atom_array.res_name == "ASN")
        & (atom_array.res_id == 165)
        & (atom_array.atom_name == "ND2")
    )
    atom_a_index = np.where(atom_a_mask)[0][0]

    atom_b_mask = (
        (atom_array.chain_id == "H")
        & (atom_array.res_name == "NAG")
        & (atom_array.res_id == 1)
        & (atom_array.atom_name == "C1")
    )
    atom_b_index = np.where(atom_b_mask)[0][0]

    atom_c_mask = (
        (atom_array.chain_id == "H")
        & (atom_array.res_name == "NAG")
        & (atom_array.res_id == 2)
        & (atom_array.atom_name == "C2")
    )
    atom_c_index = np.where(atom_c_mask)[0][0]

    # Remove the existing bond between atom A and atom B...
    atom_array.bonds.remove_bond(atom_a_index, atom_b_index)

    # ...and create a new bond between atom A and atom C
    atom_array.bonds.add_bond(atom_a_index, atom_c_index)
    result["assemblies"]["1"][0] = atom_array

    # manual test case
    # fmt: off
    test_case = {
        # Protein-protein heteromer, with glycosylation, where two glycosylated chains have the same molecule entity (same bond connectivity, despite having multiple chains covalently bound)
        "pdb_id": "1hge",
        "chains_with_same_molecule_entity": [
            [
                "B", "N", "F", "X", "D", "S",
            ],  # Two equivalent glycosylated chains, each involving one protein and one glycan chain
            [
                "A", "J", "K", "G", "L", "E", "T", "U", "I", "V",
            ],  # Two equivalent glycosylated chains, each involving one protein and four glycan chains
            [
                "C", "O", "P", "H", "Q",
            ],  # One glycosylated chain with a different bond connectivity (manual change)
            ["M", "R", "W"],  # Small molecules, all with the same entity ID
        ],
    }
    # fmt: on

    validate_molecule_entity_annotations(result, test_case)

ADD_CHAIN_ENTITY_TEST_CASES = [
    {"pdb_id": "1ivo", "equivalent_chains": [["A", "B"], ["C", "D"], ["E"], ["F", "G", "H", "I", "J"]]},
    # TODO: Add more test cases, including ones that were previously failing
]


@pytest.mark.parametrize("test_case", ADD_CHAIN_ENTITY_TEST_CASES)
def test_regenerate_and_add_chain_entity_annotation(test_case):
    """
    Tests that we:
    - Regenerate the chain entities for equivalent chains (ensure all equivalent chains have the same chain_entity)
    - Add the chain entity annotation to the atom array
    """
    path = get_digs_path(test_case["pdb_id"])
    result = CIF_PARSER_BIOTITE.parse(filename=path)

    # Check that all equivalent chains have the same chain_entity in the chain_info dictionary
    chain_info_dict = result["chain_info"]
    for equivalent_chains in test_case["equivalent_chains"]:
        chain_entities = [chain_info_dict[chain_id]["entity_id"] for chain_id in equivalent_chains]
        assert len(set(chain_entities)) == 1, f"Chains {equivalent_chains} do not have the same chain_entity"

    # Check that the chain_entity annotation maps to the correct chain_id
    atom_array = result["atom_array"]
    for equivalent_chains in test_case["equivalent_chains"]:
        chain_entity = chain_info_dict[equivalent_chains[0]]["entity_id"]
        for chain_id in equivalent_chains:
            chain_mask = atom_array.chain_id == chain_id
            assert np.all(
                atom_array.chain_entity[chain_mask] == chain_entity
            ), f"Entity ID for chain {chain_id} does not match expected chain_entity"

if __name__ == "__main__":
    pytest.main([__file__])
