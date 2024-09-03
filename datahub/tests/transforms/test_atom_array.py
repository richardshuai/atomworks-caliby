import biotite.structure as struc
import numpy as np
import pytest

from datahub.datasets.dataframe_parsers import PNUnitsDFParser, load_from_row
from datahub.transforms.atom_array import (
    AddMoleculeSymmetricIdAnnotation,
    AddProteinTerminiAnnotation,
    AddWithinPolyResIdxAnnotation,
    RemoveHydrogens,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
    RenumberNonPolymerResidueIdx,
    chain_instance_iter,
    sort_poly_then_non_poly,
)
from datahub.transforms.base import Compose
from datahub.transforms.msa._msa_constants import THREE_LETTER_TO_MSA_INTEGER
from datahub.transforms.msa.msa import LoadPolymerMSAs
from tests.conftest import CIF_PARSER, PN_UNITS_DF, PROTEIN_MSA_DIR, RNA_MSA_DIR, cached_parse
from tests.transforms.msa.test_pair_and_merge_polymer_msas import MSA_PAIRING_PIPELINE_TEST_CASES


@pytest.mark.parametrize("example_id", ["{['pdb', 'pn_units']}{4gqa}{2}{['C_3']}"])
def test_remove_unresolved_pn_units(example_id):
    row = PN_UNITS_DF[PN_UNITS_DF["example_id"] == example_id].iloc[0]
    data = load_from_row(row, PNUnitsDFParser(), cif_parser=CIF_PARSER)

    # Artificially set the occupancy for all atoms in chain_iid "G_1" to 0
    data["atom_array"].occupancy[data["atom_array"].chain_iid == "G_1"] = 0

    pipeline = Compose(
        [
            RemoveUnresolvedPNUnits(),
        ],
        track_rng_state=False,
    )
    output = pipeline(data)

    # Assert that the atom array has no unresolved PN units
    pn_unit_iids = np.unique(output["atom_array"].pn_unit_iid)
    resolved_mask = output["atom_array"].occupancy != 0
    resolved_pn_unit_iids = np.unique(output["atom_array"].pn_unit_iid[resolved_mask])

    assert set(pn_unit_iids) == set(resolved_pn_unit_iids)


def test_remove_hydrogens_original_pdb():
    atom_array = struc.info.residue("ILE")
    assert "H" in atom_array.element

    data = {"atom_array": atom_array}

    transform = RemoveHydrogens()
    data = transform(data)

    atom_array_new = data["atom_array"]
    assert "H" not in atom_array_new.element
    assert len(atom_array_new) == len(atom_array[atom_array.element != "H"])


@pytest.mark.parametrize("pdb_id", ["5ocm", "1b4y", "1tqn"])
def test_remove_hydrogens_parsed_pdb(pdb_id: str):
    data = cached_parse(pdb_id, keep_hydrogens=True)
    atom_array = data["atom_array"]
    assert "1" in atom_array.element

    transform = RemoveHydrogens()
    data = transform(data)

    atom_array_new = data["atom_array"]
    assert "1" not in atom_array_new.element
    assert len(atom_array_new) == len(atom_array[atom_array.element != "1"])


def test_remove_terminal_oxygen():
    atom_array = struc.info.residue("ILE")
    assert "OXT" in atom_array.atom_name

    data = {"atom_array": atom_array}

    transform = RemoveTerminalOxygen()
    data = transform(data)

    atom_array_new = data["atom_array"]
    assert "OXT" not in atom_array_new.atom_name
    assert len(atom_array_new) == len(atom_array) - 1


@pytest.mark.parametrize("pdb_id", ["5ocm", "6lyz"])
def test_annotate_protein_termini(pdb_id: str):
    """
    Test the AddProteinTerminiAnnotation transform on parsed PDB files.

    This test checks the following:
    1. The N-termini and C-termini annotations are added to the atom array.
    2. At least one N-terminus and one C-terminus are found in the atom array.
    3. The number of annotated N-termini and C-termini matches the number of protein chains.

    Args:
        pdb_id (str): The PDB ID of the structure to be tested.
    """
    data = cached_parse(pdb_id)
    poly_filter = struc.filter_polymer(data["atom_array"], pol_type="peptide")
    n_protein_chains = struc.get_chain_count(data["atom_array"][poly_filter])
    pipe = Compose([AddProteinTerminiAnnotation()], track_rng_state=False)
    data = pipe(data)

    assert data["atom_array"].is_N_terminus is not None, "N-termini should be annotated"
    assert data["atom_array"].get_annotation("is_C_terminus") is not None, "C-termini should be annotated"

    assert np.any(data["atom_array"].get_annotation("is_N_terminus")), "Not a single N-terminus found"
    assert np.any(data["atom_array"].get_annotation("is_C_terminus")), "Not a single C-terminus found"

    # Assert that the chain count is the same as the number of N- and C-termini
    res_n_terminus = data["atom_array"][data["atom_array"].is_N_terminus]
    res_c_terminus = data["atom_array"][data["atom_array"].is_C_terminus]
    n_ntermini = struc.get_residue_count(res_n_terminus)
    n_ctermini = struc.get_residue_count(res_c_terminus)
    assert (
        n_protein_chains == n_ntermini
    ), f"Found {n_protein_chains} protein chains, but {n_ntermini} N-termini were annotated"
    assert (
        n_protein_chains == n_ctermini
    ), f"Found {n_protein_chains} protein chains, but {n_ctermini} C-termini were annotated"


@pytest.mark.parametrize("pdb_id", MSA_PAIRING_PIPELINE_TEST_CASES)
def test_add_within_poly_res_idx_annotation(pdb_id: str):
    data = cached_parse(pdb_id)

    pipe = Compose(
        [
            AddWithinPolyResIdxAnnotation(),
            LoadPolymerMSAs(protein_msa_dir=PROTEIN_MSA_DIR, rna_msa_dir=RNA_MSA_DIR),
        ],
        track_rng_state=False,
    )
    result = pipe(data)
    atom_array = result["atom_array"]

    # Check that for polymers, the within_poly_res_idx annotation is the same as the residue index minus one
    polymer_chains = atom_array[atom_array.is_polymer]
    inferred_within_poly_res_idx = polymer_chains.res_id - 1
    assert all(polymer_chains.within_poly_res_idx == inferred_within_poly_res_idx)

    # Check that all non-polymers have empty within_poly_res_idx
    non_polymer_chains = atom_array[~atom_array.is_polymer]
    assert all(
        non_polymer_chains.within_poly_res_idx == -1
    ), "Non-polymer chains should have within_poly_res_idx set -1"

    # Check that we can use the within_poly_res_idx to index the MSA
    # Check that when we use `within_poly_res_idx` to index into the MSA, we get the same sequence as when we re-construct the one-letter sequence from the atom array
    # In a sense, this is an integration test between LoadPolymerMSAs and AddWithinPolyResIdxAnnotation
    polymer_chain_ids = np.unique(polymer_chains.chain_id)
    assert len(polymer_chain_ids) > 0, "No polymer chains found in the atom array"
    for chain_id in polymer_chain_ids:
        chain_atom_array = polymer_chains[polymer_chains.chain_id == chain_id]

        # Get the polymer sequence by indexing into the first row of the MSA (query sequence) with the within_poly_res_id
        polymer_msa = result["polymer_msas_by_chain_id"][chain_id]
        residues_starts = chain_atom_array[struc.get_residue_starts(chain_atom_array)]
        polymer_sequence_from_msa_indexing = polymer_msa["msa"][0][residues_starts.within_poly_res_idx]

        # Get the polymer sequence from the atom array, and convert it to a one-letter sequence ()
        polymer_three_letter_sequence_from_atom_array = struc.get_residues(chain_atom_array)[
            1
        ]  # Returns a tuple of (ids, names), so [1] gets the list of names
        polymer_sequence_from_atom_array = np.array(
            [str(THREE_LETTER_TO_MSA_INTEGER[res_name]) for res_name in polymer_three_letter_sequence_from_atom_array],
            dtype=np.int8,
        )

        # Assert that the MSA sequence is the same as the sequence in the atom array
        assert np.array_equal(polymer_sequence_from_msa_indexing, polymer_sequence_from_atom_array)


def test_renumber_non_polymer_residue_idx():
    # Test with a mixed polymer and non-polymer chain example
    atom_array = struc.AtomArray(10)
    atom_array.set_annotation("chain_iid", np.array(["A", "A", "A", "A", "B", "B", "B", "C", "C", "D"]))
    atom_array.set_annotation("res_id", np.array([1, 1, 2, 2, 101, 101, 102, 205, 206, 1]))
    atom_array.set_annotation("is_polymer", np.array([True, True, True, True, False, False, False, False, False, True]))
    data = {"atom_array": atom_array}

    pipe = RenumberNonPolymerResidueIdx()
    result = pipe(data)
    expected_res_ids = np.array([1, 1, 2, 2, 1, 1, 2, 1, 2, 1])
    assert np.array_equal(result["atom_array"].get_annotation("res_id"), expected_res_ids)


@pytest.mark.parametrize("pdb_id", ["1mna", "1a8o", "1hge"])
def test_add_molecule_symmetric_id_annotation(pdb_id):
    data = cached_parse(pdb_id)

    pipe = Compose(
        [
            RenumberNonPolymerResidueIdx(),
            AddMoleculeSymmetricIdAnnotation(),
        ],
        track_rng_state=False,
    )
    result = pipe(data)
    atom_array = result["atom_array"]
    molecule_entities = np.unique(atom_array.molecule_entity)

    # Check that chains with the same molecule entity all have different molecule symmetric IDs
    for molecule_entity in molecule_entities:
        molecules = atom_array[atom_array.molecule_entity == molecule_entity]

        # Count how many different compounds have this entity ID
        molecule_iids = np.unique(molecules.molecule_iid)

        # Count how many different symmetric IDs there are with this entity ID
        molecule_symmetric_ids = np.unique(molecules.molecule_symmetric_id)

        # Ensure that the number of molecule symmetric IDs is the same as the number of molecules
        assert len(molecule_iids) == len(molecule_symmetric_ids)

        # Check that the symmetric IDs are 0-indexed, with no gaps
        assert np.all(np.sort(molecule_symmetric_ids) == np.arange(len(molecule_symmetric_ids)))


CHAIN_ITER_TEST_CASES = [
    {"pdb_id": "1a8o", "ordered_chain_iids": ["A_1", "A_2"]},
    {"pdb_id": "1rxz", "ordered_chain_iids": ["A_1", "B_1", "A_2", "B_2", "A_3", "B_3"]},
]


@pytest.mark.parametrize("test_case", CHAIN_ITER_TEST_CASES)
def test_chain_iter(test_case):
    data = cached_parse(test_case["pdb_id"])

    atom_array = data["assemblies"]["1"]
    ordered_chain_iids = test_case["ordered_chain_iids"]

    num_counted = 0
    for index, chain_instance_atom_array in enumerate(chain_instance_iter(atom_array)):
        # Check that all the atoms in the chain have the same chain_iid
        assert np.all(chain_instance_atom_array.chain_iid == ordered_chain_iids[index])
        num_counted += 1

    # Ensure that all chain_iids were counted
    assert num_counted == len(ordered_chain_iids)


def test_sort_poly_then_non_poly():
    # Create a mock AtomArray with polymer and non-polymer chains
    elements = np.array(["C", "C", "C", "O", "N", "C", "C", "C"])
    chain_iid = np.array([0, 0, 0, 1, 1, 2, 2, 2])  # 0: polymer, 1: non-polymer, 2: polymer
    is_polymer = np.array([True, True, True, False, False, True, True, True])

    atom_array = struc.AtomArray(len(elements))
    atom_array.set_annotation("element", elements)
    atom_array.set_annotation("chain_iid", chain_iid)
    atom_array.set_annotation("is_polymer", is_polymer)

    # Add bonds in linear order
    bonds = struc.BondList(len(elements))
    for i in range(len(elements) - 1):
        bonds.add_bond(i, i + 1)
    atom_array.bonds = bonds

    # Sort the atom array
    sorted_atom_array = sort_poly_then_non_poly(atom_array)

    # Check the order of the sorted AtomArray
    sorted_chain_iid = sorted_atom_array.chain_iid
    sorted_is_polymer = sorted_atom_array.is_polymer

    # Check that the sorted atom array has the correct number of atoms
    assert len(sorted_atom_array) == len(atom_array)

    # Check that the first 6 atoms are from a polymer chain
    assert np.all(sorted_is_polymer[:6])

    # Check that the next 2 atoms are from a non-polymer chain
    assert not np.any(sorted_is_polymer[6:-1])

    # Check that the chain_iid annotations are correct
    assert np.array_equal(sorted_chain_iid, np.array([0, 0, 0, 2, 2, 2, 1, 1]))

    # Check that the bonds are still valid (aka no bonds were broken)
    assert len(sorted_atom_array.bonds.as_array()) == len(atom_array.bonds.as_array())


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=WARNING", __file__])
