import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray

from datahub.datasets.dataframe_parsers import PNUnitsDFParser, load_example_from_metadata_row
from datahub.preprocessing.constants import SUPPORTED_CHAIN_TYPES
from datahub.transforms.base import Compose
from datahub.transforms.filters import (
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
    RemoveUnsupportedChainTypes,
)
from tests.conftest import CIF_PARSER, PN_UNITS_DF, cached_parse


@pytest.mark.parametrize("test_case", [{"pdb_id": "1s2k"}])
def test_remove_polymers_with_too_few_resolved_residues(test_case):
    # ...load the example from the CIF parser
    data = cached_parse(test_case["pdb_id"])
    atom_array = data["atom_array"]

    MIN_RESIDUES = 4

    def get_min_num_residues(atom_array: AtomArray) -> int:
        """
        Calculate the minimum number of unique residues in each polymer chain.
        """
        unique_chain_iids = np.unique(atom_array.chain_iid[atom_array.is_polymer])
        min_num_residues = np.min(
            [len(np.unique(atom_array.res_id[atom_array.chain_iid == chain_iid])) for chain_iid in unique_chain_iids]
        )
        return min_num_residues

    # ...assert that we have a polymer with too few resolved residues
    min_num_residues = get_min_num_residues(atom_array)
    assert min_num_residues < MIN_RESIDUES

    pipeline = Compose(
        [
            RemovePolymersWithTooFewResolvedResidues(min_residues=MIN_RESIDUES),
        ],
        track_rng_state=False,
    )
    output = pipeline(data)
    output_atom_array = output["atom_array"]

    # ...assert that the polymer with too few resolved residues has been removed
    min_num_residues = get_min_num_residues(output_atom_array)
    assert min_num_residues >= MIN_RESIDUES


def test_remove_terminal_oxygen():
    atom_array = struc.info.residue("ILE")
    assert "OXT" in atom_array.atom_name

    data = {"atom_array": atom_array}

    transform = RemoveTerminalOxygen()
    data = transform(data)

    atom_array_new = data["atom_array"]
    assert "OXT" not in atom_array_new.atom_name
    assert len(atom_array_new) == len(atom_array) - 1


@pytest.mark.parametrize("pdb_id", ["4gqa"])
def test_remove_unresolved_pn_units(pdb_id):
    # ...load the example from the CIF parser
    data = cached_parse(pdb_id)
    data["atom_array"] = data["assemblies"]["1"][0]

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


UNSUPPORTED_CHAIN_TYPE_TEST_CASES = [
    "104D",  # DNA/RNA Hybrid
    "5X3O",  # polypeptide(D)
]


@pytest.mark.parametrize("pdb_id", UNSUPPORTED_CHAIN_TYPE_TEST_CASES)
def test_remove_unsupported_chain_types(pdb_id: str):
    rows = PN_UNITS_DF[
        (PN_UNITS_DF["pdb_id"] == pdb_id.lower()) & (PN_UNITS_DF["assembly_id"] == "1")
    ]  # We only need the first assembly for UNSUPPORTED_CHAIN_TYPE_TEST_CASES

    assert not rows.empty

    for _, row in rows.iterrows():
        data = load_example_from_metadata_row(row, PNUnitsDFParser(), cif_parser=CIF_PARSER)
        is_unsupported_type = row["q_pn_unit_type"] not in SUPPORTED_CHAIN_TYPES
        original_atom_array = data["atom_array"].copy()

        # Apply transforms
        # fmt: off
        pipeline = Compose([
            RemoveUnsupportedChainTypes(),
        ], track_rng_state=False)
        # fmt: on

        output = None
        if is_unsupported_type:
            with pytest.raises(AssertionError):
                output = pipeline(data)
        else:
            output = pipeline(data)

        if output:
            atom_array = output["atom_array"]
            num_unsupported_atoms = len(original_atom_array) - len(atom_array)
            assert num_unsupported_atoms > 0, "There should be some atoms removed"
            chain_types = np.unique(atom_array.chain_type)
            assert np.all(np.isin(chain_types, SUPPORTED_CHAIN_TYPES)), "All remaining chain types should be supported"


def test_handle_undesired_res_single():
    transform = HandleUndesiredResTokens(["PTR", "SEP", "SO4", "NH2"])

    for with_hydrogens in (True, False):
        # Case 1:
        res = struc.info.residue("ALA")
        res.set_annotation("is_polymer", np.ones(res.array_length(), dtype=bool))
        res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
        res.set_annotation("chain_type", 6 * np.ones(res.array_length(), dtype=int))

        if not with_hydrogens:
            res = res[res.element != "H"]

        res_out = transform({"atom_array": res})["atom_array"]
        assert np.all(res.coord == res_out.coord)
        assert np.all(res.atom_name == res_out.atom_name)
        assert np.all(res.is_polymer == np.ones(res.array_length(), dtype=bool))

    # Case 2:
    res = struc.info.residue("PTR")
    res.set_annotation("is_polymer", np.ones(res.array_length(), dtype=bool))
    res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
    res.set_annotation("chain_type", 6 * np.ones(res.array_length(), dtype=int))
    res_out_target = struc.info.residue("TYR")

    res = res[res.element != "H"]
    res_out_target = res_out_target[res_out_target.element != "H"]

    res_out = transform({"atom_array": res})["atom_array"]
    assert np.all(res_out.res_name == "TYR")
    assert np.all(res_out.is_polymer == np.ones(res_out.array_length(), dtype=bool))
    assert np.all(res_out.coord.shape == res_out_target.coord.shape)
    assert np.all(res_out.coord == res.coord[np.isin(res.atom_name, res_out_target.atom_name)])

    # Case 3:
    res = struc.info.residue("SEP")
    res.set_annotation("is_polymer", np.ones(res.array_length(), dtype=bool))
    res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
    res.set_annotation("chain_type", 6 * np.ones(res.array_length(), dtype=int))
    res_out_target = struc.info.residue("SER")

    res = res[res.element != "H"]
    res_out_target = res_out_target[res_out_target.element != "H"]

    res_out = transform({"atom_array": res})["atom_array"]
    assert np.all(res_out.res_name == "SER")
    assert np.all(res_out.is_polymer == np.ones(res_out.array_length(), dtype=bool))
    assert np.all(res_out.coord.shape == res_out_target.coord.shape)
    assert np.all(res_out.coord == res.coord[np.isin(res.atom_name, res_out_target.atom_name)])

    # Case 4 (atomize polymer bits that cannot be mapped to a canonical or unknown residue)
    res = struc.info.residue("NH2")
    res.set_annotation("is_polymer", np.ones(res.array_length(), dtype=bool))
    res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
    res.set_annotation("chain_type", 6 * np.ones(res.array_length(), dtype=int))

    res = res[res.element != "H"]

    res_out = transform({"atom_array": res})["atom_array"]
    assert np.all(res_out.res_name == "NH2")
    assert len(res_out) == 1
    assert np.all(res_out.atomize == 1)

    # Case 5 (remove non-polymer bits)
    res = struc.info.residue("SO4")
    res.set_annotation("is_polymer", np.zeros(res.array_length(), dtype=bool))
    res.set_annotation("pn_unit_iid", np.full(res.array_length(), -1, dtype=int))
    res.set_annotation("chain_type", 8 * np.ones(res.array_length(), dtype=int))

    res = res[res.element != "H"]

    res_out = transform({"atom_array": res})["atom_array"]
    assert len(res_out) == 0


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
