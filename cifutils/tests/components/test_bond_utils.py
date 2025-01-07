import biotite.structure as struc
import numpy as np
import pytest

from cifutils.template import get_empty_ccd_template
from cifutils.utils.bonds import correct_formal_charges_for_specified_atoms, get_inferred_polymer_bonds
from cifutils.utils.ccd import get_chem_comp_leaving_atom_names

LEAVING_GROUP_TEST_CASES = {
    "ALA": {"N": ("H2",), "C": ("OXT", "HXT"), "OXT": ("HXT",)},
    "TYR": {"N": ("H2",), "C": ("OXT", "HXT"), "OXT": ("HXT",)},
}


@pytest.mark.parametrize("ccd_code, expected_leaving_groups", LEAVING_GROUP_TEST_CASES.items())
def test_leaving_group_computation(ccd_code, expected_leaving_groups):
    assert get_chem_comp_leaving_atom_names(ccd_code) == expected_leaving_groups


def test_fix_formal_charge_of_deprotonated_alanine():
    ala = get_empty_ccd_template("ALA", res_id=1, remove_hydrogens=False)
    assert np.array_equal(ala.charge, np.zeros(len(ala)))

    ala_oxt_deprotonated = ala[ala.atom_name != "HXT"]
    assert (
        correct_formal_charges_for_specified_atoms(ala_oxt_deprotonated, np.ones(len(ala) - 1, dtype=bool))[
            ala_oxt_deprotonated.atom_name == "OXT"
        ].charge
        == -1
    )


def test_infer_polymer_bonds():
    residues = []
    n_intra_bonds = []
    for i, ccd_code in enumerate(["ALA", "TYR", "GLY", "SER"]):
        residue = get_empty_ccd_template(ccd_code, res_id=i + 1, chain_id="A", remove_hydrogens=False)
        residues.append(residue)
        n_intra_bonds.append(len(residue.bonds.as_array()))
    atom_array = struc.concatenate(residues)

    assert np.array_equal(atom_array.charge, np.zeros(len(atom_array)))
    assert len(atom_array[atom_array.atom_name == "OXT"]) == 4
    assert sum(n_intra_bonds) == len(atom_array.bonds.as_array())

    # ... make polymer bonds
    polymer_bonds, leaving_atom_idxs = get_inferred_polymer_bonds(atom_array)
    assert len(polymer_bonds) == 3
    assert all(atom_array[leaving_atom_idxs].is_leaving_atom)

    # ... add those bonds to the atom array and remove the leaving atoms
    atom_array.bonds = atom_array.bonds.merge(struc.BondList(len(atom_array), polymer_bonds))
    is_leaving = np.zeros(len(atom_array), dtype=bool)
    is_leaving[leaving_atom_idxs] = True
    atom_array = atom_array[~is_leaving]

    assert len(atom_array[atom_array.atom_name == "OXT"]) == 1
    assert len(atom_array[atom_array.atom_name == "HXT"]) == 1
    assert len(atom_array[atom_array.atom_name == "H2"]) == 1

    # ... fix formal charges
    atom_array = correct_formal_charges_for_specified_atoms(atom_array, np.ones(len(atom_array), dtype=bool))
    assert np.array_equal(atom_array.charge, np.zeros(len(atom_array)))
