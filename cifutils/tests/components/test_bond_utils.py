import numpy as np

from cifutils.template import get_empty_ccd_template
from cifutils.utils.bond_utils import fix_formal_charges


def test_fix_formal_charges():
    ala = get_empty_ccd_template("ALA", res_id=1, remove_hydrogens=False)
    assert np.array_equal(ala.charge, np.zeros(len(ala)))

    ala_oxt_deprotonated = ala[ala.atom_name != "HXT"]
    assert (
        fix_formal_charges(ala_oxt_deprotonated, np.ones(len(ala) - 1, dtype=bool))[
            ala_oxt_deprotonated.atom_name == "OXT"
        ].charge
        == -1
    )
