import biotite.structure as struc
import numpy as np

from datahub.transforms.atom_array import HandleUndesiredResTokens


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
