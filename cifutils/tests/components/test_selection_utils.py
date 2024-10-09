from cifutils.utils.selection_utils import get_residue_starts
import numpy as np
import biotite.structure as struc


def test_get_residue_starts():
    # fmt: off
    atom_array = struc.array([
        struc.Atom(np.array([44.869,     8.188,    36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  element="7",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([45.024,     7.456,    34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([44.142,     6.714,    34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", element="8",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.669,     8.171,    36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.812,     8.982,    38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.152,     8.296,    39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.479,     9.3  ,    40.792 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="SD", element="16", charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.232,     8.184,    42.102 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CE", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.46 ,     8.724,    36.151 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="C",  element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.339,     9.907,    35.831 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O",  element="8",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([58.656483, 34.763695, 36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  element="7",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.212917, 35.263927, 34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", element="6",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([60.296505, 34.87109 , 34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", element="8",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.271206, 33.732964, 36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", element="6",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([58.49736 , 33.451305, 38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", element="6",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.42145 , 33.22273 , 39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", element="6",  charge=0,  transformation_id="2"),
    ])
    # fmt: on

    assert len(get_residue_starts(atom_array)) == 2
