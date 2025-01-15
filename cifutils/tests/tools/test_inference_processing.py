import tempfile
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure import AtomArray
from conftest import get_pdb_path

from cifutils import parse
from cifutils.enums import ChainType
from cifutils.tools.fasta import split_generalized_fasta_sequence
from cifutils.tools.inference import (
    ChemicalComponent,
    Protein,
    SequenceComponent,
    SmilesComponent,
    components_to_atom_array,
    one_letter_to_ccd_code,
    read_chai_fasta,
)
from cifutils.utils.testing import assert_same_atom_array


@pytest.fixture
def dict_inputs():
    """Fixture providing example chemical components for testing."""
    monomer = [
        {
            "seq": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
            "chain_type": "polypeptide(l)",
            "chain_id": "A",
            "is_polymer": True,
        }
    ]

    dimer = [
        {
            "seq": "MRDTDVTVLGLGLMGQALAGAFLKDGHATTVWNRSEGKAGQLAEQGAVLASSARDAAEASPLVVVCVSDHAAVRAVLDPLGDVLAGRVLVNLTSGTSEQARATAEWAAERGITYLDGAIMAIPQVVGTADAFLLYSGPEAAYEAHEPTLRSLGAGTTYLGADHGLSSLYDVALLGIMWGTLNSFLHGAALLGTAKVEATTFAPFANRWIEAVTGFVSAYAGQVDQGAYPALDATIDTHVATVDHLIHESEAAGVNTELPRLVRTLADRALAGGQGGLGYAAMIEQFRSPSA",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
            "chain_id": "B",
        },
        {
            "seq": "MRDTDVTVLGLGLMGQALAGAFLKDGHATTVWNRSEGKAGQLAEQGAVLASSARDAAEASPLVVVCVSDHAAVRAVLDPLGDVLAGRVLVNLTSGTSEQARATAEWAAERGITYLDGAIMAIPQVVGTADAFLLYSGPEAAYEAHEPTLRSLGAGTTYLGADHGLSSLYDVALLGIMWGTLNSFLHGAALLGTAKVEATTFAPFANRWIEAVTGFVSAYAGQVDQGAYPALDATIDTHVATVDHLIHESEAAGVNTELPRLVRTLADRALAGGQGGLGYAAMIEQFRSPSA",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
            "chain_id": "C",
        },
    ]

    noncanonical = [
        {
            "seq": "KVFGRCE(SEP)AAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
        }
    ]

    ligand = [
        {
            "smiles": "O=C1OCC(=C1)C5C4(C(O)CC3C(CCC2CC(O)CCC23C)C4(O)CC5)C",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "E",
        }
    ]

    glycan_1 = [
        {
            "ccd_code": "NAG",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "F",
        }
    ]
    glycan_2 = [
        {
            "ccd_code": "NAG",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "G",
        }
    ]

    return {
        "monomer": monomer,
        "dimer": dimer,
        "noncanonical": noncanonical,
        "ligand": ligand,
        "glycan_1": glycan_1,
        "glycan_2": glycan_2,
    }


@pytest.fixture
def bonds():
    # Bond between the two NAG residues (O4 and C1 atoms) and NAG and the protein (ND2 on ASN and C1 on NAG)
    return [("F:NAG:1:O4", "G:NAG:1:C1"), ("A:ASN:19:ND2", "F:NAG:1:C1")]


@pytest.fixture
def chai_fasta_input():
    example_fasta_content = """
>protein|name=example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|name=example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|name=example-peptide
GAAL
>ligand|name=example-ligand-as-smiles
CCCCCCCCCCCCCC(=O)O
""".strip()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(example_fasta_content)
    fasta_path = Path(f.name)
    yield fasta_path
    fasta_path.unlink()


def test_split_generalized_fasta_sequence():
    """Test splitting sequences with and without parentheses."""
    assert split_generalized_fasta_sequence("ABC") == ["A", "B", "C"]
    assert split_generalized_fasta_sequence("ABC(DEF)GH") == ["A", "B", "C", "(DEF)", "G", "H"]
    assert split_generalized_fasta_sequence("A(XYZ)C(DEF)") == ["A", "(XYZ)", "C", "(DEF)"]


def test_one_letter_to_ccd_code():
    """Test conversion of one-letter codes to CCD codes."""
    # Test standard amino acids
    seq = ["A", "C", "D"]
    result = one_letter_to_ccd_code(seq, ChainType.POLYPEPTIDE_L)
    assert result == ["ALA", "CYS", "ASP"]

    # Test with modified residue
    seq = ["A", "(SEP)", "G"]
    result = one_letter_to_ccd_code(seq, ChainType.POLYPEPTIDE_L)
    assert result == ["ALA", "SEP", "GLY"]

    # Test invalid modified residue
    with pytest.raises(ValueError):
        one_letter_to_ccd_code(["A", "(THISDOESNOTEXIST)"], ChainType.POLYPEPTIDE_L)


def test_components_to_atom_array_monomer(dict_inputs):
    """Test conversion of monomer components to AtomArray."""
    atom_array = components_to_atom_array(dict_inputs["monomer"])

    assert isinstance(atom_array, AtomArray)
    assert len(atom_array) > 0

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 1
    assert "A" in chain_ids
    print(chain_ids)

    # Verify polymer annotation
    assert np.all(atom_array.is_polymer)
    assert np.all(atom_array.chain_type == ChainType.POLYPEPTIDE_L)


def test_components_to_atom_array_dimer(dict_inputs):
    """Test conversion of dimer components to AtomArray."""
    atom_array = components_to_atom_array(dict_inputs["dimer"])

    assert isinstance(atom_array, AtomArray)

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 2
    assert "B" in chain_ids
    assert "C" in chain_ids

    # Verify polymer annotation
    assert np.all(atom_array.is_polymer)
    assert np.all(atom_array.chain_type == ChainType.POLYPEPTIDE_L)


def test_components_to_atom_array_noncanonical(dict_inputs):
    """Test conversion of components with non-canonical residues to AtomArray."""
    atom_array = components_to_atom_array(dict_inputs["noncanonical"])

    assert isinstance(atom_array, AtomArray)

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 1
    assert "A" in chain_ids

    # Verify SEP residue is present
    res_names = np.unique(atom_array.res_name)
    assert "SEP" in res_names


def test_components_to_atom_array_ligand(dict_inputs):
    """Test conversion of ligand components to AtomArray."""
    atom_array = components_to_atom_array(dict_inputs["ligand"])

    assert isinstance(atom_array, AtomArray)

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 1
    assert "E" in chain_ids

    # Verify non-polymer annotation
    assert not np.any(atom_array.is_polymer)
    assert np.all(atom_array.chain_type == ChainType.NON_POLYMER)

    # Assert that all elements are letters, not string representations of numbers
    assert all(s.isupper() for s in atom_array.element)


def test_components_to_atom_array_glycan(dict_inputs, bonds):
    """Test conversion of ligand components to AtomArray."""
    components = dict_inputs["glycan_1"] + dict_inputs["glycan_2"] + dict_inputs["monomer"]
    atom_array = components_to_atom_array(components, bonds=bonds)

    assert isinstance(atom_array, AtomArray)

    # Check only one molecule ID (e.g., all atoms are connected via bonds)
    assert len(np.unique(atom_array.molecule_id)) == 1

    # Check two PN units
    assert len(np.unique(atom_array.pn_unit_id)) == 2

    # Check chain IDs
    chain_ids = np.unique(atom_array.chain_id)
    assert len(chain_ids) == 3


def test_chemical_component_from_dict():
    """Test creation of chemical components from dictionaries."""
    # Test sequence component
    seq_dict = {"seq": "ACDEF", "chain_type": "polypeptide(l)", "is_polymer": True}
    comp = ChemicalComponent.from_dict(seq_dict)
    assert isinstance(comp, SequenceComponent)

    # Test SMILES component
    smiles_dict = {"smiles": "CCCCCCCCCCCCCC(=O)O", "chain_type": "non-polymer", "is_polymer": False}
    comp = ChemicalComponent.from_dict(smiles_dict)
    assert isinstance(comp, SmilesComponent)

    # Test invalid component
    with pytest.raises(ValueError):
        ChemicalComponent.from_dict({"invalid": "component"})


def test_read_chai_fasta(chai_fasta_input):
    """Test reading components from FASTA file."""
    components = read_chai_fasta(chai_fasta_input)

    assert len(components) == 4
    assert isinstance(components[0], Protein)
    assert isinstance(components[1], Protein)
    assert isinstance(components[2], Protein)
    assert isinstance(components[3], SmilesComponent)

    # Check the ligand
    assert components[3].smiles == "CCCCCCCCCCCCCC(=O)O"
    assert components[3].chain_type == ChainType.NON_POLYMER
    assert components[1].chain_type == ChainType.POLYPEPTIDE_L
    assert not components[2].is_polymer
    assert not components[3].is_polymer


def test_sequence_component_validation():
    """Test validation of sequence components."""
    # Test invalid polymer flag for SMILES
    with pytest.raises(ValueError):
        SmilesComponent(smiles="CCCC", is_polymer=True)

    # Test invalid chain type for SMILES
    with pytest.raises(ValueError):
        SmilesComponent(smiles="CCCC", chain_type=ChainType.POLYPEPTIDE_L)

    with pytest.raises(ValueError):
        SequenceComponent(seq="(MET)ACGT", chain_type=ChainType.RNA, is_polymer=True)

    with pytest.raises(ValueError):
        SequenceComponent(seq="(DA)MKL", chain_type=ChainType.POLYPEPTIDE_L, is_polymer=True)


def test_full_chai_input(chai_fasta_input):
    components = read_chai_fasta(chai_fasta_input)
    atom_array = components_to_atom_array(components)

    assert isinstance(atom_array, AtomArray)
    assert np.unique(atom_array.chain_id).shape[0] == 4


def test_full_components_input(dict_inputs):
    components = sum(dict_inputs.values(), start=[])
    atom_array = components_to_atom_array(components)

    assert isinstance(atom_array, AtomArray)
    assert np.unique(atom_array.chain_id).shape[0] == 7  # 1 monomer, 2 dimers, 1 noncanonical, 1 ligand, 2 glycans
    assert set(np.unique(atom_array.chain_id)) == {"A", "B", "C", "D", "E", "F", "G"}
    assert set(np.unique(atom_array.chain_type)) == {ChainType.POLYPEPTIDE_L, ChainType.NON_POLYMER}

    # Assert full occupancy
    assert np.all(atom_array.occupancy == 1.0)


def test_recover_bonds_from_cif(dict_inputs):
    data = parse(
        "tests/data/test_unl_ligand_with_bonds.cif",
        fix_ligands_at_symmetry_centers=False,
    )
    atom_array = data["asym_unit"][0]

    assert all(atom_array.res_name == "UNL")
    assert len(atom_array) == 28
    assert atom_array.bonds.as_array().shape[0] == 32


def test_same_atom_array_from_cif_and_inference():
    """Tests if the bonds inferred from the components are the same as the bonds in the CIF file."""
    transformation_id = "1"
    data = parse(get_pdb_path("7rxs"), remove_hydrogens=True)
    atom_array_from_cif = data["assemblies"][transformation_id][0]

    # ... extract the sequence and build inference input
    monomer = [
        {
            "seq": data["chain_info"]["A"]["unprocessed_entity_non_canonical_sequence"],
            "chain_type": data["chain_info"]["A"]["chain_type"],
            "chain_id": "A",
            "is_polymer": data["chain_info"]["A"]["is_polymer"],
        }
    ]
    ligand = [
        {
            "smiles": "Cc1cc(cc(c1)Oc2nccc(n2)c3c(ncn3[C@H]4CCN(C4)CCN)c5ccc(cc5)I)C",
            "chain_type": "non-polymer",
            "is_polymer": False,
            "chain_id": "C",
        }
    ]

    atom_array_from_inference = components_to_atom_array(monomer + ligand)

    for chain_id in np.unique(atom_array_from_cif.chain_id):
        chain_atom_array_from_inference = atom_array_from_inference[(atom_array_from_inference.chain_id == chain_id)]
        chain_atom_array_from_cif = atom_array_from_cif[atom_array_from_cif.chain_id == chain_id]

        # Inference should have full occupancy and null b_factor
        assert np.all(chain_atom_array_from_inference.occupancy == 1.0)
        assert np.all(np.isnan(chain_atom_array_from_inference.b_factor))

        # Assert same atom array
        annotations_to_compare = list(
            set(chain_atom_array_from_cif.get_annotation_categories())
            - {
                "occupancy",
                "b_factor",
                "is_aromatic",
                "alt_atom_id",
                "molecule_id",  # The molecule_id and all entity annotations may differ between the two
                "molecule_iid",
                "molecule_entity",
                "pn_unit_entity",
            }
        )
        is_ligand = not chain_atom_array_from_cif.is_polymer[0]

        if is_ligand:
            # Renumber residues to be 0-indexed for non-polymers, 1-indexed for polymers
            chain_atom_array_from_cif.res_id = struc.spread_residue_wise(
                chain_atom_array_from_cif, np.arange(struc.get_residue_count(chain_atom_array_from_cif))
            )
            # ... and don't compare residue names, atom names, stereo annotations, as that won't work for UNL
            for item in ["res_name", "atom_name", "stereo"]:
                annotations_to_compare.remove(item)
        else:
            chain_atom_array_from_cif.res_id = struc.spread_residue_wise(
                chain_atom_array_from_cif, np.arange(1, struc.get_residue_count(chain_atom_array_from_cif) + 1)
            )

        assert_same_atom_array(
            chain_atom_array_from_inference,
            chain_atom_array_from_cif,
            compare_coords=False,
            compare_bonds=True,
            annotations_to_compare=annotations_to_compare,
            enforce_order=False,
            compare_bond_order=False,
        )


if __name__ == "__main__":
    pytest.main([__file__])
