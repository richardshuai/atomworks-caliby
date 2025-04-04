import io
import logging
import os
from abc import ABC
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import AllChem

import cifutils.transforms.atom_array as ta
from cifutils import parse
from cifutils.common import KeyToIntMapper
from cifutils.constants import (
    CCD_MIRROR_PATH,
    STANDARD_AA_ONE_LETTER,
    STANDARD_DNA_ONE_LETTER,
    STANDARD_RNA,
    UNKNOWN_LIGAND,
)
from cifutils.enums import ChainType, ChainTypeInfo
from cifutils.template import build_template_atom_array
from cifutils.tools.fasta import one_letter_to_ccd_code, split_generalized_fasta_sequence
from cifutils.utils.bonds import (
    correct_formal_charges_for_specified_atoms,
    get_inferred_polymer_bonds,
    get_struct_conn_bonds,
    hash_atom_array,
    spoof_struct_conn_dict_from_string,
)
from cifutils.utils.ccd import (
    atom_array_from_ccd_code,
    check_ccd_codes_are_available,
    get_chain_type_from_ccd_code,
    get_chem_comp_type,
)
from cifutils.utils.chain import create_chain_id_generator

logger = logging.getLogger("cifutils")


class ChemicalComponent(ABC):  # noqa: B024
    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(args_dict: dict) -> "ChemicalComponent":
        if "seq" in args_dict:
            return SequenceComponent(**args_dict)
        elif "smiles" in args_dict:
            return SmilesComponent(**args_dict)
        elif "path" in args_dict and args_dict["path"].endswith(".sdf"):
            return SDFComponent(**args_dict)
        elif "path" in args_dict and args_dict["path"].endswith(".cif"):
            return CIFFileComponent(**args_dict)
        elif "ccd_code" in args_dict:
            return CCDComponent(**args_dict)
        else:
            raise ValueError(f"Unknown chemical component type: {args_dict=}")


@dataclass
class SequenceComponent(ChemicalComponent):
    seq: str | list[str]
    chain_type: ChainType | None = None
    is_polymer: bool | None = None
    chain_id: str | None = None
    msa_path: os.PathLike | None = None

    @staticmethod
    def infer_chain_type(seq: str) -> ChainType:
        if isinstance(seq, str):
            seq = split_generalized_fasta_sequence(seq)

        hits = Counter()
        for letter in seq:
            if letter in Protein._valid_one_letter_codes():
                hits["protein"] += 1
            if letter in DNA._valid_one_letter_codes():
                hits["dna"] += 1
            if letter in RNA._valid_one_letter_codes():
                hits["rna"] += 1
            if letter.startswith("("):
                hits["unknown"] += 1

        # Heuristics:
        # If the sequence contains more protein hits than DNA or RNA hits, it's probably a protein
        if hits["protein"] > hits["dna"] and hits["protein"] > hits["rna"]:
            return ChainType.POLYPEPTIDE_L

        # Else, if the sequence is all RNA hits, it's probably RNA
        elif hits["rna"] == len(seq):
            return ChainType.RNA

        # Else, if the sequence is all DNA hits, it's probably DNA
        elif hits["dna"] == len(seq):
            return ChainType.DNA

        raise ValueError(f"Could not infer chain type from sequence: {seq=}")

    @staticmethod
    def assert_valid_chain_type(seq: list[str], chain_type: ChainType) -> bool:
        ccd_codes = set(seq)
        chem_comp_types = {get_chem_comp_type(ccd_code) for ccd_code in ccd_codes}

        valid_chem_comp_types = ChainTypeInfo.VALID_CHEM_COMP_TYPES.get(chain_type, chem_comp_types)
        if not chem_comp_types.issubset(valid_chem_comp_types):
            raise ValueError(f"Invalid {chain_type=} for {chem_comp_types=}. Valid are {valid_chem_comp_types=}")

    @staticmethod
    def from_seq(
        seq: str | list[str], *, chain_type: ChainType | str = None, is_polymer: bool | None = None
    ) -> "SequenceComponent":
        chain_type = chain_type or SequenceComponent.infer_chain_type(seq)
        is_polymer = is_polymer or chain_type in ChainType.get_polymers()

        if chain_type in ChainTypeInfo.PROTEINS:
            return Protein(seq=seq, chain_type=chain_type, is_polymer=is_polymer)
        elif chain_type == ChainType.RNA:
            return RNA(seq=seq, chain_type=chain_type, is_polymer=is_polymer)
        elif chain_type == ChainType.DNA:
            return DNA(seq=seq, chain_type=chain_type, is_polymer=is_polymer)
        else:
            return SequenceComponent(seq=seq, chain_type=chain_type, is_polymer=is_polymer)

    def __post_init__(self):
        # If the chain type is not provided, infer it from the sequence
        self.chain_type = self.chain_type or SequenceComponent.infer_chain_type(self.seq)
        self.chain_type = ChainType.as_enum(self.chain_type)

        # If the is_polymer is not provided, infer it from the sequence
        self.is_polymer = self.is_polymer or self.chain_type.is_polymer()

        # If the sequence is a string, split it into a list of one-letter codes
        if isinstance(self.seq, str):
            self.seq = split_generalized_fasta_sequence(self.seq)

        # Process sequence into CCD codes
        if isinstance(self.seq, str):
            self.seq = split_generalized_fasta_sequence(self.seq)

        self.seq = one_letter_to_ccd_code(self.seq, self.chain_type)

        # Validate chain type
        SequenceComponent.assert_valid_chain_type(self.seq, self.chain_type)


@dataclass
class LigandComponent(ChemicalComponent):
    def __post_init__(self):
        self.chain_type = ChainType.as_enum(self.chain_type)

        if self.is_polymer:
            raise ValueError(f"{self.__class__.__name__} must have 'is_polymer=False'")

        if self.chain_type != ChainType.NON_POLYMER:
            raise ValueError(f"{self.__class__.__name__} must have 'chain_type=ChainType.NON_POLYMER'")


@dataclass
class CCDComponent(LigandComponent):
    ccd_code: str
    chain_type: ChainType | str = "non-polymer"
    is_polymer: bool = False
    chain_id: str | None = None


@dataclass
class SmilesComponent(LigandComponent):
    smiles: str
    chain_type: ChainType | str = "non-polymer"
    is_polymer: bool = False
    chain_id: str | None = None
    res_name: str = UNKNOWN_LIGAND


@dataclass
class SDFComponent(LigandComponent):
    path: os.PathLike | io.StringIO
    chain_type: ChainType | str = "non-polymer"
    is_polymer: bool = False
    chain_id: str | None = None
    res_name: str = UNKNOWN_LIGAND


@dataclass
class CIFFileComponent(ChemicalComponent):
    path: os.PathLike | io.StringIO
    msa_paths: dict[str, os.PathLike] | None = None
    assembly: int = "1"
    model: int = 0

    def __post_init__(self):
        # Parse the file to set the AtomArray and extract the chain IDs
        atom_array = parse(self.path, add_missing_atoms=False)["assemblies"][self.assembly][self.model]
        self.chain_ids = np.unique(atom_array.chain_id)
        self.atom_array = atom_array


@dataclass
class Polymer(SequenceComponent):
    is_polymer: bool = True


@dataclass
class Protein(SequenceComponent):
    chain_type: ChainType = ChainType.POLYPEPTIDE_L

    @staticmethod
    def _valid_one_letter_codes() -> set[str]:
        return set(STANDARD_AA_ONE_LETTER)


@dataclass
class RNA(SequenceComponent):
    chain_type: ChainType = ChainType.RNA

    @staticmethod
    def _valid_one_letter_codes() -> set[str]:
        return set(STANDARD_RNA)


@dataclass
class DNA(SequenceComponent):
    chain_type: ChainType = ChainType.DNA

    @staticmethod
    def _valid_one_letter_codes() -> set[str]:
        return set(STANDARD_DNA_ONE_LETTER)


@dataclass
class Peptide(SequenceComponent):
    chain_type: ChainType = ChainType.POLYPEPTIDE_L
    is_polymer: bool = False


def read_chai_fasta(fasta_path: Path) -> list[ChemicalComponent]:
    from biotite.sequence.io.fasta import FastaFile

    fasta = FastaFile.read(fasta_path)

    components = []
    for metadata, content in fasta.items():
        metadata = metadata.lower()
        if metadata.startswith("ligand"):
            components.append(SmilesComponent(smiles=content))
        elif metadata.endswith(".sdf"):
            components.append(sdf_to_annotated_atom_array(path=content))
        else:
            if "protein" in metadata:
                components.append(Protein(seq=content))
            elif "rna" in metadata:
                components.append(RNA(seq=content))
            elif "dna" in metadata:
                components.append(DNA(seq=content))
            elif "peptide" in metadata:
                components.append(Peptide(seq=content))
            else:
                components.append(SequenceComponent.from_seq(content))
    return components


def sequence_to_annotated_atom_array(
    seq: list[str],
    chain_id: str,
    *,
    chain_type: ChainType | str = None,
    is_polymer: bool | None = None,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    **kwargs,
) -> AtomArray:
    if isinstance(seq, str) and is_polymer:
        seq = one_letter_to_ccd_code(split_generalized_fasta_sequence(seq), chain_type=chain_type)

    # Turn the sequence into a numpy array
    seq = np.asarray(seq)

    chain_type = chain_type or SequenceComponent.infer_chain_type(seq)
    chain_type = ChainType.as_enum(chain_type)
    is_polymer = is_polymer or chain_type.is_polymer()

    # Ensure that the sequence is a valid combination of existing 3-letter CCD codes
    ccd_codes_in_seq = set(seq)
    if UNKNOWN_LIGAND in ccd_codes_in_seq:
        raise ValueError(
            f"Unknown ligand `{UNKNOWN_LIGAND}` found in sequence. If you want to pass a ligand, that "
            f"is not in the CCD, use a SMILES string or SDF file instead."
        )
    check_ccd_codes_are_available(ccd_codes_in_seq, ccd_mirror_path=ccd_mirror_path, mode="raise")

    # ... create a list of atoms based on the reference CCD entries
    atom_array = build_template_atom_array(
        chain_info_dict={
            chain_id: {
                "res_name": seq,
                "res_id": np.arange(1, len(seq) + 1),
                "chain_type": chain_type,
                "is_polymer": is_polymer,
            }
        },
        atom_array=None,
        remove_hydrogens=False,  # we keep hydrogens here, to allow fixing formal charges
        use_ccd_charges=True,
        ccd_mirror_path=ccd_mirror_path,
    )

    # ... add the atomic number annotation (vs. element, which is a string)
    atom_array = ta.add_atomic_number_annotation(atom_array)

    # Compute bonds and leaving groups
    n_atoms = atom_array.array_length()
    polymer_bonds, polymer_bonds_leaving_atoms = get_inferred_polymer_bonds(atom_array)
    polymer_bonds = struc.BondList(n_atoms, polymer_bonds)
    # ... add bonds to the atom array
    atom_array.bonds = atom_array.bonds.merge(polymer_bonds)
    # ... remove the leaving groups
    atom_array = atom_array[np.setdiff1d(np.arange(n_atoms), polymer_bonds_leaving_atoms)]

    # ... remove index annotation and leaving group annotations
    _annotations_to_remove = (
        "is_n_terminal_atom",
        "is_c_terminal_atom",
        "is_leaving_atom",
    )
    for annotation in _annotations_to_remove:
        atom_array.del_annotation(annotation)

    # Add custom annotations
    atom_array.set_annotation("occupancy", np.ones(atom_array.array_length()))
    atom_array.set_annotation("is_polymer", np.full(atom_array.array_length(), is_polymer))
    atom_array.set_annotation("chain_type", np.full(atom_array.array_length(), chain_type))
    atom_array.set_annotation("b_factor", np.full(atom_array.array_length(), np.nan))

    return atom_array


def smiles_to_annotated_atom_array(
    smiles: str,
    chain_id: str,
    *,
    chain_type: ChainType | str = "non-polymer",
    is_polymer: bool = False,
    backend: Literal["openbabel", "rdkit"] = "rdkit",
    res_name: str = UNKNOWN_LIGAND,
) -> AtomArray:
    if backend == "rdkit":
        from cifutils.tools.rdkit import atom_array_from_rdkit, smiles_to_rdkit

        mol = smiles_to_rdkit(smiles)
        try:
            # ... generate a conformer to keep the stereochemistry encoded in the SMILES
            #   NOTE: This may stall for 40ish seconds for some difficult molecules like HEM
            #   TODO: Migrate the timeout utils to cifutils so we can timeout here.
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.maxAttempts = 1
            AllChem.EmbedMultipleConfs(mol, numConfs=1, params=params)
        except Exception:
            pass

        array = atom_array_from_rdkit(mol)
    elif backend == "openbabel":
        raise NotImplementedError("Openbabel backend not yet implemented.")
    else:
        raise ValueError(f"Unknown backend: {backend=}")

    # Update annotations
    array.set_annotation("occupancy", np.ones(array.array_length()))
    array.set_annotation("hetero", np.full(array.array_length(), True))
    array.set_annotation("res_name", np.full(array.array_length(), res_name))
    array.set_annotation("chain_id", np.full(array.array_length(), chain_id))
    array.set_annotation("is_polymer", np.full(array.array_length(), is_polymer))
    array.set_annotation("chain_type", np.full(array.array_length(), ChainType.as_enum(chain_type)))
    array.set_annotation("b_factor", np.full(array.array_length(), np.nan))
    array.set_annotation("stereo", np.full(array.array_length(), "N"))
    array.set_annotation("is_backbone_atom", np.full(array.array_length(), False))

    return array


def sdf_to_annotated_atom_array(
    path: io.StringIO | os.PathLike,
    chain_id: str,
    *,
    chain_type: ChainType | str = "non-polymer",
    is_polymer: bool = False,
    res_name: str = UNKNOWN_LIGAND,
    backend: Literal["openbabel", "rdkit"] = "rdkit",
) -> AtomArray:
    if backend == "rdkit":
        from cifutils.tools.rdkit import atom_array_from_rdkit, sdf_to_rdkit

        mol = sdf_to_rdkit(path)
        array = atom_array_from_rdkit(mol)
    elif backend == "openbabel":
        raise NotImplementedError("Openbabel backend not yet implemented.")
    else:
        raise ValueError(f"Unknown backend: {backend=}")

    # Update annotations
    array.set_annotation("occupancy", np.ones(array.array_length()))
    array.set_annotation("hetero", np.full(array.array_length(), True))
    array.set_annotation("res_name", np.full(array.array_length(), res_name))
    array.set_annotation("chain_id", np.full(array.array_length(), chain_id))
    array.set_annotation("is_polymer", np.full(array.array_length(), is_polymer))
    array.set_annotation("chain_type", np.full(array.array_length(), ChainType.as_enum(chain_type)))
    array.set_annotation("b_factor", np.full(array.array_length(), np.nan))
    array.set_annotation("stereo", np.full(array.array_length(), "N"))
    array.set_annotation("is_backbone_atom", np.full(array.array_length(), False))
    return array


def ccd_code_to_annotated_atom_array(
    ccd_code: list[str],
    chain_id: str,
    *,
    chain_type: ChainType | str = None,
    is_polymer: bool | None = None,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> AtomArray:
    check_ccd_codes_are_available([ccd_code], ccd_mirror_path=ccd_mirror_path, mode="raise")

    # ... build the atom array
    array = atom_array_from_ccd_code(ccd_code)

    # ... set or infer chain type
    chain_type = chain_type or get_chain_type_from_ccd_code(ccd_code)
    is_polymer = is_polymer or chain_type.is_polymer()

    # ... update annotations
    array.set_annotation("occupancy", np.ones(array.array_length()))
    array.set_annotation("hetero", np.full(array.array_length(), True))
    array.set_annotation("res_name", np.full(array.array_length(), ccd_code))
    array.set_annotation("chain_id", np.full(array.array_length(), chain_id))
    array.set_annotation("is_polymer", np.full(array.array_length(), is_polymer))
    array.set_annotation("chain_type", np.full(array.array_length(), ChainType.as_enum(chain_type)))

    return array


def assign_res_name_from_atom_array_hash(atom_array: AtomArray, hash_to_id: KeyToIntMapper) -> AtomArray:
    """Assigns a residue name to an array based on its hash.

    The residue names will be assigned as `L:{id}` where `id` is a unique integer assigned to each hash.

    Args:
        ligand_array (AtomArray): The ligand array to assign a residue name to.
        ligand_hash_to_id (KeyToIntMapper): A mapper from ligand hash to ligand ID.
    """
    ligand_hash = hash_atom_array(atom_array, annotations=["element", "atom_name"], bond_order=True)
    ligand_id = hash_to_id(ligand_hash)
    atom_array.res_name = np.full(atom_array.array_length(), f"L:{ligand_id}")
    return atom_array


def add_inference_iid_id_entity_annotations(atom_array: AtomArray) -> AtomArray:
    # ... annotate ids and entities
    atom_array = ta.add_id_and_entity_annotations(atom_array)

    # ... annotate iids (assumes we are only given the asym)
    atom_array.set_annotation("chain_iid", np.char.add(atom_array.chain_id, "_1"))
    atom_array.set_annotation("pn_unit_iid", np.char.add(atom_array.pn_unit_id, "_1"))
    atom_array.set_annotation("molecule_iid", atom_array.molecule_id)

    # ... set transformation ID to string "1"
    atom_array.set_annotation("transformation_id", np.full(atom_array.array_length(), "1"))

    return atom_array


def build_msa_paths_by_chain_id_from_component_list(components: list[ChemicalComponent]) -> dict[str, os.PathLike]:
    """Build a dictionary of MSA paths by chain ID from a list of ChemicalComponent objects.

    The composed dictionary may be encoded as extra metadata in the CIF file, and ultimately loaded
    into `chain_info` through `parse`.
    """
    msa_paths_by_chain_id = {}
    for component in components:
        if hasattr(component, "msa_path") and component.msa_path is not None:
            msa_paths_by_chain_id[component.chain_id] = component.msa_path
        elif hasattr(component, "msa_paths") and component.msa_paths is not None:
            for chain_id, msa_path in component.msa_paths.items():
                msa_paths_by_chain_id[chain_id] = msa_path

    return msa_paths_by_chain_id


def components_to_atom_array(
    components: list[ChemicalComponent | dict],
    bonds: list[str] | None = None,
    return_components: bool = False,
) -> AtomArray | list[ChemicalComponent]:
    """Build an AtomArray from a list of ChemicalComponent objects and, optionally, a list of bonds.

    Args:
        components (list[ChemicalComponent | dict]): List of ChemicalComponent objects or dictionaries that can be
            converted to ChemicalComponent objects using ChemicalComponent.from_dict().
        bonds (list[str]): List of tuples of atom ids to be bonded. We will add them like spoof `struct_conn` entries,
            ensuring that we remove leaving groups as appropriate. Bonds tuples must be in the format (1-indexed!):
            ```
            (CHAIN_ID / RES_NAME / RES_ID / ATOM_NAME, CHAIN_ID / RES_NAME / RES_ID / ATOM_NAME)
            ```
            e.g., [("A/THR/4/CG", "D/L:1/0/O13"), ("A/CYS/5/SG",  "A/CYS/137/SG")]
        return_components (bool): If True, return the components list as well as the AtomArray. Useful for e.g., mapping
            components to generated chain IDs or inferred chain types.

    NOTE: If manually specifying bonds, we recommend visualizing the bond graph with `matplotlib` to ensure that the bonds are correctly
    NOTE: The res_id numbering follows the RCSB convention (1-indexed)

    Returns:
        AtomArray: The assembled AtomArray, used for visualization or inference.
    """
    # TODO: Add support for chiral specifications
    # TODO: Add support for cif/pdb

    # Ensure that all components are ChemicalComponent objects
    components = [
        ChemicalComponent.from_dict(component) if isinstance(component, dict) else component for component in components
    ]

    # Extract all assigned chain IDs
    chain_ids = []
    for component in components:
        if hasattr(component, "chain_ids") and component.chain_ids:
            chain_ids.extend(component.chain_ids)
        elif hasattr(component, "chain_id") and component.chain_id:
            chain_ids.append(component.chain_id)

    chain_id_generator = create_chain_id_generator(chain_ids)

    atom_arrays = []
    ligand_hash_to_id = KeyToIntMapper()  # ... to keep track of identical ligands
    for component in components:
        if isinstance(component, CIFFileComponent):
            # ... append and skip to the next component, as we already have the
            # annotated AtomArray
            atom_arrays.append(component.atom_array)
            continue

        component.chain_id = component.chain_id or next(chain_id_generator)

        if isinstance(component, SequenceComponent):
            atom_arrays.append(sequence_to_annotated_atom_array(**component.as_dict()))
        elif isinstance(component, SmilesComponent):
            ligand_array = smiles_to_annotated_atom_array(**component.as_dict())
            atom_arrays.append(assign_res_name_from_atom_array_hash(ligand_array, ligand_hash_to_id))
        elif isinstance(component, CCDComponent):
            atom_arrays.append(ccd_code_to_annotated_atom_array(**component.as_dict()))
        elif isinstance(component, SDFComponent):
            ligand_array = sdf_to_annotated_atom_array(**component.as_dict())
            atom_arrays.append(assign_res_name_from_atom_array_hash(ligand_array, ligand_hash_to_id))
        else:
            raise ValueError(f"Unknown chemical component type: {type(component)}")

    # ... concatenate all atom arrays into a single AtomArray
    atom_array = struc.concatenate(atom_arrays)

    # TODO: We may be able to simplify by casting to a buffer and running `parse`

    if bonds:
        # ... spoof the struct_conn CIFCategory
        struct_conn_dict = spoof_struct_conn_dict_from_string(bonds)

        # ... get the bonds and leaving atoms
        struct_conn_bonds, struct_conn_leaving_atom_idxs = get_struct_conn_bonds(
            atom_array=atom_array, struct_conn_dict=struct_conn_dict, add_bond_types=["covale"], raise_on_failure=True
        )
        struct_conn_bonds = struc.BondList(atom_array.array_length(), struct_conn_bonds)

        # ... add the bonds to the AtomArray
        atom_array.bonds = atom_array.bonds.merge(struct_conn_bonds)

        # ... record which atoms make inter-residue bonds
        atoms_with_inter_bonds = np.unique(struct_conn_bonds.as_array()[:, :2])
        makes_inter_bond = np.zeros(len(atom_array), dtype=bool)
        makes_inter_bond[atoms_with_inter_bonds] = True

        # ... and remove the leaving atoms
        is_leaving = np.zeros(len(atom_array), dtype=bool)
        is_leaving[struct_conn_leaving_atom_idxs] = True
        atom_array = atom_array[~is_leaving]
        makes_inter_bond = makes_inter_bond[~is_leaving]

        # ... fix charges of newly bonded atoms, where needed
        atom_array = correct_formal_charges_for_specified_atoms(atom_array, to_update=makes_inter_bond)

    # ... remove hydrogens
    atom_array = ta.remove_hydrogens(atom_array)

    # ... add iid, id, entity annotations
    atom_array = add_inference_iid_id_entity_annotations(atom_array)

    if return_components:
        return atom_array, components

    return atom_array
