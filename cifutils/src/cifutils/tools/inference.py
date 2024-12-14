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

import cifutils.transforms.atom_array as ta
from cifutils.common import exists
from cifutils.constants import (
    CCD_MIRROR_PATH,
    PEPTIDE_MAX_RESIDUES,
    STANDARD_AA_ONE_LETTER,
    STANDARD_DNA_ONE_LETTER,
    STANDARD_RNA,
    UNKNOWN_LIGAND,
)
from cifutils.enums import ChainType, ChainTypeInfo
from cifutils.template import build_template_atom_array
from cifutils.tools.fasta import one_letter_to_ccd_code, split_generalized_fasta_sequence
from cifutils.utils.bonds import (
    fix_formal_charges,
    get_inferred_polymer_bonds,
    get_struct_conn_bonds,
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
    def infer_is_polymer(seq: str | list[str]) -> bool:
        """Infer if the sequence is a polymer based on length. Arbitrarily choose 25 residues as threshold."""
        return len(seq) > PEPTIDE_MAX_RESIDUES

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
        is_polymer = is_polymer or SequenceComponent.infer_is_polymer(seq)

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
        self.is_polymer = self.is_polymer or SequenceComponent.infer_is_polymer(self.seq)

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


@dataclass
class SDFComponent(LigandComponent):
    path: os.PathLike | io.StringIO
    chain_type: ChainType | str = "non-polymer"
    is_polymer: bool = False
    chain_id: str | None = None


@dataclass
class CIFFileComponent(ChemicalComponent):
    path: os.PathLike | io.StringIO


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
            components.append(sdf_to_annotated_atom_array(sdf=content))
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
) -> AtomArray:
    if isinstance(seq, str) and is_polymer:
        seq = one_letter_to_ccd_code(split_generalized_fasta_sequence(seq), chain_type=chain_type)

    # Turn the sequence into a numpy array
    seq = np.asarray(seq)

    chain_type = chain_type or SequenceComponent.infer_chain_type(seq)
    chain_type = ChainType.as_enum(chain_type)
    is_polymer = is_polymer or SequenceComponent.infer_is_polymer(seq)

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
        remove_hydrogens=True,
        use_ccd_charges=True,
        ccd_mirror_path=ccd_mirror_path,
    )

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
        "is_backbone_atom",
        "is_n_terminal_atom",
        "is_c_terminal_atom",
    )
    for annotation in _annotations_to_remove:
        atom_array.del_annotation(annotation)

    # Add custom annotations
    atom_array.set_annotation("occupancy", np.ones(atom_array.array_length()))
    atom_array.set_annotation("is_polymer", np.full(atom_array.array_length(), is_polymer))
    atom_array.set_annotation("chain_type", np.full(atom_array.array_length(), chain_type))

    return atom_array


def smiles_to_annotated_atom_array(
    smiles: str,
    chain_id: str,
    *,
    chain_type: ChainType | str = "non-polymer",
    is_polymer: bool = False,
    backend: Literal["openbabel", "rdkit"] = "rdkit",
) -> AtomArray:
    if backend == "rdkit":
        from cifutils.tools.rdkit import atom_array_from_rdkit, smiles_to_rdkit

        mol = smiles_to_rdkit(smiles)
        array = atom_array_from_rdkit(mol)
    elif backend == "openbabel":
        raise NotImplementedError("Openbabel backend not yet implemented.")
    else:
        raise ValueError(f"Unknown backend: {backend=}")

    # Update annotations
    array.set_annotation("occupancy", np.ones(array.array_length()))
    array.set_annotation("hetero", np.full(array.array_length(), True))
    array.set_annotation("res_name", np.full(array.array_length(), UNKNOWN_LIGAND))
    array.set_annotation("chain_id", np.full(array.array_length(), chain_id))
    array.set_annotation("is_polymer", np.full(array.array_length(), is_polymer))
    array.set_annotation("chain_type", np.full(array.array_length(), ChainType.as_enum(chain_type)))
    return array


def sdf_to_annotated_atom_array(
    sdf: io.StringIO | os.PathLike,
    chain_id: str,
    *,
    chain_type: ChainType | str = "non-polymer",
    is_polymer: bool = False,
    backend: Literal["openbabel", "rdkit"] = "rdkit",
) -> AtomArray:
    if backend == "rdkit":
        from cifutils.tools.rdkit import atom_array_from_rdkit, sdf_to_rdkit

        mol = sdf_to_rdkit(sdf)
        array = atom_array_from_rdkit(mol)
    elif backend == "openbabel":
        raise NotImplementedError("Openbabel backend not yet implemented.")
    else:
        raise ValueError(f"Unknown backend: {backend=}")

    # Update annotations
    array.set_annotation("occupancy", np.ones(array.array_length()))
    array.set_annotation("hetero", np.full(array.array_length(), True))
    array.set_annotation("res_name", np.full(array.array_length(), UNKNOWN_LIGAND))
    array.set_annotation("chain_id", np.full(array.array_length(), chain_id))
    array.set_annotation("is_polymer", np.full(array.array_length(), is_polymer))
    array.set_annotation("chain_type", np.full(array.array_length(), ChainType.as_enum(chain_type)))
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


def add_inference_iid_id_entity_annotations(atom_array: AtomArray) -> AtomArray:
    # ... annotate ids and entities
    atom_array = ta.add_id_and_entity_annotations(atom_array)

    # ... annotate iids (assumes we are only given the asym)
    atom_array.set_annotation("chain_iid", atom_array.chain_id)
    atom_array.set_annotation("pn_unit_iid", atom_array.pn_unit_id)
    atom_array.set_annotation("molecule_iid", atom_array.molecule_id)

    return atom_array


def components_to_atom_array(components: list[ChemicalComponent | dict], bonds: list[str] | None = None) -> AtomArray:
    """Build an AtomArray from a list of ChemicalComponent objects and, optionally, a list of bonds.

    Args:
        components (list[ChemicalComponent | dict]): List of ChemicalComponent objects or dictionaries that can be
            converted to ChemicalComponent objects using ChemicalComponent.from_dict().
        bonds (list[str]): List of tuples of atom ids to be bonded. We will add them like spoof `struct_conn` entries,
            ensuring that we remove leaving groups as appropriate. Bonds tuples must be in the format (1-indexed!):
            ```
            (CHAIN_ID:RES_NAME:RES_ID:ATOM_NAME, CHAIN_ID:RES_NAME:RES_ID:ATOM_NAME)
            ```
            e.g., [("A:THR:4:CG", "D:UNL:0:O13"), ("A:CYS:5:SG",  "A:CYS:137:SG")]

    NOTE: We recommend visualizing the AtomArray to ensure that the bonds are correctly added before using it for inference.
    NOTE: The ResID numbering follows the RCSB convention and is 1-indexed!

    Returns:
        AtomArray: The assembled AtomArray, used for visualization or inference.
    """
    # TODO: Add support for chiral specifications
    # TODO: Add support for ligands inserted inside sequence polymer chains (e.g. non-CCD coded noncanonicals)
    # TODO: Add support for cif/pdb

    # Ensure that all components are ChemicalComponent objects
    components = [
        ChemicalComponent.from_dict(component) if isinstance(component, dict) else component for component in components
    ]

    # Extract all chain ids
    chain_ids = [component.chain_id for component in components if exists(component.chain_id)]
    chain_id_generator = create_chain_id_generator(chain_ids)

    atom_arrays = []
    for component in components:
        component.chain_id = component.chain_id or next(chain_id_generator)

        # TODO: Can we add support for SDFComponent?
        if isinstance(component, SequenceComponent):
            atom_arrays.append(sequence_to_annotated_atom_array(**component.as_dict()))
        elif isinstance(component, SmilesComponent):
            atom_arrays.append(smiles_to_annotated_atom_array(**component.as_dict()))
        elif isinstance(component, CCDComponent):
            atom_arrays.append(ccd_code_to_annotated_atom_array(**component.as_dict()))
        else:
            raise ValueError(f"Unknown chemical component type: {type(component)}")

    # TODO: Rewrite using Biotite's new `concatenate` method
    atom_array = atom_arrays[0]
    for arr in atom_arrays[1:]:
        atom_array += arr

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
        atom_array = fix_formal_charges(atom_array, to_update=makes_inter_bond)

    atom_array = add_inference_iid_id_entity_annotations(atom_array)

    return atom_array
