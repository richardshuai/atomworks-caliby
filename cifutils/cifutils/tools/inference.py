import io
import os
from abc import ABC
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from string import ascii_uppercase
from typing import Iterator, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from cifutils.constants import (
    CCD_MIRROR_PATH,
    PEPTIDE_MAX_RESIDUES,
    STANDARD_AA_ONE_LETTER,
    STANDARD_RNA,
    STANDARD_DNA_ONE_LETTER,
    UNKNOWN_LIGAND,
)
from cifutils.common import exists
from cifutils.enums import ChainType
import cifutils.transforms.atom_array as ta
from cifutils.utils.bond_utils import get_inferred_polymer_bonds
from cifutils.utils.residue_utils import build_chem_comp_atom_list, get_chem_comp_type
from cifutils.enums import ChainTypeInfo
from cifutils.tools.fasta import split_generalized_fasta_sequence, one_letter_to_ccd_code
from cifutils.utils.io_utils import read_any, get_structure
from cifutils.utils.selection_utils import get_residue_starts
from cifutils.utils.sequence_utils import get_1_from_3_letter_code
from cifutils.utils.ccd import check_ccd_codes_are_available
import logging

logger = logging.getLogger("cifutils")


class ChemicalComponent(ABC):
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
        chem_comp_types = set(get_chem_comp_type(ccd_code) for ccd_code in ccd_codes)

        valid_chem_comp_types = ChainTypeInfo.VALID_CHEM_COMP_TYPES.get(chain_type, chem_comp_types)
        if not chem_comp_types.issubset(valid_chem_comp_types):
            raise ValueError(f"Invalid {chain_type=} for {chem_comp_types=}. Valid are {valid_chem_comp_types=}")

    @staticmethod
    def from_seq(
        seq: str | list[str], *, chain_type: ChainType | str = None, is_polymer: bool = None
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
            components.append(sdf_to_atom_array(sdf=content))
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


def sequence_to_atom_array(
    seq: list[str],
    chain_id: str,
    *,
    chain_type: ChainType | str = None,
    is_polymer: bool = None,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> AtomArray:
    if isinstance(seq, str):
        seq = one_letter_to_ccd_code(split_generalized_fasta_sequence(seq), chain_type=chain_type)

    # Turn the sequence into a numpy array
    seq = np.asarray(seq)

    chain_type = chain_type or SequenceComponent.infer_chain_type(seq)
    chain_type = ChainType.as_enum(chain_type)
    is_polymer = is_polymer or SequenceComponent.infer_is_polymer(seq)

    # Ensure that the sequence is a valid combination of existing 3-letter CCD codes
    ccd_codes_in_seq = set(seq)
    check_ccd_codes_are_available(ccd_codes_in_seq, ccd_mirror_path=ccd_mirror_path, mode="raise")

    # ... create a list of atoms based on the reference CCD entries
    atom_list = []
    atom_lens = []
    for res_id, ccd_code in enumerate(seq):
        atoms_in_chem_comp = build_chem_comp_atom_list(ccd_code, keep_hydrogens=False, ccd_mirror_path=ccd_mirror_path)
        atom_list.extend(atoms_in_chem_comp)
        atom_lens.append(len(atoms_in_chem_comp))

    # ... convert to AtomArray
    atom_array = struc.array(atom_list)
    atom_array.set_annotation("chain_id", [chain_id] * len(atom_array))
    atom_array.set_annotation("res_id", np.repeat(np.arange(len(atom_lens)), atom_lens))
    atom_array.set_annotation("index", np.arange(len(atom_array)))

    # Compute bonds and leaving groups
    bond_list, leaving_atom_indices = get_inferred_polymer_bonds(
        atom_array,
        chain_id=chain_id,
        chain_type=chain_type,
        keep_hydrogens=False,
        ccd_mirror_path=ccd_mirror_path,
    )
    # ... add bonds to the atom array
    atom_array.bonds = struc.BondList(atom_array.array_length(), bond_list)
    # ... remove the leaving groups
    atom_array = atom_array[np.setdiff1d(atom_array.index, leaving_atom_indices)]

    # ... remove index annotation and leaving group annotations
    atom_array.del_annotation("index")
    atom_array.del_annotation("leaving_group")
    atom_array.del_annotation("leaving_atom_flag")

    # Add annotations
    atom_array.set_annotation("occupancy", np.ones(atom_array.array_length()))
    atom_array.set_annotation("is_polymer", np.full(atom_array.array_length(), is_polymer))
    atom_array.set_annotation("chain_type", np.full(atom_array.array_length(), chain_type))

    return atom_array


def smiles_to_atom_array(
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
        from cifutils.tools.obabel import atom_array_from_openbabel, smiles_to_openbabel

        mol = smiles_to_openbabel(smiles)
        array = atom_array_from_openbabel(mol)
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


def sdf_to_atom_array(
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
        from cifutils.tools.obabel import atom_array_from_openbabel, sdf_to_openbabel

        mol = sdf_to_openbabel(sdf)
        array = atom_array_from_openbabel(mol)
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


def cif_to_atom_array(cif: io.StringIO | os.PathLike) -> AtomArray:
    return get_structure(read_any(cif), assume_residues_all_resolved=True, include_bonds=True, model=1)


def get_next_chain_id_generator(occupied_chain_ids: list[str] = []) -> Iterator[str]:
    """
    Generate the next available chain ID that is not in the occupied_chain_ids list.

    Args:
        - occupied_chain_ids (list[str]): List of already occupied chain IDs.

    Yields:
        - str: The next available chain ID.

    Example:
        >>> occupied = ["A", "B", "C", "AA", "AB"]
        >>> next_id = get_next_chain_id_generator(occupied)
        >>> print(next(next_id), next(next_id), next(next_id))
        D E F
    """
    occupied_set = set(occupied_chain_ids)

    def chain_id_generator():
        for length in range(1, 100):  # Adjust the upper limit if needed
            for combo in product(ascii_uppercase, repeat=length):
                yield "".join(combo)

    for chain_id in chain_id_generator():
        if chain_id not in occupied_set:
            yield chain_id


def add_inference_iid_id_entity_annotations(atom_array: AtomArray) -> AtomArray:
    # ... annotate ids and entities
    atom_array = ta.add_id_and_entity_annotations(atom_array)

    # ... annotate iids (assumes we are only given the asym)
    atom_array.set_annotation("chain_iid", atom_array.chain_id)
    atom_array.set_annotation("pn_unit_iid", atom_array.pn_unit_id)
    atom_array.set_annotation("molecule_iid", atom_array.molecule_id)

    return atom_array


def components_to_atom_array(components: list[ChemicalComponent | dict]) -> AtomArray:
    # TODO: Add support for covalent bonds between components (including chirals specification)
    # TODO: Add support for ligands inserted inside sequence polymer chains (e.g. non-CCD coded noncanonicals)
    # TODO: Add support for cif/pdb

    # Ensure that all components are ChemicalComponent objects
    components = [
        ChemicalComponent.from_dict(component) if isinstance(component, dict) else component for component in components
    ]

    # Extract all chain ids
    chain_ids = [component.chain_id for component in components if exists(component.chain_id)]
    chain_id_generator = get_next_chain_id_generator(chain_ids)

    atom_arrays = []
    for component in components:
        component.chain_id = component.chain_id or next(chain_id_generator)

        if isinstance(component, SequenceComponent):
            atom_arrays.append(sequence_to_atom_array(**component.as_dict()))
        elif isinstance(component, SmilesComponent):
            atom_arrays.append(smiles_to_atom_array(**component.as_dict()))
        else:
            raise ValueError(f"Unknown chemical component type: {type(component)}")

    atom_array = atom_arrays[0]
    for arr in atom_arrays[1:]:
        atom_array += arr

    atom_array = add_inference_iid_id_entity_annotations(atom_array)

    return atom_array


def chain_info_from_atom_array(atom_array: AtomArray) -> dict[str, dict[str, str]]:
    chain_ids = np.unique(atom_array.chain_id)
    chain_info = {}
    for chain_id in chain_ids:
        chain = atom_array[atom_array.chain_id == chain_id]
        seq = chain.res_name[get_residue_starts(chain)]
        chain_type = ChainType.as_enum(chain.chain_type[0])
        chain_info[chain_id] = dict(type=chain_type.to_string())
        if chain_type.is_polymer():
            chain_info[chain_id] |= dict(
                processed_entity_non_canonical_sequence="".join(
                    get_1_from_3_letter_code(residue, chain_type, use_closest_canonical=False) for residue in seq
                ),
                processed_entity_canonical_sequence="".join(
                    get_1_from_3_letter_code(residue, chain_type, use_closest_canonical=True) for residue in seq
                ),
            )
    return chain_info
