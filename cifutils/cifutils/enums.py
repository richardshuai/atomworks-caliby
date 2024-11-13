"""Enums used in the `cifutils` package."""

from enum import IntEnum
from typing import Union, Final
from types import MappingProxyType
from cifutils.constants import (
    AA_LIKE_CHEM_TYPES,
    DNA_LIKE_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
    POLYPEPTIDE_D_CHEM_TYPES,
    POLYPEPTIDE_L_CHEM_TYPES,
)
from toolz import keymap
import numpy as np


class ChainType(IntEnum):
    """
    IntEnum representing the type of chain in a RCSB mmCIF file from the Protein Data Bank (PDB).

    Useful constants relating to ChainType are defined in ChainTypeInfo.
    """

    CYCLIC_PSEUDO_PEPTIDE = 0
    OTHER_POLYMER = 1
    PEPTIDE_NUCLEIC_ACID = 2
    DNA = 3
    DNA_RNA_HYBRID = 4
    POLYPEPTIDE_D = 5
    POLYPEPTIDE_L = 6
    RNA = 7
    NON_POLYMER = 8

    @classmethod
    def from_string(cls, str_value: str) -> "ChainType":
        """Convert a string to a ChainType enum."""
        try:
            return ChainTypeInfo.STRING_TO_ENUM[str_value.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid chain type: {str_value=}. Allowed values: {set(ChainTypeInfo.STRING_TO_ENUM.keys())}"
            )

    @staticmethod
    def get_chain_type_strings() -> list[str]:
        """Get a list of all chain type strings."""
        return list(ChainTypeInfo.STRING_TO_ENUM.keys())

    @staticmethod
    def get_polymers() -> list["ChainType"]:
        """Get a list of all polymer chain types."""
        return ChainTypeInfo.POLYMERS

    @staticmethod
    def get_proteins() -> list["ChainType"]:
        """Get a list of all protein chain types."""
        return ChainTypeInfo.PROTEINS

    @staticmethod
    def get_nucleic_acids() -> list["ChainType"]:
        """Get a list of all nucleic acid chain types."""
        return ChainTypeInfo.NUCLEIC_ACIDS

    def __eq__(self, other) -> bool:
        """Check if two ChainType enums are equal."""
        if isinstance(other, ChainType):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        elif isinstance(other, str):
            try:
                # Attempt to convert the string to a ChainType
                other_chain_type = ChainType.from_string(other)
                return self.value == other_chain_type.value
            except ValueError:
                # Could not convert the string to a ChainType
                return False
        return NotImplemented

    def __hash__(self):
        """Hash a ChainType enum."""
        return hash(self.value)

    def __str__(self) -> str:
        """Convert a ChainType enum to a string."""
        return self.to_string()

    def get_valid_chem_comp_types(self) -> set[str]:
        """Get the set of valid chemical component types for a ChainType."""
        return ChainTypeInfo.VALID_CHEM_COMP_TYPES[self]

    def is_protein(self) -> bool:
        """Check if a ChainType is a protein."""
        return self in ChainTypeInfo.PROTEINS

    def is_nucleic_acid(self) -> bool:
        """Check if a ChainType is a nucleic acid."""
        return self in ChainTypeInfo.NUCLEIC_ACIDS

    def is_polymer(self) -> bool:
        """Check if a ChainType is a polymer."""
        return self in ChainTypeInfo.POLYMERS

    def is_non_polymer(self) -> bool:
        """Check if a ChainType is a non-polymer."""
        return self == ChainType.NON_POLYMER

    def to_string(self) -> str:
        """
        Convert a ChainType enum to a string.

        NOTE: Returns UPPERCASE string (e.g., "POLYPEPTIDE(D)" instead of "polypeptide(D)")
        """
        return ChainTypeInfo.ENUM_TO_STRING[self]

    @staticmethod
    def as_enum(value: Union[str, int, "ChainType"]) -> "ChainType":
        """Convert a string, int, or ChainType to a ChainType enum."""
        if isinstance(value, ChainType):
            return value
        elif isinstance(value, str):
            return ChainType.from_string(value)
        elif isinstance(value, (int, np.integer)):
            return ChainType(value)
        else:
            raise ValueError(f"Invalid value: {value}")


class ChainTypeInfo:
    """
    Companion class containing metadata and helper methods for ChainType enum.

    This class should not be instantiated - it serves as a namespace for ChainType-related constants and utilities.
    """

    POLYMERS: Final[tuple[ChainType, ...]] = (
        ChainType.POLYPEPTIDE_D,
        ChainType.POLYPEPTIDE_L,
        ChainType.DNA,
        ChainType.DNA_RNA_HYBRID,
        ChainType.RNA,
        ChainType.PEPTIDE_NUCLEIC_ACID,
        ChainType.CYCLIC_PSEUDO_PEPTIDE,
        ChainType.OTHER_POLYMER,
    )

    PROTEINS: Final[tuple[ChainType, ...]] = (
        ChainType.POLYPEPTIDE_D,
        ChainType.POLYPEPTIDE_L,
        ChainType.CYCLIC_PSEUDO_PEPTIDE,
    )

    NUCLEIC_ACIDS: Final[tuple[ChainType, ...]] = (ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID)

    # Define a mapping from chain_type strings to ChainType enums
    # Entity types found at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity.type.html
    # Polymer types found at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v5_next.dic/Items/_entity_poly.type.html
    STRING_TO_ENUM: Final[MappingProxyType[str, ChainType]] = MappingProxyType(
        keymap(
            str.upper,
            {
                "CYCLIC-PSEUDO-PEPTIDE": ChainType.CYCLIC_PSEUDO_PEPTIDE,
                "PEPTIDE NUCLEIC ACID": ChainType.PEPTIDE_NUCLEIC_ACID,
                "POLYDEOXYRIBONUCLEOTIDE": ChainType.DNA,
                "POLYDEOXYRIBONUCLEOTIDE/POLYRIBONUCLEOTIDE HYBRID": ChainType.DNA_RNA_HYBRID,
                "POLYPEPTIDE(D)": ChainType.POLYPEPTIDE_D,
                "POLYPEPTIDE(L)": ChainType.POLYPEPTIDE_L,
                "POLYRIBONUCLEOTIDE": ChainType.RNA,
                "NON-POLYMER": ChainType.NON_POLYMER,
                "BRANCHED": ChainType.NON_POLYMER,  # The PDB does not consider oligosaccharides to be polymers
                "MACROLIDE": ChainType.NON_POLYMER,
                "WATER": ChainType.NON_POLYMER,
                "OTHER": ChainType.OTHER_POLYMER,  # WARNING! Paradoxically, "other" is a polymer type.
            },
        )
    )

    # Compute the reverse mapping, from ChainType enums to chain_type strings
    ENUM_TO_STRING: Final[MappingProxyType[ChainType, str]] = MappingProxyType(
        {chain_type: chain_type_str for chain_type_str, chain_type in STRING_TO_ENUM.items()}
        | {ChainType.NON_POLYMER: "NON-POLYMER"}
    )

    VALID_CHEM_COMP_TYPES: Final[MappingProxyType[ChainType, set[str]]] = MappingProxyType(
        {
            ChainType.CYCLIC_PSEUDO_PEPTIDE: AA_LIKE_CHEM_TYPES,
            ChainType.PEPTIDE_NUCLEIC_ACID: AA_LIKE_CHEM_TYPES | DNA_LIKE_CHEM_TYPES | RNA_LIKE_CHEM_TYPES,
            ChainType.DNA: DNA_LIKE_CHEM_TYPES,
            ChainType.DNA_RNA_HYBRID: DNA_LIKE_CHEM_TYPES | RNA_LIKE_CHEM_TYPES,
            ChainType.POLYPEPTIDE_D: POLYPEPTIDE_D_CHEM_TYPES
            | {"PEPTIDE LINKING"},  # GLY counts as a peptide linking without L/D
            ChainType.POLYPEPTIDE_L: POLYPEPTIDE_L_CHEM_TYPES
            | {"PEPTIDE LINKING"},  # GLY counts as a peptide linking without L/D
            ChainType.RNA: RNA_LIKE_CHEM_TYPES,
        }
    )
