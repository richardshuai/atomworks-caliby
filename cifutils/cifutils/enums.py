"""Enums used in the `cifutils` package."""

from enum import IntEnum


class ChainType(IntEnum):
    """
    IntEnum representing the type of chain in a RCSB mmCIF file from the Protein Data Bank (PDB).
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
        try:
            return CHAIN_TYPE_STRING_TO_ENUM_MAPPING[str_value.lower()]
        except KeyError:
            raise ValueError(f"Invalid chain type: {str_value.lower()}")

    @staticmethod
    def get_chain_type_strings() -> list[str]:
        return list(CHAIN_TYPE_STRING_TO_ENUM_MAPPING.keys())

    @staticmethod
    def get_polymers() -> list["ChainType"]:
        return POLYMERS

    @staticmethod
    def get_proteins() -> list["ChainType"]:
        return PROTEINS

    @staticmethod
    def get_nucleic_acids() -> list["ChainType"]:
        return NUCLEIC_ACIDS

    def __eq__(self, other) -> bool:
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
        return hash(self.value)

    def is_protein(self) -> bool:
        return self in PROTEINS

    def is_nucleic_acid(self) -> bool:
        return self in NUCLEIC_ACIDS

    def is_polymer(self) -> bool:
        return self in POLYMERS

    def is_non_polymer(self) -> bool:
        return self == ChainType.NON_POLYMER

    def to_string(self) -> str:
        return ENUM_TO_CHAIN_TYPE_STRING_MAPPING[self]


POLYMERS = [
    ChainType.POLYPEPTIDE_D,
    ChainType.POLYPEPTIDE_L,
    ChainType.DNA,
    ChainType.DNA_RNA_HYBRID,
    ChainType.RNA,
    ChainType.PEPTIDE_NUCLEIC_ACID,
    ChainType.CYCLIC_PSEUDO_PEPTIDE,
    ChainType.OTHER_POLYMER,
]
PROTEINS = [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L, ChainType.CYCLIC_PSEUDO_PEPTIDE]
NUCLEIC_ACIDS = [ChainType.DNA, ChainType.RNA, ChainType.DNA_RNA_HYBRID]

# Define a mapping from chain_type strings to ChainType enums
# Entity types found at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity.type.html
# Polymer types found at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v5_next.dic/Items/_entity_poly.type.html
CHAIN_TYPE_STRING_TO_ENUM_MAPPING: dict[str, ChainType] = {
    "cyclic-pseudo-peptide": ChainType.CYCLIC_PSEUDO_PEPTIDE,
    "peptide nucleic acid": ChainType.PEPTIDE_NUCLEIC_ACID,
    "polydeoxyribonucleotide": ChainType.DNA,
    "polydeoxyribonucleotide/polyribonucleotide hybrid": ChainType.DNA_RNA_HYBRID,
    "polypeptide(d)": ChainType.POLYPEPTIDE_D,
    "polypeptide(l)": ChainType.POLYPEPTIDE_L,
    "polyribonucleotide": ChainType.RNA,
    "non-polymer": ChainType.NON_POLYMER,
    "branched": ChainType.NON_POLYMER,  # The PDB does not consider oligosaccharides to be polymers
    "macrolide": ChainType.NON_POLYMER,
    "water": ChainType.NON_POLYMER,
}

# Compute the reverse mapping, from ChainType enums to chain_type strings
ENUM_TO_CHAIN_TYPE_STRING_MAPPING: dict[ChainType, str] = {
    chain_type: chain_type_str for chain_type_str, chain_type in CHAIN_TYPE_STRING_TO_ENUM_MAPPING.items()
}
