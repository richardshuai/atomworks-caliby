from enum import Enum
from typing import Dict, List


class ChainType(Enum):
    """
    Enum representing the type of chain in a PDB file.
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
    def from_int(cls, int_value: int) -> "ChainType":
        return cls(int_value)

    @classmethod
    def from_string(cls, str_value: str) -> "ChainType":
        try:
            return CHAIN_TYPE_STRING_TO_ENUM_MAPPING[str_value.lower()]
        except KeyError:
            raise ValueError(f"Invalid chain type: {str_value.lower()}")

    @staticmethod
    def to_int(chain_type: "ChainType") -> int:
        return chain_type.value

    @staticmethod
    def get_chain_type_strings() -> List[str]:
        return CHAIN_TYPE_STRING_TO_ENUM_MAPPING.keys()

    @staticmethod
    def get_polymers() -> List["ChainType"]:
        return POLYMERS

    @staticmethod
    def get_proteins() -> List["ChainType"]:
        return PROTEINS

    @staticmethod
    def get_nucleic_acids() -> List["ChainType"]:
        return NUCLEIC_ACIDS

    def __eq__(self, other) -> bool:
        if isinstance(other, ChainType):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return NotImplemented

    def __hash__(self):
        return hash(f"ChainType.{self.name.lower()}")

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

# At the time of writing, the supported chain types are: NON_POLYMER, POLYPEPTIDE_L, DNA, RNA
SUPPORTED_CHAIN_TYPES = [ChainType.NON_POLYMER, ChainType.POLYPEPTIDE_L, ChainType.DNA, ChainType.RNA]

# Define a mapping from chain_type strings to ChainType enums
# Entity types found at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity.type.html
# Polymer types found at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v5_next.dic/Items/_entity_poly.type.html
CHAIN_TYPE_STRING_TO_ENUM_MAPPING: Dict[str, ChainType] = {
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
ENUM_TO_CHAIN_TYPE_STRING_MAPPING: Dict[ChainType, str] = {
    chain_type: chain_type_str for chain_type_str, chain_type in CHAIN_TYPE_STRING_TO_ENUM_MAPPING.items()
}


class ClashSeverity(Enum):
    """
    Enum representing the severity of clashes in a PDB file.
    """

    SEVERE = "severe"  # More than 50% of polymers are clashing
    MODERATE = "moderate"  # Any polymers are clashing
    MILD = "mild"  # Any clashes (polymer or non-polymer)
    NO_CLASH = "no-clash"  # No clashes


# Atomic numbers
OXYGEN_ATOMIC_NUMBER = 8
FLUORINE_ATOMIC_NUMBER = 9

# For building the CellList
CELL_SIZE = 4.5

# Entries to exclude from the dataset
ENTRIES_TO_EXCLUDE = [
    # TODO: Deduplicate this list
    "7bho",  # DNA origami
    "7lhd",
    "6vyr",
    "3zif",
    "7nwh",  # Ribosome
    "6nu3",
    "7tbi",  # Nuclear pore complex
    "7lhd",  # Hetero 180-mer
    "4l3b",  # Hetero 180-mer
    "7bsi",  # Hetero 2,820-mer
    "6nhj",  # Hetero 3,300-mer
    "5j7v",  # Homo 8,280-mer
    "6w19",  # Hetero 3,060-mer
    "6cgr",  # Hetero 2,760-mer
    "6b43",  # Hetero 306-mer
    "6cgr",  # Hetero 2,760-mer
    "7bw6",
    "6lgl",
    "7as5",
    "7fj1",
    "7r5k",
    "3k1q",
    "6ncl",
    "5zap",
    "6b43",
    "1m4x",
    "6q1f",
    "7qiz",
    "4f5x",
    "7tbk",
    "5jus",
    "1uf2",
    "6ftg",
    "7tbk",
    "7v4t",
    "7r5j",
    "3jbp",
    "6ylh",
    "7btb",
    "6zvk",
    "4u3n",
    "6woo",
    "6ftj",
    "4u3n",
    "3jan",
    "4u3n",
    "5tbw",
    "6tb3",
    "6zvk",
    "5dat",
    "3jan",
    "4v5x",
]
