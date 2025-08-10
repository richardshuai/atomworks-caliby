from enum import Flag, StrEnum
from typing import Self


class FlagWithCast(Flag):
    """
    A version of the Flag enum that can take in a string value and attempt to cast it to bool and then to the enum
    """

    @classmethod
    def _missing_(cls, value: str | bool | int | float) -> Self:
        try:
            value = bool(value)
        except ValueError:
            raise ValueError(f"Attempted to cast {value} of type {type(value)} to bool and failed")  # noqa: B904
        return cls(value)


class ProblemType(StrEnum):
    """
    A type of problem / interaction measured.
    """

    PPI = "protein-protein"
    DNA_BINDING = "protein-dna"
    RNA_BINDING = "protein-rna"
    SMALL_MOLECULE_BINDING = "protein-ligand"
    # ENZYME = "enzyme"  # NOT SUPPORTED YET


class BindingLabel(FlagWithCast):
    """
    Whether or not the binding interaction happens.
    """

    BIND = True
    NO_BIND = False


class ConfidenceLabel(StrEnum):
    """
    The confidence in the binding label.

    TODO: maybe we could do high, medium, low? Enum keeps it flexible in case we want to do this later.
    """

    CONFIDENT = "confident"
    NOT_CONFIDENT = "not confident"


class ExperimentType(StrEnum):
    """
    The type of experiment used to measure the binding interaction.
    """

    SPR = "spr"
    BLI = "bli"
    YEAST_DISPLAY = "yeast-display"
    # TODO: need way more


class TagType(StrEnum):
    """
    The type of tag used to measure the binding interaction.
    """

    BIOTIN = "biotin"
    STREP_TAG = "strep"
    # TODO: More??? (I am dry lab pls help)


class StructureMethod(StrEnum):
    """
    The method used to determine the structure of the binding interaction.
    """

    X_RAY = "x-ray"
    NMR = "nmr"
    CRYO_EM = "cryo-em"
    AF3 = "af3"
    AF2 = "af2"
    RF3 = "rf3"
    RFD1 = "rfd1"
    RFD2 = "rfd2"
    RFD3 = "rfd3"


class StructureType(StrEnum):
    """
    The type of structure used to measure the binding interaction.
    """

    COMPUTATIONAL = "computational"
    EXPERIMENTAL = "experimental"


class DataSourceType(StrEnum):
    """
    The type of data source for bind / no-bind measurements.
    """

    EXPERIMENT_IN_HOUSE = "experiment-in-house"
    EXPERIMENT_COLLABORATOR = "experiment-collaborator"
    EXPERIMENT_LITERATURE = "experiment-literature"
    DATABASE = "database"
