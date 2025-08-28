from enum import Flag, StrEnum


class FlagWithCast(Flag):
    """
    A version of the Flag enum that can take in a string value and attempt to cast it to bool and then to the enum
    """

    @classmethod
    def _missing_(cls, value: str | bool | int | float):
        try:
            value = bool(value)
        except ValueError:
            raise ValueError(f"Attempted to cast {value} of type {type(value)} to bool and failed")
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


class ExpressionConfidenceLabel(StrEnum):
    """
    The confidence that this thing actually expressed

    """

    EXPRESSED = "expressed"
    UNCERTAIN = "uncertain"


class FoldingConfidenceLabel(StrEnum):
    """
    The confidence that the protein folds.

    For some non-binding proteins, we may have doubts about whether they fold whereas others might have evidence of folding
    for example, if they bind a different target. This flag allows for making this distinction.
    """

    FOLDS = "folds"
    UNCERTAIN = "uncertain"


class ExperimentType(StrEnum):
    """
    The type of experiment used to measure the binding interaction.
    """

    SPR = "spr"
    BLI = "bli"
    YEAST_DISPLAY = "yeast-display"
    NGS_SEQUENCING = "ngs"
    SELEX = "selex"
    # TODO: need way more


class TagType(StrEnum):
    """
    The type of tag used to measure the binding interaction.
    """

    BIOTIN = "biotin"
    STREP_TAG = "strep"
    FLAG = "flag"
    HIS = "his"
    CONJUGATED_FLUOROPHORE = "conjugated-fluorophore"
    AVI = "avi"
    MYC = "myc"
    SPY = "spy"
    SNOOP = "snoop"
    DOG = "dog"
    HALO = "halo"
    FC = "fc"


class StructureMethod(StrEnum):
    """
    The method used to determine the structure of the binding interaction.
    """

    DESIGN_MODEL = "design_model"
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


class StructureDockType(StrEnum):
    """
    How confident are you that this thing is docked correctly
    """

    BOUND = "bound"
    UNCERTAIN = "uncertain"


class DataSourceType(StrEnum):
    """
    The type of data source for bind / no-bind measurements.
    """

    EXPERIMENT_IN_HOUSE = "experiment-in-house"
    EXPERIMENT_COLLABORATOR = "experiment-collaborator"
    EXPERIMENT_LITERATURE = "experiment-literature"
    DATABASE = "database"
