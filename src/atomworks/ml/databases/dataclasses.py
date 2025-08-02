"""
This file defines dataclasses for a database. For now, we only use cif databases, which are
where we save data points as individual cif files with sequence, structure (optionally),
and metadata desired for that data point.
"""

from abc import ABC
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, List, Optional, Type

from biotite.structure import AtomArray
from atomworks.io.tools.inference import ChemicalComponent

from atomworks.ml.databases.enums import (
    BindingLabel,
    ConfidenceLabel,
    DataSourceType,
    ExperimentType,
    ProblemType,
    StructureMethod,
    StructureType,
    TagType,
)


@dataclass
class DataObject(ABC):
    """
    A base class for all data objects in a database.
    """

    def validate(self) -> None:
        """
        Override this method to add custom validation logic to your data object. I.e. if the field "name" is a string but cannot have any spaces
        in it, you can do something like:

        if not any(char.isspace() for char in self.name):
            raise ValueError(f"Name cannot have spaces: {self.name}")

        NOTE: Best practice is to raise a ValueError with a helpful message. There can be many reasons a data object is not valid.

        Returns:
            Raises a ValueError if the data object is not valid.
        """
        pass

    def __post_init__(self):
        self.validate()

    @classmethod
    def get_field_info(cls) -> Dict[str, Type]:
        """
        Get all field names and their types from the dataclass.

        Returns:
            Dict[str, Type]: A dictionary mapping field names to their types.
        """
        return {field.name: field.type for field in fields(cls)}

    @classmethod
    def get_required_fields(cls) -> Dict[str, Type]:
        """
        Get only the required fields (those without defaults) and their types.

        Returns:
            Dict[str, Type]: A dictionary mapping required field names to their types.
        """
        return {
            field.name: field.type
            for field in fields(cls)
            if field.default is MISSING and field.default_factory is MISSING
        }

    @classmethod
    def get_optional_fields(cls) -> Dict[str, Type]:
        """
        Get only the optional fields (those with defaults) and their types.

        Returns:
            Dict[str, Type]: A dictionary mapping optional field names to their types.
        """
        return {
            field.name: field.type
            for field in fields(cls)
            if field.default is not MISSING or field.default_factory is not MISSING
        }


@dataclass
class BindNoBindMeasurement(DataObject):
    """
    A dataclass for a single measurement of a binding interaction between a protein and another molecule
    (protein, DNA, ligand, etc.). Notably we do not require a structure for the complex nor do we require
    the sequences involved to be unique. We may have the same binding interaction measured multiple times
    across different experimental conditions, and each measurement + complex combo is saved separately.

    Required Fields:
        atom_array: (AtomArray | List[ChemicalComponent]) The atom array of the complex. If you have no structure, you can pass a list of ChemicalComponents for each
            chain / molecule and a dummy structure will be created upon saving to a cif file.
        data_source_id: (str) The id of the data source that this measurement belongs to.
        target: (str) The name of the target protein. (e.g. "VEGFR2")
        binding_label: (BindingLabel) Whether or not the binding interaction happens. (either BindingLabel.BIND or BindingLabel.NO_BIND)
        label_confidence: (ConfidenceLabel) The confidence in the binding label. (either ConfidenceLabel.CONFIDENT or ConfidenceLabel.NOT_CONFIDENT)
        partners: (List[List[str]]) The partners of the binding interaction. The order here doesn't matter. (e.g. [["H", "L"], ["T"]])

    Optional Fields:
        target_chains: (List[str]) The chains of the target protein. (e.g. ["H", "L"])
        binder_chains: (List[str]) The chains of the binder protein. (e.g. ["T"])
        fitness: (float) The fitness of the binding interaction in a.u. This may be something like an enrichment score in a yeast display experiment. (e.g. 0.35)
        affinity: (float) The affinity of the binding interaction in nM. (e.g. 0.5)
        affinity_std: (float) The standard deviation of the affinity in nM. (e.g. 0.1)
        label_threshold: (float) The threshold for the bind/no-bind label in nM. (e.g. 10 nM)
        pH: (float) The pH of the measurement. (e.g. 7.4)
        temperature: (float) The temperature of the measurement in C. (e.g. 25.0)
        tag_type: (TagType) The type of tag used to measure the binding interaction. (e.g. TagType.BIOTIN, TagType.STREP_TAG, etc.)
        structure_method: (StructureMethod) The method used to determine the structure included in the measurement if there is one. (e.g. StructureMethod.X_RAY, StructureMethod.AF3, etc.)
        structure_type: (StructureType) The type of structure included in the measurement if there is one. (e.g. StructureType.COMPUTATIONAL, StructureType.EXPERIMENTAL)
        measurement_description: (str) A catch-all for anything else about this individual measurement you may want to annotate.

    Derived Fields:
        See DataSource for derived fields.
    """

    # ============ REQUIRED FIELDS ============
    atom_array: AtomArray | List[ChemicalComponent]
    data_source_id: str
    target: str
    binding_label: BindingLabel
    label_confidence: ConfidenceLabel
    partners: List[List[str]]

    # ============ OPTIONAL FIELDS ============
    target_chains: Optional[List[str]] = None
    binder_chains: Optional[List[str]] = None  # TODO: should these be required?
    fitness: Optional[float] = None
    affinity: Optional[float] = None
    affinity_std: Optional[float] = None
    label_threshold: Optional[float] = None
    pH: Optional[float] = None
    temperature: Optional[float] = None
    tag_type: Optional[TagType] = None
    structure_method: Optional[StructureMethod] = None
    structure_type: Optional[StructureType] = None
    measurement_description: Optional[str] = None

    # ==== FIELDS DERIVED FROM DATA SOURCE ====
    # NOTE: These are all optional fields in BindNoBindMeasurement, but they are guarenteed to be present if you use
    # datahub.databases.io_utils.save_measurement_to_cif() to save this measurement
    # NOTE: Generally these should be left blank upon initialization, but will be populated during saving and loading of measurement cif files.
    author: Optional[str] = None
    year: Optional[int] = None
    data_source_tag: Optional[str] = None
    problem: Optional[ProblemType] = None
    data_source_type: Optional[DataSourceType] = None
    experiment_type: Optional[ExperimentType] = None
    experiment_metadata: Optional[Dict[str, Any]] = None
    data_source_description: Optional[str] = None


@dataclass
class DataSource(DataObject):
    """
    A dataclass for data sources for bind / no-bind measurements.

    Examples may include individual experiments run in house or by a collaborator, or a database of collated measurements.

    Required Fields:
        author: (str) The author of the data source.
        year: (int) The year of the data source.
        data_source_tag: (str) A tag for the data source. that will make {author}_{year}_{data_source_tag} unique. (e.g. "attempt5")
        problem: (ProblemType) The type of problem / interaction measured. (e.g. ProblemType.PPI, ProblemType.DNA_BINDING, etc.)
        data_source_type: (DataSourceType) The type of data source. (e.g. DataSourceType.EXPERIMENT_IN_HOUSE, DataSourceType.DATABASE, etc.)

    Optional Fields:
        experiment_type: (ExperimentType) The type of experiment used to measure the binding interaction. (e.g. ExperimentType.SPR, ExperimentType.BLI, etc.)
        experiment_metadata: (Dict[str, Any]) Any other metadata about the experiment you wish to annotate. This is a catch-all for experimental conditions you wish to annotate.
        data_source_description: (str) A description of the data source.
    """

    # ======== REQUIRED FIELDS ========
    author: str
    year: int
    data_source_tag: str
    problem: ProblemType
    data_source_type: DataSourceType

    # ======== DERIVED FIELDS ========
    data_source_id: str = field(init=False)

    # ======== OPTIONAL FIELDS ========
    experiment_type: Optional[ExperimentType] = None
    experiment_metadata: Optional[Dict[str, Any]] = None
    data_source_description: Optional[str] = None

    def __post_init__(self):
        self.data_source_id = f"{self.author}_{self.year}_{self.data_source_tag}"
        self.validate()
