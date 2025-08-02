"""
Tutorial: Uploading Bind/No-Bind Data to DataHub
================================================

This script demonstrates the recommended workflow for uploading new bind/no-bind data to the DataHub system.
It is intended as a tutorial for new users and as a test script for developers.

**Workflow Overview:**
1. Create a DataSource object describing your experiment or database.
2. Upload the DataSource to the database (must be done before adding measurements).
3. Create individual BindNoBindMeasurement objects for each measurement.
4. Save measurements to CIF files, which include all metadata.
5. Load measurements back from CIF files and inspect their contents.

**Key Points:**
- Every measurement must reference an existing DataSource via its `data_source_id`.
- All metadata is stored in the CIF file for reproducibility and downstream analysis. We also merge in the DataSource fields into the CIF file
    so you never have to reference the DataSource database directly at run time.
- The DataSource database is a CSV file (see DATA_SOURCE_DB_PATH in data_source_utils.py).

For more details, see the docstrings in the relevant modules.
"""

from cifutils.tools.inference import SequenceComponent

from datahub.databases.data_source_utils import get_data_source, get_data_source_db, upload_data_source
from datahub.databases.dataclasses import BindNoBindMeasurement, DataSource
from datahub.databases.enums import (
    BindingLabel,
    ConfidenceLabel,
    DataSourceType,
    ProblemType,
    StructureMethod,
    StructureType,
    TagType,
)
from datahub.databases.io_utils import load_measurement_from_cif, save_measurement_to_cif

# =============================
# 1. Create a DataSource object
# =============================

# Fill in the details for your experiment or database. The combination of author, year, and data_source_tag must be unique.
data_source = DataSource(
    author="Max Kazman",  # Your name or lab
    year=2025,  # Year of the experiment or data collection
    data_source_tag="test_dummy",  # Unique tag for this data source
    problem=ProblemType.PPI,  # Type of problem (e.g., protein-protein interaction)
    data_source_type=DataSourceType.EXPERIMENT_IN_HOUSE,  # Where the data came from
    # Optional fields:
    # experiment_type=ExperimentType.SPR,
    # experiment_metadata={"buffer": "PBS", "temp": 25},
    # data_source_description="My SPR experiment on VEGFR2"
)

# =====================================
# 2. Upload the DataSource to the database
# =====================================

# This must be done before you can add measurements referencing this data source.
# If you run this line twice with the same data_source_id, you'll get an error.
try:
    upload_data_source(data_source)
    print(f"DataSource '{data_source.data_source_id}' uploaded successfully.")
except ValueError as e:
    print(f"Warning: {e}")

# You can view the current DataSource database (a CSV file) as a pandas DataFrame:
data_source_db = get_data_source_db()
print("\nCurrent DataSource database:")
print(data_source_db)

# Retrieve the DataSource object from the database (to ensure it was saved correctly):
data_source = get_data_source(data_source.data_source_id)
print("\nLoaded DataSource from database:")
print(data_source)

# =====================================
# 3. Create a BindNoBindMeasurement object
# =====================================

# Prepare your structure. Here we use a dummy SequenceComponent, but in practice this could be a biotite AtomArray or a list of meaningful ChemicalComponent objects.
atom_array = [SequenceComponent("VQEVG")]
partners = [["H", "L"], ["T"]]  # Example: two binding partners

bind_no_bind_measurement = BindNoBindMeasurement(
    atom_array=atom_array,
    data_source_id=data_source.data_source_id,  # Link to your DataSource
    target="VEGFR2",
    binding_label=BindingLabel.BIND,
    label_confidence=ConfidenceLabel.CONFIDENT,
    partners=partners,
    # ============ OPTIONAL FIELDS ============
    target_chains=["H", "L"],
    binder_chains=["T"],
    fitness=0.35,
    affinity=0.5,
    affinity_std=0.1,
    label_threshold=10,
    pH=7.8,
    temperature=25.0,
    tag_type=TagType.BIOTIN,
    structure_method=StructureMethod.NMR,
    structure_type=StructureType.COMPUTATIONAL,
    measurement_description="This is a test measurement",
)

# =====================================
# 4. Save the measurement to a CIF file
# =====================================

cif_path = "/projects/ml/datahub/experimental_data/test_dummy.cif"  # Change this path as needed
save_measurement_to_cif(bind_no_bind_measurement, cif_path)
print(f"\nMeasurement saved to CIF file: {cif_path}")

# =====================================
# 5. Load the measurement back from the CIF file
# =====================================

loaded_measurement = load_measurement_from_cif(cif_path)
print("\nLoaded measurement from CIF file:")
print(loaded_measurement)

# =====================================
# 6. Inspect the output CIF file
# =====================================

print("""
---
You can open the output CIF file in a text editor.
Scroll to the bottom of the file and look for a block labeled 'database_fields'.
This block contains all the metadata and annotations you provided, including both the measurement and its associated DataSource fields.
This makes your data fully reproducible and self-describing!
---
""")

# =============================
# Additional Demonstrations
# =============================

# Show that the loaded measurement contains all the information, including the DataSource fields
print("\nMeasurement fields:")
for field, value in loaded_measurement.__dict__.items():
    print(f"{field}: {value}")
