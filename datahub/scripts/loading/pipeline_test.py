"""
This script tests the data loading pipeline for RF2AA (RoseTTAFold2 All-Atom).

It performs the following:
- Loads examples from the dataset
- Runs them through the full preprocessing pipeline
- Verifies that the output shapes are correct for input to RF2AA

Usage:
To run in main process, run for example:
python scripts/loading/pipeline_test.py /mnt/projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/pn_units_df.parquet 4 rf2aa
python scripts/loading/pipeline_test.py /mnt/projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/interfaces_df.parquet 4 rf2aa
python scripts/loading/pipeline_test.py /mnt/projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/pn_units_df.parquet 4 af3
python scripts/loading/pipeline_test.py /mnt/projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/interfaces_df.parquet 4 af3
To run silently in the background:
sbatch -c 4 --mem=32g --time=08:00:00 --output=test_pipeline_$(date +%Y%m%d).log \
    --wrap='export PYTHONPATH="${PYTHONPATH}:${PWD}"; python scripts/loading/pipeline_test.py /mnt/projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/interfaces_df.parquet 4 rf2aa'

# Or for AF3:
sbatch -c 4 --mem=32g --time=08:00:00 --output=test_pipeline_af3_$(date +%Y%m%d).log \
    --wrap='export PYTHONPATH="${PYTHONPATH}:${PWD}"; python scripts/loading/pipeline_test.py /mnt/projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/interfaces_df.parquet 4 af3'

Adjust paths and resource requirements as needed.
"""

import logging
import os
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Literal

import fire
import pandas as pd
from cifutils import CIFParser
from cifutils.constants import AF3_EXCLUDED_LIGANDS_REGEX
from toolz.curried import assoc, compose, keyfilter, map
from torch.utils.data import DataLoader
from tqdm import tqdm

from datahub.datasets.dataframe_parsers import InterfacesDFParser, PNUnitsDFParser
from datahub.datasets.pdb_dataset import PDBDataset
from datahub.preprocessing.constants import SUPPORTED_CHAIN_TYPES_INTS
from datahub.utils.rng import (
    capture_rng_states,
    create_rng_state_from_seeds,
    rng_state,
    serialize_rng_state_dict,
)
from tests.conftest import PROTEIN_MSA_DIRS, RNA_MSA_DIRS

logger = logging.getLogger(__name__)

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["RLIMIT_NPROC"] = "100"

_USER = os.getenv("USER")

_SHARED_COLUMNS_TO_LOAD = ["example_id", "pdb_id", "assembly_id", "method"]
PN_UNITS_COLUMNS_TO_LOAD = _SHARED_COLUMNS_TO_LOAD + [
    "q_pn_unit_iid",
    "q_pn_unit_type",
    "q_pn_unit_non_polymer_res_names",
]
INTERFACES_COLUMNS_TO_LOAD = _SHARED_COLUMNS_TO_LOAD + [
    "pn_unit_1_iid",
    "pn_unit_2_iid",
    "pn_unit_1_type",
    "pn_unit_2_type",
    "pn_unit_1_non_polymer_res_names",
    "pn_unit_2_non_polymer_res_names",
]

_SHARED_FILTERS = [
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
]

PN_UNITS_FILTERS = _SHARED_FILTERS + [
    f"q_pn_unit_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
]

INTERFACES_FILTERS = _SHARED_FILTERS + [
    f"pn_unit_1_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"pn_unit_2_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
    f"~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
]


def _get_rf2aa_dataset_from_path(dataset_path: Path) -> PDBDataset:
    from datahub.pipelines.rf2aa import build_rf2aa_transform_pipeline

    assert (
        "pn_unit" in dataset_path.name or "interface" in dataset_path.name
    ), "Dataset must be a pn_unit or interface dataset."

    pdb_dataset = PDBDataset(
        name="dataset",
        dataset_path=dataset_path,
        cif_parser=CIFParser(),
        filters=(PN_UNITS_FILTERS if "pn_unit" in dataset_path.name else INTERFACES_FILTERS),
        columns_to_load=PN_UNITS_COLUMNS_TO_LOAD if "pn_unit" in dataset_path.name else INTERFACES_COLUMNS_TO_LOAD,
        dataset_parser=PNUnitsDFParser() if "pn_unit" in dataset_path.name else InterfacesDFParser(),
        id_column="example_id",
        transform=build_rf2aa_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            n_recycles=5,
            crop_size=256,
            crop_contiguous_probability=1 / 3 if "pn_unit" in dataset_path.name else 0,
            crop_spatial_probability=2 / 3 if "pn_unit" in dataset_path.name else 1,
            assert_rf2aa_assumptions=True,
            convert_feats_to_rf2aa_input_tuple=False,
        ),
        save_failed_examples_to_dir=f"/net/scratch/{_USER}/failures/pipeline_test/rf2aa",
    )
    return pdb_dataset


def _get_af3_dataset_from_path(dataset_path: Path) -> PDBDataset:
    from datahub.pipelines.af3 import build_af3_transform_pipeline

    assert (
        "pn_unit" in dataset_path.name or "interface" in dataset_path.name
    ), "Dataset must be a pn_unit or interface dataset."

    pdb_dataset = PDBDataset(
        name="dataset",
        dataset_path=dataset_path,
        cif_parser=CIFParser(),
        filters=PN_UNITS_FILTERS if "pn_unit" in dataset_path.name else INTERFACES_FILTERS,
        columns_to_load=PN_UNITS_COLUMNS_TO_LOAD if "pn_unit" in dataset_path.name else INTERFACES_COLUMNS_TO_LOAD,
        dataset_parser=PNUnitsDFParser() if "pn_unit" in dataset_path.name else InterfacesDFParser(),
        id_column="example_id",
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
        ),
        save_failed_examples_to_dir=f"/net/scratch/{_USER}/failures/pipeline_test/af3",
    )
    return pdb_dataset


def test_dataset(dataset_path: Path, num_processes: int, pipeline_name: Literal["rf2aa", "af3"]):
    """
    Test the full dataset by running through the data loading pipeline and ensuring the examples come out with the
    correct shapes to plug into RF2AA.

    Args:
        - pdb_dataset: The dataset to test.
        - num_processes (int): Number of processes to use for parallel processing.
    """
    dataset_path = Path(dataset_path)

    match pipeline_name:
        case "rf2aa":
            pdb_dataset = _get_rf2aa_dataset_from_path(dataset_path)
        case "af3":
            pdb_dataset = _get_af3_dataset_from_path(dataset_path)
        case _:
            raise ValueError(f"Invalid pipeline name: {pipeline_name}. Must be 'rf2aa' or 'af3'.")

    today = date.today().strftime("%Y-%m-%d")
    logfile = Path(__file__).with_stem(f"{Path(__file__).stem}_{pipeline_name}_{today}").with_suffix(".log")
    print(f"Logfile: {logfile}")

    # Ensure the logger writes to the logfile
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.ERROR)
    logger.addHandler(file_handler)
    logfile.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    data_loader = DataLoader(
        pdb_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_processes,
    )
    data_iter = iter(data_loader)

    with open(logfile, "w") as f:
        f.write("=" * 80 + "\n")

    timings = []
    _filter_name_and_time = keyfilter(lambda x: x in ["name", "processing_time"])  # noqa
    _add_id = lambda id: assoc(key="example_id", value=id)  # noqa

    failure_indices = []
    processed_examples = 0
    with rng_state(create_rng_state_from_seeds(1, 1, 1)):
        for index in tqdm(range(len(pdb_dataset))):
            example_id = pdb_dataset.idx_to_id(index)
            rng_state_dict = capture_rng_states(include_cuda=False)
            processed_examples += 1
            try:
                data = next(data_iter)
                timings.extend(
                    list(map(compose(_add_id(data["example_id"]), _filter_name_and_time), data.__transform_history__))
                )
            except KeyboardInterrupt as e:
                raise e
            except StopIteration:
                logger.error(
                    f"StopIteration at index {index}/{len(pdb_dataset)} {example_id}. Time taken: {time.time() - start_time:.2f} seconds."
                )
                break
            except Exception as e:
                fail_msg = "=" * 80 + "\n"
                fail_msg += f"Failed at index {index} {example_id}: {e} \n{serialize_rng_state_dict(rng_state_dict)}\n"
                fail_msg += "\t" + traceback.format_exc().replace("\n", "\n\t")
                fail_msg += "=" * 80 + "\n"
                logger.error(fail_msg)
                failure_indices.append(index)

            if index % 100 == 0:
                # Save intermediate timings to disk
                timings_df = pd.DataFrame(timings)

                # ... ensure example_id is treated as a string (collated due to batching)
                timings_df["example_id"] = timings_df["example_id"].astype(str)

                timings_df.to_csv(logfile.with_suffix(".timings.csv"))
                avg_time = timings_df.groupby("example_id")["processing_time"].sum().mean()
                print(f"Current average processing time per example: {avg_time:.2f} seconds")
                print(f"Current failure rate: {100 * len(failure_indices) / processed_examples:.2f}%")

    # Save timings to disk
    if timings:  # Check if timings list is not empty before saving
        pd.DataFrame(timings).to_csv(logfile.with_suffix(".timings.csv"))  # Updated suffix to be valid

    # record success and time taken in logfile
    with open(logfile, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Success: {True}\n")
        f.write(f"Time taken: {time.time() - start_time:.2f} seconds\n")
        f.write("=" * 80 + "\n")

    print(
        f"Completed {len(pdb_dataset)} examples in {time.time() - start_time:.2f} seconds on {num_processes} processes with {len(failure_indices)} failures."
    )
    print(f"Failed indices: {failure_indices}")


if __name__ == "__main__":
    fire.Fire(test_dataset)
