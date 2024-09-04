"""
Includes tests to:
- Run through the data loading pipeline and ensure the examples come out with the
  correct shapes to plug into RF2AA

To run silently in the background, use (adapt the paths as necessary):
sbatch -c 12 --mem=128g --time=16:00:00 --output=job_output.log --wrap='export PYTHONPATH="${PYTHONPATH}:/home/{USER}/code/RF2-allatom"; pytest /home/smathis/code/RF2-allatom/rf2aa/data_new/tests/pipeline/test_rf2aa_pipeline_on_full_dataset.py -s -m very_slow'
"""

import datetime
import logging
import os
import time
import traceback
from pathlib import Path

import fire
import pandas as pd
from cifutils import CIFParser
from toolz.curried import assoc, compose, keyfilter, map
from torch.utils.data import DataLoader
from tqdm import tqdm

from datahub.datasets.dataframe_parsers import InterfacesDFParser, PNUnitsDFParser
from datahub.datasets.pdb_dataset import PDBDataset
from datahub.pipelines.rf2aa import assert_satisfies_rf2aa_assumptions, build_rf2aa_transform_pipeline
from datahub.utils.debug import save_failed_example_to_disk
from datahub.utils.rng import (
    capture_rng_states,
    create_rng_state_from_seeds,
    rng_state,
    serialize_rng_state_dict,
)
from tests.conftest import PROTEIN_MSA_DIRS, RNA_MSA_DIRS

logger = logging.getLogger(__name__)
logfile = Path(__file__).with_suffix(".log")

# Ensure the logger writes to the logfile
file_handler = logging.FileHandler(logfile)
file_handler.setLevel(logging.ERROR)
logger.addHandler(file_handler)

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["RLIMIT_NPROC"] = "100"


def _get_dataset_from_path(dataset_path: Path) -> PDBDataset:
    assert (
        "pn_unit" in dataset_path.name or "interface" in dataset_path.name
    ), "Dataset must be a pn_unit or interface dataset."

    pdb_dataset = PDBDataset(
        name="dataset",
        dataset_path=dataset_path,
        cif_parser=CIFParser(),
        filters=None,
        dataset_parser=PNUnitsDFParser() if "pn_unit" in dataset_path.name else InterfacesDFParser(),
        id_column="example_id",
        transform=build_rf2aa_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            n_recycles=5,
            crop_size=256,
            crop_contiguous_probability=1 / 3 if "pn_unit" in dataset_path.name else 0,
            crop_spatial_probability=2 / 3 if "pn_unit" in dataset_path.name else 1,
        ),
        unpack_data_dict=False,
    )
    return pdb_dataset


def test_dataset(dataset_path: Path, num_processes: int):
    """
    Test the full dataset by running through the data loading pipeline and ensuring the examples come out with the
    correct shapes to plug into RF2AA.

    Args:
        - pdb_dataset: The dataset to test.
        - num_processes (int): Number of processes to use for parallel processing.
    """
    dataset_path = Path(dataset_path)
    pdb_dataset = _get_dataset_from_path(dataset_path)

    start_time = time.time()
    logfile.parent.mkdir(parents=True, exist_ok=True)

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
    with rng_state(create_rng_state_from_seeds(1, 1, 1)):
        for index in tqdm(range(len(pdb_dataset))):
            example_id = pdb_dataset.idx_to_id(index)
            rng_state_dict = capture_rng_states(include_cuda=False)
            try:
                data = next(data_iter)
                timings.extend(
                    list(map(compose(_add_id(data["example_id"]), _filter_name_and_time), data.__transform_history__))
                )
                assert_satisfies_rf2aa_assumptions(data)
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
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                save_failed_example_to_disk(
                    example_id,
                    data={},  # Do not save data since it is memory intensive & can be recreated from the RNG state
                    rng_state_dict=rng_state_dict,
                    error_msg=str(e) + "\n" + traceback.format_exc(),
                    fail_dir=f"/net/scratch/failures/{dataset_path.stem}-{current_date}",
                )
                failure_indices.append(index)

            if index % 100 == 0:
                # Save intermediate timings to disk
                pd.DataFrame(timings).to_csv(logfile.with_suffix(".timings.csv"))

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
    assert len(failure_indices) == 0, f"Failed indices: {failure_indices}"


if __name__ == "__main__":
    fire.Fire(test_dataset)
