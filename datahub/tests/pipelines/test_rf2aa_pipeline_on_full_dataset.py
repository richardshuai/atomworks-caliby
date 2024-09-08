"""
Includes tests to:
- Run through the data loading pipeline and ensure the examples come out with the
  correct shapes to plug into RF2AA

To run silently in the background, use (adapt the paths as necessary):
sbatch -c 12 --mem=128g --time=16:00:00 --output=job_output.log --wrap='export PYTHONPATH="${PYTHONPATH}:/home/{USER}/code/RF2-allatom"; pytest /home/smathis/code/RF2-allatom/rf2aa/data_new/tests/pipeline/test_rf2aa_pipeline_on_full_dataset.py -s -m very_slow'
"""

import logging
import os
import time
import traceback
from pathlib import Path

import pytest
from torch.utils.data import DataLoader
from tqdm import tqdm

from datahub.transforms.rf2aa_assumptions import assert_satisfies_rf2aa_assumptions
from datahub.utils.debug import save_failed_example_to_disk
from datahub.utils.rng import (
    capture_rng_states,
    create_rng_state_from_seeds,
    rng_state,
    serialize_rng_state_dict,
)
from tests.datasets.conftest import RF2AA_PDB_DATASET

logger = logging.getLogger(__name__)
logfile = Path(__file__).with_suffix(".log")

# Ensure the logger writes to the logfile
file_handler = logging.FileHandler(logfile)
file_handler.setLevel(logging.ERROR)
logger.addHandler(file_handler)

# Set OPENBLAS_NUM_THREADS to 4 to avoid using all cores
# (needed to avoid jojo crashing)
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["RLIMIT_NPROC"] = "100"


@pytest.mark.very_slow
def test_full_dataset(pdb_dataset=RF2AA_PDB_DATASET, num_processes=5):
    """
    Test the full dataset by running through the data loading pipeline and ensuring the examples come out with the
    correct shapes to plug into RF2AA.

    Args:
        - pdb_dataset: The dataset to test.
        - num_processes (int): Number of processes to use for parallel processing.
    """
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

    failure_indices = []
    with rng_state(create_rng_state_from_seeds(1, 1, 1)):
        for index in tqdm(range(len(pdb_dataset))):
            example_id = pdb_dataset.idx_to_id(index)
            rng_state_dict = capture_rng_states(include_cuda=False)
            try:
                data = next(data_iter)
                assert (
                    data["example_id"][0] == example_id
                ), f"Example ID mismatch: {data['example_id'][0]} != {example_id}."
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
                rng_state_dict = e.rng_state_dict if hasattr(e, "rng_state_dict") else rng_state_dict
                save_failed_example_to_disk(
                    example_id,
                    data={},  # Do not save data since it is memory intensive & can be recreated from the RNG state
                    rng_state_dict=rng_state_dict,
                    error_msg=str(e) + "\n" + traceback.format_exc(),
                )
                failure_indices.append(index)

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
    pytest.main(["-s", "-v", "-m", "very_slow", __file__])
