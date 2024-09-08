import os
import pickle
import re

try:
    import wandb
except ImportError:
    wandb = None

    import re


def _remove_special_characters(s: str) -> str:
    # Remove unwanted characters using regex
    clean_s = re.sub(r"[^a-zA-Z0-9_]", "", s)
    return f"{clean_s}"


def save_failed_example_to_disk(
    example_id: str,
    data: dict = {},
    rng_state_dict: dict = {},
    error_msg: str = "",
    fail_dir: str = "/net/scratch/failures",
):
    """
    Saves a failed example to disk as a pickle file.

    Args:
        - example_id (str): The ID of the example.
        - rng_state_dict (dict): The random number generator state dictionary.
        - error_msg (str): The error message associated with the failure.
        - fail_dir (str): The directory where the failed example should be saved. Defaults to a specific path.

    Returns:
        None
    """
    # Get wandb run ID if currently in a wandb run
    if wandb is not None and hasattr(wandb, "run") and wandb.run is not None:
        run_id = wandb.run.id
    else:
        run_id = "unknown"

    # Ensure the fail directory exists
    file_path = os.path.join(fail_dir, run_id, _remove_special_characters(example_id) + ".pkl")
    os.makedirs(os.path.dirname(file_path), exist_ok=True, mode=0o777)  # Allow everyone to read/write

    with open(file_path, "wb") as f:
        data = {
            "example_id": example_id,
            "rng_state_dict": rng_state_dict,
            "error_msg": error_msg,
            "wandb_run_id": run_id,
        } | data
        pickle.dump(data, f)
