APP=/projects/ml/RF2_allatom/spec_files/cifutils_20241028.sif
# The `-m` flag in `-mpytest` tells Python to run the `pytest` module as a script.
# It's equivalent to running `python -m pytest`.
# This ensures we're using the correct Python interpreter and environment.
PYTHONPATH=$APP python -m pytest tests --durations=10