set -e  # Exit on error

# Ensure correct lint
PYTHONPATH=$PWD:$PYTHONPATH /projects/ml/RF2_allatom/spec_files/cifutils_latest.sif ruff check .
