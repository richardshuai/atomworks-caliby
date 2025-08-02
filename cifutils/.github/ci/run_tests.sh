set -e  # Exit on error

echo "Running from $PWD"

source ./.ipd/setup.sh

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Get max processes from environment variable, default to 24 if not set
N_CPU=${N_CPU:-24}
echo "Running tests with max $N_CPU CPUs"

# Ensure we can collect all tests (i.e. imports succeed)
echo "Testing imports by trying to collect all tests"
apptainer exec ./.ipd/cifutils.sif pytest -m "not benchmark" --collect-only tests/

# Run the tests in coverage mode
apptainer exec ./.ipd/cifutils.sif pytest -m "not benchmark" --cov=cifutils --cov-report=xml --cov-report=term-missing -n=auto --maxprocesses=$N_CPU --dist=worksteal tests/

# Require at least 85% coverage
apptainer exec ./.ipd/cifutils.sif coverage report --fail-under=80

# Output the coverage in a format GitLab can parse
apptainer exec ./.ipd/cifutils.sif coverage report | tail -n 1 | awk '{print "TOTAL", $NF}'
