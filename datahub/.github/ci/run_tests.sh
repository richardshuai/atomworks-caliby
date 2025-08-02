set -e  # Exit on error

echo "Running from $PWD"

source ./.ipd/setup.sh

# Mount the `squash/af2_distillation_facebook` directory by cd-ing into it and back
original_dir=$(pwd)
cd /squash/af2_distillation_facebook
cd $original_dir

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Get max processes from environment variable, default to 24 if not set
N_CPU=${N_CPU:-24}
echo "Running tests with max $N_CPU CPUs"

# Ensure we can collect all tests (i.e. imports succeed)
echo "Testing imports by trying to collect all tests"
apptainer exec ./.ipd/datahub.sif pytest --collect-only tests/

# Clean previous coverage data
echo "Cleaning previous coverage data"
apptainer exec ./.ipd/datahub.sif coverage erase

# ... fast tests
echo "Running fast tests"
apptainer exec ./.ipd/datahub.sif pytest --cov=datahub --cov-append --cov-report= -n=auto --maxprocesses=$N_CPU --dist=worksteal tests/ 

# ... slow tests
echo "Running slow tests"
apptainer exec ./.ipd/datahub.sif pytest --cov=datahub --cov-append --cov-report= -n=auto --maxprocesses=$N_CPU --dist=worksteal tests/ -m "slow"

# Generate the combined reports at the end
echo "Generating combined reports"
apptainer exec ./.ipd/datahub.sif coverage report --fail-under=85
apptainer exec ./.ipd/datahub.sif coverage html
apptainer exec ./.ipd/datahub.sif coverage xml

# Print the total coverage
apptainer exec ./.ipd/datahub.sif coverage report | tail -n 1 | awk '{print "TOTAL", $NF}'