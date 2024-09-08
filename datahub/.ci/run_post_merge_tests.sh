set -e  # Exit on error

# Ensure we can collect all tests (i.e. imports succeed)
PYTHONPATH=$PWD:$PYTHONPATH /projects/ml/RF2_allatom/spec_files/datahub_latest.sif pytest --collect-only tests/

# Run the tests in coverage mode
PYTHONPATH=$PWD:$PYTHONPATH /projects/ml/RF2_allatom/spec_files/datahub_latest.sif pytest --cov=datahub --cov-report=term-missing -m "not very_slow" tests/

# Output the coverage in a format GitLab can parse
PYTHONPATH=$PWD:$PYTHONPATH /projects/ml/RF2_allatom/spec_files/datahub_latest.sif coverage report | tail -n 1 | awk '{print "TOTAL", $NF}'
