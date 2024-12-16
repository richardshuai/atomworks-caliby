set -e  # Exit on error

# cd to the cifutils directory
echo "Running from $PWD"

# Source IPD setup script
source $PWD/.ipd/setup.sh

# Ensure we can collect all tests (i.e. imports succeed)
$PWD/.ipd/cifutils_latest.sif pytest --collect-only tests/

# Run the tests in coverage mode
$PWD/.ipd/cifutils_latest.sif pytest --cov=cifutils --cov-report=xml --cov-report=term-missing tests/

# Output the coverage in a format GitLab can parse
$PWD/.ipd/cifutils_latest.sif coverage report | tail -n 1 | awk '{print "TOTAL", $NF}'
