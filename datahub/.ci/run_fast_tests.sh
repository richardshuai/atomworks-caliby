PYTHONPATH=$PWD:$PYTHONPATH /projects/ml/RF2_allatom/spec_files/datahub_latest.sif pytest --cov=datahub --cov-report=term-missing tests/
# Output the coverage in a format GitLab can parse
PYTHONPATH=$PWD:$PYTHONPATH /projects/ml/RF2_allatom/spec_files/datahub_latest.sif coverage report | tail -n 1 | awk '{print "TOTAL", $NF}'