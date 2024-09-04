/projects/ml/RF2_allatom/spec_files/cifutils_latest.sif pytest --cov=cifutils --cov-report=term-missing tests/
# Output the coverage in a format GitLab can parse
/projects/ml/RF2_allatom/spec_files/cifutils_latest.sif coverage report | tail -n 1 | awk '{print "TOTAL", $NF}'