echo "This script will only work if you are at the IPD."
echo "Running test for $1"
echo
echo "###############################################################################"
echo "#                                                                             #"
echo "#                               ⚠️  WARNING ⚠️                                  #"
echo "#                     RUNNING WITH DEBUGPY ON PORT 2345                       #"
echo "#                      DON'T FORGET TO ATTACH A DEBUGGER                      #"
echo "#                                                                             #"
echo "###############################################################################"
echo
source ./.ipd/setup.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 apptainer exec .ipd/datahub.sif python -m debugpy --listen 2345 --wait-for-client -m pytest "$1"