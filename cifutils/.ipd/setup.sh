echo "Adding cifutils to your PYTHONPATH"
export PYTHONPATH=$PWD/src:$PYTHONPATH

# check if we are at the IPD
echo
echo "Verifying that you are on the IPD cluster"
IPD_FILE="/software/containers/versions/rf_diffusion_aa/ipd.txt"
if [ -f $IPD_FILE ]; then
    echo "... you are on the IPD cluster!"
else
    echo "ERROR! You appear not to be on the IPD cluster. The setup script is only available on the IPD cluster."
    exit 1
fi

# Ensure a .env file exists, otherwise create it
if [ ! -f .env ]; then
    echo "Creating a .env file"
    touch .env
fi

# Read the .env file and check if the CCD_MIRROR_PATH and PDB_MIRROR_PATH are already set
echo
echo "Adding CCD MIRROR to your environment variables"
if ! grep -q "CCD_MIRROR_PATH" .env; then
    echo "... setting CCD_MIRROR_PATH in your .env file"
    echo "" >> .env
    echo "CCD_MIRROR_PATH=$CCD_MIRROR_PATH" >> .env
else
    echo "... CCD_MIRROR_PATH already set in your .env file"
fi

echo
echo "Adding PDB MIRROR to your environment variables"
if ! grep -q "PDB_MIRROR_PATH" .env; then
    echo "... setting PDB_MIRROR_PATH in your .env file"
    echo "" >> .env
    echo "PDB_MIRROR_PATH=$PDB_MIRROR_PATH" >> .env
else
    echo "... PDB_MIRROR_PATH already set in your .env file"
fi

echo
echo "Your .env file is:"
echo "================================================"
cat .env
echo "================================================"
echo
echo "Done!"
