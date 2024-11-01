set -e  # Exit on error

DATE=$(date +%Y-%m-%d)
echo "Building apptainer image for $DATE"

make apptainer

# TODO: Currently `bench` does not have permissions to write to `/projects/ml/RF2_allatom/spec_files`
mv cifutils_$DATE.sif /net/scratch/smathis/workers
echo "Apptainer image is available at /net/scratch/smathis/workers/cifutils_$DATE.sif"