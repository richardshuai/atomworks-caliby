set -e  # Exit on error

DATE=$(date +%Y-%m-%d)
echo "Building apptainer image for $DATE"

make apptainer

# TODO: Currently `bench` does not have permissiosn to write to `/projects/ml/RF2_allatom/spec_files`
mv datahub_$DATE.sif /net/scratch/smathis/workers
