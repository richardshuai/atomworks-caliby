set -e  # Exit on error

echo "Running from $PWD"

source ./.ipd/setup.sh

# Mount the `squash/af2_distillation_facebook` directory by cd-ing into it and back
original_dir=$(pwd)
cd /squash/af2_distillation_facebook
cd $original_dir

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

DATE=$(date +%Y-%m-%d)
echo "Building apptainer image for $DATE"

make apptainer

# TODO: Currently `bench` does not have permissions to write to `/projects/ml/RF2_allatom/spec_files`
mv datahub_$DATE.sif /net/scratch/smathis/workers
echo "Apptainer image is available at /net/scratch/smathis/workers/datahub_$DATE.sif"