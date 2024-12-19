#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem-per-cpu=64g
#SBATCH -a 1-200
#SBATCH -t 0-12:00:00
#SBATCH -c 1
#SBATCH -o slurm_logs/slurm_%A_%a.log
#SBATCH -e slurm_logs/slurm_%A_%a.err

# Calculate task ID and number of tasks
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))
NUM_TASKS=$SLURM_ARRAY_TASK_COUNT

# Set PYTHONPATH to the base project directory
PROJECT_ROOT=~/projects/datahub # +------- Change this to the base project directory -------+
export PYTHONPATH=${PROJECT_ROOT}
echo "PYTHONPATH: ${PYTHONPATH}"

# Set the CCD path environment variables (we will pass PDB paths directly from `base_dir`)
# WARNING: Ensure that `cifutils` is using the correct CCD path!
export CCD_MIRROR_PATH="${PROJECT_ROOT}/.ipd/ccd"

# Default values for the arguments
out_dir="/projects/ml/datahub/dfs/parsing_outputs/2024_12_01_pdb/" # +------- Change this to the output directory -------+
base_dir="/projects/ml/frozen_pdb_copies/2024_12_01_pdb" # +------- Change this to the base directory -------+

from_rcsb=True # We are parsing from the PDB...
file_extension=".cif.gz" # ...with the file extension ".cif.gz"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --out_dir) out_dir="$2"; shift ;;
        --base_dir) base_dir="$2"; shift ;;
        --from_rcsb) from_rcsb=$2; shift ;;
        --file_extension) file_extension="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Make the directory to store the files, if it doesn't exist
mkdir -p "${out_dir}"

# Create a README file if it doesn't exist, and add timestamp, PDB directory path, and base_dir
readme_file="${out_dir}/README"
if [ ! -f "${readme_file}" ]; then
    echo "Creating README file at ${readme_file}"
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    {
        echo "Timestamp: ${timestamp}"
        echo "Output Directory Path: ${out_dir}"
        echo "Base Directory Path: ${base_dir}"
        echo "CCD Path: ${CCD_MIRROR_PATH}"
    } >> "${readme_file}"
fi

# Print the parameters
echo "Loading files from: ${base_dir}"
echo "Saving files to: ${out_dir}"
echo "CCD path: ${CCD_MIRROR_PATH}"
echo "Base directory: ${base_dir}"

### Run the Python script
command="apptainer exec ${PROJECT_ROOT}/.ipd/datahub.sif python get_csvs_from_structures.py \
  --task_id ${TASK_ID} \
  --num_tasks ${NUM_TASKS} \
  --out_dir '${out_dir}' \
  --from_rcsb '${from_rcsb}' \
  --file_extension '${file_extension}' \
  --selection 'all' \
  --base_dir '${base_dir}'"
echo -e "command\t$command"
eval $command

# Example usage:
# sbatch data/scripts/get_csvs_from_structures.sh --out_dir /path/to/output/directory --base_dir /path/to/base/directory
