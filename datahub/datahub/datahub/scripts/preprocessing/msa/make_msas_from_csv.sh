#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem-per-cpu=64g
#SBATCH -a 1-1000
#SBATCH -t 0-12:00:00
#SBATCH -C DB
#SBATCH -c 1
#SBATCH -o slurm_logs/slurm_%A_%a.log
#SBATCH -e slurm_logs/slurm_%A_%a.err

echo "Starting task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}"

# Add rf2aa-dev to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/ncorley/projects/rf2aa-dev:/home/ncorley/projects/rf2aa-dev/datahub/"

# Print the path
echo "PYTHONPATH: ${PYTHONPATH}"

# Calculate task ID and number of tasks
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))
NUM_TASKS=$SLURM_ARRAY_TASK_COUNT

# Default values for the arguments
out_dir="/projects/msa/rf2aa_af3/2024_08_12"
csv_file="/projects/ml/RF2_allatom/data_preprocessing/msa/missing_protein_sequences_2024_08_12.csv"

echo "TASK_ID: ${TASK_ID}"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --out_dir) out_dir="$2"; shift ;;
        --csv_file) csv_file="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Make the directory to store the files, if it doesn't exist
echo "Making directory: ${out_dir}"
mkdir -p "${out_dir}"

# Print the parameters
echo "Saving files to: ${out_dir}"

# Get the directory of the current script
echo "Script directory: ${PWD}"

# Run the Python script
apptainer exec /projects/ml/RF2_allatom/spec_files/SE3nv_new_2024_08_18.sif python "${PWD}/make_msa.py" --out-dir "${out_dir}" --num-workers "${SLURM_CPUS_PER_TASK}" --csv-file "${csv_file}" --mem 32 --task-id "${TASK_ID}" --num-tasks "${NUM_TASKS}"

# Example usage:
# sbatch data/scripts/msa/make_msas_from_csv.sh --out_dir /path/to/output/directory --csv_file /path/to/csv/csv_file.csv
